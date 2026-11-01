"""Latest 13F-HR holdings from SEC EDGAR, mapped to Yahoo tickers.

SEC fair-access rules require every request to carry a declared
User-Agent with a contact, so there is deliberately no built-in
default: callers pass their own ``user_agent`` (for example
``"My App admin@example.com"``). The investor registry (section name
to ``{"name": ..., "cik": ...}``) and the on-disk cache directory are
likewise caller-supplied; the cache defaults to a ``toolbox_13f``
folder in the system temp directory.
"""

from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir

import requests

logger = logging.getLogger(__name__)

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
TOP_HOLDINGS = 30
_SEC_SLEEP = 0.2
_OPENFIGI_SLEEP = 2.5
_US_EXCHANGES = {"US", "UN", "UW", "UA", "N", "UQ", "UJ", "UV"}
_SKIP_SEC_TYPES = {"Option", "Warrant", "Right", "Unit", "Preferred Stock"}
_EQUITY_TYPES = {"Common Stock", "ETP", "REIT", "ADR", "MLP"}


def sec_headers(user_agent: str) -> dict[str, str]:
    """SEC request headers carrying the caller's declared User-Agent."""
    return {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}


def _cache_dir(cache_dir: Path | None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else Path(gettempdir()) / "toolbox_13f"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _sec_get(url: str, user_agent: str) -> requests.Response:
    time.sleep(_SEC_SLEEP)
    response = requests.get(url, headers=sec_headers(user_agent), timeout=30)
    response.raise_for_status()
    return response


def load_13f_investors(source) -> dict[str, dict]:
    """Investor registry from a mapping or a YAML file path.

    Keeps only sections that are mappings carrying a ``cik``. Reading a
    YAML file needs PyYAML, imported lazily.
    """
    if isinstance(source, (str, Path)):
        import yaml

        raw = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
    else:
        raw = source or {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict) and v.get("cik")}


def _latest_13f(cik: str, user_agent: str) -> dict:
    padded = str(int(cik)).zfill(10)
    subs = _sec_get(
        f"https://data.sec.gov/submissions/CIK{padded}.json", user_agent
    ).json()
    recent = subs.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    for i, form in enumerate(forms):
        if form in ("13F-HR", "13F-HR/A"):
            report = recent["reportDate"][i] if "reportDate" in recent else ""
            return {
                "cik": str(int(cik)),
                "accession": recent["accessionNumber"][i],
                "filing_date": recent.get("filingDate", [""])[i],
                "report_period": report,
            }
    raise ValueError(f"No 13F-HR filing for CIK {cik}")


def _holdings_xml_url(cik: str, accession: str, user_agent: str) -> str:
    acc = accession.replace("-", "")
    idx = _sec_get(
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/index.json",
        user_agent,
    ).json()
    items = idx.get("directory", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    xml_names = [
        item["name"]
        for item in items
        if str(item.get("name", "")).lower().endswith(".xml")
        and "primary" not in str(item.get("name", "")).lower()
        and "xsl" not in str(item.get("name", "")).lower()
    ]
    if not xml_names:
        raise ValueError(f"No holdings XML for {accession}")
    name = next(
        (n for n in xml_names if re.search(r"info|holding|13f", n, re.I)),
        xml_names[0],
    )
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{name}"


def _parse_holdings_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    rows: list[dict] = []
    for node in root.iter():
        if _local(node.tag) != "infoTable":
            continue
        fields: dict[str, str] = {}
        for child in node.iter():
            fields[_local(child.tag)] = (child.text or "").strip()
        if fields.get("putCall", "").upper() in {"PUT", "CALL"}:
            continue
        title = fields.get("titleOfClass", "").upper()
        if "PUT" in title or "CALL" in title:
            continue
        try:
            value = int(float(fields.get("value", "0") or 0))
            shares = int(float(fields.get("sshPrnamt", "0") or 0))
        except ValueError:
            continue
        cusip = fields.get("cusip", "").strip().upper()
        if not cusip or value <= 0:
            continue
        rows.append(
            {
                "issuer": fields.get("nameOfIssuer", ""),
                "cusip": cusip,
                "value": value,
                "shares": shares,
            }
        )
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    by_cusip: dict[str, dict] = {}
    for row in rows:
        item = by_cusip.setdefault(
            row["cusip"],
            {"cusip": row["cusip"], "issuer": row["issuer"], "value": 0, "shares": 0},
        )
        item["value"] += row["value"]
        item["shares"] += row["shares"]
    return sorted(by_cusip.values(), key=lambda r: r["value"], reverse=True)


def _cached_holdings(
    cik: str, user_agent: str, cache_dir: Path | None
) -> tuple[list[dict], dict]:
    meta = _latest_13f(cik, user_agent)
    path = _cache_dir(cache_dir) / f"holdings_{meta['cik']}.json"
    cached = _load_json(path, {})
    if cached.get("accession") == meta["accession"] and cached.get("rows"):
        return cached["rows"], meta
    xml_text = _sec_get(
        _holdings_xml_url(cik, meta["accession"], user_agent), user_agent
    ).text
    rows = _aggregate(_parse_holdings_xml(xml_text))
    if rows:
        _save_json(path, {**meta, "rows": rows})
    return rows, meta


def format_13f_asof_note(meta: dict | None) -> str:
    """Caption: holdings quarter-end and SEC filing date."""
    if not meta:
        return ""
    period = _pretty_sec_date(meta.get("report_period") or "")
    filed = _pretty_sec_date(meta.get("filing_date") or "")
    if period and filed:
        return f"Holdings as of {period} · filed {filed}"
    if period:
        return f"Holdings as of {period}"
    if filed:
        return f"Filed {filed}"
    return ""


def _pretty_sec_date(value: str) -> str:
    raw = str(value).strip()
    if not raw:
        return ""
    compact = raw.replace("-", "")[:8]
    try:
        return datetime.strptime(compact, "%Y%m%d").strftime("%b %d, %Y")
    except ValueError:
        return raw[:10]


def _yahoo_ticker(raw: str) -> str:
    ticker = raw.strip().upper().replace("/", "-").replace(".", "-")
    return re.sub(r"[^A-Z0-9-]", "", ticker)


def _pick_figi_ticker(records: list[dict]) -> str | None:
    scored: list[tuple[int, int, str]] = []
    for rec in records:
        ticker = rec.get("ticker")
        if not ticker:
            continue
        market = rec.get("marketSector") or ""
        if market and market != "Equity":
            continue
        sec_type = rec.get("securityType") or rec.get("securityType2") or ""
        if sec_type in _SKIP_SEC_TYPES:
            continue
        exch = rec.get("exchCode") or ""
        yahoo = _yahoo_ticker(ticker)
        if not yahoo:
            continue
        score = 0
        if exch == "US":
            score += 6
        elif exch in _US_EXCHANGES:
            score += 4
        if sec_type in _EQUITY_TYPES:
            score += 3
        if re.fullmatch(r"[A-Z]{1,5}", yahoo):
            score += 2
        score -= yahoo.count("-")
        scored.append((score, len(yahoo), yahoo))
    if not scored:
        return None
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    return scored[0][2]


def _openfigi_map(id_type: str, values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i in range(0, len(values), 10):
        batch = values[i : i + 10]
        jobs = [{"idType": id_type, "idValue": c} for c in batch]
        time.sleep(_OPENFIGI_SLEEP)
        try:
            resp = requests.post(
                OPENFIGI_URL,
                json=jobs,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if resp.status_code == 429:
                time.sleep(30)
                resp = requests.post(
                    OPENFIGI_URL,
                    json=jobs,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
            resp.raise_for_status()
            results = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("OpenFIGI %s map failed: %s", id_type, exc)
            continue
        for cusip, item in zip(batch, results):
            records = item.get("data") if isinstance(item, dict) else None
            out[cusip] = _pick_figi_ticker(records or []) or ""
    return out


def _map_cusips(cusips: list[str], cache_dir: Path | None) -> dict[str, str]:
    path = _cache_dir(cache_dir) / "cusip_tickers.json"
    mapping: dict[str, str] = _load_json(path, {})
    missing = [c for c in cusips if c not in mapping]
    if missing:
        mapping.update(_openfigi_map("ID_CUSIP", missing))
    still = [c for c in cusips if not mapping.get(c)]
    if still:
        mapping.update(_openfigi_map("ID_CINS", still))
    if missing or still:
        _save_json(path, mapping)
    return mapping


def load_13f_lens(
    investors,
    user_agent: str,
    top_n: int = TOP_HOLDINGS,
    cache_dir: Path | None = None,
) -> tuple[
    dict[str, list[str]],
    dict[str, str],
    dict[str, dict[str, float]],
    dict[str, dict],
]:
    """Tickers, labels, weights (% of filing), and EDGAR filing meta per investor.

    ``investors`` is the registry mapping (or a YAML path) accepted by
    :func:`load_13f_investors`; ``user_agent`` is the caller's declared
    SEC User-Agent.
    """
    sections: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    weights: dict[str, dict[str, float]] = {}
    asof: dict[str, dict] = {}
    for section, spec in load_13f_investors(investors).items():
        title = str(spec.get("name") or section).split(" (")[0].strip()
        try:
            rows, meta = _cached_holdings(str(spec["cik"]), user_agent, cache_dir)
        except (requests.RequestException, ValueError, OSError, ET.ParseError) as exc:
            logger.warning("13F %s: %s", title, exc)
            continue
        total = sum(row["value"] for row in rows) or 1
        candidates = rows[: max(top_n * 2, top_n)]
        mapping = _map_cusips([row["cusip"] for row in candidates], cache_dir)
        tickers: list[str] = []
        section_weights: dict[str, float] = {}
        for row in candidates:
            ticker = mapping.get(row["cusip"]) or ""
            if not ticker or ticker in tickers:
                continue
            tickers.append(ticker)
            section_weights[ticker] = 100.0 * row["value"] / total
            if row.get("issuer") and ticker not in labels:
                labels[ticker] = str(row["issuer"]).title()
            if len(tickers) >= top_n:
                break
        if tickers:
            sections[title] = tickers
            weights[title] = section_weights
            asof[title] = meta
        else:
            logger.warning("13F %s: no Yahoo tickers mapped", title)
    return sections, labels, weights, asof
