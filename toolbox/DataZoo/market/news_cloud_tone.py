"""Model tone for the news cloud: Twitter-RoBERTa
(cardiffnlp/twitter-roberta-base-sentiment-latest, a RoBERTa
fine-tuned on general social/news text) run through its
ONNX export with onnxruntime — no torch anywhere in the
stack.

`model_headline_tones(titles)` returns P(positive) -
P(negative) per headline, in [-1, 1], or None when the
model is unavailable (deps missing, download failed,
inference error): the caller falls back to the lexicon
scorer, so the cloud's colors never depend on the model
being reachable.

The weights are the int8-quantized ONNX export from the
Xenova/twitter-roberta-base-sentiment-latest mirror
(~126 MB), downloaded once into
~/.cache/toolbox/twitter-roberta-sentiment/ — the same
cache the briefs' news_sentiment adapter shares, so the
weights exist on disk exactly once. Set
NEWS_CLOUD_TONE_MODEL=off to disable the model entirely.
Labels follow this model's config: 0 negative, 1 neutral,
2 positive. (FinBERT's order — 0 positive, 1 negative,
2 neutral — is DIFFERENT; the indices below are this
model's own.)

History: until 2026-11-24 the scorer was FinBERT
(ProsusAI/finbert, financial text). A 2026-11-24 benchmark
on 39 brief-style items scored Twitter-RoBERTa 84.6%
against FinBERT's 69.2% (it caught all 17 negative
stories; FinBERT let four fade into the neutral band),
and the user directed the swap the same day — first for
the briefs' News map, then here: "Use this new model in
the streamlit app too."
"""
import os
import re
import urllib.request
from pathlib import Path

_BASE = ("https://huggingface.co/Xenova/"
         "twitter-roberta-base-sentiment-latest/resolve/main/")
_FILES = {"tokenizer.json": "tokenizer.json",
          "twitter_roberta_int8.onnx": "onnx/model_int8.onnx"}
_BATCH = 32
_state = None  # None = untried, False = unavailable

# Process-wide tone cache (speed pass 2026-11-05): a
# headline's model tone never changes, and consecutive
# cloud builds overlap heavily — the same stories sit in
# the feeds for hours — so scored tones are kept and only
# new headlines reach the model. Keyed by the normalized
# title, the same form the cloud dedupes on. Bounded:
# the oldest entries drop past _TONE_CACHE_MAX.
_TONE_CACHE: dict = {}
_TONE_CACHE_MAX = 20_000


def _tone_key(title):
    return re.sub(r"\W+", " ", title.lower()).strip()


def _cache_dir():
    d = Path.home() / ".cache" / "toolbox" / "twitter-roberta-sentiment"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load():
    """(session, tokenizer) or None. Memoized, including the
    failure: a container that cannot reach the model does not
    re-try the download on every build."""
    global _state
    if _state is not None:
        return _state or None
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = _cache_dir()
        for local, remote in _FILES.items():
            dst = d / local
            if not dst.exists():
                tmp = dst.with_suffix(".part")
                urllib.request.urlretrieve(_BASE + remote, tmp)
                tmp.rename(dst)
        sess = ort.InferenceSession(str(d / "twitter_roberta_int8.onnx"),
                                    providers=["CPUExecutionProvider"])
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        # Headlines never approach the model's 512-token
        # window, but the briefs' adapter passes headline +
        # summary strings through this same function —
        # truncation keeps an over-long one from erroring
        # the whole call into the None fallback.
        tok.enable_truncation(max_length=512)
        _state = (sess, tok)
    except Exception:  # noqa: BLE001 - memoized unavailable-on-any-failure contract (native load/download failures are open-ended)
        _state = False
    return _state or None


def model_headline_tones(titles, progress=None):
    """Model tone per headline: P(positive) - P(negative)
    under Twitter-RoBERTa (labels 0 negative, 1 neutral,
    2 positive). None when the model is disabled or
    unavailable.

    ``progress`` (optional) is called as
    ``progress(done, total, label)`` — once with ``done == 0``
    (labelled with the model load when this call has to load
    it: the cold-container download is the longest silent
    stretch in the cloud build), then after each inference
    batch. Tones are cached process-wide (see _TONE_CACHE):
    cached headlines resolve without the model, the first
    report then starts at the cached count, and a fully
    cached call reports nothing. The default ``None``
    reports nothing."""
    if os.environ.get("NEWS_CLOUD_TONE_MODEL", "").lower() == "off":
        return None
    keys = [_tone_key(t) for t in titles]
    out: list = [None] * len(titles)
    miss_idx = []
    for i, k in enumerate(keys):
        v = _TONE_CACHE.get(k)
        if v is None:
            miss_idx.append(i)
        else:
            out[i] = v
    if not miss_idx:
        return out
    if progress is not None and _state is None:
        progress(0, len(titles), "loading the tone model (first run)")
    loaded = _load()
    if loaded is None:
        return None
    if progress is not None:
        progress(len(titles) - len(miss_idx), len(titles), "")
    sess, tok = loaded
    import numpy as np
    names = {i.name for i in sess.get_inputs()}
    resolved = len(titles) - len(miss_idx)
    try:
        for j in range(0, len(miss_idx), _BATCH):
            batch_idx = miss_idx[j:j + _BATCH]
            encs = [tok.encode(titles[i]) for i in batch_idx]
            mx = max(len(e.ids) for e in encs)
            ids = np.array(
                [e.ids + [0] * (mx - len(e.ids)) for e in encs],
                dtype=np.int64)
            att = np.array(
                [e.attention_mask + [0] * (mx - len(e.attention_mask))
                 for e in encs], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": att}
            if "token_type_ids" in names:
                feed["token_type_ids"] = np.zeros_like(ids)
            logits = sess.run(None, feed)[0]
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            for i2, v in zip(batch_idx,
                             (p[:, 2] - p[:, 0]).tolist(), strict=True):
                out[i2] = v
                _TONE_CACHE[keys[i2]] = v
            resolved += len(batch_idx)
            if progress is not None:
                progress(resolved, len(titles), "")
    except Exception:  # noqa: BLE001 - any inference failure returns None; caller falls back to the lexicon scorer
        return None
    while len(_TONE_CACHE) > _TONE_CACHE_MAX:
        _TONE_CACHE.pop(next(iter(_TONE_CACHE)))
    return out
