"""Model tone for the news cloud: FinBERT (ProsusAI/finbert,
a BERT fine-tuned on financial text) run through its ONNX
export with onnxruntime — no torch anywhere in the stack.

`model_headline_tones(titles)` returns P(positive) -
P(negative) per headline, in [-1, 1], or None when the model
is unavailable (deps missing, download failed, inference
error): the caller falls back to the lexicon scorer, so the
cloud's colors never depend on the model being reachable.

The weights are the int8-quantized ONNX export from the
Xenova/finbert mirror of ProsusAI/finbert (~110 MB),
downloaded once into ~/.cache/toolbox/finbert/. Set
NEWS_CLOUD_TONE_MODEL=off to disable the model entirely.
Labels follow the FinBERT config: 0 positive, 1 negative,
2 neutral.
"""
import os
import urllib.request
from pathlib import Path

_BASE = "https://huggingface.co/Xenova/finbert/resolve/main/"
_FILES = {"tokenizer.json": "tokenizer.json",
          "finbert_int8.onnx": "onnx/model_int8.onnx"}
_BATCH = 32
_state = None  # None = untried, False = unavailable


def _cache_dir():
    d = Path.home() / ".cache" / "toolbox" / "finbert"
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
        sess = ort.InferenceSession(str(d / "finbert_int8.onnx"),
                                    providers=["CPUExecutionProvider"])
        _state = (sess, Tokenizer.from_file(str(d / "tokenizer.json")))
    except Exception:
        _state = False
    return _state or None


def model_headline_tones(titles, progress=None):
    """FinBERT tone per headline: P(positive) - P(negative).
    None when the model is disabled or unavailable.

    ``progress`` (optional) is called as
    ``progress(done, total, label)`` — once with ``done == 0``
    (labelled with the model load when this call has to load
    it: the cold-container download is the longest silent
    stretch in the cloud build), then after each inference
    batch. The default ``None`` reports nothing."""
    if os.environ.get("NEWS_CLOUD_TONE_MODEL", "").lower() == "off":
        return None
    if progress is not None and _state is None:
        progress(0, len(titles), "loading the tone model (first run)")
    loaded = _load()
    if loaded is None:
        return None
    if progress is not None:
        progress(0, len(titles), "")
    sess, tok = loaded
    import numpy as np
    names = {i.name for i in sess.get_inputs()}
    out = []
    try:
        for i in range(0, len(titles), _BATCH):
            encs = [tok.encode(t) for t in titles[i:i + _BATCH]]
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
            out.extend((p[:, 0] - p[:, 1]).tolist())
            if progress is not None:
                progress(len(out), len(titles), "")
    except Exception:
        return None
    return out
