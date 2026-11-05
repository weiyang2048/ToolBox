"""FinBERT tone cache (speed pass 2026-11-05): scored tones
are kept process-wide; repeat and overlapping calls score
only the headlines the cache has never seen."""
import numpy as np

from toolbox.DataZoo.market import news_cloud_tone as nt


class _Enc:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1] * len(ids)


class _Tok:
    def encode(self, text):
        return _Enc([len(text) % 7 + 2, 3, 4])


class _Input:
    name = "input_ids"


class _Sess:
    def __init__(self):
        self.batches = []

    def get_inputs(self):
        return [_Input()]

    def run(self, _outs, feed):
        n = feed["input_ids"].shape[0]
        self.batches.append(n)
        # positive logit grows with the row's first id: tones
        # differ per headline and are cheap to recompute here
        logits = np.zeros((n, 3), dtype=np.float64)
        logits[:, 0] = feed["input_ids"][:, 0] / 10.0
        return [logits]


def _install(monkeypatch):
    # test_news_cloud.py's kill-switch test sets the env var
    # directly (line ~838) and it leaks across the session
    monkeypatch.delenv("NEWS_CLOUD_TONE_MODEL", raising=False)
    sess = _Sess()
    monkeypatch.setattr(nt, "_state", (sess, _Tok()))
    monkeypatch.setattr(nt, "_load", lambda: (sess, _Tok()))
    nt._TONE_CACHE.clear()
    return sess


def test_tones_cached_across_calls(monkeypatch):
    sess = _install(monkeypatch)
    titles = ["Stocks rally on strong earnings", "Markets fall hard"]
    first = nt.model_headline_tones(titles)
    assert first is not None and len(first) == 2
    assert sess.batches == [2]
    second = nt.model_headline_tones(titles)
    assert second == first
    assert sess.batches == [2]  # no new inference


def test_overlap_scores_only_new_headlines(monkeypatch):
    sess = _install(monkeypatch)
    nt.model_headline_tones(["Old headline one", "Old headline two"])
    assert sess.batches == [2]
    out = nt.model_headline_tones(
        ["Old headline one", "Fresh headline three"])
    assert out is not None and len(out) == 2
    assert sess.batches == [2, 1]  # only the fresh one scored
    assert out[0] == nt._TONE_CACHE[nt._tone_key("Old headline one")]


def test_progress_starts_at_cached_count(monkeypatch):
    _install(monkeypatch)
    nt.model_headline_tones(["Cached headline"])
    seen = []
    nt.model_headline_tones(
        ["Cached headline", "Uncached headline"],
        progress=lambda d, t, s: seen.append((d, t)))
    assert seen[0] == (1, 2)
    assert seen[-1] == (2, 2)
