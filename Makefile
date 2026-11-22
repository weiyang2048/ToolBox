# Canonical test entry point. The extra set is pinned because the suite's
# green state depends on it: pytest/ruff come from `dev`, the QuantStats
# indicator tests need `quantstats` (a bare environment fails collection),
# and the audio tests need `audio`. The vision extra is deliberately NOT
# included — its torch-gated tests skip here and are run for real by the
# dedicated CI job. See the README's Tests section.
UV_EXTRAS := --extra dev --extra quantstats --extra audio

.PHONY: test-full
test-full:
	uv run $(UV_EXTRAS) -m pytest
	uv run $(UV_EXTRAS) python -m unittest discover -s energy_optimization_challenge/tests -t.
