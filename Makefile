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

# Lint with the pinned ruff: the version is locked in the `dev` extra
# (ruff==0.17.3 in pyproject.toml, the same pin CI installs) and the
# enforced rule set is pyproject's [tool.ruff.lint]. Mirrors CI's bare
# `ruff check` at the repo root.
.PHONY: lint
lint:
	uv run --extra dev ruff check
