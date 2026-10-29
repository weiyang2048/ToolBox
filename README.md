# ToolBox

```bash
pip install git+https://github.com/weiyang2048/ToolBox.git
```

Reusable signal-processing utilities with an interactive Fourier transform tutorial.

## Install

### uv (recommended)

```bash
uv sync --all-extras
```

Core only:

```bash
uv sync
```

Run commands in the project environment:

```bash
uv run pytest
uv run jupyter notebook
```

### pip

```bash
pip install -e .
```

With Jupyter support for the tutorial notebook:

```bash
pip install -e ".[notebook]"
```

Development (tests, lint):

```bash
pip install -e ".[dev]"
```

## Package layout

```
toolbox/
  fourier/
    audio.py    # FFT coefficient compression
    images.py   # 2D frequency-domain filters
    noise.py    # notch filter, spectral subtraction, SNR
  vision/       # torch image classification (requires the `vision` extra)
    data/               # folder scanning + ImageLoader Dataset
    models/             # EfficientNet wrapper (timm, b0-b7)
    transformations/    # train/test image transforms
  automation/   # browser/file automation bots (requires the `automation` extra)
    LinkedInHelper/     # saved-jobs scraper + CLI (Selenium)
    ArXivSearcher.py    # arXiv search + PDF downloads
    SeleniumInitializer.py  # Selenium login helper
    random_folder_opener.py # random folder opener + Click CLI (cli.py)
  DataZoo/      # sample and synthetic datasets for demos and tests
    market/             # FearGreed, StockIndices, Yahoo prices, Zillow data
    image/              # synthetic image data
  net/          # networking helpers (stdlib only)
    ssh_proxy_connect.py  # HTTP CONNECT ProxyCommand tunnel for ssh/git
```

With the vision pipeline:

```bash
pip install -e ".[vision]"
```

With the automation bots:

```bash
pip install -e ".[automation]"
```

## Energy market optimization (salvaged from the Optimization repo)

```
energy_optimization_challenge/  # day-ahead vs real-time bidding optimizer for wind assets
  tests/                        # unittest suite + run_tests.py
```

Requires numpy, pandas, scipy, matplotlib, seaborn, pyomo, pulp and scikit-learn
(the last four are declared in the `energy` extra: `pip install -e ".[energy]"`).

The package itself is declared in `pyproject.toml`, so it is installed
alongside `toolbox` (not just importable from a repo checkout).

Test coverage note: the suite covers market clearing, objective functions,
the optimizer, risk management and variable definitions. `data_loader` has
no test coverage — its test module (`tests/test_data_loader.py`) was never
salvaged: the source no longer exists upstream (only a compiled `.pyc`
survived in the deleted Optimization repo).

## Notebook

Open `notebooks/Stock_Indices_Walkthrough.ipynb` after installing with the `notebook` extra.

## Tests

```bash
uv run pytest
```

`pyproject.toml`'s `testpaths` covers two suites: `tests/` (core `toolbox`
packages — currently `fourier`, pure numpy/scipy, and `DataZoo`, whose
network sources are all mocked so the suite runs offline) and
`energy_optimization_challenge/tests/` (needs the `energy` extra and the
GLPK solver for the risk-management tests). CI
(`.github/workflows/tests.yml`) runs both on Python 3.10–3.12.
`toolbox/vision` and `toolbox/automation` have no test
suites yet; they are covered locally by import smoke tests under their
optional extras (`pip install -e ".[vision]"` / `".[automation]"`, then
import the submodules).
