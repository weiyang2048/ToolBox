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
```

With the vision pipeline:

```bash
pip install -e ".[vision]"
```

## Notebook

Open `Fourier_Transform_Introduction.ipynb` after installing with the `notebook` extra.

## Tests

```bash
uv run pytest
```
