"""FFT-based audio compression helpers."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft, ifft


def compress_fft(
    signal: np.ndarray, keep_ratio: float = 0.05
) -> tuple[np.ndarray, int, int]:
    """Keep only the largest |FFT coefficients| and zero the rest."""
    spectrum = fft(signal)
    n_keep = max(1, int(len(spectrum) * keep_ratio))
    magnitudes = np.abs(spectrum)
    threshold_idx = np.argpartition(magnitudes, -n_keep)[-n_keep]
    threshold = magnitudes[threshold_idx]
    mask = magnitudes >= threshold
    compressed = spectrum * mask
    return ifft(compressed).real, int(mask.sum()), len(spectrum)
