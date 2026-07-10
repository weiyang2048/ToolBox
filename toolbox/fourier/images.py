"""2D Fourier filtering for images."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift


def freq_filter_2d(
    image: np.ndarray, filter_fn: Callable[[np.ndarray], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a radial filter in the frequency domain."""
    spectrum = fftshift(fft2(image))
    h, w = image.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = filter_fn(radius)
    filtered = ifft2(ifftshift(spectrum * mask)).real
    return np.clip(filtered, 0, 1), mask
