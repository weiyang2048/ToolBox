"""Frequency-domain noise reduction."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft, fftfreq, ifft


def notch_filter(
    signal: np.ndarray,
    fs: float,
    notch_freqs: list[float],
    bandwidth: float = 3,
) -> np.ndarray:
    """Attenuate narrow bands around specified frequencies."""
    spectrum = fft(signal)
    freqs = fftfreq(len(signal), d=1 / fs)
    for f0 in notch_freqs:
        spectrum[np.abs(np.abs(freqs) - f0) < bandwidth] = 0
    return ifft(spectrum).real


def spectral_subtraction(
    signal: np.ndarray, noise_mag: np.ndarray, alpha: float = 1.5
) -> np.ndarray:
    """Subtract a pre-estimated noise magnitude spectrum; keep original phase."""
    spectrum = fft(signal)
    mag = np.abs(spectrum)
    phase = np.angle(spectrum)
    clean_mag = np.maximum(mag - alpha * noise_mag, 0)
    return ifft(clean_mag * np.exp(1j * phase)).real


def snr(clean: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-noise ratio in dB (higher is better)."""
    error_power = np.sum((clean - estimate) ** 2)
    if error_power == 0:
        return float("inf")
    return float(10 * np.log10(np.sum(clean**2) / error_power))
