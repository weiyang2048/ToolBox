"""Tests for toolbox.fourier (audio compression, 2D filtering, noise).

Pure numpy/scipy, headless, deterministic — safe for CI.
"""

import numpy as np
import pytest

from toolbox.fourier import (
    compress_fft,
    freq_filter_2d,
    notch_filter,
    snr,
    spectral_subtraction,
)


def _sine(cycles: float, n: int = 256) -> np.ndarray:
    t = np.arange(n) / n
    return np.sin(2 * np.pi * cycles * t)


# --- compress_fft -------------------------------------------------------


def test_compress_fft_full_ratio_is_exact():
    signal = _sine(5)
    recovered, n_kept, n_total = compress_fft(signal, keep_ratio=1.0)
    assert n_total == len(signal)
    assert n_kept == n_total
    np.testing.assert_allclose(recovered, signal, atol=1e-12)


def test_compress_fft_keeps_dominant_bins_of_pure_tone():
    # A pure tone's energy sits in exactly two FFT bins (k and N-k).
    signal = _sine(5)
    recovered, n_kept, _ = compress_fft(signal, keep_ratio=0.01)
    assert n_kept == 2
    np.testing.assert_allclose(recovered, signal, atol=1e-12)


def test_compress_fft_output_shape_and_minimum_one_kept():
    signal = np.random.default_rng(0).normal(size=64)
    recovered, n_kept, _ = compress_fft(signal, keep_ratio=0.0)
    assert recovered.shape == signal.shape
    assert n_kept >= 1


def test_compress_fft_compression_reduces_kept_count():
    signal = np.random.default_rng(1).normal(size=256)
    _, n_kept, n_total = compress_fft(signal, keep_ratio=0.05)
    assert 1 <= n_kept < n_total


# --- freq_filter_2d -----------------------------------------------------


def test_freq_filter_2d_identity_filter_returns_input():
    rng = np.random.default_rng(2)
    image = rng.random((16, 20))
    filtered, mask = freq_filter_2d(image, lambda r: np.ones_like(r))
    assert mask.shape == image.shape
    np.testing.assert_allclose(filtered, image, atol=1e-12)


def test_freq_filter_2d_zero_filter_returns_zeros():
    image = np.full((8, 8), 0.7)
    filtered, _ = freq_filter_2d(image, lambda r: np.zeros_like(r))
    np.testing.assert_allclose(filtered, 0.0, atol=1e-12)


def test_freq_filter_2d_lowpass_keeps_constant_image():
    image = np.full((16, 16), 0.5)
    filtered, mask = freq_filter_2d(image, lambda r: (r <= 3).astype(float))
    np.testing.assert_allclose(filtered, image, atol=1e-12)
    assert mask[8, 8] == 1.0  # DC passes
    assert mask[0, 0] == 0.0  # far corner is outside the radius


def test_freq_filter_2d_output_is_clipped_to_unit_range():
    image = np.full((8, 8), 2.0)
    filtered, _ = freq_filter_2d(image, lambda r: np.ones_like(r))
    assert filtered.max() <= 1.0


# --- notch_filter -------------------------------------------------------


def test_notch_filter_removes_target_frequency_only():
    fs = 200.0
    t = np.arange(400) / fs
    keep = np.sin(2 * np.pi * 10 * t)
    signal = keep + np.sin(2 * np.pi * 50 * t)
    filtered = notch_filter(signal, fs, [50.0])
    np.testing.assert_allclose(filtered, keep, atol=1e-10)


def test_notch_filter_no_frequencies_is_identity():
    signal = _sine(3, n=128)
    # Notch far outside the signal's content leaves it (near-)unchanged:
    # use an empty notch list, which must be an exact identity.
    filtered = notch_filter(signal, fs=128.0, notch_freqs=[])
    np.testing.assert_allclose(filtered, signal, atol=1e-12)


# --- spectral_subtraction -----------------------------------------------


def test_spectral_subtraction_zero_noise_is_identity():
    signal = _sine(4, n=128)
    out = spectral_subtraction(signal, np.zeros_like(signal))
    np.testing.assert_allclose(out, signal, atol=1e-12)


def test_spectral_subtraction_full_magnitude_removes_signal():
    signal = _sine(4, n=128)
    spectrum_mag = np.abs(np.fft.fft(signal))
    out = spectral_subtraction(signal, spectrum_mag, alpha=1.0)
    np.testing.assert_allclose(out, 0.0, atol=1e-10)


def test_spectral_subtraction_never_amplifies():
    rng = np.random.default_rng(3)
    signal = rng.normal(size=128)
    noise_mag = np.abs(np.fft.fft(rng.normal(size=128))) * 0.1
    out = spectral_subtraction(signal, noise_mag)
    assert np.sum(out**2) <= np.sum(signal**2) + 1e-9


# --- snr ----------------------------------------------------------------


def test_snr_identical_signals_is_infinite():
    signal = _sine(2, n=64)
    assert snr(signal, signal) == float("inf")


def test_snr_known_value():
    clean = np.ones(100)
    estimate = np.ones(100)
    estimate[0] = 0.0  # one unit of squared error; signal power 100
    assert snr(clean, estimate) == pytest.approx(20.0)  # 10*log10(100/1)


def test_snr_worse_estimate_scores_lower():
    rng = np.random.default_rng(4)
    clean = rng.normal(size=256)
    good = clean + rng.normal(scale=0.01, size=256)
    bad = clean + rng.normal(scale=1.0, size=256)
    assert snr(clean, good) > snr(clean, bad)
