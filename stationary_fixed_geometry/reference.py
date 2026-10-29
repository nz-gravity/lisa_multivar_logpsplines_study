"""Mean tapered/block spectrum for the generator's finite periodic ensemble.

Truth is used only for validation, never for response factoring or inference.
"""

from __future__ import annotations

import h5py
import numpy as np
from scipy.signal import windows


def expected_block_psd(truth, *, n_samples, fs, Nb, window):
    """Convolve the full-record matrix PSD with the block/taper response.

    Global mean subtraction removes the full-record DC Fourier mode. There is
    no block detrending. This computes a mean, not a claim of independent
    Wishart observations. One-sided DC and Nyquist powers are undoubled.
    """
    n = int(n_samples)
    if n % Nb or n % 2:
        raise ValueError("Require an even full-record length divisible by Nb")
    length = n // Nb
    truth = np.asarray(truth, dtype=np.complex128).copy()
    if truth.shape[0] != n // 2 + 1:
        raise ValueError("Truth must lie on the full-record rFFT grid")
    truth[0] = 0  # The only Fourier mode removed by global mean subtraction.
    taper = (
        np.ones(length)
        if window is None
        else windows.get_window(window, length, fftbins=True)
    )
    energy = float(taper @ taper)
    indices = np.arange(1, length // 2 + 1) * Nb
    padded = np.zeros(n)
    padded[:length] = taper
    response = np.fft.fft(padded)
    kernel_fft = np.fft.fft(abs(response) ** 2)
    p = truth.shape[1]
    expected = np.empty((len(indices), p, p), complex)
    refined = []
    for i in range(p):
        for j in range(p):
            source = np.concatenate((truth[:, i, j], truth[-2:0:-1, i, j].conj()))
            source[0] *= 2
            source[n // 2] *= 2
            expected[:, i, j] = np.fft.ifft(np.fft.fft(source) * kernel_fft)[
                indices
            ] / (n * energy)
        # Quiet T can be many orders below the largest power. Recompute small
        # diagonal values as nonnegative sums to avoid FFT cancellation.
        source = np.concatenate((truth[:, i, i].real, truth[-2:0:-1, i, i].real))
        source[0] *= 2
        source[n // 2] *= 2
        small = np.flatnonzero(
            (abs(expected[:, i, i].real) < 1e-10 * np.max(abs(source)))
            | (expected[:, i, i].real < 0)
        )
        full_indices = np.arange(n)
        for target in small:
            weights = abs(response[(indices[target] - full_indices) % n]) ** 2
            expected[target, i, i] = np.dot(source, weights) / (n * energy)
        refined.append(int(len(small)))
    if length % 2 == 0:
        expected[-1] *= 0.5
    expected = 0.5 * (expected + expected.conj().swapaxes(-1, -2))
    return (
        np.fft.rfftfreq(length, 1 / fs)[1:],
        expected,
        {
            "full_record_samples": n,
            "block_samples": length,
            "Nb": Nb,
            "window": window,
            "mean_removal": "global_per_channel",
            "direct_diagonal_refinement_bins": refined,
            "expectation": "finite-periodic Gaussian ensemble, exact block/taper mean",
            "independent_Wishart_distribution_established": False,
        },
    )


def align_reference(source, basis, fine_frequency, *, Nb=8, window=("tukey", 0.1)):
    """Average the expectation over the exact fine-bin supports used by a fit."""
    with h5py.File(source) as file:
        truth = file[f"truth_{basis}"][:]
        n, fs = int(file.attrs["size"]), float(file.attrs["fs"])
    frequency, expected, metadata = expected_block_psd(
        truth, n_samples=n, fs=fs, Nb=Nb, window=window
    )
    fine_frequency = np.asarray(fine_frequency)
    block_indices = np.rint(fine_frequency * (n // Nb) / fs).astype(int)
    np.testing.assert_allclose(
        frequency[block_indices - 1], fine_frequency, atol=2e-17, rtol=0
    )
    underlying = truth[block_indices * Nb]
    midpoint = (fine_frequency.shape[1] - 1) // 2
    return {
        "expected_psd": expected[block_indices - 1].mean(axis=1),
        "coarse_true_psd": underlying.mean(axis=1),
        "midpoint_true_psd": underlying[:, midpoint],
        "metadata": metadata,
    }


def validate_direct_response():
    """Independent small Fourier-matrix calculation including complex CSDs."""
    rng = np.random.default_rng(520)
    errors = []
    for n, nb in ((48, 3), (60, 4), (64, 1)):
        h = rng.normal(size=(n // 2 + 1, 3, 3)) + 1j * rng.normal(
            size=(n // 2 + 1, 3, 3)
        )
        truth = h @ h.conj().swapaxes(-1, -2)
        truth[0] = truth[0].real
        truth[-1] = truth[-1].real
        length = n // nb
        for window in (None, "hann", ("tukey", 0.1)):
            w = (
                np.ones(length)
                if window is None
                else windows.get_window(window, length)
            )
            modes = np.exp(2j * np.pi * np.outer(np.arange(length), np.arange(n)) / n)
            modes[:, 0] = 0  # Whole-record centering, not centering these blocks.
            transform = np.exp(
                -2j
                * np.pi
                * np.outer(np.arange(1, length // 2 + 1), np.arange(length))
                / length
            )
            response = (transform * w) @ modes
            source = np.concatenate((truth, truth[-2:0:-1].conj()))
            source[0] *= 2
            source[n // 2] *= 2
            direct = np.einsum("kf,fij->kij", abs(response) ** 2, source) / (
                n * (w @ w)
            )
            if length % 2 == 0:
                direct[-1] *= 0.5
            _, actual, _ = expected_block_psd(
                truth, n_samples=n, fs=0.5, Nb=nb, window=window
            )
            np.testing.assert_allclose(actual, direct, atol=2e-12, rtol=2e-12)
            errors.append(float(np.max(abs(actual - direct)) / np.max(abs(direct))))
    return {"direct_response_cases": len(errors), "maximum_relative_error": max(errors)}


if __name__ == "__main__":
    import json

    print(json.dumps(validate_direct_response(), indent=2))
