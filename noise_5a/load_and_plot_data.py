#!/usr/bin/env python3
"""
Generate and analyse example LISA X/Y/Z noise data for the asymmetric noise-5a set.

This mirrors `noise_4a/load_and_plot_data.py`, but the `noise_5a` realization is
not permutation-symmetric in X/Y/Z because the six MOSA noise levels are drawn
independently. Rather than reusing the equal-arm closed-form covariance from
`noise_4a`, we use the trimmed full-series periodogram as the reference spectral
matrix and compare it against a Welch estimate from the same time-domain data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import csd, welch

TRIANGLE_PNG = Path("spectra_triangle.png")
EDGE_TRIM = 500
LOW_FREQ_BIN_TRIM = 5
DEFAULT_NPERSEG = 2**16
DEFAULT_WINDOW = ("kaiser", 30)


def welch_spectral_matrix_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fs: float,
    nperseg: int = DEFAULT_NPERSEG,
    *,
    window: Union[str, Tuple[str, float]] = DEFAULT_WINDOW,
    detrend: Optional[str] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Welch PSD/CSD estimator matched to the noise notebooks."""
    nperseg = min(int(nperseg), len(x))
    kwargs = {
        "fs": fs,
        "window": window,
        "nperseg": nperseg,
        "detrend": detrend,
    }
    freq, Sxx = welch(x, **kwargs)
    _, Syy = welch(y, **kwargs)
    _, Szz = welch(z, **kwargs)
    _, Sxy = csd(x, y, **kwargs)
    _, Syz = csd(y, z, **kwargs)
    _, Szx = csd(z, x, **kwargs)
    return (
        freq[LOW_FREQ_BIN_TRIM:],
        Sxx[LOW_FREQ_BIN_TRIM:],
        Syy[LOW_FREQ_BIN_TRIM:],
        Szz[LOW_FREQ_BIN_TRIM:],
        Sxy[LOW_FREQ_BIN_TRIM:],
        Syz[LOW_FREQ_BIN_TRIM:],
        Szx[LOW_FREQ_BIN_TRIM:],
    )


def spectral_matrix_from_components(
    Sxx: np.ndarray,
    Syy: np.ndarray,
    Szz: np.ndarray,
    Sxy: np.ndarray,
    Syz: np.ndarray,
    Szx: np.ndarray,
) -> np.ndarray:
    """Assemble a 3x3 spectral matrix Sigma(f) from auto- and cross-spectra."""
    nf = len(Sxx)
    cov = np.zeros((nf, 3, 3), dtype=np.complex128)
    cov[:, 0, 0] = Sxx
    cov[:, 1, 1] = Syy
    cov[:, 2, 2] = Szz

    cov[:, 0, 1] = Sxy
    cov[:, 1, 0] = np.conj(Sxy)

    cov[:, 1, 2] = Syz
    cov[:, 2, 1] = np.conj(Syz)

    cov[:, 2, 0] = Szx
    cov[:, 0, 2] = np.conj(Szx)
    return cov


def coherence(Sii: np.ndarray, Sjj: np.ndarray, Sij: np.ndarray) -> np.ndarray:
    """Return magnitude coherence, guarded against zero/negative diagonal bins."""
    sii = np.maximum(np.asarray(Sii).real, 0.0)
    sjj = np.maximum(np.asarray(Sjj).real, 0.0)
    denom = np.sqrt(sii * sjj)
    coh = np.divide(
        np.abs(Sij),
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=denom > 0.0,
    )
    return np.clip(coh, 0.0, 1.0)


def plot_psd_coherence(
    freq: np.ndarray,
    S_true: np.ndarray,
    S_emp: Dict[str, np.ndarray],
    fname: Optional[Union[Path, str]] = None,
    *,
    psd_unit_label: str = "1/Hz",
    empirical_label: str = "Welch",
    reference_label: str = "Reference",
) -> None:
    """
    Plot PSDs on the diagonal and coherences on the lower triangle.

    Parameters
    ----------
    freq
        Frequency array.
    S_true
        Reference spectral matrix with shape `(N, 3, 3)`.
    S_emp
        Dict with keys `Sxx`, `Syy`, `Szz`, `Sxy`, `Syz`, `Szx`.
    """

    Sxx_true = S_true[:, 0, 0]
    Syy_true = S_true[:, 1, 1]
    Szz_true = S_true[:, 2, 2]

    Sxy_true = S_true[:, 0, 1]
    Syz_true = S_true[:, 1, 2]
    Szx_true = S_true[:, 2, 0]

    Sxx_emp = S_emp["Sxx"]
    Syy_emp = S_emp["Syy"]
    Szz_emp = S_emp["Szz"]

    Sxy_emp = S_emp["Sxy"]
    Syz_emp = S_emp["Syz"]
    Szx_emp = S_emp["Szx"]

    coh_xy_true = coherence(Sxx_true, Syy_true, Sxy_true)
    coh_yz_true = coherence(Syy_true, Szz_true, Syz_true)
    coh_zx_true = coherence(Szz_true, Sxx_true, Szx_true)

    coh_xy_emp = coherence(Sxx_emp, Syy_emp, Sxy_emp)
    coh_yz_emp = coherence(Syy_emp, Szz_emp, Syz_emp)
    coh_zx_emp = coherence(Szz_emp, Sxx_emp, Szx_emp)

    channels = ["X", "Y", "Z"]
    true_psd = [Sxx_true, Syy_true, Szz_true]
    emp_psd = [Sxx_emp, Syy_emp, Szz_emp]

    true_coh = [
        [None, coh_xy_true, coh_zx_true],
        [coh_xy_true, None, coh_yz_true],
        [coh_zx_true, coh_yz_true, None],
    ]

    emp_coh = [
        [None, coh_xy_emp, coh_zx_emp],
        [coh_xy_emp, None, coh_yz_emp],
        [coh_zx_emp, coh_yz_emp, None],
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]

            if i < j:
                ax.axis("off")
                continue

            if i == j:
                ax.loglog(freq, np.abs(true_psd[i]), label=reference_label)
                ax.loglog(
                    freq,
                    np.abs(emp_psd[i].real),
                    alpha=0.55,
                    label=f"{empirical_label} PSD",
                )
                ax.set_title(f"{channels[i]} PSD")
                ax.set_ylabel(f"PSD [{psd_unit_label}]")
                ax.grid(True, which="both", ls="--", alpha=0.3)
                emp_pos = np.abs(emp_psd[i].real)
                emp_pos = emp_pos[emp_pos > 0]
                if len(emp_pos):
                    ax.set_ylim(bottom=emp_pos.min() * 0.1)
                if i == 0:
                    ax.legend()
                continue

            ax.semilogx(freq, true_coh[i][j], label=f"{reference_label} coh")
            ax.semilogx(
                freq,
                emp_coh[i][j],
                alpha=0.55,
                label=f"{empirical_label} coh",
            )
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Coherence")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.set_title(f"{channels[i]}-{channels[j]}")
            if i == 1 and j == 0:
                ax.legend()

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Frequency [Hz]")

    fig.tight_layout()
    if fname is not None:
        out_path = Path(fname)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def periodogram_covariance_from_timeseries(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a reference covariance from the full trimmed periodogram.

    For `noise_5a`, the six link noises are intentionally unequal, so the tidy
    equal-arm/equal-link closed form used in `noise_4a` is not appropriate.
    Using the full-series periodogram preserves the channel asymmetry while
    keeping the same `(freq, 3, 3)` interface expected by downstream code.
    """
    dt = float(t[1] - t[0])
    n = len(t)
    freq_full = np.fft.rfftfreq(n, dt)
    fft_x = np.fft.rfft(x)
    fft_y = np.fft.rfft(y)
    fft_z = np.fft.rfft(z)

    scale = dt / n
    Sxx = scale * np.abs(fft_x) ** 2
    Syy = scale * np.abs(fft_y) ** 2
    Szz = scale * np.abs(fft_z) ** 2
    Sxy = scale * fft_x * np.conj(fft_y)
    Syz = scale * fft_y * np.conj(fft_z)
    Szx = scale * fft_z * np.conj(fft_x)

    for arr in (Sxx, Syy, Szz, Sxy, Syz, Szx):
        arr[1:-1] *= 2.0

    freq = freq_full[1 + LOW_FREQ_BIN_TRIM :]
    return freq, spectral_matrix_from_components(
        Sxx[1 + LOW_FREQ_BIN_TRIM :],
        Syy[1 + LOW_FREQ_BIN_TRIM :],
        Szz[1 + LOW_FREQ_BIN_TRIM :],
        Sxy[1 + LOW_FREQ_BIN_TRIM :],
        Syz[1 + LOW_FREQ_BIN_TRIM :],
        Szx[1 + LOW_FREQ_BIN_TRIM :],
    )


def interpolate_spectral_matrix(
    src_freq: np.ndarray,
    src_cov: np.ndarray,
    dst_freq: np.ndarray,
) -> np.ndarray:
    """Interpolate each covariance entry onto a target frequency grid."""
    out = np.zeros((len(dst_freq), 3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            out[:, i, j] = np.interp(dst_freq, src_freq, src_cov[:, i, j].real) + 1j * np.interp(
                dst_freq, src_freq, src_cov[:, i, j].imag
            )
    return out


def load_tdi_timeseries(
    h5_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read `t`, `X2`, `Y2`, `Z2` arrays from the HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        t = np.array(f["t"])
        X2 = np.array(f["X2"])
        Y2 = np.array(f["Y2"])
        Z2 = np.array(f["Z2"])
    return t, X2, Y2, Z2


@dataclass
class LISAData:
    """Container for frequency- and time-domain LISA spectra and helpers."""

    time: np.ndarray
    data: np.ndarray
    freq: np.ndarray
    matrix: np.ndarray
    true_matrix: np.ndarray
    delta_t: float

    @classmethod
    def load(
        cls,
        data_path: Union[Path, str],
        welch_length: int = DEFAULT_NPERSEG,
    ) -> "LISAData":
        return cls._from_hdf5(
            data_path=data_path,
            welch_length=welch_length,
        )

    @classmethod
    def _from_hdf5(
        cls,
        data_path: Union[Path, str],
        welch_length: int,
    ) -> "LISAData":
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        t, X2, Y2, Z2 = load_tdi_timeseries(path)
        dt = float(t[1] - t[0])
        fs = 1.0 / dt

        sl = slice(EDGE_TRIM, -EDGE_TRIM)
        t_trim = t[sl]
        X2_trim = X2[sl]
        Y2_trim = Y2[sl]
        Z2_trim = Z2[sl]
        data = np.vstack((X2_trim, Y2_trim, Z2_trim)).T

        freq_est, Sxx, Syy, Szz, Sxy, Syz, Szx = welch_spectral_matrix_xyz(
            X2_trim,
            Y2_trim,
            Z2_trim,
            fs=fs,
            nperseg=welch_length,
        )
        empirical_cov = spectral_matrix_from_components(
            Sxx, Syy, Szz, Sxy, Syz, Szx
        )

        freq_ref, true_cov_native = periodogram_covariance_from_timeseries(
            t_trim,
            X2_trim,
            Y2_trim,
            Z2_trim,
        )
        true_cov = interpolate_spectral_matrix(freq_ref, true_cov_native, freq_est)

        print(
            f"Loaded {len(t)} samples from {path} (fs={fs:.6f} Hz); "
            f"using {len(t_trim)} after trimming {EDGE_TRIM} samples from each edge."
        )

        return cls(
            time=t_trim,
            freq=freq_est,
            matrix=empirical_cov,
            true_matrix=true_cov,
            data=data,
            delta_t=dt,
        )

    def plot(
        self,
        fname: Union[Path, str] = TRIANGLE_PNG,
    ) -> None:
        """Produce the diagnostic PSD/CSD plots for the stored spectra."""
        S_emp = {
            "Sxx": self.matrix[:, 0, 0].real,
            "Syy": self.matrix[:, 1, 1].real,
            "Szz": self.matrix[:, 2, 2].real,
            "Sxy": self.matrix[:, 0, 1],
            "Syz": self.matrix[:, 1, 2],
            "Szx": self.matrix[:, 2, 0],
        }

        plot_psd_coherence(
            self.freq,
            self.true_matrix,
            S_emp,
            fname=fname,
            reference_label="Full periodogram",
        )
        print(f"Wrote PSD/coherence plot to {Path(fname).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("tdi.h5"),
        help="Path to the HDF5 file containing t, X2, Y2, Z2.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("triangle.png"),
        help="Output path for the PSD/coherence triangle plot.",
    )
    parser.add_argument(
        "--welch-length",
        type=int,
        default=DEFAULT_NPERSEG,
        help="Welch segment length.",
    )
    args = parser.parse_args()

    lisa_data = LISAData.load(
        data_path=args.data_path,
        welch_length=args.welch_length,
    )
    lisa_data.plot(args.plot_path)


if __name__ == "__main__":
    main()
