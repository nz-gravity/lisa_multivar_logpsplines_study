#!/usr/bin/env python3
"""
Generate and analyse example LISA X/Y/Z noise data.

Load XYZ TDI time series from an HDF5 file, build the analytic PSD/CSD
matrix from the legacy LDC noise model, and compare it against Welch
estimates from the time-domain data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import csd, resample_poly, welch

# --- Constants and default paths -------------------------------------------------
C_LIGHT = 299_792_458.0  # m / s
L_ARM = 2.5e9  # m
LIGHT_TRAVEL_TIME = L_ARM / C_LIGHT  # ≈ 8.33 s

# Match the `noise-4.ipynb` generation notebook.
OMS_ASD = 7.9e-12
OMS_FKNEE = 2e-3
PM_ASD = 2.4e-15
PM_LOW_FKNEE = 4e-4
LASER_FREQ = 2.81e14  # Hz

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
    """Welch PSD/CSD estimator matched to `noise-4.ipynb`."""
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
    """Assemble a 3×3 spectral matrix Σ(f) from auto- and cross-spectra."""
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
    # Guard against exact/null bins so plotting/diagnostics stay finite.
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


def analytic_covariance_from_model(
    freq: np.ndarray,
    dt: float,
    n: int,
) -> np.ndarray:
    """Build the analytic XYZ spectral matrix from the legacy LDC model."""
    fs = 1.0 / dt
    fmin = 1.0 / (n * dt)
    Spm, Sop = lisa_link_noises_ldc(freq, fs=fs, fmin=fmin)
    diag, csd = tdi2_psd_and_csd(freq, Spm, Sop)
    return covariance_matrix(diag, csd)


def plot_psd_coherence(
    freq: np.ndarray,
    S_true: np.ndarray,
    S_emp: Dict[str, np.ndarray],
    fname: Optional[Union[Path, str]] = None,
    *,
    psd_unit_label: str = "1/Hz",
    empirical_label: str = "Welch",
) -> None:
    """
    Plot PSDs on the diagonal and coherences on the lower triangle.

    freq: frequency array
    S_true: (N, 3, 3) analytic spectral matrix
    S_emp: dict with keys "Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"
    """

    Sxx_true = S_true[:, 0, 0].real
    Syy_true = S_true[:, 1, 1].real
    Szz_true = S_true[:, 2, 2].real

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
                ax.loglog(freq, true_psd[i], label="True PSD")
                ax.loglog(
                    freq,
                    emp_psd[i],
                    alpha=0.5,
                    label=f"{empirical_label} PSD",
                )
                ax.set_title(f"{channels[i]} PSD")
                ax.set_ylabel(f"PSD [{psd_unit_label}]")
                ax.grid(True, which="both", ls="--", alpha=0.3)
                if i == 0:
                    ax.legend()
                continue

            ax.semilogx(freq, true_coh[i][j], label="True coh")
            ax.semilogx(
                freq,
                emp_coh[i][j],
                alpha=0.5,
                label=f"{empirical_label} coh",
            )
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Coherence")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.set_title(f"{channels[i]}–{channels[j]}")
            if i == 1 and j == 0:
                ax.legend()

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Frequency [Hz]")

    for ax in axes.flatten():
        ax.set_xlim(freq[0], 0.1)
        
    # for all diagonal axes set ylim to [1e-40, 1e-20]
    for i in range(3):
        ax = axes[i, i]
        if ax.has_data():
            ax.set_ylim(1e-17, 1e-9)

    fig.tight_layout()
    if fname is not None:
        out_path = Path(fname)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


# --- Noise and transfer helpers --------------------------------------------------
def lisa_link_noises_ldc(
    freq: np.ndarray,
    fs: float,
    fmin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reproduce the single-link proof-mass (Spm) and optical-path (Sop) PSDs
    used in the LDC noise realizations.

    See https://zenodo.org/doi/10.5281/zenodo.15698080
    """
    exp_term = np.exp(-2.0 * np.pi * fmin / fs) * np.exp(
        -2j * np.pi * freq / fs
    )
    denom_mag2 = np.abs(1.0 - exp_term) ** 2

    psd_tm_high = (
        (2.0 * PM_ASD * LASER_FREQ / (2.0 * np.pi * C_LIGHT)) ** 2
        * (2.0 * np.pi * fmin) ** 2
        / denom_mag2
        / (fs * fmin) ** 2
    )
    psd_tm_low = (
        (2.0 * PM_ASD * LASER_FREQ * PM_LOW_FKNEE / (2.0 * np.pi * C_LIGHT))
        ** 2
        * (2.0 * np.pi * fmin) ** 2
        / denom_mag2
        / (fs * fmin) ** 2
        * np.abs(1.0 / (1.0 - np.exp(-2j * np.pi * freq / fs))) ** 2
        * (2.0 * np.pi / fs) ** 2
    )
    Spm = psd_tm_high + psd_tm_low

    psd_oms_high = (OMS_ASD * fs * LASER_FREQ / C_LIGHT) ** 2 * np.sin(
        2.0 * np.pi * freq / fs
    ) ** 2
    psd_oms_low = (
        (2.0 * np.pi * OMS_ASD * LASER_FREQ * OMS_FKNEE**2 / C_LIGHT) ** 2
        * (2.0 * np.pi * fmin) ** 2
        / denom_mag2
        / (fs * fmin) ** 2
    )
    Sop = psd_oms_high + psd_oms_low
    return Spm, Sop


def tdi2_psd_and_csd(
    freq: np.ndarray, Spm: np.ndarray, Sop: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute diagonal PSD (X2) and cross-term CSD (XY) for TDI2 combinations.

    Because of the symmetries of the equal-arm constellation:
        S_X2 = S_Y2 = S_Z2
        S_XY = S_YZ = S_ZX

    See e.g. Eq. (54-55) of https://arxiv.org/pdf/2211.02539

    The diagonal PSD is matched to the `noise-4.ipynb` validation notebook:
    `tdi_tf_oms^2 = 4 * tdi_common` and
    `tdi_tf_testmass^2 = tdi_common * (3 + cos(2x))`.
    """
    x = 2.0 * np.pi * LIGHT_TRAVEL_TIME * freq
    sinx = np.sin(x)
    sin2x = np.sin(2.0 * x)
    cos2x = np.cos(2.0 * x)
    common = 16.0 * sinx**2 * sin2x**2

    diag = 4.0 * common * Sop
    diag += common * (3.0 + cos2x) * Spm

    # The notebook only validates the diagonal ASD. Keep the existing
    # equal-arm symmetry structure for the off-diagonal terms, but scale the
    # test-mass contribution consistently with the notebook's single-link model.
    csd = -16.0 * sinx * (sin2x**3) * (Spm + Sop)
    return diag, csd


def covariance_matrix(diag: np.ndarray, csd: np.ndarray) -> np.ndarray:
    """Assemble the 3×3 covariance matrix Σ(f) for each frequency."""
    nf = diag.size
    cov = np.zeros((nf, 3, 3), dtype=np.complex128)
    cov[:, 0, 0] = cov[:, 1, 1] = cov[:, 2, 2] = diag
    cov[:, 0, 1] = cov[:, 1, 0] = csd
    cov[:, 1, 2] = cov[:, 2, 1] = csd
    cov[:, 0, 2] = cov[:, 2, 0] = csd
    return cov


def periodogram_covariance(
    auto_psd: Dict[str, np.ndarray], cross_csd: Dict[str, np.ndarray]
) -> np.ndarray:
    """Build the empirical 3×3 spectral matrix from auto/cross periodograms."""
    nf = len(next(iter(auto_psd.values())))
    cov = np.zeros((nf, 3, 3), dtype=np.complex128)
    cov[:, 0, 0] = auto_psd["X"]
    cov[:, 1, 1] = auto_psd["Y"]
    cov[:, 2, 2] = auto_psd["Z"]

    cov[:, 0, 1] = cross_csd["XY"]
    cov[:, 1, 0] = np.conj(cov[:, 0, 1])

    cov[:, 1, 2] = cross_csd["YZ"]
    cov[:, 2, 1] = np.conj(cov[:, 1, 2])

    cov[:, 2, 0] = cross_csd["ZX"]
    cov[:, 0, 2] = np.conj(cov[:, 2, 0])
    return cov


# --- Data handling + spectral estimates -----------------------------------------
def load_tdi_timeseries(
    h5_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read t, X2, Y2, Z2 arrays from the HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        t = np.array(f["t"])
        X2 = np.array(f["X2"])
        Y2 = np.array(f["Y2"])
        Z2 = np.array(f["Z2"])
    return t, X2, Y2, Z2


def compute_periodograms(
    t: np.ndarray,
    X2: np.ndarray,
    Y2: np.ndarray,
    Z2: np.ndarray,
) -> Tuple[
    np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]
]:
    """Return (freq, auto-PSD dict, cross-CSD dict, metadata) from the time-domain data."""
    dt = t[1] - t[0]
    n = len(t)
    freq_full = np.fft.rfftfreq(n, dt)
    fft_x = np.fft.rfft(X2)
    fft_y = np.fft.rfft(Y2)
    fft_z = np.fft.rfft(Z2)

    scale = dt / n
    auto = {
        "X": scale * np.abs(fft_x) ** 2,
        "Y": scale * np.abs(fft_y) ** 2,
        "Z": scale * np.abs(fft_z) ** 2,
    }
    cross = {
        "XY": scale * fft_x * np.conj(fft_y),
        "YZ": scale * fft_y * np.conj(fft_z),
        "ZX": scale * fft_z * np.conj(fft_x),
    }

    # Double the positive-frequency interior to account for two-sided FFT.
    for arr in list(auto.values()) + list(cross.values()):
        arr[1:-1] *= 2.0

    # Drop the DC bin for plotting (log axis incompatible with zero frequency).
    freq = freq_full[1:]
    for key in auto:
        auto[key] = auto[key][1:]
    for key in cross:
        cross[key] = cross[key][1:]

    meta = {"dt": dt, "fs": 1.0 / dt, "n": n, "fmin": 1.0 / (n * dt)}
    return freq, auto, cross, meta


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
        dt = t[1] - t[0]
        fs = 1.0 / dt
        sl = slice(EDGE_TRIM, -EDGE_TRIM)
        X2_trim = X2[sl]
        Y2_trim = Y2[sl]
        Z2_trim = Z2[sl]
        t_trim = t[sl]
        data = np.vstack((X2_trim, Y2_trim, Z2_trim)).T
        freq_est, Sxx, Syy, Szz, Sxy, Syz, Szx = welch_spectral_matrix_xyz(
            X2_trim,
            Y2_trim,
            Z2_trim,
            fs=fs,
            nperseg=welch_length,
        )

        true_cov = analytic_covariance_from_model(
            freq_est,
            dt=dt,
            n=len(t),
        )
        empirical_cov = spectral_matrix_from_components(
            Sxx, Syy, Szz, Sxy, Syz, Szx
        )
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

        print(f"dt={self.delta_t:.6f} s, fs={1.0 / self.delta_t:.6f} Hz, n={len(self.time):,}")
        plot_psd_coherence(self.freq, self.true_matrix, S_emp, fname=fname)
        print(f"Wrote PSD/coherence plot to {Path(fname).resolve()}")


    def downsample(self, new_dt: int, fmax_fraction: float = 1.0) -> None:
        """Downsample the time series to a new sampling interval.

        Strategy
        --------
        - Downsample only the time-domain data.
        - Recompute the empirical Welch spectral matrix on the new grid.
        - Keep the analytic model tied to the original physics/model, and
        interpolate it onto the new frequency grid rather than rebuilding it.
        - Restrict comparison to a conservative passband below the new Nyquist,
        where the anti-alias filter is close to flat.

        Parameters
        ----------
        new_dt
            New sampling interval in seconds. Must be an integer multiple of
            the current delta_t.
        fmax_fraction
            Fraction of the new sampling rate used as a safe comparison band.
            Default 0.4 means compare only up to 0.4 / new_dt, which stays
            below the new Nyquist 0.5 / new_dt.
        """
        ratio = float(new_dt) / float(self.delta_t)
        factor = int(round(ratio))
        if factor <= 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1e-12):
            raise ValueError(
                "new_dt must be an integer multiple of the current "
                f"delta_t={self.delta_t:.6f} s; got new_dt={new_dt}."
            )

        # Cache the current analytic model before changing anything.
        freq_true_old = self.freq.copy()
        true_old = self.true_matrix.copy()

        # Downsample the data with anti-alias filtering.
        self.data = resample_poly(self.data, up=1, down=factor, axis=0)
        self.time = self.time[0] + np.arange(self.data.shape[0]) * float(new_dt)
        self.delta_t = float(new_dt)

        fs = 1.0 / self.delta_t
        freq_est, Sxx, Syy, Szz, Sxy, Syz, Szx = welch_spectral_matrix_xyz(
            self.data[:, 0],
            self.data[:, 1],
            self.data[:, 2],
            fs=fs,
            nperseg=min(DEFAULT_NPERSEG, len(self.data)),
        )

        empirical_cov = spectral_matrix_from_components(
            Sxx,
            Syy,
            Szz,
            Sxy,
            Syz,
            Szx,
        )

        # Conservative passband: avoid the anti-alias filter rolloff region.
        fmax_safe = fmax_fraction / self.delta_t
        mask = freq_est <= fmax_safe
        freq_use = freq_est[mask]
        empirical_cov = empirical_cov[mask]

        # Interpolate the original analytic matrix onto the new frequency grid.
        true_interp = np.zeros((len(freq_use), 3, 3), dtype=np.complex128)
        for i in range(3):
            for j in range(3):
                re = np.interp(freq_use, freq_true_old, true_old[:, i, j].real)
                im = np.interp(freq_use, freq_true_old, true_old[:, i, j].imag)
                true_interp[:, i, j] = re + 1j * im

        self.freq = freq_use
        self.matrix = empirical_cov
        self.true_matrix = true_interp


def main() -> None:
    lisa_data = LISAData.load(
        data_path=Path("tdi.h5"),
        welch_length=DEFAULT_NPERSEG,
    )
    lisa_data.plot(Path("triangle.png"))
    lisa_data.downsample(new_dt=8)
    lisa_data.plot(Path("triangle_downsampled.png"))


if __name__ == "__main__":
    main()
