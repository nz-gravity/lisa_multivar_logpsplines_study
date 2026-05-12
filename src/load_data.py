"""Unified LISA TDI data loading for noise_4a and noise_5a datasets.

Provides a single LISAData class whose load() classmethod selects the
reference spectral matrix based on the dataset:
  - "analytic": equal-arm LDC analytic model (noise_4a)
  - "welch":    heavily-averaged Welch estimate (~1000 averages) (noise_5a)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
from scipy.signal import csd, welch

# Physical constants
C_LIGHT = 299_792_458.0          # m/s
L_ARM = 2.5e9                    # m
LIGHT_TRAVEL_TIME = L_ARM / C_LIGHT   # ≈ 8.33 s

# LDC noise model parameters (noise_4a)
OMS_ASD = 7.9e-12
OMS_FKNEE = 2e-3
PM_ASD = 2.4e-15
PM_LOW_FKNEE = 4e-4
LASER_FREQ = 2.81e14             # Hz

# Data loading
EDGE_TRIM = 500
LOW_FREQ_BIN_TRIM = 5
DEFAULT_NPERSEG = 2**16
DEFAULT_WINDOW: Tuple[str, float] = ("kaiser", 30)


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------

def load_tdi_timeseries(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        t = np.array(f["t"])
        X2 = np.array(f["X2"])
        Y2 = np.array(f["Y2"])
        Z2 = np.array(f["Z2"])
    return t, X2, Y2, Z2


# ---------------------------------------------------------------------------
# Spectral estimation helpers
# ---------------------------------------------------------------------------

def welch_spectral_matrix_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fs: float,
    nperseg: int = DEFAULT_NPERSEG,
    *,
    window: Union[str, Tuple] = DEFAULT_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nperseg = min(int(nperseg), len(x))
    kw = {"fs": fs, "window": window, "nperseg": nperseg, "detrend": None}
    freq, Sxx = welch(x, **kw)
    _, Syy = welch(y, **kw)
    _, Szz = welch(z, **kw)
    _, Sxy = csd(x, y, **kw)
    _, Syz = csd(y, z, **kw)
    _, Szx = csd(z, x, **kw)
    sl = slice(LOW_FREQ_BIN_TRIM, None)
    return freq[sl], Sxx[sl], Syy[sl], Szz[sl], Sxy[sl], Syz[sl], Szx[sl]


def spectral_matrix_from_components(
    Sxx: np.ndarray, Syy: np.ndarray, Szz: np.ndarray,
    Sxy: np.ndarray, Syz: np.ndarray, Szx: np.ndarray,
) -> np.ndarray:
    nf = len(Sxx)
    cov = np.zeros((nf, 3, 3), dtype=np.complex128)
    cov[:, 0, 0] = Sxx; cov[:, 1, 1] = Syy; cov[:, 2, 2] = Szz
    cov[:, 0, 1] = Sxy; cov[:, 1, 0] = np.conj(Sxy)
    cov[:, 1, 2] = Syz; cov[:, 2, 1] = np.conj(Syz)
    cov[:, 2, 0] = Szx; cov[:, 0, 2] = np.conj(Szx)
    return cov


def interpolate_spectral_matrix(
    src_freq: np.ndarray,
    src_cov: np.ndarray,
    dst_freq: np.ndarray,
) -> np.ndarray:
    out = np.zeros((len(dst_freq), 3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            out[:, i, j] = (
                np.interp(dst_freq, src_freq, src_cov[:, i, j].real)
                + 1j * np.interp(dst_freq, src_freq, src_cov[:, i, j].imag)
            )
    return out


# ---------------------------------------------------------------------------
# Analytic noise model — equal-arm LDC (noise_4a only)
# ---------------------------------------------------------------------------

def _lisa_link_noises_ldc(
    freq: np.ndarray, fs: float, fmin: float
) -> Tuple[np.ndarray, np.ndarray]:
    exp_term = np.exp(-2.0 * np.pi * fmin / fs) * np.exp(-2j * np.pi * freq / fs)
    denom_mag2 = np.abs(1.0 - exp_term) ** 2
    scale = (2.0 * np.pi * fmin) ** 2 / (denom_mag2 * (fs * fmin) ** 2)

    psd_tm_high = (2.0 * PM_ASD * LASER_FREQ / (2.0 * np.pi * C_LIGHT)) ** 2 * scale
    psd_tm_low = (
        (2.0 * PM_ASD * LASER_FREQ * PM_LOW_FKNEE / (2.0 * np.pi * C_LIGHT)) ** 2
        * scale
        * np.abs(1.0 / (1.0 - np.exp(-2j * np.pi * freq / fs))) ** 2
        * (2.0 * np.pi / fs) ** 2
    )
    Spm = psd_tm_high + psd_tm_low

    psd_oms_high = (OMS_ASD * fs * LASER_FREQ / C_LIGHT) ** 2 * np.sin(2.0 * np.pi * freq / fs) ** 2
    psd_oms_low = (
        (2.0 * np.pi * OMS_ASD * LASER_FREQ * OMS_FKNEE**2 / C_LIGHT) ** 2 * scale
    )
    Sop = psd_oms_high + psd_oms_low
    return Spm, Sop


def analytic_covariance_from_model(
    freq: np.ndarray, dt: float, n: int
) -> np.ndarray:
    """Equal-arm LDC XYZ spectral matrix at given frequencies."""
    fs = 1.0 / dt
    fmin = 1.0 / (n * dt)
    Spm, Sop = _lisa_link_noises_ldc(freq, fs=fs, fmin=fmin)

    x = 2.0 * np.pi * LIGHT_TRAVEL_TIME * freq
    sinx = np.sin(x)
    sin2x = np.sin(2.0 * x)
    common = 16.0 * sinx**2 * sin2x**2
    diag = 4.0 * common * Sop + common * (3.0 + np.cos(2.0 * x)) * Spm
    off = -16.0 * sinx * sin2x**3 * (Spm + Sop)

    nf = len(freq)
    cov = np.zeros((nf, 3, 3), dtype=np.complex128)
    cov[:, 0, 0] = cov[:, 1, 1] = cov[:, 2, 2] = diag
    cov[:, 0, 1] = cov[:, 1, 0] = off
    cov[:, 1, 2] = cov[:, 2, 1] = off
    cov[:, 0, 2] = cov[:, 2, 0] = off
    return cov


# ---------------------------------------------------------------------------
# Unified LISAData container
# ---------------------------------------------------------------------------

@dataclass
class LISAData:
    """Time- and frequency-domain LISA TDI data for one noise realisation."""

    time: np.ndarray
    data: np.ndarray       # shape (N, 3): X2, Y2, Z2
    freq: np.ndarray
    matrix: np.ndarray     # Welch spectral matrix (~1000 averages)
    true_matrix: np.ndarray  # reference: analytic (noise_4a) or Welch (noise_5a)
    delta_t: float

    @classmethod
    def load(
        cls,
        data_path: Union[Path, str],
        reference: str = "analytic",
        welch_length: int = DEFAULT_NPERSEG,
    ) -> "LISAData":
        """Load LISA TDI data.

        Parameters
        ----------
        data_path:
            Path to the tdi.h5 HDF5 file.
        reference:
            ``"analytic"`` — equal-arm LDC analytic covariance (noise_4a).
            ``"welch"``    — interpolated heavily-averaged Welch estimate (noise_5a).
            ``"segwo"``    — SEGWO analytic XYZ covariance (noise_5a); requires
                            ``segwo_true_cov.npz`` next to the tdi.h5 file.
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        t, X2, Y2, Z2 = load_tdi_timeseries(path)
        dt = float(t[1] - t[0])
        fs = 1.0 / dt
        sl = slice(EDGE_TRIM, -EDGE_TRIM)
        t_trim = t[sl]
        X2t, Y2t, Z2t = X2[sl], Y2[sl], Z2[sl]
        data = np.vstack((X2t, Y2t, Z2t)).T

        freq, Sxx, Syy, Szz, Sxy, Syz, Szx = welch_spectral_matrix_xyz(
            X2t, Y2t, Z2t, fs=fs, nperseg=welch_length
        )
        matrix = spectral_matrix_from_components(Sxx, Syy, Szz, Sxy, Syz, Szx)

        if reference == "analytic":
            true_matrix = analytic_covariance_from_model(freq, dt=dt, n=len(t))
        elif reference == "welch":
            true_matrix = matrix  # Welch estimate (~1000 averages) is the best available reference
        elif reference == "segwo":
            # Resolve relative to this file's project root so it works from any cwd.
            segwo_path = Path(__file__).parent.parent / "data" / "noise5a_segwo_ref.npz"
            if not segwo_path.exists():
                raise FileNotFoundError(
                    f"SEGWO covariance not found at {segwo_path}; run src/segwo.py first."
                )
            _d = np.load(segwo_path)
            true_matrix = interpolate_spectral_matrix(_d["freq"], _d["cov"], freq)
        else:
            raise ValueError(f"reference must be 'analytic', 'welch', or 'segwo'; got {reference!r}")

        print(
            f"Loaded {len(t)} samples from {path} (fs={fs:.6f} Hz); "
            f"using {len(t_trim)} after edge trim. reference='{reference}'."
        )
        return cls(
            time=t_trim, data=data, freq=freq,
            matrix=matrix, true_matrix=true_matrix, delta_t=dt,
        )
