"""Build and save the SEGWO analytic XYZ covariance for noise_5a.

Reads data/noise5a.h5, constructs the noise covariance using SEGWO and PyTDI,
and writes the reference to data/noise5a_segwo_ref.npz.

Requires: pip install -e sgwb-renate-noise-models/

Usage:
    python src/segwo.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import csd, welch

import segwo
from lisaconstants import c
from lisaconstants.indexing import LINKS, MOSAS
from pytdi import LISATDICombination
from pytdi.michelson import X2_ETA, Y2_ETA, Z2_ETA


DATA = Path("data")
OUT = Path("out/segwo_ref")

# Simulation parameters
fs = 0.5           # Hz
oms_fknee = 2e-3   # Hz
tm_fknee = 0.4e-3  # Hz
laser_freq = 2.816e14  # Hz
ltts = np.repeat(8.3, 6)[None]  # light-travel times (1, 6)

# Per-MOSA noise ASDs (from simulation metadata)
tm_asds = {
    12: 3.2987207830707333e-15,
    23: 2.2068318236082305e-15,
    31: 4.799588250657558e-15,
    13: 3.711602738525377e-15,
    32: 4.271678793058393e-15,
    21: 4.467581058832328e-15,
}
oms_asds = {
    12: 1.3592816495174599e-11,
    23: 1.1705105384539884e-11,
    31: 1.1098305430366561e-11,
    13: 9.41502170206022e-12,
    32: 1.0832545004320958e-11,
    21: 7.6801489202984e-12,
}


def tm_model_psd(freq: NDArray, asd: float) -> NDArray:
    """TM noise PSD in TMI carrier beatnote fluctuations (filter approximation)."""
    asd /= 2.0  # ASD is acceleration noise; account for back-and-forth optical path
    duration = 1.0 / freq[0]
    fmin = 1.0 / duration
    psd_high = (
        (2 * asd * laser_freq / (2 * np.pi * c)) ** 2
        * np.abs((2 * np.pi * fmin) / (1 - np.exp(-2 * np.pi * fmin / fs) * np.exp(-2j * np.pi * freq / fs))) ** 2
        / (fs * fmin) ** 2
    )
    psd_low = (
        (2 * asd * laser_freq * tm_fknee / (2 * np.pi * c)) ** 2
        * np.abs((2 * np.pi * fmin) / (1 - np.exp(-2 * np.pi * fmin / fs) * np.exp(-2j * np.pi * freq / fs))) ** 2
        / (fs * fmin) ** 2
        * np.abs(1 / (1 - np.exp(-2j * np.pi * freq / fs))) ** 2
        * (2 * np.pi / fs) ** 2
    )
    return psd_high + psd_low


def oms_model_psd(freq: NDArray, asd: float) -> NDArray:
    """OMS noise PSD in ISI carrier beatnote fluctuations (filter approximation)."""
    duration = 1.0 / freq[0]
    fmin = 1.0 / duration
    psd_high = (asd * fs * laser_freq / c) ** 2 * np.sin(2 * np.pi * freq / fs) ** 2
    psd_low = (
        (2 * np.pi * asd * laser_freq * oms_fknee**2 / c) ** 2
        * np.abs((2 * np.pi * fmin) / (1 - np.exp(-2 * np.pi * fmin / fs) * np.exp(-2j * np.pi * freq / fs))) ** 2
        / (fs * fmin) ** 2
    )
    return psd_high + psd_low


def build_xyz_covariance(freq: NDArray) -> NDArray:
    """Build XYZ noise covariance matrix via SEGWO + PyTDI mixing."""
    noise_cov = segwo.cov.construct_covariance_from_psds(
        [tm_model_psd(freq, tm_asds[mosa]) for mosa in MOSAS]
        + [oms_model_psd(freq, oms_asds[mosa]) for mosa in MOSAS]
    )

    ETA_COMBS: dict[int, LISATDICombination] = {}
    ETA_COMBS[12] = LISATDICombination(
        {"N_tm_21": [(1, ["D_12"])], "N_tm_12": [(1, [])], "N_oms_12": [(1, [])]}
    )
    ETA_COMBS[23] = ETA_COMBS[12].rotated()
    ETA_COMBS[31] = ETA_COMBS[23].rotated()
    ETA_COMBS[13] = ETA_COMBS[12].reflected(1)
    ETA_COMBS[21] = ETA_COMBS[23].reflected(2)
    ETA_COMBS[32] = ETA_COMBS[31].reflected(3)

    noise_list = [f"N_tm_{mosa}" for mosa in LINKS] + [f"N_oms_{mosa}" for mosa in LINKS]
    noise2eta = segwo.cov.construct_mixing_from_pytdi(
        freq, measurements=noise_list,
        tdi_combinations=[ETA_COMBS[mosa] for mosa in LINKS], ltts=ltts,
    )
    eta_list = [f"eta_{mosa}" for mosa in LINKS]
    eta2xyz = segwo.cov.construct_mixing_from_pytdi(freq, eta_list, [X2_ETA, Y2_ETA, Z2_ETA], ltts)

    noise_cov_xyz = segwo.cov.project_covariance(noise_cov, [noise2eta, eta2xyz])
    # Shape: (1, nfreq, 3, 3) — squeeze time axis; ensure diagonal is real
    S = noise_cov_xyz[0].copy()
    for i in range(3):
        S[:, i, i] = np.abs(noise_cov_xyz[0, :, i, i])
    return S


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Reading noise5a.h5...")
    with h5py.File(DATA / "noise5a.h5", "r") as f:
        xyz = np.stack([f["X2"][:], f["Y2"][:], f["Z2"][:]], axis=-1)

    welch_kw = dict(fs=fs, axis=0, window=("kaiser", 31), detrend=False, nperseg=2**18)
    freq_all, xyz_psd = welch(xyz, **welch_kw)
    freq = freq_all[3:]  # drop DC and very low frequencies
    xyz_psd = xyz_psd[3:]

    print("Building SEGWO covariance...")
    S_true = build_xyz_covariance(freq)

    out_npz = DATA / "noise5a_segwo_ref.npz"
    np.savez(out_npz, freq=freq, cov=S_true)
    print(f"Saved SEGWO reference covariance to {out_npz}")

    # Diagnostic plots
    _band = freq >= 1e-4
    f_plot = freq[_band]
    S_plot = S_true[_band]
    psd_plot = xyz_psd[_band]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for i, ch in enumerate("XYZ"):
        axes[i].loglog(f_plot, np.abs(S_plot[:, i, i]), label="SEGWO model", c="black", lw=2)
        axes[i].loglog(f_plot, psd_plot[:, i], alpha=0.7, label="Data Welch")
        axes[i].set_xlabel("Frequency [Hz]")
        axes[i].set_title(f"TDI {ch}")
        axes[i].legend()
    axes[0].set_ylabel("PSD [1/Hz]")
    fig.tight_layout()
    fig.savefig(OUT / "segwo_vs_data.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic plot to {OUT / 'segwo_vs_data.png'}")


if __name__ == "__main__":
    main()
