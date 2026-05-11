#!/usr/bin/env python3
"""Overlay analytic AET true PSD on existing univar inference_data.nc runs.

Usage:
    python plot_aet_diagnostic.py noise_4a/bf_runs/duration_30d_eta0.5/univar
    python plot_aet_diagnostic.py noise_4a/bf_runs/duration_30d_eta0.5/univar --out diag.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

_HERE = Path(__file__).resolve().parent       # noise_4a/
_ROOT = _HERE.parent                          # lisa_logpspline/
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from load_and_plot_data import (             # noqa: E402
    LISAData,
    covariance_matrix,
    lisa_link_noises_ldc,
    tdi2_psd_and_csd,
)
from log_psplines.psplines import LogPSplines  # noqa: E402
from run_univar_aet import xyz_to_aet_matrix   # noqa: E402

DATA_PATH = _HERE / "tdi.h5"
CHANNEL_LABELS = ("A", "E", "T")


def _compute_aet_true_psd(
    freq: np.ndarray, dt: float, n_total: int
) -> list[np.ndarray]:
    """Return analytic diagonal AET PSDs [S_A, S_E, S_T] at each frequency."""
    fs = 1.0 / dt
    fmin = 1.0 / (n_total * dt)
    Spm, Sop = lisa_link_noises_ldc(freq, fs=fs, fmin=fmin)
    diag, off = tdi2_psd_and_csd(freq, Spm, Sop)
    xyz_cov = covariance_matrix(diag, off)                # (nf, 3, 3)
    aet_cov = xyz_to_aet_matrix(xyz_cov)                  # (nf, 3, 3)
    return [aet_cov[:, i, i].real for i in range(3)]


def _reconstruct_posterior_psd(
    weights: np.ndarray,
    spline: LogPSplines,
    scaling_factor: float,
    n_max_draws: int = 500,
) -> np.ndarray:
    """Return (3, n_freq) array: [p05, p50, p95] in physical units."""
    S_total = weights.shape[0] * weights.shape[1]
    w_flat = weights.reshape(S_total, -1)           # (S, n_knots)
    if S_total > n_max_draws:
        idx = np.random.default_rng(0).choice(S_total, n_max_draws, replace=False)
        w_flat = w_flat[idx]
    basis = np.asarray(spline.basis, dtype=np.float64)               # (n_freq, n_knots)
    log_para = np.asarray(spline.log_parametric_model, dtype=np.float64)  # (n_freq,)
    # ln_psd_std: (S, n_freq) in standardized units
    ln_psd_std = (basis @ w_flat.T).T + log_para[np.newaxis, :]
    psd_phys = np.exp(ln_psd_std) * scaling_factor                   # (S, n_freq)
    return np.percentile(psd_phys, [5, 50, 95], axis=0)              # (3, n_freq)


def plot_aet_diagnostic(run_dir: Path, data_path: Path, out_path: Path) -> None:
    lisa = LISAData.load(data_path)
    dt = lisa.delta_t
    n_total = len(lisa.time)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, ch in zip(axes, CHANNEL_LABELS):
        nc_path = run_dir / ch / "inference_data.nc"
        if not nc_path.exists():
            ax.set_title(f"{ch} — not found")
            ax.axis("off")
            continue

        tree = xr.open_datatree(nc_path)
        sf = float(tree.attrs["scaling_factor"])

        pdgrm_da = tree["observed_data"].ds["periodogram"]
        freq = pdgrm_da.coords["freq"].values.astype(np.float64)
        power_phys = pdgrm_da.values * sf

        sm_dict = {k: v.values for k, v in tree["spline_model"].ds.data_vars.items()}
        spline = LogPSplines.from_storage_dataset(sm_dict)

        weights = tree["posterior"].ds["weights"].values   # (chain, draw, n_knots)
        psd_q = _reconstruct_posterior_psd(weights, spline, sf)  # (3, n_freq)

        true_psds = _compute_aet_true_psd(freq, dt, n_total)
        true_ch = true_psds[CHANNEL_LABELS.index(ch)]

        # Data
        ax.loglog(freq, power_phys, color="0.65", lw=0.4, alpha=0.5, label="Data")
        # Posterior
        ax.loglog(freq, psd_q[1], color="C0", lw=1.5, label="Posterior median")
        ax.fill_between(freq, psd_q[0], psd_q[2], alpha=0.25, color="C0",
                        label="90% CI")
        # True
        ax.loglog(freq, true_ch, color="C1", lw=1.8, ls="--", label="Analytic truth")

        # Parametric model used during fit (in physical units)
        pm = np.asarray(spline.parametric_model, dtype=np.float64) * sf
        if not np.allclose(pm, sf):   # only plot if not trivially all-ones
            ax.loglog(freq, pm, color="C3", lw=1.0, ls=":", label="Parametric model (fit)")

        ax.set_title(f"Channel {ch}  (σ²={sf:.2e})")
        ax.set_xlabel("Frequency [Hz]")
        if ax is axes[0]:
            ax.set_ylabel("PSD [1/Hz]")
        ax.grid(True, which="both", ls="--", alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(f"Univariate AET diagnostic — {run_dir}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to univar run root containing A/, E/, T/ subdirectories.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help=f"Path to tdi.h5 (default: {DATA_PATH}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <run_dir>/aet_diagnostic.png).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_path = args.out or (run_dir / "aet_diagnostic.png")
    plot_aet_diagnostic(run_dir, args.data_path.resolve(), out_path)


if __name__ == "__main__":
    main()
