#!/usr/bin/env python3
"""Log P-Spline MCMC runner for LISA TDI noise datasets.

Usage
-----
  python main.py --dataset noise4a --model H4 --duration-days 30
  python main.py --dataset noise5a --model H0 --duration-days 180 --eta 0.03
  python main.py --dataset noise4a --model H1 --compute-lnz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpyro
from scipy.signal import csd, welch

from src.utils import (
    choose_coarse_grain_nc,
    choose_nb_for_duration,
    compute_retained_frequency_count,
    duration_slug,
    light_travel_null_exclusion_bands,
)
from src.load_data import (
    LIGHT_TRAVEL_TIME,
    LOW_FREQ_BIN_TRIM,
    LISAData,
    interpolate_spectral_matrix,
)
from src.aet import run_univar_aet_analysis, xyz_to_aet_timeseries, xyz_to_aet_matrix
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.arviz_utils import save_inference_data
from log_psplines.arviz_utils.from_arviz import get_multivar_posterior_psd_quantiles
from log_psplines.pipeline.config import PipelineConfig
from log_psplines.pipeline.make_pipeline import make_pipeline
from log_psplines.pipeline.preprocessing import align_true_psd_to_freq
from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig


# ---------------------------------------------------------------------------
# Per-dataset configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    data_path: Path
    outdir_base: Path
    reference: str       # "analytic" | "segwo"
    theta_im_knots: int


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "noise4a": DatasetConfig(
        data_path=Path("data/noise4a.h5"),
        outdir_base=Path("out/noise4a"),
        reference="analytic",
        theta_im_knots=2,
    ),
    "noise5a": DatasetConfig(
        data_path=Path("data/noise5a.h5"),
        outdir_base=Path("out/noise5a"),
        reference="segwo",
        theta_im_knots=50,
    ),
}


# ---------------------------------------------------------------------------
# Hypothesis model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    description: str
    runner: str        # "multivar" | "multivar_aet" | "univar"
    k_theta: int | None
    outdir_name: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "H0": ModelSpec(
        description="univar(AET, 3×p=1, Wishart) — exactly diagonal AET",
        runner="univar",
        k_theta=None,
        outdir_name="mcmc_output_H0",
    ),
    "H1": ModelSpec(
        description="multivar(AET, p=3, k_theta=2) — effectively diagonal AET",
        runner="multivar_aet",
        k_theta=2,
        outdir_name="mcmc_output_H1",
    ),
    "H2": ModelSpec(
        description="multivar(AET, p=3, k_theta=50) — full AET covariance",
        runner="multivar_aet",
        k_theta=50,
        outdir_name="mcmc_output_H2",
    ),
    "H4": ModelSpec(
        description="multivar(XYZ, p=3, k_theta=50) — full XYZ covariance",
        runner="multivar",
        k_theta=None,
        outdir_name="mcmc_output_H4",
    ),
}


# ---------------------------------------------------------------------------
# Shared analysis constants
# ---------------------------------------------------------------------------

FMIN = 1e-4
FMAX = 1e-1
BLOCK_DAYS = 7.0
NUM_CHAINS = 4
ETA = 0.03
ANALYSIS_DURATION_DAYS = 10.0
NC_TARGET = 1024
VI_NC_TARGET = 256
NULL_EXCISION_HALFWIDTH = 1e-4

DIFF_ORDER = 2
KNOT_METHOD = "density"
N_SPLINE_KNOTS_DELTA = 50
N_SPLINE_KNOTS_THETA_RE = 50
N_SPLINE_KNOTS_THETA_AET = 10

N_WARMUP = 2000
N_SAMPLES = 1500
TARGET_ACCEPT_PROB = 0.85
MAX_TREE_DEPTH = 12
DENSE_MASS = True

ALPHA_DELTA = 3.0
BETA_DELTA = 3.0
WISHART_FLOOR_FRACTION = 1e-6
WISHART_WINDOW = ("tukey", 0.1)
DIAG_WELCH_MIN_BLOCKS = 8
DIAG_WELCH_OVERLAP_FRACTION = 0.5

USE_VI = True
VI_STEPS = 50_000
VI_LR = 1e-2
VI_POSTERIOR_DRAWS = 256

COMPUTE_LNZ = False
LNZ_N_RESAMPLES = 512
LNZ_N_ESTIMATIONS = 3


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _set_symlog_scale(ax: Any, values: list[np.ndarray]) -> None:
    finite = np.concatenate(
        [np.abs(np.asarray(v, dtype=np.float64).ravel()) for v in values]
    )
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    vmax = float(np.max(finite))
    if vmax <= 0.0:
        return
    ax.set_yscale("symlog", linthresh=max(vmax * 1e-4, 1e-30))


def _coherence_from_matrix(s: np.ndarray, i: int, j: int) -> np.ndarray:
    sii = np.maximum(np.asarray(s[:, i, i].real, dtype=np.float64), 0.0)
    sjj = np.maximum(np.asarray(s[:, j, j].real, dtype=np.float64), 0.0)
    denom = sii * sjj
    return np.divide(
        np.abs(np.asarray(s[:, i, j], dtype=np.complex128)) ** 2,
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=denom > 0.0,
    )


def _posterior_coherence_from_sd_quantiles(sd_quantiles: np.ndarray) -> np.ndarray:
    if sd_quantiles.ndim != 4 or sd_quantiles.shape[0] != 3:
        raise ValueError(
            f"sd_quantiles must have shape (3, n_freq, 3, 3); got {sd_quantiles.shape}."
        )
    coherence = np.zeros(sd_quantiles.shape, dtype=np.float64)
    for q in range(3):
        sq = np.asarray(sd_quantiles[q], dtype=np.complex128)
        for i in range(3):
            for j in range(3):
                coherence[q, :, i, j] = _coherence_from_matrix(sq, i, j)
    return np.sort(np.clip(coherence, 0.0, 1.0), axis=0)


def _interp_complex_series(
    src_freq: np.ndarray, values: np.ndarray, dst_freq: np.ndarray
) -> np.ndarray:
    return np.interp(dst_freq, src_freq, values.real) + 1j * np.interp(
        dst_freq, src_freq, values.imag
    )


def _interp_complex_matrix(
    src_freq: np.ndarray, values: np.ndarray, dst_freq: np.ndarray
) -> np.ndarray:
    out = np.zeros((len(dst_freq),) + values.shape[1:], dtype=np.complex128)
    for i in range(values.shape[1]):
        for j in range(values.shape[2]):
            out[:, i, j] = _interp_complex_series(src_freq, values[:, i, j], dst_freq)
    return out


def plot_psd_triangle(
    freq: np.ndarray,
    s_ref: np.ndarray,
    s_emp: dict[str, np.ndarray],
    fname: Path,
    *,
    empirical_label: str,
    reference_label: str = "Reference",
    posterior_quantiles: dict[str, Any] | None = None,
) -> None:
    """Plot XYZ PSDs and coherences, optionally overlaying posterior quantiles."""
    if posterior_quantiles is not None:
        freq_plot = np.asarray(posterior_quantiles["freq"], dtype=np.float64)
        s_ref = _interp_complex_matrix(freq, s_ref, freq_plot)
        s_emp = {
            k: _interp_complex_series(freq, np.asarray(v), freq_plot)
            for k, v in s_emp.items()
        }
    else:
        freq_plot = np.asarray(freq, dtype=np.float64)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    channels = ["X", "Y", "Z"]
    emp_diag = [s_emp["Sxx"], s_emp["Syy"], s_emp["Szz"]]
    coh_ref = {
        (1, 0): _coherence_from_matrix(s_ref, 1, 0),
        (2, 1): _coherence_from_matrix(s_ref, 2, 1),
        (2, 0): _coherence_from_matrix(s_ref, 2, 0),
    }
    coh_emp = {
        (1, 0): np.clip(
            np.abs(s_emp["Sxy"]) / np.sqrt(
                np.maximum(s_emp["Sxx"].real, 0.0) * np.maximum(s_emp["Syy"].real, 0.0)
            ), 0.0, 1.0),
        (2, 1): np.clip(
            np.abs(s_emp["Syz"]) / np.sqrt(
                np.maximum(s_emp["Syy"].real, 0.0) * np.maximum(s_emp["Szz"].real, 0.0)
            ), 0.0, 1.0),
        (2, 0): np.clip(
            np.abs(s_emp["Szx"]) / np.sqrt(
                np.maximum(s_emp["Szz"].real, 0.0) * np.maximum(s_emp["Sxx"].real, 0.0)
            ), 0.0, 1.0),
    }
    post_real = (
        None if posterior_quantiles is None
        else np.asarray(posterior_quantiles["real"], dtype=np.float64)
    )
    post_coh = (
        None if posterior_quantiles is None
        else np.asarray(posterior_quantiles["coherence"], dtype=np.float64)
    )

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue
            if i == j:
                ax.loglog(freq_plot, np.maximum(s_ref[:, i, i].real, 1e-30), label=f"{reference_label} PSD")
                ax.loglog(freq_plot, np.maximum(emp_diag[i].real, 1e-30), alpha=0.5, label=f"{empirical_label} PSD")
                if post_real is not None:
                    ax.fill_between(freq_plot, np.maximum(post_real[0, :, i, i], 1e-30),
                                    np.maximum(post_real[2, :, i, i], 1e-30), alpha=0.2, color="C2")
                    ax.loglog(freq_plot, np.maximum(post_real[1, :, i, i], 1e-30), color="C2", lw=1.5, label="Posterior median")
                ax.set_title(f"{channels[i]} PSD")
                ax.set_ylabel("PSD [1/Hz]")
                ax.grid(True, which="both", ls="--", alpha=0.3)
                if i == 0:
                    ax.legend()
                continue

            pair = (i, j)
            ax.semilogx(freq_plot, coh_ref[pair], label=f"{reference_label} coh")
            ax.semilogx(freq_plot, coh_emp[pair], alpha=0.5, label=f"{empirical_label} coh")
            if post_coh is not None:
                ax.fill_between(freq_plot, np.clip(post_coh[0, :, i, j], 0.0, 1.0),
                                np.clip(post_coh[2, :, i, j], 0.0, 1.0), alpha=0.2, color="C2")
                ax.semilogx(freq_plot, np.clip(post_coh[1, :, i, j], 0.0, 1.0), color="C2", lw=1.5, label="Posterior median")
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
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_psd_csd_matrix(
    freq: np.ndarray,
    s_ref: np.ndarray,
    s_emp: dict[str, np.ndarray],
    fname: Path,
    *,
    reference_label: str = "Reference",
    empirical_label: str = "Welch",
    posterior_quantiles: dict[str, Any] | None = None,
) -> None:
    """Plot diagonal PSDs and off-diagonal real/imaginary CSDs."""
    if posterior_quantiles is not None:
        freq_plot = np.asarray(posterior_quantiles["freq"], dtype=np.float64)
        s_ref = _interp_complex_matrix(freq, s_ref, freq_plot)
        s_emp = {
            k: _interp_complex_series(freq, np.asarray(v), freq_plot)
            for k, v in s_emp.items()
        }
    else:
        freq_plot = np.asarray(freq, dtype=np.float64)

    post_real = (
        None if posterior_quantiles is None
        else np.asarray(posterior_quantiles["real"], dtype=np.float64)
    )
    post_imag = (
        None if posterior_quantiles is None
        else np.asarray(posterior_quantiles["imag"], dtype=np.float64)
    )
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    channels = ["X", "Y", "Z"]
    pairs = {(0, 1): "Sxy", (1, 2): "Syz", (2, 0): "Szx"}

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if i == j:
                ref_psd = np.maximum(s_ref[:, i, i].real, 1e-30)
                emp_psd = np.maximum(s_emp[f"S{channels[i].lower() * 2}"].real, 1e-30)
                ax.loglog(freq_plot, ref_psd, label=reference_label)
                ax.loglog(freq_plot, emp_psd, alpha=0.55, label=empirical_label)
                if post_real is not None:
                    ax.fill_between(freq_plot, np.maximum(post_real[0, :, i, i], 1e-30),
                                    np.maximum(post_real[2, :, i, i], 1e-30), alpha=0.2, color="C2")
                    ax.loglog(freq_plot, np.maximum(post_real[1, :, i, i], 1e-30), color="C2", lw=1.5, label="Posterior median")
                ax.set_title(f"{channels[i]} PSD")
                ax.grid(True, which="both", ls="--", alpha=0.3)
                if i == 0:
                    ax.legend()
                continue

            key = pairs.get((i, j)) or pairs[(j, i)]
            ref_csd = s_ref[:, i, j]
            emp_csd = s_emp[key]
            component = np.real if i > j else np.imag
            label = "Re" if i > j else "Im"
            ax.semilogx(freq_plot, component(ref_csd), label=f"{reference_label} {label}")
            ax.semilogx(freq_plot, component(emp_csd), alpha=0.55, label=f"{empirical_label} {label}")
            if post_real is not None and post_imag is not None:
                post_arr = post_real if label == "Re" else post_imag
                ax.fill_between(freq_plot, post_arr[0, :, i, j], post_arr[2, :, i, j], alpha=0.2, color="C2")
                ax.semilogx(freq_plot, post_arr[1, :, i, j], color="C2", lw=1.5, label="Posterior median")
                _set_symlog_scale(ax, [component(ref_csd), component(emp_csd), post_arr[0, :, i, j], post_arr[2, :, i, j]])
            else:
                _set_symlog_scale(ax, [component(ref_csd), component(emp_csd)])
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.set_title(f"{channels[i]}-{channels[j]} {label} CSD")
            if (i, j) == (1, 0):
                ax.legend()

    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency [Hz]")
    for ax in axes[:, 0]:
        ax.set_ylabel("PSD / CSD")
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_plot_inputs(
    lisa: LISAData,
    n_trim: int,
    nb: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], str]:
    """Return block-Welch frequency grid, reference matrix, and empirical spectra."""
    plot_blocks = max(nb, DIAG_WELCH_MIN_BLOCKS)
    plot_blocks = min(plot_blocks, n_trim)
    nperseg = n_trim // plot_blocks
    noverlap = min(int(DIAG_WELCH_OVERLAP_FRACTION * nperseg), max(0, nperseg - 1))
    fs = 1.0 / lisa.delta_t
    x, y, z = lisa.data[:n_trim].T
    kwargs = {
        "fs": fs, "window": WISHART_WINDOW, "nperseg": nperseg,
        "noverlap": noverlap, "detrend": False,
    }
    freq, Sxx = welch(x, **kwargs)
    _, Syy = welch(y, **kwargs)
    _, Szz = welch(z, **kwargs)
    _, Sxy = csd(x, y, **kwargs)
    _, Syz = csd(y, z, **kwargs)
    _, Szx = csd(z, x, **kwargs)
    freq = freq[LOW_FREQ_BIN_TRIM:]
    true_cov = interpolate_spectral_matrix(lisa.freq, lisa.true_matrix, freq)
    s_emp = {
        "Sxx": Sxx[LOW_FREQ_BIN_TRIM:], "Syy": Syy[LOW_FREQ_BIN_TRIM:],
        "Szz": Szz[LOW_FREQ_BIN_TRIM:], "Sxy": Sxy[LOW_FREQ_BIN_TRIM:],
        "Syz": Syz[LOW_FREQ_BIN_TRIM:], "Szx": Szx[LOW_FREQ_BIN_TRIM:],
    }
    overlap_pct = int(round(100.0 * noverlap / nperseg)) if nperseg > 0 else 0
    return freq, true_cov, s_emp, f"Welch ({plot_blocks} blocks, {overlap_pct}% overlap)"


def make_block_welch_plots(
    lisa: LISAData, n_trim: int, nb: int, outdir: Path, *, idata: Any | None = None,
) -> None:
    freq, true_cov, s_emp, empirical_label = build_plot_inputs(lisa, n_trim, nb)
    _raw_q = get_multivar_posterior_psd_quantiles(idata) if idata is not None else None
    if _raw_q is not None:
        _sd = np.asarray(_raw_q["spectral_density"], dtype=np.complex128)
        posterior_quantiles = {
            "freq": _raw_q["freq"],
            "real": np.asarray(_sd.real, dtype=np.float64),
            "imag": np.asarray(_sd.imag, dtype=np.float64),
            "coherence": _posterior_coherence_from_sd_quantiles(_sd),
        }
    else:
        posterior_quantiles = None

    plot_psd_triangle(
        freq, true_cov, s_emp, outdir / "welch_psd_triangle.png",
        empirical_label=empirical_label, reference_label="Reference",
        posterior_quantiles=posterior_quantiles,
    )
    plot_psd_csd_matrix(
        freq, true_cov, s_emp, outdir / "welch_psd_csd_matrix.png",
        reference_label="Reference", empirical_label=empirical_label,
        posterior_quantiles=posterior_quantiles,
    )


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def build_multivar_config(
    cfg: DatasetConfig,
    true_psd_freq: np.ndarray,
    true_psd_matrix: np.ndarray,
    nb: int,
    main_nc: int,
    vi_nc: int,
    eta: float,
    outdir: Path,
    compute_lnz: bool,
    lnz_n_resamples: int,
    lnz_n_estimations: int,
    lnz_max_iter: int,
    lnz_tol: float,
    lnz_kde_bw: str,
    theta_re_knots: int | None = None,
    theta_im_knots: int | None = None,
) -> PipelineConfig:
    exclude_bands = light_travel_null_exclusion_bands(
        FMIN, FMAX, light_travel_time=LIGHT_TRAVEL_TIME, halfwidth=NULL_EXCISION_HALFWIDTH,
    )
    n_knots = {
        "delta": N_SPLINE_KNOTS_DELTA,
        "theta_re": theta_re_knots if theta_re_knots is not None else N_SPLINE_KNOTS_THETA_RE,
        "theta_im": theta_im_knots if theta_im_knots is not None else cfg.theta_im_knots,
    }
    return PipelineConfig(
        n_samples=N_SAMPLES,
        n_warmup=N_WARMUP,
        num_chains=NUM_CHAINS,
        chain_method="parallel",
        Nb=nb,
        alpha_delta=ALPHA_DELTA,
        beta_delta=BETA_DELTA,
        eta=eta,
        coarse_grain_config=CoarseGrainConfig(enabled=True, Nc=main_nc, Nh=None),
        wishart_window=WISHART_WINDOW,
        wishart_floor_fraction=WISHART_FLOOR_FRACTION,
        n_knots=n_knots,
        degree=2,
        diffMatrixOrder=DIFF_ORDER,
        knot_kwargs={"method": KNOT_METHOD},
        true_psd=(true_psd_freq, true_psd_matrix),
        fmin=FMIN,
        fmax=FMAX,
        exclude_freq_bands=exclude_bands,
        verbose=True,
        outdir=str(outdir),
        only_vi=False,
        init_from_vi=USE_VI,
        vi_steps=max(VI_STEPS, nb * 1500),
        vi_lr=VI_LR,
        vi_guide=f"lowrank:{max(20, nb * 2)}",
        vi_posterior_draws=VI_POSTERIOR_DRAWS,
        vi_progress_bar=True,
        coarse_grain_config_vi=CoarseGrainConfig(enabled=USE_VI, Nc=vi_nc, Nh=None),
        target_accept_prob=TARGET_ACCEPT_PROB,
        max_tree_depth=MAX_TREE_DEPTH,
        dense_mass=DENSE_MASS,
        design_from_vi=False,
        design_from_vi_tau=3.0,
        compute_lnz=compute_lnz,
        extra_kwargs={
            "lnz_kwargs": {
                "morph_type": "indep",
                "thin": 1,
                "n_resamples": lnz_n_resamples,
                "n_estimations": lnz_n_estimations,
                "kde_bw": lnz_kde_bw,
                "max_iter": lnz_max_iter,
                "tol": lnz_tol,
            }
        } if compute_lnz else {},
    )


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def _prepare_run(
    args: argparse.Namespace, cfg: DatasetConfig, lisa: LISAData,
) -> tuple[float, float, str, int, int, int, int]:
    """Resolve shared run parameters; return (eta, duration_days, slug, nb, n_trim, main_nc, vi_nc)."""
    duration_days = float(args.duration_days)
    eta = float(args.eta) if args.eta is not None else ETA
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"eta must be in (0, 1]; got {eta}.")

    duration_label = duration_slug(duration_days)
    nb = choose_nb_for_duration(duration_days, block_days=BLOCK_DAYS)

    n_raw = min(lisa.data.shape[0], int(duration_days * 86400.0 / lisa.delta_t))
    n_trim = (n_raw // (2 * nb)) * (2 * nb)
    if n_trim <= 0:
        raise ValueError(f"After trimming for Nb={nb}, got n_trim={n_trim}.")

    n_band_raw = compute_retained_frequency_count(
        n_trim=n_trim, dt=lisa.delta_t, nb=nb, fmin=FMIN, fmax=FMAX
    )
    main_nc = choose_coarse_grain_nc(n_retained=n_band_raw, nc_target=NC_TARGET)
    vi_nc = choose_coarse_grain_nc(n_retained=n_band_raw, nc_target=VI_NC_TARGET)
    return eta, duration_days, duration_label, nb, n_trim, main_nc, vi_nc


def run_multivar(
    args: argparse.Namespace, cfg: DatasetConfig, lisa: LISAData, *, outdir_name: str,
) -> None:
    eta, duration_days, duration_label, nb, n_trim, main_nc, vi_nc = _prepare_run(args, cfg, lisa)
    outdir = args.outdir or (cfg.outdir_base / outdir_name / duration_label / f"eta{eta:g}")
    outdir.mkdir(parents=True, exist_ok=True)
    numpyro.set_host_device_count(NUM_CHAINS)

    print(f"n_trim={n_trim}, Nb={nb} (~{duration_days / nb:.2f} days/block)")
    make_block_welch_plots(lisa, n_trim, nb, outdir)

    ts = MultivariateTimeseries(y=lisa.data[:n_trim].astype(np.float64), t=lisa.time[:n_trim].astype(np.float64))
    config = build_multivar_config(
        cfg=cfg, true_psd_freq=lisa.freq, true_psd_matrix=lisa.true_matrix,
        nb=nb, main_nc=main_nc, vi_nc=vi_nc, eta=eta, outdir=outdir,
        compute_lnz=bool(args.compute_lnz),
        lnz_n_resamples=int(args.lnz_n_resamples), lnz_n_estimations=int(args.lnz_n_estimations),
        lnz_max_iter=int(args.lnz_max_iter), lnz_tol=float(args.lnz_tol), lnz_kde_bw=str(args.lnz_kde_bw),
    )
    pipeline = make_pipeline(ts, config)
    result = pipeline.run()
    result.save(str(outdir), true_psd=align_true_psd_to_freq(config.true_psd, pipeline.data))
    idata = result.idata

    if bool(args.compute_lnz):
        print("lnZ:", {k: idata.attrs.get(k) for k in ("lnz", "lnz_err", "lnz_valid")})

    make_block_welch_plots(lisa, n_trim, nb, outdir, idata=idata)
    out_nc = outdir / "idata.nc"
    save_inference_data(idata, out_nc)
    print(f"Saved inference data to {out_nc.resolve()}")


def run_multivar_aet(
    args: argparse.Namespace, cfg: DatasetConfig, lisa: LISAData,
    *, outdir_name: str, k_theta: int,
) -> None:
    eta, duration_days, duration_label, nb, n_trim, main_nc, vi_nc = _prepare_run(args, cfg, lisa)
    outdir = args.outdir or (cfg.outdir_base / outdir_name / duration_label / f"eta{eta:g}")
    outdir.mkdir(parents=True, exist_ok=True)
    numpyro.set_host_device_count(NUM_CHAINS)

    print(f"multivar(AET): duration={duration_days:g}d, Nb={nb}, eta={eta:.6g}, theta knots={k_theta}")

    y_aet = xyz_to_aet_timeseries(lisa.data[:n_trim].astype(np.float64))
    ts = MultivariateTimeseries(y=y_aet, t=lisa.time[:n_trim].astype(np.float64))
    true_psd_matrix_aet = xyz_to_aet_matrix(lisa.true_matrix)

    config = build_multivar_config(
        cfg=cfg, true_psd_freq=lisa.freq, true_psd_matrix=true_psd_matrix_aet,
        nb=nb, main_nc=main_nc, vi_nc=vi_nc, eta=eta, outdir=outdir,
        compute_lnz=bool(args.compute_lnz),
        lnz_n_resamples=int(args.lnz_n_resamples), lnz_n_estimations=int(args.lnz_n_estimations),
        lnz_max_iter=int(args.lnz_max_iter), lnz_tol=float(args.lnz_tol), lnz_kde_bw=str(args.lnz_kde_bw),
        theta_re_knots=k_theta, theta_im_knots=k_theta,
    )
    pipeline = make_pipeline(ts, config)
    result = pipeline.run()
    result.save(str(outdir), true_psd=align_true_psd_to_freq(config.true_psd, pipeline.data))
    idata = result.idata

    if bool(args.compute_lnz):
        print("lnZ:", {k: idata.attrs.get(k) for k in ("lnz", "lnz_err", "lnz_valid")})

    out_nc = outdir / "idata.nc"
    save_inference_data(idata, out_nc)
    print(f"Saved inference data to {out_nc.resolve()}")


def run_univar(
    args: argparse.Namespace, cfg: DatasetConfig, lisa: LISAData, *, outdir_name: str,
) -> None:
    eta, duration_days, duration_label, nb, n_trim, main_nc, vi_nc = _prepare_run(args, cfg, lisa)
    root_outdir = args.outdir or (cfg.outdir_base / outdir_name / duration_label / f"eta{eta:g}")
    numpyro.set_host_device_count(NUM_CHAINS)

    exclude_bands = light_travel_null_exclusion_bands(
        FMIN, FMAX, light_travel_time=LIGHT_TRAVEL_TIME, halfwidth=NULL_EXCISION_HALFWIDTH,
    )
    print(f"univar(AET): duration={duration_days:g}d, Nb={nb}, eta={eta:.6g}")

    run_univar_aet_analysis(
        lisa=lisa, n_trim=n_trim, root_outdir=root_outdir,
        fmin=FMIN, fmax=FMAX, nb=nb, main_nc=main_nc, vi_nc=vi_nc,
        n_samples=N_SAMPLES, n_warmup=N_WARMUP, num_chains=NUM_CHAINS,
        chain_method="parallel",
        alpha_delta=ALPHA_DELTA, beta_delta=BETA_DELTA, eta=eta,
        n_knots=N_SPLINE_KNOTS_DELTA, degree=2, diff_order=DIFF_ORDER,
        knot_method=KNOT_METHOD, use_vi=USE_VI,
        vi_steps=max(VI_STEPS, nb * 1500), vi_lr=VI_LR,
        vi_guide=f"lowrank:{max(20, nb * 2)}", vi_posterior_draws=VI_POSTERIOR_DRAWS,
        vi_progress_bar=True, target_accept_prob=TARGET_ACCEPT_PROB,
        max_tree_depth=MAX_TREE_DEPTH, dense_mass=DENSE_MASS,
        exclude_freq_bands=exclude_bands, compute_lnz=bool(args.compute_lnz),
        lnz_kwargs={
            "n_resamples": int(args.lnz_n_resamples),
            "n_estimations": int(args.lnz_n_estimations),
            "kde_bw": str(args.lnz_kde_bw),
            "max_iter": int(args.lnz_max_iter),
            "tol": float(args.lnz_tol),
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Log P-Spline MCMC on LISA TDI noise data.")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    _model_help = "  |  ".join(f"{k}: {v.description}" for k, v in MODEL_SPECS.items())
    parser.add_argument("--model", required=True, choices=list(MODEL_SPECS), help=_model_help)
    parser.add_argument("--duration-days", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--compute-lnz", action="store_true", default=COMPUTE_LNZ)
    parser.add_argument("--lnz-n-resamples", type=int, default=LNZ_N_RESAMPLES)
    parser.add_argument("--lnz-n-estimations", type=int, default=LNZ_N_ESTIMATIONS)
    parser.add_argument("--lnz-max-iter", type=int, default=5000)
    parser.add_argument("--lnz-tol", type=float, default=1e-2)
    parser.add_argument("--lnz-kde-bw", type=str, default="silverman")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    if args.duration_days is None:
        args.duration_days = ANALYSIS_DURATION_DAYS

    spec = MODEL_SPECS[args.model]
    print(f"Dataset: {args.dataset}  Model {args.model}: {spec.description}")

    lisa = LISAData.load(cfg.data_path, reference=cfg.reference)

    if spec.runner == "multivar":
        run_multivar(args, cfg, lisa, outdir_name=spec.outdir_name)
    elif spec.runner == "multivar_aet":
        run_multivar_aet(args, cfg, lisa, outdir_name=spec.outdir_name, k_theta=spec.k_theta)
    else:
        run_univar(args, cfg, lisa, outdir_name=spec.outdir_name)


if __name__ == "__main__":
    main()
