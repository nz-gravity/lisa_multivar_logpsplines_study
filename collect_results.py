#!/usr/bin/env python3
"""
Collect eta=0.5 LISA multivariate PSD results and make comparison plots.

Outputs written to the repo root:
  - results_eta0p5.csv
  - results_eta0p5_summary.csv
  - triangle_noise4a_eta0p5.png
  - triangle_noise5a_eta0p5.png
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr

from log_psplines.arviz_utils.from_arviz import (
    _build_psd_dataset,
    _get_multivar_frequency_grid,
    _get_multivar_reconstruction_inputs_from_dataset,
    _quantiles_from_psd_draws,
    _rescale_multivar_psd,
    get_multivar_spline_model,
    get_sample_dataset,
)
from log_psplines.diagnostics._utils import (
    compute_matrix_riae,
    interior_frequency_slice,
)


ROOT = Path(__file__).resolve().parent
NOISE_4A = ROOT / "noise_4a"
NOISE_5A = ROOT / "noise_5a"
CACHE_DIR = ROOT / ".collect_results_cache"

ETA = "eta0.5"
DURATIONS = ("1m", "6m", "1y")
MAX_POSTERIOR_DRAWS = 20
PSD_WELCH_DAYS = 7.0
COHERENCE_WELCH_DAYS = 56.0
CHANNELS = ("X", "Y", "Z")
CHANNEL_PAIRS = (
    (0, 0, "X"),
    (1, 1, "Y"),
    (2, 2, "Z"),
    (1, 0, "Y-X"),
    (2, 0, "Z-X"),
    (2, 1, "Z-Y"),
)

DURATION_COLORS = {
    "1m": "tab:blue",
    "6m": "tab:orange",
    "1y": "tab:green",
}
DURATION_LABELS = {
    "1m": "1 month",
    "6m": "6 months",
    "1y": "12 months",
}

FMIN_PLOT = 1e-4
FMAX_PLOT = 1e-1
PSD_FLOOR = 1e-50
FIG_DPI = 300

POSTERIOR_FILL_ALPHA = 0.26
POSTERIOR_FILL_ALPHA_PSD = 0.34
POSTERIOR_EDGE_ALPHA = 0.95
POSTERIOR_EDGE_WIDTH = 1.0
WELCH_COLOR = "0.78"
WELCH_ALPHA = 0.95
WELCH_WIDTH = 1.1
ANALYTIC_WIDTH = 1.2
LABEL_FONT_SIZE = 15


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


@dataclass(frozen=True)
class RunData:
    """Posterior quantiles for one `(noise, duration, eta)` run."""

    noise: str
    duration: str
    eta: str
    freq: np.ndarray  # (F,)
    q05: np.ndarray  # (F, 3, 3)
    q50: np.ndarray  # (F, 3, 3)
    q95: np.ndarray  # (F, 3, 3)
    coh_q05: np.ndarray | None  # (F, 3, 3)
    coh_q50: np.ndarray | None  # (F, 3, 3)
    coh_q95: np.ndarray | None  # (F, 3, 3)


def import_module_from_path(name: str, path: Path):
    """Import a module from an explicit filesystem path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lp4 = import_module_from_path("noise4_load_and_plot_data", NOISE_4A / "load_and_plot_data.py")
lp5 = import_module_from_path("noise5_load_and_plot_data", NOISE_5A / "load_and_plot_data.py")


def _coherence_from_matrix(s: np.ndarray, i: int, j: int) -> np.ndarray:
    """Return magnitude coherence for spectral matrices with shape `(F, 3, 3)`."""
    sii = np.maximum(np.asarray(s[:, i, i].real, dtype=np.float64), 0.0)
    sjj = np.maximum(np.asarray(s[:, j, j].real, dtype=np.float64), 0.0)
    denom = np.sqrt(sii * sjj)
    return np.clip(
        np.divide(
            np.abs(np.asarray(s[:, i, j], dtype=np.complex128)),
            denom,
            out=np.zeros_like(denom, dtype=np.float64),
            where=denom > 0.0,
        ),
        0.0,
        1.0,
    )


def _find_inference_file(run_dir: Path) -> Path | None:
    """Return the canonical inference NetCDF if present."""
    candidates = (
        run_dir / "inference_data.nc",
        run_dir / "idata.nc",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _cache_is_fresh(cache_path: Path, source_paths: tuple[Path, ...]) -> bool:
    """Return whether `cache_path` is newer than all `source_paths`."""
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    return all(path.exists() and cache_mtime >= path.stat().st_mtime for path in source_paths)


def _load_pickle(cache_path: Path):
    """Load a cached Python object from disk."""
    with cache_path.open("rb") as handle:
        return pickle.load(handle)


def _save_pickle(cache_path: Path, payload) -> None:
    """Persist a cached Python object to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _collect_runtime_seconds(run_dir: Path) -> float:
    """Best-effort runtime estimate from earliest and latest output mtimes."""
    if not run_dir.exists():
        return float("nan")

    candidates = [
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".nc", ".png", ".npy", ".csv", ".txt"}
    ]
    if len(candidates) < 2:
        return float("nan")

    mtimes = [path.stat().st_mtime for path in candidates]
    runtime = max(mtimes) - min(mtimes)
    return float(runtime) if runtime >= 0.0 else float("nan")


def load_run(noise_dir: Path, duration: str, eta: str) -> RunData | None:
    """Load posterior PSD/coherence quantiles from one run directory."""
    run_dir = noise_dir / "mcmc_output" / duration / eta
    nc_path = _find_inference_file(run_dir)
    if nc_path is None:
        print(f"  [skip] {run_dir} has no inference_data.nc/idata.nc")
        return None

    cache_path = CACHE_DIR / f"{noise_dir.name}_{duration}_{eta}_quantiles.pkl"
    if _cache_is_fresh(cache_path, (nc_path, Path(__file__))):
        try:
            return _load_pickle(cache_path)
        except (AttributeError, ModuleNotFoundError, pickle.UnpicklingError, EOFError):
            cache_path.unlink(missing_ok=True)

    print(f"  Loading {nc_path}")
    idata = xr.open_datatree(str(nc_path))
    try:
        n_keep_attr = idata.attrs.get("posterior_psd_max_draws")
        n_keep = MAX_POSTERIOR_DRAWS
        if n_keep_attr is not None:
            n_keep = min(int(n_keep_attr), MAX_POSTERIOR_DRAWS)
        quantiles = _get_fast_multivar_quantiles(
            idata,
            n_keep=n_keep,
            percentiles=(5.0, 50.0, 95.0),
        )
    finally:
        idata.close()

    spectral_density = np.asarray(
        quantiles["spectral_density"], dtype=np.complex128
    )
    coherence = quantiles.get("coherence")
    coherence_arr = (
        np.asarray(coherence, dtype=np.float64)
        if coherence is not None
        else None
    )

    run = RunData(
        noise=noise_dir.name,
        duration=duration,
        eta=eta,
        freq=np.asarray(quantiles["freq"], dtype=np.float64),
        q05=np.asarray(spectral_density[0].real, dtype=np.float64),
        q50=np.asarray(spectral_density[1].real, dtype=np.float64),
        q95=np.asarray(spectral_density[2].real, dtype=np.float64),
        coh_q05=None if coherence_arr is None else coherence_arr[0],
        coh_q50=None if coherence_arr is None else coherence_arr[1],
        coh_q95=None if coherence_arr is None else coherence_arr[2],
    )
    _save_pickle(cache_path, run)
    return run


def _get_fast_multivar_quantiles(
    idata: xr.DataTree,
    *,
    n_keep: int,
    percentiles: tuple[float, ...],
) -> dict[str, np.ndarray | None]:
    """Reconstruct capped posterior PSD draws before computing quantiles."""
    posterior = get_sample_dataset(idata, source="posterior")
    spline_model = get_multivar_spline_model(idata)
    params = _get_multivar_reconstruction_inputs_from_dataset(
        posterior,
        spline_model,
        n_keep=n_keep,
    )

    n_samples = int(params["log_delta_sq"].shape[0])
    spectral_density = spline_model.reconstruct_psd_matrix(
        params["log_delta_sq"],
        params["theta_re"],
        params["theta_im"],
        n_samples_max=n_samples,
    )
    spectral_density = np.moveaxis(np.asarray(spectral_density), 1, -1)
    spectral_density = _rescale_multivar_psd(idata, spectral_density[:, None, ...])

    dataset = _build_psd_dataset(
        spectral_density=spectral_density,
        freq=_get_multivar_frequency_grid(idata),
        chain_count=int(spectral_density.shape[0]),
        draw_count=int(spectral_density.shape[1]),
    )
    return _quantiles_from_psd_draws(
        dataset,
        n_keep=None,
        percentiles=percentiles,
        freq_idx=None,
    )


def _compute_welch_matrix(
    noise_dir: Path,
    lpmod,
    duration_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a Welch spectral matrix from the first `duration_days` of trimmed data."""
    lisa = lpmod.LISAData.load(noise_dir / "tdi.h5")
    n_samples = min(
        lisa.data.shape[0],
        int(duration_days * 86400.0 / lisa.delta_t),
    )
    if n_samples < 2:
        raise ValueError(
            f"Not enough samples for {duration_days:g} days of Welch data."
        )
    fs = 1.0 / lisa.delta_t
    x = lisa.data[:n_samples, 0]
    y = lisa.data[:n_samples, 1]
    z = lisa.data[:n_samples, 2]
    freq, sxx, syy, szz, sxy, syz, szx = lpmod.welch_spectral_matrix_xyz(
        x,
        y,
        z,
        fs=fs,
        nperseg=lpmod.DEFAULT_NPERSEG,
    )
    matrix = lpmod.spectral_matrix_from_components(sxx, syy, szz, sxy, syz, szx)
    return np.asarray(freq, dtype=np.float64), np.asarray(matrix, dtype=np.complex128)


def load_reference_4a() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(psd_freq, psd_welch, coh_freq, coh_welch, analytic_matrix)`."""
    cache_path = CACHE_DIR / "noise_4a_reference.pkl"
    source_paths = (
        NOISE_4A / "tdi.h5",
        NOISE_4A / "load_and_plot_data.py",
        Path(__file__),
    )
    if _cache_is_fresh(cache_path, source_paths):
        return _load_pickle(cache_path)

    lisa = lp4.LISAData.load(NOISE_4A / "tdi.h5")
    psd_freq, psd_welch = _compute_welch_matrix(NOISE_4A, lp4, PSD_WELCH_DAYS)
    coh_freq, coh_welch = _compute_welch_matrix(
        NOISE_4A,
        lp4,
        COHERENCE_WELCH_DAYS,
    )
    n_full = len(lisa.time) + 2 * lp4.EDGE_TRIM
    analytic_matrix = lp4.analytic_covariance_from_model(
        lisa.freq,
        dt=lisa.delta_t,
        n=n_full,
    )
    reference = (
        psd_freq,
        psd_welch,
        coh_freq,
        coh_welch,
        np.asarray(analytic_matrix, dtype=np.complex128),
    )
    _save_pickle(cache_path, reference)
    return reference


def load_reference_5a() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(psd_freq, psd_welch, coh_freq, coh_welch)` for `noise_5a`."""
    cache_path = CACHE_DIR / "noise_5a_reference.pkl"
    source_paths = (
        NOISE_5A / "tdi.h5",
        NOISE_5A / "load_and_plot_data.py",
        Path(__file__),
    )
    if _cache_is_fresh(cache_path, source_paths):
        return _load_pickle(cache_path)

    reference = (
        *_compute_welch_matrix(NOISE_5A, lp5, PSD_WELCH_DAYS),
        *_compute_welch_matrix(NOISE_5A, lp5, COHERENCE_WELCH_DAYS),
    )
    _save_pickle(cache_path, reference)
    return reference


def build_results_dataframe(runs: list[RunData]) -> pd.DataFrame:
    """Build one long-format DataFrame covering PSD and coherence summaries."""
    rows: list[dict[str, float | str | int]] = []

    for run in runs:
        for i, j, pair_label in CHANNEL_PAIRS:
            if i == j:
                quantity = "psd"
                q05 = np.maximum(run.q05[:, i, j], PSD_FLOOR)
                q50 = np.maximum(run.q50[:, i, j], PSD_FLOOR)
                q95 = np.maximum(run.q95[:, i, j], PSD_FLOOR)
                width = q95 - q05
            else:
                quantity = "coherence"
                if run.coh_q05 is not None and run.coh_q50 is not None and run.coh_q95 is not None:
                    q05 = np.clip(run.coh_q05[:, i, j], 0.0, 1.0)
                    q50 = np.clip(run.coh_q50[:, i, j], 0.0, 1.0)
                    q95 = np.clip(run.coh_q95[:, i, j], 0.0, 1.0)
                else:
                    q05 = _coherence_from_matrix(run.q05, i, j)
                    q50 = _coherence_from_matrix(run.q50, i, j)
                    q95 = _coherence_from_matrix(run.q95, i, j)
                width = np.clip(q95 - q05, 0.0, 1.0)

            for k, freq in enumerate(run.freq):
                rows.append(
                    {
                        "noise": run.noise,
                        "duration": run.duration,
                        "eta": run.eta,
                        "quantity": quantity,
                        "channel_i": i,
                        "channel_j": j,
                        "channel_label": pair_label,
                        "freq_hz": float(freq),
                        "q05": float(q05[k]),
                        "q50": float(q50[k]),
                        "q95": float(q95[k]),
                        "posterior_width": float(width[k]),
                    }
                )

    return pd.DataFrame(rows)


def _plot_reference_curve(
    ax: plt.Axes,
    freq: np.ndarray,
    values: np.ndarray,
    *,
    is_psd: bool,
    color: str,
    label: str,
    linestyle: str = "-",
    linewidth: float = 1.0,
    alpha: float = 1.0,
    zorder: int = 1,
) -> None:
    """Draw a PSD or coherence reference curve with consistent styling."""
    if is_psd:
        ax.loglog(
            freq,
            np.maximum(np.asarray(values, dtype=np.float64), PSD_FLOOR),
            color=color,
            ls=linestyle,
            lw=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )
    else:
        ax.semilogx(
            freq,
            np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0),
            color=color,
            ls=linestyle,
            lw=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )


def _plot_posterior_band(
    ax: plt.Axes,
    freq: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    is_psd: bool,
    color: str,
    label: str | None = None,
) -> None:
    """Plot a 90% CI band with thin bounds, without a median curve."""
    low_arr = np.asarray(low, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)

    if is_psd:
        low_arr = np.maximum(low_arr, PSD_FLOOR)
        high_arr = np.maximum(high_arr, PSD_FLOOR)
        ax.fill_between(
            freq,
            low_arr,
            high_arr,
            color=color,
            alpha=POSTERIOR_FILL_ALPHA_PSD,
            zorder=4,
            label=label,
        )
        # ax.loglog(
        #     freq,
        #     low_arr,
        #     color=color,
        #     lw=0,
        #     alpha=POSTERIOR_EDGE_ALPHA,
        #     zorder=5,
        # )
        # ax.loglog(
        #     freq,
        #     high_arr,
        #     color=color,
        #     lw=POSTERIOR_EDGE_WIDTH,
        #     alpha=POSTERIOR_EDGE_ALPHA,
        #     zorder=5,
        # )
        return

    low_arr = np.clip(low_arr, 0.0, 1.0)
    high_arr = np.clip(high_arr, 0.0, 1.0)
    ax.fill_between(
        freq,
        low_arr,
        high_arr,
        color=color,
        alpha=POSTERIOR_FILL_ALPHA,
        zorder=4,
        label=label,
    )
    # ax.semilogx(
    #     freq,
    #     low_arr,
    #     color=color,
    #     lw=0,
    #     alpha=POSTERIOR_EDGE_ALPHA,
    #     zorder=5,
    # )
    # ax.semilogx(
    #     freq,
    #     high_arr,
    #     color=color,
    #     lw=0,
    #     alpha=POSTERIOR_EDGE_ALPHA,
    #     zorder=5,
    # )


def _make_legend_handles(include_analytic: bool) -> list[object]:
    """Build a compact figure-level legend for the triangle plots."""
    handles: list[object] = [
        Line2D(
            [0],
            [0],
            color=WELCH_COLOR,
            lw=WELCH_WIDTH,
            alpha=WELCH_ALPHA,
            label="Welch PSD",
        )
    ]
    for duration in DURATIONS:
        handles.append(
            Patch(
                facecolor=DURATION_COLORS[duration],
                edgecolor=DURATION_COLORS[duration],
                alpha=POSTERIOR_FILL_ALPHA,
                label=f"{DURATION_LABELS[duration]} 90% CI",
            )
        )
    return handles


def _panel_math_label(i: int, j: int) -> str:
    """Return manuscript-style math label for one panel."""
    if i == j:
        ch = CHANNELS[i]
        return rf"$S_{{{ch}{ch}}}$"
    return rf"$C_{{{CHANNELS[i]}{CHANNELS[j]}}}$"


def _set_psd_axis_limits(ax: plt.Axes, values: list[np.ndarray]) -> None:
    """Set robust PSD limits without letting exact nulls dominate the range."""
    positive: list[np.ndarray] = []
    for arr in values:
        flat = np.asarray(arr, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat) & (flat > 0.0)]
        if flat.size > 0:
            positive.append(flat)
    if not positive:
        return
    merged = np.concatenate(positive)
    ylo = max(float(np.quantile(merged, 0.01)) * 0.35, PSD_FLOOR)
    yhi = float(np.quantile(merged, 0.995)) * 4.0
    if yhi > ylo:
        ax.set_ylim(ylo, yhi)


def _interpolate_spectral_matrix(
    src_freq: np.ndarray,
    src_matrix: np.ndarray,
    dst_freq: np.ndarray,
) -> np.ndarray:
    """Interpolate a complex spectral matrix onto a new frequency grid."""
    out = np.zeros((len(dst_freq), src_matrix.shape[1], src_matrix.shape[2]), dtype=np.complex128)
    for i in range(src_matrix.shape[1]):
        for j in range(src_matrix.shape[2]):
            out[:, i, j] = np.interp(dst_freq, src_freq, src_matrix[:, i, j].real)
            if np.iscomplexobj(src_matrix):
                out[:, i, j] = out[:, i, j] + 1j * np.interp(
                    dst_freq,
                    src_freq,
                    src_matrix[:, i, j].imag,
                )
    return out


def build_summary_dataframe(
    runs: list[RunData],
    *,
    analytic_freq_4a: np.ndarray | None = None,
    analytic_matrix_4a: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build one summary row per run with CI width and analytical diagnostics."""
    rows: list[dict[str, float | str]] = []

    for run in runs:
        diag_q05 = np.stack(
            [np.maximum(run.q05[:, ch, ch], PSD_FLOOR) for ch in range(3)],
            axis=1,
        )
        diag_q50 = np.stack(
            [np.maximum(run.q50[:, ch, ch], PSD_FLOOR) for ch in range(3)],
            axis=1,
        )
        diag_q95 = np.stack(
            [np.maximum(run.q95[:, ch, ch], PSD_FLOOR) for ch in range(3)],
            axis=1,
        )
        diag_width = np.maximum(diag_q95 - diag_q05, 0.0)
        diag_rel_width = np.divide(
            diag_width,
            diag_q50,
            out=np.full_like(diag_width, np.nan, dtype=np.float64),
            where=diag_q50 > 0.0,
        )
        diag_log_ratio = np.log(diag_q95 / diag_q05)
        coh_width = np.stack(
            [
                (
                    np.clip(run.coh_q95[:, i, j] - run.coh_q05[:, i, j], 0.0, 1.0)
                    if run.coh_q05 is not None and run.coh_q95 is not None
                    else np.clip(
                        _coherence_from_matrix(run.q95, i, j)
                        - _coherence_from_matrix(run.q05, i, j),
                        0.0,
                        1.0,
                    )
                )
                for i, j in ((1, 0), (2, 0), (2, 1))
            ],
            axis=1,
        )

        row: dict[str, float | str] = {
            "noise": run.noise,
            "duration": run.duration,
            "eta": run.eta,
            "runtime_seconds_est": _collect_runtime_seconds(
                ROOT / run.noise / "mcmc_output" / run.duration / run.eta
            ),
            "ci_width_psd_diag_mean": float(np.nanmean(diag_width)),
            "ci_width_psd_diag_median": float(np.nanmedian(diag_width)),
            "ci_width_psd_diag_rel_mean": float(np.nanmean(diag_rel_width)),
            "ci_width_psd_diag_rel_median": float(np.nanmedian(diag_rel_width)),
            "ci_width_psd_diag_logratio_mean": float(np.nanmean(diag_log_ratio)),
            "ci_width_psd_diag_logratio_median": float(np.nanmedian(diag_log_ratio)),
            "ci_width_coherence_mean": float(np.nanmean(coh_width)),
            "ci_width_coherence_median": float(np.nanmedian(coh_width)),
            "riae_analytic": float("nan"),
        }

        if (
            run.noise == "noise_4a"
            and analytic_freq_4a is not None
            and analytic_matrix_4a is not None
        ):
            analytic_on_run_freq = _interpolate_spectral_matrix(
                analytic_freq_4a,
                analytic_matrix_4a,
                run.freq,
            )
            freq_slice = interior_frequency_slice(len(run.freq))
            posterior_stack = np.stack([run.q05, run.q50, run.q95], axis=0)
            coherence_stack = (
                np.stack([run.coh_q05, run.coh_q50, run.coh_q95], axis=0)
                if run.coh_q05 is not None and run.coh_q50 is not None and run.coh_q95 is not None
                else None
            )

            row["riae_analytic"] = float(
                compute_matrix_riae(
                    run.q50[freq_slice],
                    analytic_on_run_freq[freq_slice],
                    run.freq[freq_slice],
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def plot_triangle(
    *,
    noise_label: str,
    runs: list[RunData],
    psd_welch_freq: np.ndarray,
    psd_welch_matrix: np.ndarray,
    coh_welch_freq: np.ndarray,
    coh_welch_matrix: np.ndarray,
    analytic_matrix: np.ndarray | None,
    out_path: Path,
) -> None:
    """Plot PSDs on the diagonal and coherences on the lower triangle."""
    fig, axes = plt.subplots(3, 3, figsize=(9.8, 8.4), constrained_layout=False)

    runs_by_duration = {
        run.duration: run
        for run in sorted(runs, key=lambda item: DURATIONS.index(item.duration))
    }

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]

            if i < j:
                ax.axis("off")
                continue

            is_psd = i == j

            if is_psd:
                welch_freq = psd_welch_freq
                welch_values = psd_welch_matrix[:, i, i].real
            else:
                welch_freq = coh_welch_freq
                welch_values = _coherence_from_matrix(coh_welch_matrix, i, j)
            _plot_reference_curve(
                ax,
                welch_freq,
                welch_values,
                is_psd=is_psd,
                color=WELCH_COLOR,
                label="Welch",
                linewidth=WELCH_WIDTH,
                alpha=WELCH_ALPHA,
                zorder=2,
            )

            for duration in DURATIONS:
                run = runs_by_duration.get(duration)
                if run is None:
                    continue

                color = DURATION_COLORS[duration]
                label = DURATION_LABELS[duration]

                if is_psd:
                    q05 = np.maximum(run.q05[:, i, i], PSD_FLOOR)
                    q95 = np.maximum(run.q95[:, i, i], PSD_FLOOR)
                    _plot_posterior_band(
                        ax,
                        run.freq,
                        q05,
                        q95,
                        is_psd=True,
                        color=color,
                        label=label,
                    )
                else:
                    if run.coh_q05 is not None and run.coh_q50 is not None and run.coh_q95 is not None:
                        cq05 = np.clip(run.coh_q05[:, i, j], 0.0, 1.0)
                        cq95 = np.clip(run.coh_q95[:, i, j], 0.0, 1.0)
                    else:
                        cq05 = _coherence_from_matrix(run.q05, i, j)
                        cq95 = _coherence_from_matrix(run.q95, i, j)
                    _plot_posterior_band(
                        ax,
                        run.freq,
                        cq05,
                        cq95,
                        is_psd=False,
                        color=color,
                        label=label,
                    )

            ax.set_xlim(FMIN_PLOT, FMAX_PLOT)
            ax.grid(True, which="major", ls=":", alpha=0.35)
            ax.grid(True, which="minor", ls=":", alpha=0.18)

            if is_psd:
                _set_psd_axis_limits(
                    ax,
                    [
                        psd_welch_matrix[:, i, i].real,
                        *[
                            np.maximum(run.q05[:, i, i], PSD_FLOOR)
                            for run in runs_by_duration.values()
                        ],
                        *[
                            np.maximum(run.q95[:, i, i], PSD_FLOOR)
                            for run in runs_by_duration.values()
                        ],
                    ],
                )
                ax.set_ylabel("PSD [1/Hz]")
            else:
                ax.set_ylim(-0.01, 1.0)
                ax.set_ylabel("Coherence")

            ax.text(
                0.04,
                0.93,
                _panel_math_label(i, j),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=LABEL_FONT_SIZE,
            )

            if i == 2:
                ax.set_xlabel("Frequency [Hz]")
            else:
                ax.tick_params(labelbottom=False)

    fig.legend(
        handles=_make_legend_handles(include_analytic=False),
        loc="upper right",
        ncol=1,
        bbox_to_anchor=(0.975, 0.985),
        frameon=False,
        columnspacing=0.9,
        handlelength=2.2,
        handletextpad=0.6,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985), h_pad=0.55, w_pad=0.65)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main() -> None:
    """Collect eta=0.5 results, write CSV, and generate triangle plots."""
    runs_4a: list[RunData] = []
    runs_5a: list[RunData] = []

    print("Loading noise_4a eta=0.5 runs...")
    for duration in DURATIONS:
        run = load_run(NOISE_4A, duration, ETA)
        if run is not None:
            runs_4a.append(run)

    print("Loading noise_5a eta=0.5 runs...")
    for duration in DURATIONS:
        run = load_run(NOISE_5A, duration, ETA)
        if run is not None:
            runs_5a.append(run)

    all_runs = runs_4a + runs_5a
    if not all_runs:
        raise FileNotFoundError("No eta=0.5 runs were found under noise_4a/noise_5a.")

    print("Building aggregated DataFrame...")
    results_df = build_results_dataframe(all_runs)
    csv_path = ROOT / "results_eta0p5.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path} ({len(results_df):,} rows)")

    print("Loading reference spectra...")
    psd_freq_4a, psd_welch_4a, coh_freq_4a, coh_welch_4a, analytic_4a = load_reference_4a()
    psd_freq_5a, psd_welch_5a, coh_freq_5a, coh_welch_5a = load_reference_5a()

    print("Building summary DataFrame...")
    summary_df = build_summary_dataframe(
        all_runs,
        analytic_freq_4a=psd_freq_4a,
        analytic_matrix_4a=analytic_4a,
    )
    summary_csv_path = ROOT / "results_eta0p5_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  Saved {summary_csv_path} ({len(summary_df):,} rows)")

    if runs_4a:
        print("Plotting noise_4a triangle...")
        plot_triangle(
            noise_label="noise_4a (eta=0.5)",
            runs=runs_4a,
            psd_welch_freq=psd_freq_4a,
            psd_welch_matrix=psd_welch_4a,
            coh_welch_freq=coh_freq_4a,
            coh_welch_matrix=coh_welch_4a,
            analytic_matrix=analytic_4a,
            out_path=ROOT / "triangle_noise4a_eta0p5.png",
        )

    if runs_5a:
        print("Plotting noise_5a triangle...")
        plot_triangle(
            noise_label="noise_5a (eta=0.5)",
            runs=runs_5a,
            psd_welch_freq=psd_freq_5a,
            psd_welch_matrix=psd_welch_5a,
            coh_welch_freq=coh_freq_5a,
            coh_welch_matrix=coh_welch_5a,
            analytic_matrix=None,
            out_path=ROOT / "triangle_noise5a_eta0p5.png",
        )

    print("Done.")


if __name__ == "__main__":
    main()
