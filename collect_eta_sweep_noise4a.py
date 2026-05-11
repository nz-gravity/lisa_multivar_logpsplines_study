#!/usr/bin/env python3
"""
Collect eta-sweep diagnostics for `noise_4a`.

Outputs written to the repo root:
  - results_eta_sweep_noise4a.csv

One row per `(duration, eta)` run, with:
  - analytical RIAE
  - analytical coverage
  - relative PSD CI width summaries
  - divergence / max-tree-depth failure flags
"""

from __future__ import annotations

import pickle
from pathlib import Path

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


ROOT = Path(__file__).resolve().parent
NOISE_4A = ROOT / "noise_4a"
OUT_CSV = ROOT / "results_eta_sweep_noise4a.csv"
CACHE_DIR = ROOT / ".collect_results_cache"

DURATIONS = ("1m", "6m", "1y")
PERCENTILES = (5.0, 50.0, 95.0)
PSD_FLOOR = 1e-50
MAX_POSTERIOR_DRAWS = 20

def _eta_value(eta_label: str) -> float:
    """Parse labels like `eta0.03`, `eta0.5`, `eta1`."""
    return float(eta_label.removeprefix("eta"))


def _find_inference_file(run_dir: Path) -> Path | None:
    """Return the primary inference NetCDF if present."""
    for name in ("inference_data.nc", "idata.nc"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def _find_eta_dirs(duration: str) -> list[Path]:
    """Return eta subdirectories for one duration, sorted numerically."""
    duration_dir = NOISE_4A / "mcmc_output" / duration
    eta_dirs = [
        path
        for path in duration_dir.iterdir()
        if path.is_dir() and path.name.startswith("eta")
    ]
    return sorted(eta_dirs, key=lambda path: _eta_value(path.name))


def _cache_is_fresh(cache_path: Path, source_paths: tuple[Path, ...]) -> bool:
    """Return whether `cache_path` is newer than every file in `source_paths`."""
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


def _load_quantiles(nc_path: Path, duration: str, eta_label: str) -> dict[str, np.ndarray | None]:
    """Load cached posterior PSD quantiles from one NetCDF."""
    cache_path = CACHE_DIR / f"noise_4a_{duration}_{eta_label}_eta_sweep_quantiles.pkl"
    if _cache_is_fresh(cache_path, (nc_path, Path(__file__))):
        return _load_pickle(cache_path)

    idata = xr.open_datatree(str(nc_path))
    try:
        quantiles = _get_fast_multivar_quantiles(
            idata,
            n_keep=MAX_POSTERIOR_DRAWS,
            percentiles=PERCENTILES,
        )
    finally:
        idata.close()

    _save_pickle(cache_path, quantiles)
    return quantiles


def _relative_psd_ci_widths(psd_quantiles: np.ndarray) -> tuple[float, float]:
    """Return mean/median relative width over diagonal PSD bins."""
    q05 = np.maximum(np.asarray(psd_quantiles[0].real, dtype=np.float64), PSD_FLOOR)
    q50 = np.maximum(np.asarray(psd_quantiles[1].real, dtype=np.float64), PSD_FLOOR)
    q95 = np.maximum(np.asarray(psd_quantiles[2].real, dtype=np.float64), PSD_FLOOR)

    diag_q05 = np.stack([q05[:, ch, ch] for ch in range(3)], axis=1)
    diag_q50 = np.stack([q50[:, ch, ch] for ch in range(3)], axis=1)
    diag_q95 = np.stack([q95[:, ch, ch] for ch in range(3)], axis=1)

    rel_width = np.divide(
        np.maximum(diag_q95 - diag_q05, 0.0),
        diag_q50,
        out=np.full_like(diag_q50, np.nan, dtype=np.float64),
        where=diag_q50 > 0.0,
    )
    return float(np.nanmean(rel_width)), float(np.nanmedian(rel_width))


def _read_diagnostics(run_dir: Path) -> dict[str, float]:
    """Read saved sampler diagnostics for one eta run."""
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_csv = diagnostics_dir / "diagnostics.csv"
    nuts_summary_csv = diagnostics_dir / "nuts_summary.csv"

    metrics = {
        "nuts_riae": float("nan"),
        "nuts_coverage": float("nan"),
        "nuts_divergences": 0.0,
        "nuts_rhat_max": float("nan"),
        "max_treedepth_hits": 0.0,
        "failed": False,
    }

    if diagnostics_csv.exists():
        df = pd.read_csv(diagnostics_csv)
        if not df.empty:
            row = df.iloc[0]
            metrics["nuts_riae"] = float(row.get("nuts_riae", np.nan))
            metrics["nuts_coverage"] = float(row.get("nuts_coverage", np.nan))
            metrics["nuts_divergences"] = float(row.get("nuts_divergences", 0.0))
            metrics["nuts_rhat_max"] = float(row.get("nuts_rhat_max", np.nan))

    if nuts_summary_csv.exists():
        df = pd.read_csv(nuts_summary_csv)
        if not df.empty:
            metrics["max_treedepth_hits"] = float(
                pd.to_numeric(df.get("max_treedepth_hits", 0.0), errors="coerce").fillna(0.0).sum()
            )
            metrics["nuts_divergences"] = float(
                pd.to_numeric(df.get("divergences", 0.0), errors="coerce").fillna(0.0).sum()
            )
            metrics["nuts_rhat_max"] = float(
                pd.to_numeric(df.get("rhat_max", np.nan), errors="coerce").max()
            )

    metrics["failed"] = bool(
        metrics["nuts_divergences"] > 0.0 or metrics["max_treedepth_hits"] > 0.0
    )
    return metrics


def main() -> None:
    """Collect eta-sweep rows and write a summary CSV."""
    rows: list[dict[str, float | str | bool]] = []

    for duration in DURATIONS:
        for eta_dir in _find_eta_dirs(duration):
            eta_label = eta_dir.name
            nc_path = _find_inference_file(eta_dir)
            if nc_path is None:
                print(f"[skip] {duration} {eta_label}: no inference NetCDF")
                continue

            print(f"Loading {duration} {eta_label} from {nc_path}")
            quantiles = _load_quantiles(nc_path, duration, eta_label)
            psd_quantiles = np.asarray(
                quantiles["spectral_density"],
                dtype=np.complex128,
            )
            diag_rel_mean, diag_rel_median = _relative_psd_ci_widths(psd_quantiles)
            diagnostics = _read_diagnostics(eta_dir)

            rows.append(
                {
                    "duration": duration,
                    "eta": eta_label,
                    "eta_value": _eta_value(eta_label),
                    "riae": diagnostics["nuts_riae"],
                    "coverage": diagnostics["nuts_coverage"],
                    "ci_width_rel_psd_diag_mean": diag_rel_mean,
                    "ci_width_rel_psd_diag_median": diag_rel_median,
                    "nuts_divergences": diagnostics["nuts_divergences"],
                    "max_treedepth_hits": diagnostics["max_treedepth_hits"],
                    "nuts_rhat_max": diagnostics["nuts_rhat_max"],
                    "failed": diagnostics["failed"],
                }
            )

    df = pd.DataFrame(rows).sort_values(
        by=["duration", "eta_value"],
        kind="stable",
    )
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
