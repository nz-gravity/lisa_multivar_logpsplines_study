"""Post-processing pipeline for the LISA baseline/full-covariance comparison.

Scans completed runs and writes:
  out/results.csv              per-run RIAE, credible-band widths, runtimes
  out/triangle_{dataset}_full_xyz.png  full-XYZ spectral matrix CI plots

Usage:
    python postproc.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import xarray as xr

from src.aet import aet_to_xyz_matrix
from src.load_data import (
    DEFAULT_NPERSEG,
    LISAData,
    interpolate_spectral_matrix,
    spectral_matrix_from_components,
    welch_spectral_matrix_xyz,
)
from log_psplines.arviz_utils.from_arviz import (
    _build_psd_dataset,
    _get_multivar_frequency_grid,
    _get_multivar_reconstruction_inputs_from_dataset,
    _quantiles_from_psd_draws,
    _rescale_multivar_psd,
    get_multivar_spline_model,
    get_sample_dataset,
)
from log_psplines.diagnostics._utils import compute_matrix_riae, interior_frequency_slice


OUT = Path("out")
DATASETS = ("noise4a", "noise5a")
DATASET_REFS = {"noise4a": "analytic", "noise5a": "segwo"}
RESULTS_CSV = OUT / "results.csv"

MAX_POSTERIOR_DRAWS = 20
PSD_FLOOR = 1e-50
DURATIONS = ("14d", "1m", "6m", "1y")
DURATION_COLORS = {"14d": "tab:purple", "1m": "tab:blue", "6m": "tab:orange", "1y": "tab:green"}
DURATION_LABELS = {"14d": "14 days", "1m": "1 month", "6m": "6 months", "1y": "12 months"}
FMIN_PLOT, FMAX_PLOT = 1e-4, 1e-1
CHANNELS = ("X", "Y", "Z")
WELCH_COLOR = "0.78"
POSTERIOR_FILL_ALPHA = 0.26
POSTERIOR_FILL_ALPHA_PSD = 0.34
FIG_DPI = 300
LABEL_FONT_SIZE = 15

RESULTS_FIELDS = (
    "dataset",
    "model",
    "duration",
    "eta",
    "riae",
    "median_psd_rel_width",
    "median_coh_rel_width",
    "vi_runtime_s",
    "mcmc_runtime_s",
    "pipeline_runtime_s",
)


class ModelOutput(NamedTuple):
    label: str
    outdirs: tuple[str, ...]
    basis: str


MODEL_OUTPUTS = (
    ModelOutput("baseline_aet", ("mcmc_output_baseline_aet",), "AET"),
    ModelOutput("full_xyz", ("mcmc_output_full_xyz",), "XYZ"),
)


def _eta_sort_value(eta: str) -> float:
    if not eta.startswith("eta"):
        return float("-inf")
    try:
        return float(eta[3:])
    except ValueError:
        return float("-inf")


@dataclass(frozen=True)
class RunData:
    dataset: str
    model: str
    duration: str
    eta: str
    freq: np.ndarray
    q05: np.ndarray
    q50: np.ndarray
    q95: np.ndarray
    coh_q05: np.ndarray | None
    coh_q50: np.ndarray | None
    coh_q95: np.ndarray | None


def _find_idata(run_dir: Path) -> Path | None:
    for name in ("idata.nc", "inference_data.nc"):
        p = run_dir / name
        if p.exists():
            return p
    return None


def _load_runtime_metadata(run_dir: Path) -> dict[str, float]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {
            "vi_runtime_s": float("nan"),
            "mcmc_runtime_s": float("nan"),
            "pipeline_runtime_s": float("nan"),
        }
    data = json.loads(path.read_text())
    return {
        "vi_runtime_s": float(data.get("vi_total_runtime_s", data.get("vi_runtime_s", float("nan")))),
        "mcmc_runtime_s": float(data.get("mcmc_runtime_s", float("nan"))),
        "pipeline_runtime_s": float(data.get("pipeline_runtime_s", float("nan"))),
    }


def _load_posterior_quantiles(nc_path: Path) -> dict | None:
    idata = xr.open_datatree(str(nc_path), engine="h5netcdf")
    try:
        posterior = get_sample_dataset(idata, source="posterior")
        spline_model = get_multivar_spline_model(idata)
        params = _get_multivar_reconstruction_inputs_from_dataset(
            posterior, spline_model, n_keep=MAX_POSTERIOR_DRAWS,
        )
        n_samples = int(params["log_delta_sq"].shape[0])
        spectral_density = spline_model.reconstruct_psd_matrix(
            params["log_delta_sq"], params["theta_re"], params["theta_im"],
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
            dataset, n_keep=None, percentiles=(5.0, 50.0, 95.0), freq_idx=None,
        )
    except Exception as exc:
        print(f"  [warn] Could not load posteriors from {nc_path}: {exc}")
        return None
    finally:
        idata.close()


def _safe_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _median_psd_rel_width(sd: np.ndarray) -> float:
    q05, q50, q95 = sd[0].real, sd[1].real, sd[2].real
    widths = []
    for i in range(3):
        denom = np.abs(q50[:, i, i])
        widths.append(
            np.divide(
                q95[:, i, i] - q05[:, i, i],
                denom,
                out=np.full_like(denom, np.nan, dtype=np.float64),
                where=denom > 0.0,
            )
        )
    return _safe_median(np.concatenate(widths))


def _median_coh_rel_width(coh: np.ndarray | None) -> float:
    if coh is None:
        return float("nan")
    q05, q50, q95 = np.asarray(coh[0]), np.asarray(coh[1]), np.asarray(coh[2])
    widths = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        denom = np.abs(q50[:, i, j])
        widths.append(
            np.divide(
                q95[:, i, j] - q05[:, i, j],
                denom,
                out=np.full_like(denom, np.nan, dtype=np.float64),
                where=denom > 0.0,
            )
        )
    return _safe_median(np.concatenate(widths))


def _compute_riae(
    q: dict,
    model: ModelOutput,
    ref_freq: np.ndarray,
    ref_matrix: np.ndarray,
) -> float:
    freq = np.asarray(q["freq"], dtype=np.float64)
    q50 = np.asarray(q["spectral_density"], dtype=np.complex128)[1]
    if model.basis == "AET":
        q50 = aet_to_xyz_matrix(q50)
    ref = interpolate_spectral_matrix(ref_freq, ref_matrix, freq)
    sl = interior_frequency_slice(len(freq))
    return float(compute_matrix_riae(q50[sl], ref[sl], freq[sl]))


def _iter_run_dirs(dataset_dir: Path, model: ModelOutput):
    seen: set[Path] = set()
    for outdir_name in model.outdirs:
        model_dir = dataset_dir / outdir_name
        if not model_dir.exists():
            continue
        for dur_dir in sorted(model_dir.iterdir()):
            if not dur_dir.is_dir():
                continue
            for eta_dir in sorted(dur_dir.iterdir()):
                if not eta_dir.is_dir() or eta_dir in seen:
                    continue
                seen.add(eta_dir)
                yield outdir_name, dur_dir, eta_dir


def collect_results(refs: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    rows: list[dict] = []

    for dataset in DATASETS:
        ds_dir = OUT / dataset
        ref_freq, ref_matrix = refs[dataset]
        for model in MODEL_OUTPUTS:
            for outdir_name, dur_dir, eta_dir in _iter_run_dirs(ds_dir, model):
                nc = _find_idata(eta_dir)
                if nc is None:
                    continue
                print(f"  {dataset}/{outdir_name}/{dur_dir.name}/{eta_dir.name}")
                q = _load_posterior_quantiles(nc)
                if q is None:
                    continue
                sd = np.asarray(q["spectral_density"], dtype=np.complex128)
                runtime = _load_runtime_metadata(eta_dir)
                rows.append({
                    "dataset": dataset,
                    "model": model.label,
                    "duration": dur_dir.name,
                    "eta": eta_dir.name,
                    "riae": _compute_riae(q, model, ref_freq, ref_matrix),
                    "median_psd_rel_width": _median_psd_rel_width(sd),
                    "median_coh_rel_width": _median_coh_rel_width(q.get("coherence")),
                    **runtime,
                })

    if not rows:
        print("  No completed runs found.")
        return

    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {RESULTS_CSV}")


def _to_run_data(dataset: str, model: str, duration: str, eta: str, q: dict) -> RunData:
    sd = np.asarray(q["spectral_density"], dtype=np.complex128)
    coh = q.get("coherence")
    return RunData(
        dataset=dataset, model=model, duration=duration, eta=eta,
        freq=np.asarray(q["freq"], dtype=np.float64),
        q05=sd[0].real, q50=sd[1].real, q95=sd[2].real,
        coh_q05=None if coh is None else np.asarray(coh[0]),
        coh_q50=None if coh is None else np.asarray(coh[1]),
        coh_q95=None if coh is None else np.asarray(coh[2]),
    )


def _coherence_from_matrix(s: np.ndarray, i: int, j: int) -> np.ndarray:
    sii = np.maximum(s[:, i, i].real, 0.0)
    sjj = np.maximum(s[:, j, j].real, 0.0)
    denom = sii * sjj
    return np.clip(
        np.divide(np.abs(s[:, i, j]) ** 2, denom, out=np.zeros(len(s)), where=denom > 0.0),
        0.0, 1.0,
    )


def _panel_math_label(i: int, j: int) -> str:
    ch = CHANNELS[i]
    return rf"$S_{{{ch}{ch}}}$" if i == j else rf"$C_{{{CHANNELS[i]}{CHANNELS[j]}}}$"


def _set_psd_limits(ax: plt.Axes, arrays: list[np.ndarray]) -> None:
    pos = np.concatenate([a.ravel() for a in arrays])
    pos = pos[np.isfinite(pos) & (pos > 0.0)]
    if pos.size == 0:
        return
    ax.set_ylim(
        max(float(np.quantile(pos, 0.01)) * 0.35, PSD_FLOOR),
        float(np.quantile(pos, 0.995)) * 4.0,
    )


def _plot_triangle(
    runs: list[RunData],
    psd_welch_freq: np.ndarray,
    psd_welch_matrix: np.ndarray,
    coh_welch_freq: np.ndarray,
    coh_welch_matrix: np.ndarray,
    out_path: Path,
) -> None:
    runs_by_dur = {r.duration: r for r in runs}
    fig, axes = plt.subplots(3, 3, figsize=(9.8, 8.4), constrained_layout=False)

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue
            is_psd = (i == j)

            if is_psd:
                ax.loglog(
                    psd_welch_freq,
                    np.maximum(psd_welch_matrix[:, i, i].real, PSD_FLOOR),
                    color=WELCH_COLOR,
                    lw=1.1,
                    alpha=0.95,
                    zorder=2,
                )
            else:
                ax.semilogx(
                    coh_welch_freq,
                    _coherence_from_matrix(coh_welch_matrix, i, j),
                    color=WELCH_COLOR,
                    lw=1.1,
                    alpha=0.95,
                    zorder=2,
                )

            for duration in DURATIONS:
                run = runs_by_dur.get(duration)
                if run is None:
                    continue
                color = DURATION_COLORS[duration]
                if is_psd:
                    ax.fill_between(
                        run.freq,
                        np.maximum(run.q05[:, i, i], PSD_FLOOR),
                        np.maximum(run.q95[:, i, i], PSD_FLOOR),
                        color=color,
                        alpha=POSTERIOR_FILL_ALPHA_PSD,
                        zorder=4,
                    )
                else:
                    cq05 = (
                        np.clip(run.coh_q05[:, i, j], 0.0, 1.0)
                        if run.coh_q05 is not None else _coherence_from_matrix(run.q05, i, j)
                    )
                    cq95 = (
                        np.clip(run.coh_q95[:, i, j], 0.0, 1.0)
                        if run.coh_q95 is not None else _coherence_from_matrix(run.q95, i, j)
                    )
                    ax.fill_between(run.freq, cq05, cq95, color=color, alpha=POSTERIOR_FILL_ALPHA, zorder=4)

            ax.set_xlim(FMIN_PLOT, FMAX_PLOT)
            ax.grid(True, which="major", ls=":", alpha=0.35)
            ax.grid(True, which="minor", ls=":", alpha=0.18)
            ax.text(0.04, 0.93, _panel_math_label(i, j), transform=ax.transAxes,
                    ha="left", va="top", fontsize=LABEL_FONT_SIZE)

            if is_psd:
                _set_psd_limits(
                    ax,
                    [psd_welch_matrix[:, i, i].real]
                    + [np.maximum(r.q95[:, i, i], PSD_FLOOR) for r in runs_by_dur.values()],
                )
                ax.set_ylabel("PSD [1/Hz]")
            else:
                ax.set_ylim(-0.01, 1.0)
                ax.set_ylabel("Coherence")

            if i == 2:
                ax.set_xlabel("Frequency [Hz]")
            else:
                ax.tick_params(labelbottom=False)

    handles = [Line2D([0], [0], color=WELCH_COLOR, lw=1.1, alpha=0.95, label="Welch PSD")]
    for dur in DURATIONS:
        if dur in runs_by_dur:
            handles.append(Patch(facecolor=DURATION_COLORS[dur], alpha=POSTERIOR_FILL_ALPHA,
                                 label=f"{DURATION_LABELS[dur]} 90% CI"))
    fig.legend(handles=handles, loc="upper right", ncol=1,
               bbox_to_anchor=(0.975, 0.985), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985), h_pad=0.55, w_pad=0.65)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _welch_for_triangle(lisa: LISAData, days: float) -> tuple[np.ndarray, np.ndarray]:
    fs = 1.0 / lisa.delta_t
    n = min(lisa.data.shape[0], int(days * 86400 / lisa.delta_t))
    freq, *comps = welch_spectral_matrix_xyz(*lisa.data[:n].T, fs=fs, nperseg=DEFAULT_NPERSEG)
    return freq, spectral_matrix_from_components(*comps)


def _select_plot_eta(ds_dir: Path, model: ModelOutput) -> str | None:
    eta_durations: dict[str, set[str]] = {}
    for _, dur_dir, eta_dir in _iter_run_dirs(ds_dir, model):
        if _find_idata(eta_dir) is None:
            continue
        eta_durations.setdefault(eta_dir.name, set()).add(dur_dir.name)
    if not eta_durations:
        return None
    return max(
        eta_durations,
        key=lambda eta: (len(eta_durations[eta]), _eta_sort_value(eta), eta),
    )


def plot_full_xyz_triangles() -> None:
    full_xyz = next(model for model in MODEL_OUTPUTS if model.label == "full_xyz")
    for dataset in DATASETS:
        ds_dir = OUT / dataset
        plot_eta = _select_plot_eta(ds_dir, full_xyz)
        if plot_eta is None:
            print(f"  No completed full_xyz runs for {dataset}.")
            continue

        runs: list[RunData] = []
        for _, dur_dir, eta_dir in _iter_run_dirs(ds_dir, full_xyz):
            if eta_dir.name != plot_eta:
                continue
            nc = _find_idata(eta_dir)
            if nc is None:
                continue
            q = _load_posterior_quantiles(nc)
            if q is not None:
                runs.append(_to_run_data(dataset, full_xyz.label, dur_dir.name, eta_dir.name, q))

        if not runs:
            print(f"  No completed full_xyz {plot_eta} runs for {dataset}.")
            continue

        print(f"  Plotting {dataset} full_xyz {plot_eta}: {', '.join(sorted(r.duration for r in runs))}")
        lisa = LISAData.load(Path("data") / f"{dataset}.h5", reference=DATASET_REFS[dataset])
        psd_freq, psd_matrix = _welch_for_triangle(lisa, days=7.0)
        coh_freq, coh_matrix = _welch_for_triangle(lisa, days=56.0)
        _plot_triangle(
            runs,
            psd_freq,
            psd_matrix,
            coh_freq,
            coh_matrix,
            OUT / f"triangle_{dataset}_full_xyz.png",
        )


def _load_refs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    refs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dataset, ref_type in DATASET_REFS.items():
        lisa = LISAData.load(Path(f"data/{dataset}.h5"), reference=ref_type)
        refs[dataset] = (lisa.freq, lisa.true_matrix)
    return refs


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for ds in DATASETS:
        (OUT / ds).mkdir(exist_ok=True)

    print("Loading reference spectra...")
    refs = _load_refs()

    print("\nCollecting run diagnostics...")
    collect_results(refs)

    print("\nPlotting full_xyz triangle plots...")
    plot_full_xyz_triangles()

    print("\nDone.")


if __name__ == "__main__":
    main()
