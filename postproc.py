"""Post-processing pipeline: collect results, compare lnZ, plot triangle plots.

Scans out/ for completed MCMC runs, writes:
  out/results.csv      — per-run diagnostics (lnZ, RIAE, runtime)
  out/lnz_summary.csv  — logBF table (lnZ(H4) − lnZ(H{N}) per dataset/duration)
  out/triangle_{dataset}_H4.png — spectral matrix CI triangle plots

Usage:
    python postproc.py
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import xarray as xr

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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT = Path("out")
DATASETS = ("noise4a", "noise5a")
DATASET_REFS = {"noise4a": "analytic", "noise5a": "segwo"}
H0_OUTDIR = "mcmc_output_H0"
MULTIVAR_OUTDIRS = ("mcmc_output_H1", "mcmc_output_H2", "mcmc_output_H4")
RESULTS_CSV = OUT / "results.csv"
LNZ_CSV = OUT / "lnz_summary.csv"

MAX_POSTERIOR_DRAWS = 20
PSD_FLOOR = 1e-50
DURATIONS = ("10d", "1m", "6m", "1y")
DURATION_COLORS = {"10d": "tab:purple", "1m": "tab:blue", "6m": "tab:orange", "1y": "tab:green"}
DURATION_LABELS = {"10d": "10 days", "1m": "1 month", "6m": "6 months", "1y": "12 months"}
FMIN_PLOT, FMAX_PLOT = 1e-4, 1e-1
CHANNELS = ("X", "Y", "Z")
WELCH_COLOR = "0.78"
POSTERIOR_FILL_ALPHA = 0.26
POSTERIOR_FILL_ALPHA_PSD = 0.34
FIG_DPI = 300
LABEL_FONT_SIZE = 15

RESULTS_FIELDS = ("dataset", "model", "duration", "eta", "lnz", "lnz_err", "lnz_valid", "riae", "runtime_s")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunData:
    dataset: str
    model: str
    duration: str
    eta: str
    freq: np.ndarray
    q05: np.ndarray   # (F, 3, 3)
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


def _collect_runtime(run_dir: Path) -> float:
    candidates = [p for p in run_dir.rglob("*") if p.is_file() and p.suffix in {".nc", ".png", ".npy", ".csv"}]
    if len(candidates) < 2:
        return float("nan")
    mtimes = [p.stat().st_mtime for p in candidates]
    return float(max(mtimes) - min(mtimes))


def _read_lnz_from_nc(nc_path: Path) -> dict[str, object]:
    ds = xr.open_datatree(str(nc_path), engine="h5netcdf")
    try:
        attrs = dict(ds.attrs)
    finally:
        ds.close()
    return {k: attrs.get(k, float("nan")) for k in ("lnz", "lnz_err", "lnz_valid")}


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


# ---------------------------------------------------------------------------
# Step 1 — collect results
# ---------------------------------------------------------------------------

def _existing_keys(csv_path: Path) -> set[tuple[str, str, str, str]]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        return {(r["dataset"], r["model"], r["duration"], r["eta"]) for r in csv.DictReader(f)}


def collect_results(refs: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Scan out/ and append new per-run diagnostics to out/results.csv."""
    existing = _existing_keys(RESULTS_CSV)
    new_rows: list[dict] = []

    for dataset in DATASETS:
        ds_dir = OUT / dataset
        ref_freq, ref_matrix = refs[dataset]

        # H1, H2, H4 — each run has a single idata.nc
        for outdir_name in MULTIVAR_OUTDIRS:
            model_dir = ds_dir / outdir_name
            if not model_dir.exists():
                continue
            for dur_dir in sorted(model_dir.iterdir()):
                if not dur_dir.is_dir():
                    continue
                for eta_dir in sorted(dur_dir.iterdir()):
                    if not eta_dir.is_dir():
                        continue
                    nc = _find_idata(eta_dir)
                    if nc is None:
                        continue
                    key = (dataset, outdir_name, dur_dir.name, eta_dir.name)
                    if key in existing:
                        continue
                    print(f"  {dataset}/{outdir_name}/{dur_dir.name}/{eta_dir.name}")
                    attrs = _read_lnz_from_nc(nc)
                    # Compute RIAE from posterior median
                    riae = float("nan")
                    q = _load_posterior_quantiles(nc)
                    if q is not None:
                        freq = np.asarray(q["freq"], dtype=np.float64)
                        q50 = np.asarray(q["spectral_density"], dtype=np.complex128)[1].real
                        ref = interpolate_spectral_matrix(ref_freq, ref_matrix, freq)
                        sl = interior_frequency_slice(len(freq))
                        riae = float(compute_matrix_riae(q50[sl], ref[sl], freq[sl]))
                    new_rows.append({
                        "dataset": dataset, "model": outdir_name,
                        "duration": dur_dir.name, "eta": eta_dir.name,
                        "lnz": attrs["lnz"], "lnz_err": attrs["lnz_err"],
                        "lnz_valid": attrs["lnz_valid"],
                        "riae": riae, "runtime_s": _collect_runtime(eta_dir),
                    })

        # H0 — combined lnZ from lnz_summary.json
        h0_dir = ds_dir / H0_OUTDIR
        if h0_dir.exists():
            for dur_dir in sorted(h0_dir.iterdir()):
                if not dur_dir.is_dir():
                    continue
                for eta_dir in sorted(dur_dir.iterdir()):
                    if not eta_dir.is_dir():
                        continue
                    summary_json = eta_dir / "lnz_summary.json"
                    if not summary_json.exists():
                        continue
                    key = (dataset, H0_OUTDIR, dur_dir.name, eta_dir.name)
                    if key in existing:
                        continue
                    summary = json.loads(summary_json.read_text())
                    new_rows.append({
                        "dataset": dataset, "model": H0_OUTDIR,
                        "duration": dur_dir.name, "eta": eta_dir.name,
                        "lnz": summary.get("combined_lnz", float("nan")),
                        "lnz_err": summary.get("combined_lnz_err", float("nan")),
                        "lnz_valid": summary.get("all_channels_valid", False),
                        "riae": float("nan"),
                        "runtime_s": _collect_runtime(eta_dir),
                    })

    if not new_rows:
        print("  No new runs to add.")
        return

    mode = "a" if RESULTS_CSV.exists() else "w"
    with RESULTS_CSV.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if mode == "w":
            writer.writeheader()
        writer.writerows(new_rows)
    print(f"  Added {len(new_rows)} rows → {RESULTS_CSV}")


# ---------------------------------------------------------------------------
# Step 2 — lnZ comparison
# ---------------------------------------------------------------------------

def compare_lnz() -> None:
    """Compute logBF = lnZ(H4) − lnZ(H{N}) and print + write out/lnz_summary.csv."""
    if not RESULTS_CSV.exists():
        print("  No results.csv found; skipping lnZ comparison.")
        return

    with RESULTS_CSV.open() as f:
        rows = {(r["dataset"], r["duration"], r["eta"], r["model"]): r for r in csv.DictReader(f)}

    summary_rows: list[dict] = []
    for dataset in DATASETS:
        for duration in DURATIONS:
            etas = {k[2] for k in rows if k[0] == dataset and k[1] == duration}
            for eta_label in sorted(etas):
                h4 = rows.get((dataset, duration, eta_label, "mcmc_output_H4"))
                if h4 is None:
                    continue
                for denom_model in ("mcmc_output_H0", "mcmc_output_H1", "mcmc_output_H2"):
                    comp = rows.get((dataset, duration, eta_label, denom_model))
                    if comp is None:
                        continue
                    lnz_h4 = float(h4["lnz"])
                    lnz_comp = float(comp["lnz"])
                    err = math.sqrt(float(h4["lnz_err"]) ** 2 + float(comp["lnz_err"]) ** 2)
                    log_bf = lnz_h4 - lnz_comp
                    summary_rows.append({
                        "dataset": dataset, "duration": duration, "eta": eta_label,
                        "H_num": "H4", "H_denom": denom_model.replace("mcmc_output_", ""),
                        "logBF": f"{log_bf:.3f}", "logBF_err": f"{err:.3f}",
                    })

    if not summary_rows:
        print("  No logBF pairs found (need both H4 and comparison model for same dataset/duration/eta).")
        return

    with LNZ_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "duration", "eta", "H_num", "H_denom", "logBF", "logBF_err"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Wrote {len(summary_rows)} rows → {LNZ_CSV}\n")
    print(f"  {'dataset':<10} {'dur':<5} {'eta':<10} {'comparison':<12} {'logBF':>12}  {'±':>2}")
    print(f"  {'-'*60}")
    for r in summary_rows:
        print(f"  {r['dataset']:<10} {r['duration']:<5} {r['eta']:<10} H4/{r['H_denom']:<8} {r['logBF']:>12}  ± {r['logBF_err']}")


# ---------------------------------------------------------------------------
# Step 3+4 — load RunData and plot triangle plots
# ---------------------------------------------------------------------------

def _coherence_from_matrix(s: np.ndarray, i: int, j: int) -> np.ndarray:
    sii = np.maximum(s[:, i, i].real, 0.0)
    sjj = np.maximum(s[:, j, j].real, 0.0)
    denom = np.sqrt(sii * sjj)
    return np.clip(
        np.divide(np.abs(s[:, i, j]), denom, out=np.zeros(len(s)), where=denom > 0.0),
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
    ax.set_ylim(max(float(np.quantile(pos, 0.01)) * 0.35, PSD_FLOOR),
                float(np.quantile(pos, 0.995)) * 4.0)


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
                ax.loglog(psd_welch_freq, np.maximum(psd_welch_matrix[:, i, i].real, PSD_FLOOR),
                          color=WELCH_COLOR, lw=1.1, alpha=0.95, zorder=2)
            else:
                ax.semilogx(coh_welch_freq, _coherence_from_matrix(coh_welch_matrix, i, j),
                            color=WELCH_COLOR, lw=1.1, alpha=0.95, zorder=2)

            for duration in DURATIONS:
                run = runs_by_dur.get(duration)
                if run is None:
                    continue
                color = DURATION_COLORS[duration]
                if is_psd:
                    ax.fill_between(run.freq, np.maximum(run.q05[:, i, i], PSD_FLOOR),
                                    np.maximum(run.q95[:, i, i], PSD_FLOOR),
                                    color=color, alpha=POSTERIOR_FILL_ALPHA_PSD, zorder=4)
                else:
                    cq05 = np.clip(run.coh_q05[:, i, j], 0.0, 1.0) if run.coh_q05 is not None else _coherence_from_matrix(run.q05, i, j)
                    cq95 = np.clip(run.coh_q95[:, i, j], 0.0, 1.0) if run.coh_q95 is not None else _coherence_from_matrix(run.q95, i, j)
                    ax.fill_between(run.freq, cq05, cq95, color=color, alpha=POSTERIOR_FILL_ALPHA, zorder=4)

            ax.set_xlim(FMIN_PLOT, FMAX_PLOT)
            ax.grid(True, which="major", ls=":", alpha=0.35)
            ax.grid(True, which="minor", ls=":", alpha=0.18)
            ax.text(0.04, 0.93, _panel_math_label(i, j), transform=ax.transAxes,
                    ha="left", va="top", fontsize=LABEL_FONT_SIZE)

            if is_psd:
                _set_psd_limits(ax, [psd_welch_matrix[:, i, i].real]
                                + [np.maximum(r.q95[:, i, i], PSD_FLOOR) for r in runs_by_dur.values()])
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


def _welch_for_triangle(
    lisa: LISAData, days: float,
) -> tuple[np.ndarray, np.ndarray]:
    fs = 1.0 / lisa.delta_t
    n = min(lisa.data.shape[0], int(days * 86400 / lisa.delta_t))
    freq, *comps = welch_spectral_matrix_xyz(*lisa.data[:n].T, fs=fs, nperseg=DEFAULT_NPERSEG)
    return freq, spectral_matrix_from_components(*comps)


def plot_triangles(refs: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Plot H4 triangle plots for each dataset showing all available durations."""
    for dataset in DATASETS:
        h4_dir = OUT / dataset / "mcmc_output_H4"
        if not h4_dir.exists():
            print(f"  No H4 runs for {dataset}.")
            continue

        # Collect RunData for eta0.03 runs across all durations
        runs: list[RunData] = []
        for dur_dir in sorted(h4_dir.iterdir()):
            if not dur_dir.is_dir():
                continue
            eta_dir = dur_dir / "eta0.03"
            if not eta_dir.is_dir():
                continue
            nc = _find_idata(eta_dir)
            if nc is None:
                continue
            q = _load_posterior_quantiles(nc)
            if q is not None:
                runs.append(_to_run_data(dataset, "mcmc_output_H4", dur_dir.name, "eta0.03", q))

        if not runs:
            print(f"  No completed H4 eta0.03 runs for {dataset}.")
            continue

        # Load reference data for Welch overlays
        data_path = Path("data") / f"{dataset}.h5"
        lisa = LISAData.load(data_path, reference=DATASET_REFS[dataset])
        psd_freq, psd_matrix = _welch_for_triangle(lisa, days=7.0)
        coh_freq, coh_matrix = _welch_for_triangle(lisa, days=56.0)

        out_path = OUT / f"triangle_{dataset}_H4.png"
        _plot_triangle(runs, psd_freq, psd_matrix, coh_freq, coh_matrix, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    print("\nStep 1: Collecting run diagnostics...")
    collect_results(refs)

    print("\nStep 2: Computing lnZ comparisons...")
    compare_lnz()

    print("\nStep 3+4: Plotting triangle plots (H4, eta0.03)...")
    plot_triangles(refs)

    print("\nDone.")


if __name__ == "__main__":
    main()
