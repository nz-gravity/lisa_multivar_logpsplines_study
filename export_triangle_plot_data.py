#!/usr/bin/env python3
"""
Export triangle-plot inputs for the eta=0.5 noise_4a/noise_5a comparison.

This script is local/project-aware: it reuses the existing loaders in
`collect_results.py`, then writes one portable HDF5 bundle that can be shared
with the standalone plotting script.

Outputs:
  - triangle_plot_data_eta0p5.h5
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

import collect_results as cr

RunData = cr.RunData


ROOT = Path(__file__).resolve().parent
OUT_H5 = ROOT / "triangle_plot_data_eta0p5.h5"


def _write_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    """Write one array dataset with gzip compression when appropriate."""
    arr = np.asarray(values)
    kwargs = {}
    if arr.ndim > 0 and arr.size > 16:
        kwargs = {"compression": "gzip", "shuffle": True}
    group.create_dataset(name, data=arr, **kwargs)


def _write_run(group: h5py.Group, run: cr.RunData) -> None:
    """Write one run group with posterior PSD/coherence quantiles."""
    group.attrs["noise"] = run.noise
    group.attrs["duration"] = run.duration
    group.attrs["eta"] = run.eta
    _write_dataset(group, "freq", run.freq)
    _write_dataset(group, "q05", run.q05)
    _write_dataset(group, "q50", run.q50)
    _write_dataset(group, "q95", run.q95)
    if run.coh_q05 is not None:
        _write_dataset(group, "coh_q05", run.coh_q05)
    if run.coh_q50 is not None:
        _write_dataset(group, "coh_q50", run.coh_q50)
    if run.coh_q95 is not None:
        _write_dataset(group, "coh_q95", run.coh_q95)


def _write_reference(
    group: h5py.Group,
    *,
    psd_welch_freq: np.ndarray,
    psd_welch_matrix: np.ndarray,
    coh_welch_freq: np.ndarray,
    coh_welch_matrix: np.ndarray,
    analytic_matrix: np.ndarray | None = None,
) -> None:
    """Write reference curves and optional analytic spectrum."""
    _write_dataset(group, "psd_welch_freq", psd_welch_freq)
    _write_dataset(group, "psd_welch_matrix", psd_welch_matrix)
    _write_dataset(group, "coh_welch_freq", coh_welch_freq)
    _write_dataset(group, "coh_welch_matrix", coh_welch_matrix)
    if analytic_matrix is not None:
        _write_dataset(group, "analytic_matrix", analytic_matrix)


def _load_noise_runs(noise_dir: Path) -> list[cr.RunData]:
    """Load all eta=0.5 duration runs for one noise directory."""
    runs: list[cr.RunData] = []
    for duration in cr.DURATIONS:
        run = cr.load_run(noise_dir, duration, cr.ETA)
        if run is not None:
            runs.append(run)
    return runs


def main() -> None:
    """Export one portable HDF5 bundle for the triangle plots."""
    print("Loading noise_4a eta=0.5 runs...")
    runs_4a = _load_noise_runs(cr.NOISE_4A)
    print("Loading noise_5a eta=0.5 runs...")
    runs_5a = _load_noise_runs(cr.NOISE_5A)

    if not runs_4a and not runs_5a:
        raise FileNotFoundError("No eta=0.5 runs were found under noise_4a/noise_5a.")

    print("Loading reference spectra...")
    psd_freq_4a, psd_welch_4a, coh_freq_4a, coh_welch_4a, analytic_4a = cr.load_reference_4a()
    psd_freq_5a, psd_welch_5a, coh_freq_5a, coh_welch_5a = cr.load_reference_5a()

    print(f"Writing {OUT_H5} ...")
    with h5py.File(OUT_H5, "w") as h5:
        h5.attrs["bundle_type"] = "lisa_triangle_plot_data"
        h5.attrs["eta"] = cr.ETA
        h5.attrs["durations"] = np.asarray(cr.DURATIONS, dtype="S8")
        h5.attrs["channels"] = np.asarray(cr.CHANNELS, dtype="S8")
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

        noise_4a = h5.create_group("noise_4a")
        noise_4a.attrs["label"] = "noise_4a"
        noise_4a.attrs["has_analytic_reference"] = True
        _write_reference(
            noise_4a.create_group("reference"),
            psd_welch_freq=psd_freq_4a,
            psd_welch_matrix=psd_welch_4a,
            coh_welch_freq=coh_freq_4a,
            coh_welch_matrix=coh_welch_4a,
            analytic_matrix=analytic_4a,
        )
        runs_group_4a = noise_4a.create_group("runs")
        for run in runs_4a:
            _write_run(runs_group_4a.create_group(run.duration), run)

        noise_5a = h5.create_group("noise_5a")
        noise_5a.attrs["label"] = "noise_5a"
        noise_5a.attrs["has_analytic_reference"] = False
        _write_reference(
            noise_5a.create_group("reference"),
            psd_welch_freq=psd_freq_5a,
            psd_welch_matrix=psd_welch_5a,
            coh_welch_freq=coh_freq_5a,
            coh_welch_matrix=coh_welch_5a,
        )
        runs_group_5a = noise_5a.create_group("runs")
        for run in runs_5a:
            _write_run(runs_group_5a.create_group(run.duration), run)

    print(f"Wrote {OUT_H5}")


if __name__ == "__main__":
    main()
