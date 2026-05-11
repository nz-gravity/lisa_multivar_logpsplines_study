#!/usr/bin/env python3
"""Shared helpers for per-channel univariate LISA AET MCMC runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import csd, welch

from log_psplines.arviz_utils import save_inference_data
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.datatypes.multivar import EmpiricalPSD, _get_coherence
from log_psplines.pipeline.config import PipelineConfig
from log_psplines.pipeline.make_pipeline import make_pipeline
from log_psplines.pipeline.preprocessing import align_true_psd_to_freq
from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig
from log_psplines.plotting import plot_psd_matrix, PSDMatrixPlotSpec
from log_psplines.plotting.psd_matrix import extract_plotting_data
from src.load_data import LOW_FREQ_BIN_TRIM, interpolate_spectral_matrix


M_AET: np.ndarray = np.array(
    [
        [-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
        [1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)],
        [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
    ],
    dtype=np.float64,
)

CHANNEL_LABELS_AET = ("A", "E", "T")
CHANNEL_LABELS_XYZ = ("X", "Y", "Z")


def xyz_to_aet_timeseries(y_xyz: np.ndarray) -> np.ndarray:
    """Transform XYZ timeseries with shape ``(N, 3)`` to AET."""
    y_xyz = np.asarray(y_xyz, dtype=np.float64)
    if y_xyz.ndim != 2 or y_xyz.shape[1] != 3:
        raise ValueError(f"Expected XYZ timeseries with shape (N, 3), got {y_xyz.shape}.")
    return (M_AET @ y_xyz.T).T


def xyz_to_aet_matrix(s_xyz: np.ndarray) -> np.ndarray:
    """Transform a spectral matrix with shape ``(..., 3, 3)`` to AET."""
    s_xyz = np.asarray(s_xyz)
    if s_xyz.shape[-2:] != (3, 3):
        raise ValueError(f"Expected XYZ matrix with trailing shape (3, 3), got {s_xyz.shape}.")
    matrix = M_AET.astype(s_xyz.dtype, copy=False)
    return matrix @ s_xyz @ matrix.conj().T


def aet_to_xyz_matrix(s_aet: np.ndarray) -> np.ndarray:
    """Transform a spectral matrix with shape ``(..., 3, 3)`` from AET to XYZ."""
    s_aet = np.asarray(s_aet)
    if s_aet.shape[-2:] != (3, 3):
        raise ValueError(f"Expected AET matrix with trailing shape (3, 3), got {s_aet.shape}.")
    matrix_t = M_AET.T.astype(s_aet.dtype, copy=False)
    return matrix_t @ s_aet @ matrix_t.conj().T


def _welch_matrix_xyz(
    y_xyz: np.ndarray,
    dt: float,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch XYZ spectral matrix used for empirical plotting overlays."""
    fs = 1.0 / float(dt)
    x, y, z = np.asarray(y_xyz[:, 0]), np.asarray(y_xyz[:, 1]), np.asarray(y_xyz[:, 2])
    kwargs = {
        "fs": fs,
        "window": ("tukey", 0.1),
        "nperseg": int(nperseg),
        "noverlap": 0,
        "detrend": False,
    }
    freq, sxx = welch(x, **kwargs)
    _, syy = welch(y, **kwargs)
    _, szz = welch(z, **kwargs)
    _, sxy = csd(x, y, **kwargs)
    _, syz = csd(y, z, **kwargs)
    _, szx = csd(z, x, **kwargs)

    sl = slice(LOW_FREQ_BIN_TRIM, None)
    freq = np.asarray(freq[sl], dtype=np.float64)
    mat = np.zeros((freq.size, 3, 3), dtype=np.complex128)
    mat[:, 0, 0] = sxx[sl]
    mat[:, 1, 1] = syy[sl]
    mat[:, 2, 2] = szz[sl]
    mat[:, 0, 1] = sxy[sl]
    mat[:, 1, 0] = np.conj(sxy[sl])
    mat[:, 1, 2] = syz[sl]
    mat[:, 2, 1] = np.conj(syz[sl])
    mat[:, 2, 0] = szx[sl]
    mat[:, 0, 2] = np.conj(szx[sl])
    return freq, mat


def _write_summary(summary_path: Path, channel_rows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "channels": channel_rows,
        "combined_lnz": float(sum(row["lnz"] for row in channel_rows)),
        "combined_lnz_err": float(
            np.sqrt(sum(float(row["lnz_err"]) ** 2 for row in channel_rows))
        ),
        "all_channels_valid": bool(all(bool(row["lnz_valid"]) for row in channel_rows)),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    csv_path = summary_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("channel", "lnz", "lnz_err", "lnz_valid", "outdir"),
        )
        writer.writeheader()
        writer.writerows(channel_rows)


def run_univar_aet_analysis(
    *,
    lisa: Any,
    n_trim: int,
    root_outdir: Path,
    fmin: float,
    fmax: float,
    nb: int,
    main_nc: int,
    vi_nc: int,
    n_samples: int,
    n_warmup: int,
    num_chains: int,
    chain_method: str | None,
    alpha_delta: float,
    beta_delta: float,
    eta: float,
    n_knots: int,
    degree: int,
    diff_order: int,
    knot_method: str,
    use_vi: bool,
    vi_steps: int,
    vi_lr: float,
    vi_guide: str,
    vi_posterior_draws: int,
    vi_progress_bar: bool,
    target_accept_prob: float,
    max_tree_depth: int,
    dense_mass: bool,
    exclude_freq_bands: tuple[tuple[float, float], ...],
    compute_lnz: bool,
    lnz_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run independent univariate MCMC fits for the A, E, and T channels."""
    root_outdir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(lisa.time[:n_trim], dtype=np.float64)
    y_xyz = np.asarray(lisa.data[:n_trim], dtype=np.float64)
    y_aet = xyz_to_aet_timeseries(y_xyz)

    nperseg = max(1, n_trim // max(1, nb))
    freq_emp, mat_emp_xyz = _welch_matrix_xyz(y_xyz, dt=float(lisa.delta_t), nperseg=nperseg)

    true_freq = np.asarray(lisa.freq, dtype=np.float64)
    true_matrix_aet = xyz_to_aet_matrix(np.asarray(lisa.true_matrix))

    channel_rows: list[dict[str, Any]] = []
    channel_diag_ci: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    combined_freq: np.ndarray | None = None

    def _diag_ci_from_quantiles(
        quantiles: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        percentiles = np.asarray(quantiles["percentile"], dtype=np.float64)
        spectral_density = np.asarray(quantiles["spectral_density"])
        diag_real = np.asarray(spectral_density.real[:, :, 0, 0], dtype=np.float64)

        def _grab(target: float) -> np.ndarray:
            idx = int(np.argmin(np.abs(percentiles - target)))
            return np.asarray(diag_real[idx], dtype=np.float64)

        return _grab(5.0), _grab(50.0), _grab(95.0)

    for channel_index, channel_label in enumerate(CHANNEL_LABELS_AET):
        channel_outdir = root_outdir / channel_label
        channel_outdir.mkdir(parents=True, exist_ok=True)

        # p=1 MultivariateTimeseries: goes through Wishart with the same Nb
        # blocks as the full 3-channel model, making lnZ directly comparable.
        ts = MultivariateTimeseries(
            y=y_aet[:, channel_index : channel_index + 1].astype(np.float64),
            t=t,
        )
        true_psd_1d = np.asarray(
            true_matrix_aet[:, channel_index, channel_index].real,
            dtype=np.float64,
        )

        config = PipelineConfig(
            n_samples=n_samples,
            n_warmup=n_warmup,
            num_chains=num_chains,
            chain_method=chain_method,
            Nb=nb,
            alpha_delta=alpha_delta,
            beta_delta=beta_delta,
            eta=eta,
            coarse_grain_config=CoarseGrainConfig(
                enabled=True,
                Nc=main_nc,
                Nh=None,
            ),
            wishart_window=("tukey", 0.1),
            wishart_floor_fraction=1e-6,
            n_knots=n_knots,
            degree=degree,
            diffMatrixOrder=diff_order,
            knot_kwargs={"method": knot_method},
            true_psd=(true_freq, true_psd_1d),
            fmin=fmin,
            fmax=fmax,
            exclude_freq_bands=exclude_freq_bands,
            verbose=True,
            outdir=str(channel_outdir),
            compute_lnz=compute_lnz,
            only_vi=False,
            init_from_vi=use_vi,
            vi_steps=vi_steps,
            vi_lr=vi_lr,
            vi_guide=vi_guide,
            vi_posterior_draws=vi_posterior_draws,
            vi_progress_bar=vi_progress_bar,
            coarse_grain_config_vi=CoarseGrainConfig(
                enabled=use_vi,
                Nc=vi_nc,
                Nh=None,
            ),
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            dense_mass=dense_mass,
            extra_kwargs={"lnz_kwargs": dict(lnz_kwargs or {})} if compute_lnz else {},
        )

        print(f"Running univar(AET, p=1, Wishart) channel {channel_label} -> {channel_outdir}")
        pipeline = make_pipeline(ts, config)
        result = pipeline.run()
        true_psd_aligned = align_true_psd_to_freq(config.true_psd, pipeline.data)
        result.save(str(channel_outdir), true_psd=true_psd_aligned)
        idata = result.idata
        idata.attrs["channel_basis"] = "AET"
        idata.attrs["channel_label"] = channel_label
        idata.attrs["channel_index"] = int(channel_index)
        idata.attrs["source_basis"] = "XYZ"
        idata.attrs["transformed_from_xyz"] = True
        save_inference_data(idata, channel_outdir / "inference_data.nc")

        extracted = extract_plotting_data(idata)
        channel_freq = np.asarray(extracted.get("frequencies"), dtype=np.float64)
        quantiles = extracted.get("posterior_psd_matrix_quantiles")
        if quantiles is None:
            quantiles = extracted.get("vi_psd_matrix_quantiles")
        if quantiles is None:
            raise ValueError(
                f"Missing posterior/VI PSD quantiles for channel {channel_label}."
            )

        if combined_freq is None:
            combined_freq = channel_freq
        elif combined_freq.shape != channel_freq.shape or not np.allclose(
            combined_freq, channel_freq
        ):
            raise ValueError(
                f"Frequency grid mismatch for channel {channel_label}; "
                "cannot build combined A/E/T PSD matrix."
            )
        channel_diag_ci[channel_index] = _diag_ci_from_quantiles(quantiles)

        channel_rows.append(
            {
                "channel": channel_label,
                "lnz": float(idata.attrs.get("lnz", np.nan)),
                "lnz_err": float(idata.attrs.get("lnz_err", np.nan)),
                "lnz_valid": bool(idata.attrs.get("lnz_valid", False)),
                "outdir": str(channel_outdir),
            }
        )

    summary_path = root_outdir / "lnz_summary.json"
    _write_summary(summary_path, channel_rows)

    combined_lnz = float(sum(row["lnz"] for row in channel_rows))
    combined_lnz_err = float(
        np.sqrt(sum(float(row["lnz_err"]) ** 2 for row in channel_rows))
    )
    all_channels_valid = bool(all(bool(row["lnz_valid"]) for row in channel_rows))

    print(
        "Combined univariate AET lnZ summary:",
        {
            "lnz": combined_lnz,
            "lnz_err": combined_lnz_err,
            "lnz_valid": all_channels_valid,
            "summary_path": str(summary_path),
        },
    )

    if combined_freq is None or len(channel_diag_ci) != len(CHANNEL_LABELS_AET):
        raise ValueError("Could not assemble combined A/E/T PSD matrix inputs.")

    n_freq = combined_freq.size
    zeros = np.zeros(n_freq, dtype=np.float64)
    ci_dict: dict[str, dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "psd": {},
        "coh": {},
        "re": {},
        "im": {},
        "mag": {},
    }

    for i in range(len(CHANNEL_LABELS_AET)):
        ci_dict["psd"][(i, i)] = channel_diag_ci[i]

    for i in range(len(CHANNEL_LABELS_AET)):
        for j in range(len(CHANNEL_LABELS_AET)):
            if i == j:
                continue
            ci_dict["re"][(i, j)] = (zeros, zeros, zeros)
            ci_dict["im"][(i, j)] = (zeros, zeros, zeros)
            if i > j:
                ci_dict["coh"][(i, j)] = (zeros, zeros, zeros)

    mat_emp_xyz_interp = interpolate_spectral_matrix(freq_emp, mat_emp_xyz, combined_freq)
    mat_emp_aet_interp = xyz_to_aet_matrix(mat_emp_xyz_interp)
    empirical_aet = EmpiricalPSD(
        freq=combined_freq,
        psd=np.asarray(mat_emp_aet_interp, dtype=np.complex128),
        coherence=np.asarray(_get_coherence(mat_emp_aet_interp), dtype=np.float64),
        channels=np.asarray(CHANNEL_LABELS_AET),
    )

    print(
        "Plotting combined A/E/T PSD matrix from univariate channels "
        "(off-diagonals set to zero by independence assumption)."
    )
    plot_psd_matrix(
        PSDMatrixPlotSpec(
            ci_dict=ci_dict,
            freq=combined_freq,
            true_psd=np.asarray(true_matrix_aet, dtype=np.complex128),
            outdir=str(root_outdir),
            filename="psd_matrix_loglog.png",
            xscale="log",
            diag_yscale="log",
            offdiag_yscale="linear",
            show_knots=False,
            empirical_psd=empirical_aet,
            show_empirical=True,
            show_coherence=True,
            channel_labels=list(CHANNEL_LABELS_AET),
            freq_range=(fmin, fmax),
            excluded_bands=exclude_freq_bands,
        )
    )

    # Build an approximate XYZ posterior summary by mapping AET quantiles
    # through S_xyz = M^T S_aet M, with independent A/E/T channels.
    q05_aet = np.zeros((n_freq, 3, 3), dtype=np.complex128)
    q50_aet = np.zeros((n_freq, 3, 3), dtype=np.complex128)
    q95_aet = np.zeros((n_freq, 3, 3), dtype=np.complex128)
    for i in range(len(CHANNEL_LABELS_AET)):
        q05_i, q50_i, q95_i = channel_diag_ci[i]
        q05_aet[:, i, i] = q05_i
        q50_aet[:, i, i] = q50_i
        q95_aet[:, i, i] = q95_i

    q05_xyz = np.asarray(aet_to_xyz_matrix(q05_aet), dtype=np.complex128)
    q50_xyz = np.asarray(aet_to_xyz_matrix(q50_aet), dtype=np.complex128)
    q95_xyz = np.asarray(aet_to_xyz_matrix(q95_aet), dtype=np.complex128)

    ci_xyz: dict[str, dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "psd": {},
        "coh": {},
        "re": {},
        "im": {},
        "mag": {},
    }
    for i in range(len(CHANNEL_LABELS_XYZ)):
        ci_xyz["psd"][(i, i)] = (
            np.asarray(q05_xyz[:, i, i].real, dtype=np.float64),
            np.asarray(q50_xyz[:, i, i].real, dtype=np.float64),
            np.asarray(q95_xyz[:, i, i].real, dtype=np.float64),
        )

    tiny = np.finfo(np.float64).tiny
    for i in range(len(CHANNEL_LABELS_XYZ)):
        for j in range(len(CHANNEL_LABELS_XYZ)):
            if i == j:
                continue
            re_q05 = np.asarray(q05_xyz[:, i, j].real, dtype=np.float64)
            re_q50 = np.asarray(q50_xyz[:, i, j].real, dtype=np.float64)
            re_q95 = np.asarray(q95_xyz[:, i, j].real, dtype=np.float64)
            im_q05 = np.asarray(q05_xyz[:, i, j].imag, dtype=np.float64)
            im_q50 = np.asarray(q50_xyz[:, i, j].imag, dtype=np.float64)
            im_q95 = np.asarray(q95_xyz[:, i, j].imag, dtype=np.float64)

            ci_xyz["re"][(i, j)] = (re_q05, re_q50, re_q95)
            ci_xyz["im"][(i, j)] = (im_q05, im_q50, im_q95)

            if i > j:
                psd_i = np.asarray(q50_xyz[:, i, i].real, dtype=np.float64)
                psd_j = np.asarray(q50_xyz[:, j, j].real, dtype=np.float64)
                coh_q50 = (re_q50**2 + im_q50**2) / np.maximum(psd_i * psd_j, tiny)
                coh_q50 = np.clip(coh_q50, 0.0, 1.0)
                ci_xyz["coh"][(i, j)] = (coh_q50, coh_q50, coh_q50)

    true_matrix_xyz = np.asarray(lisa.true_matrix, dtype=np.complex128)
    empirical_xyz = EmpiricalPSD(
        freq=combined_freq,
        psd=np.asarray(mat_emp_xyz_interp, dtype=np.complex128),
        coherence=np.asarray(_get_coherence(mat_emp_xyz_interp), dtype=np.float64),
        channels=np.asarray(CHANNEL_LABELS_XYZ),
    )
    print(
        "Plotting transformed XYZ PSD matrix derived from univariate A/E/T posteriors."
    )
    plot_psd_matrix(
        PSDMatrixPlotSpec(
            ci_dict=ci_xyz,
            freq=combined_freq,
            true_psd=true_matrix_xyz,
            outdir=str(root_outdir),
            filename="psd_matrix_loglog_xyz_from_aet.png",
            xscale="log",
            diag_yscale="log",
            offdiag_yscale="linear",
            show_knots=False,
            empirical_psd=empirical_xyz,
            show_empirical=True,
            show_coherence=True,
            channel_labels=list(CHANNEL_LABELS_XYZ),
            freq_range=(fmin, fmax),
            excluded_bands=exclude_freq_bands,
        )
    )

    return {
        "channels": channel_rows,
        "combined_lnz": combined_lnz,
        "combined_lnz_err": combined_lnz_err,
        "all_channels_valid": all_channels_valid,
        "summary_path": str(summary_path),
    }
