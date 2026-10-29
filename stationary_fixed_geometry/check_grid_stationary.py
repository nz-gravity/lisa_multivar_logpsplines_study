"""Stationary LISA data through the experimental time-varying ANOVA model.

The likelihood statistics are matched exactly to the stationary fitter. The
ANOVA smoothing and interaction priors differ, so posterior equality is not
asserted. This driver does not generate or fit time-varying noise.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import arviz_stats as azs
import h5py
import jax
import numpy as np

from log_psplines import (
    PowerConfig,
    PSDResult,
    TimeSeries,
    WishartGridData,
    fit,
    local_wishart_grid,
)
from log_psplines.data.spectral_utils import Y_to_U
from log_psplines.diagnostics.sampling import _channel_idata, _tree_depth_hits
from log_psplines.models.reconstruction import wishart_grid_draws

from fit_response import (
    configuration,
    package_provenance,
    prepare_data,
    sha256,
    write_json,
)
from reference import align_reference

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent


def prepare(source):
    physical, stationary, fine, factor, generator = prepare_data(
        source, "aet", configuration(bins=128)
    )
    with h5py.File(source) as file:
        raw = np.column_stack([file[name][:] for name in ("A2", "E2", "T2")])
        time = file["t"][:]
    std = raw.std(axis=0)
    local = local_wishart_grid(
        TimeSeries(raw / std, time),
        len(time) // 8,
        window=("tukey", 0.1),
        units="standardized/Hz",
    )
    indices = np.rint(fine * stationary.duration).astype(int) - 1
    np.testing.assert_allclose(local.frequency[indices], fine, atol=2e-17, rtol=0)
    coefficients = local.U[..., 0][:, indices]
    y = np.einsum("tfhc,tfhd->tfcd", coefficients, coefficients.conj())
    y /= factor[None, :, None, None]
    pooled = y.reshape(4, 2, *y.shape[1:]).sum(axis=1)
    u = Y_to_U(pooled.reshape(-1, 3, 3)).reshape(pooled.shape)
    data = WishartGridData(
        u.real,
        u.imag,
        local.time.reshape(4, 2).mean(axis=1),
        physical.freq,
        counts=2 * stationary.Nh,
        reference_time=local.time,
        reference_frequency=physical.freq,
        enbw=local.enbw,
        preprocessing=local.preprocessing,
        units="standardized residual spectrum/Hz",
        normalization="sum of normalized local FFT outer products divided by known bin-mean G",
    )
    target = stationary.Y / stationary.duration
    # Normalize elementwise by auto-power products, so quiet T is checked too.
    diag = np.diagonal(target, axis1=-2, axis2=-1).real
    normalization = np.sqrt(diag[:, :, None] * diag[:, None, :])
    max_error = float(np.max(abs(data.Y.sum(axis=0) - target) / normalization))
    np.testing.assert_allclose(
        data.Y.sum(axis=0) / normalization,
        target / normalization,
        atol=5e-12,
        rtol=5e-12,
    )
    np.testing.assert_allclose(std, stationary.channel_stds, rtol=2e-14)
    np.testing.assert_allclose(local.enbw, stationary.enbw, atol=2e-15)
    np.testing.assert_array_equal(
        data.counts.sum(axis=0), stationary.Nb * stationary.Nh
    )
    checks = {
        "summed_sufficient_statistics_max_error_normalized_by_autos": max_error,
        "matched_counts": int(stationary.Nb * stationary.Nh),
        "fine_frequencies_per_bin": int(stationary.Nh),
        "observed_time_bins": len(data.time),
        "reference_time_bins": len(data.reference_time),
        "retained_frequency_bins": len(data.frequency),
        "enbw": data.enbw,
        "mean_removal": data.preprocessing["mean_removal"],
        "passed": True,
    }
    return data, fine, factor, std, generator, checks


def diagnostic_summary(result):
    rows = []
    for channel in range(3):
        view = _channel_idata(result, channel)
        statistics = azs.summary(view, kind="diagnostics", round_to="none")
        numeric = statistics[["r_hat", "ess_bulk", "ess_tail"]].to_numpy(dtype=float)
        stats = view["sample_stats"].dataset
        energy = np.asarray(stats["energy"])
        bfmi = np.mean(np.diff(energy, axis=1) ** 2, axis=1) / np.var(
            energy, axis=1, ddof=1
        )
        worst = {}
        for metric, ascending in (
            ("r_hat", False),
            ("ess_bulk", True),
            ("ess_tail", True),
        ):
            selected = statistics.sort_values(
                metric, ascending=ascending, na_position="first"
            ).head(10)
            worst[metric] = [
                {"parameter": str(name), **values.to_dict()}
                for name, values in selected[
                    ["r_hat", "ess_bulk", "ess_tail"]
                ].iterrows()
            ]
        n_steps = np.asarray(stats["n_steps"])
        step_size = np.asarray(stats["step_size"])
        row = {
            "row": channel,
            "divergences": int(np.asarray(stats["diverging"]).sum()),
            "rhat_max": float(np.max(numeric[:, 0])),
            "ess_bulk_min": float(np.min(numeric[:, 1])),
            "ess_tail_min": float(np.min(numeric[:, 2])),
            "all_parameter_diagnostics_finite": bool(np.isfinite(numeric).all()),
            "max_depth_hits": _tree_depth_hits(view, result.metadata["max_tree_depth"]),
            "bfmi_by_chain": bfmi,
            "worst_parameter_diagnostics": worst,
            "n_steps_quantiles": dict(
                zip(
                    ("p50", "p90", "p99", "max"),
                    np.percentile(n_steps, [50, 90, 99, 100]),
                )
            ),
            "n_steps_median_by_chain": np.median(n_steps, axis=1),
            "step_size_median_by_chain": np.median(step_size, axis=1),
            "sampling_and_conditional_recovery_seconds": result.metadata[
                "row_fit_seconds"
            ][channel],
        }
        row["diagnostic_pass"] = bool(
            row["all_parameter_diagnostics_finite"]
            and row["divergences"] == 0
            and row["rhat_max"] < 1.01
            and row["ess_bulk_min"] >= 400
            and row["ess_tail_min"] >= 400
            and row["max_depth_hits"] == 0
            and np.all(np.isfinite(bfmi) & (bfmi >= 0.3))
        )
        rows.append(row)
    return {
        "rows": rows,
        "all_factors_pass": all(row["diagnostic_pass"] for row in rows),
    }


def analyze(result, source, output, fine, factor, std):
    restore = factor[:, None, None] * np.outer(std, std)[None]
    quantiles = np.asarray(result.quantiles()) * restore[None, None]
    reference = align_reference(source, "aet", fine, Nb=8, window=("tukey", 0.1))
    expected = reference["expected_psd"]
    assert np.isfinite(quantiles).all() and np.isfinite(expected).all()
    diagonal = np.diagonal(expected, axis1=-2, axis2=-1).real
    norm = np.sqrt(diagonal[:, :, None] * diagonal[:, None, :])
    error = abs(quantiles[1] - expected[None]) / norm[None]
    ratio_q, coherence_re_q, coherence_im_q = [], [], []
    minimum_eigenvalue = np.inf
    bt = np.asarray(result.model_data["basis_time"])
    bf = np.asarray(result.model_data["basis_frequency"])
    # Ratios and positive definiteness use all posterior draws, not the preview.
    for first in range(0, len(factor), 16):
        section = slice(first, first + 16)
        draws = wishart_grid_draws(result.posterior, bt, bf[section], 3)
        assert np.isfinite(draws).all()
        diag = np.diagonal(draws, axis1=-2, axis2=-1).real
        assert np.isfinite(diag).all() and np.all(diag > 0)
        correlation = draws / np.sqrt(diag[..., :, None] * diag[..., None, :])
        assert np.isfinite(correlation).all()
        eigenvalues = np.linalg.eigvalsh(correlation)
        assert np.isfinite(eigenvalues).all()
        minimum_eigenvalue = min(minimum_eigenvalue, float(eigenvalues.min()))
        ratio = diag[:, :, -1] / diag[:, :, 0]
        ratio_q.append(np.percentile(ratio, [5, 50, 95], axis=(0, 1)))
        # Complex coherency C_ij = S_ij / sqrt(S_ii S_jj), in AE, AT, ET order.
        coherence = correlation[..., [0, 0, 1], [1, 2, 2]]
        contrast = coherence[:, :, -1] - coherence[:, :, 0]
        coherence_re_q.append(np.percentile(contrast.real, [5, 50, 95], axis=(0, 1)))
        coherence_im_q.append(np.percentile(contrast.imag, [5, 50, 95], axis=(0, 1)))
    ratio_q = np.concatenate(ratio_q, axis=1)
    coherence_re_q = np.concatenate(coherence_re_q, axis=1)
    coherence_im_q = np.concatenate(coherence_im_q, axis=1)
    assert minimum_eigenvalue > 0
    report = {
        "analysis_source_sha256": sha256(__file__),
        "prior_comparison": "ANOVA and stationary spline priors differ; this is stationary-data recovery, not posterior-equivalence testing",
        "reference": reference["metadata"],
        "contrast_times": "First and last reference-grid endpoints, bracketing the pooled observation centers",
        "contrast_endpoint_seconds": np.asarray(result.time)[[0, -1]],
        "observed_time_centers_seconds": np.asarray(result.observed_data["time"]),
        "coherence_definition": "Complex S_ij / sqrt(S_ii S_jj), not magnitude-squared coherence",
        "coherence_pair_order": ["AE", "AT", "ET"],
        "mean_normalized_absolute_matrix_error_over_time_and_frequency": error.mean(
            axis=(0, 1)
        ),
        "median_diagonal_fit_to_expected_ratio_over_time_and_frequency": np.median(
            np.diagonal(quantiles[1], axis1=-2, axis2=-1).real / diagonal[None],
            axis=(0, 1),
        ),
        "last_to_first_auto_psd_ratio_median_across_frequency": np.median(
            ratio_q[1], axis=0
        ),
        "last_to_first_ratio_90pct_contains_one_fraction": np.mean(
            (ratio_q[0] <= 1) & (ratio_q[2] >= 1), axis=0
        ),
        "minimum_correlation_eigenvalue_all_draws": minimum_eigenvalue,
        "all_spectrum_draws_finite": True,
        "last_minus_first_coherence_real_90pct_contains_zero_fraction": np.mean(
            (coherence_re_q[0] <= 0) & (coherence_re_q[2] >= 0), axis=0
        ),
        "last_minus_first_coherence_imag_90pct_contains_zero_fraction": np.mean(
            (coherence_im_q[0] <= 0) & (coherence_im_q[2] >= 0), axis=0
        ),
        "claim_scope": "One stationary realization; frequency containment is descriptive and not repeated-noise coverage",
    }
    np.savez_compressed(
        output / "physical_summaries.npz",
        frequency=result.frequency,
        time=result.time,
        quantiles=quantiles,
        expected_psd=expected,
        last_to_first_auto_psd_ratio_quantiles=ratio_q,
        last_minus_first_coherence_re_quantiles=coherence_re_q,
        last_minus_first_coherence_im_quantiles=coherence_im_q,
        coherence_pair_order=["AE", "AT", "ET"],
        factor=factor,
        std=std,
        fine_frequency=fine,
    )
    write_json(output / "recovery.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data" / "asymmetric.h5")
    parser.add_argument(
        "--output", type=Path, default=HERE / "output" / "grid_stationary"
    )
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data, fine, factor, std, generator, checks = prepare(args.data)
    write_json(args.output / "preparation_checks.json", checks)
    print(checks, flush=True)
    if args.prepare_only:
        return
    config = PowerConfig(
        structure="anova",
        n_interior_knots_time=0,
        degree_time=1,
        penalty_order_time=1,
        interior_knots_freq=np.geomspace(data.frequency[0], data.frequency[-1], 14)[
            1:-1
        ],
        degree_freq=3,
        penalty_order_freq=2,
        roughness_scale=1.28,
        interaction_scale=0.5,
        centered=False,
        dense_mass=True,
        num_chains=args.chains,
        n_warmup=args.warmup,
        n_samples=args.samples,
        target_accept_prob=0.95,
        max_tree_depth=10,
        progress_bar=True,
        spectrum_draws=50,
        seed=20261029,
    )
    if args.analyze_only:
        result = PSDResult.from_netcdf(args.output / "residual_inference_data.nc")
    else:
        write_json(
            args.output / "provenance.json",
            {
                "source_file": args.data.name,
                "source_sha256": sha256(args.data),
                "generator": generator,
                "driver_sha256": sha256(__file__),
                "package": package_provenance(),
                "config": asdict(config),
                "stationary_preprocessing_match": checks,
                "prior_comparison": "ANOVA priors differ from StationaryConfig; only sufficient statistics and conditional limiting likelihoods are exactly matched",
            },
        )
        print("Sampling stationary LISA data with the flexible ANOVA model", flush=True)
        start = perf_counter()
        result = fit(data, config)
        result.metadata["fit_seconds"] = perf_counter() - start
        result.to_netcdf(args.output / "residual_inference_data.nc")
    summary = diagnostic_summary(result)
    summary["fit_seconds"] = result.metadata.get("fit_seconds")
    summary["analysis_source_sha256"] = sha256(__file__)
    summary["source_sha256_at_analysis"] = package_provenance()["source_sha256"]
    write_json(args.output / "summary.json", summary)
    report = analyze(result, args.data, args.output, fine, factor, std)
    print({"sampling": summary, "recovery": report}, flush=True)


if __name__ == "__main__":
    main()
