"""Validate restored draws and compare with the matched mean matrix spectrum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from log_psplines.results import PSDResult
from fit_response import diagnostics, sha256, write_json
from reference import align_reference, validate_direct_response

HERE = Path(__file__).resolve().parent


def relative_error(actual, expected):
    scale = np.max(abs(expected), axis=(-2, -1), keepdims=True)
    return float(np.max(abs(actual - expected) / scale))


def analyze(source, output, basis):
    with np.load(output / "physical_spectrum.npz") as saved:
        physical = {name: saved[name] for name in saved.files}
    provenance = json.loads((output / "provenance.json").read_text())
    residual = PSDResult.from_netcdf(output / "residual_inference_data.nc")
    summary = diagnostics(residual)
    summary.update(
        case=provenance["generator"]["case"],
        basis=basis,
        n_frequency=len(physical["frequency"]),
        Nb=provenance["config"]["Nb"],
        Nh=physical["fine_frequency"].shape[1],
        fit_seconds=residual.metadata.get("fit_seconds"),
        diagnostics_recomputed_from_saved_posterior=True,
        diagnostics_source_sha256=sha256(HERE / "fit_response.py"),
        claim_scope=provenance["claim_scope"],
    )
    restored = residual.spectral_density * physical["factor"][None, None, :, None, None]
    draws, q = physical["draws"], physical["quantiles"]
    restore_error = relative_error(draws, restored)
    assert restore_error < 2e-13
    computed_q = np.percentile(
        draws.real, [5, 50, 95], axis=(0, 1)
    ) + 1j * np.percentile(draws.imag, [5, 50, 95], axis=(0, 1))
    q_error = relative_error(q, computed_q)
    assert q_error < 2e-13
    diagonals = np.diagonal(draws, axis1=-2, axis2=-1).real
    assert np.all(np.isfinite(draws)) and np.all(diagonals > 0)
    summary["finite_physical_draws"] = True
    write_json(output / "summary.json", summary)
    normalized = draws / np.sqrt(diagonals[..., :, None] * diagonals[..., None, :])
    minimum_eigenvalue = float(np.linalg.eigvalsh(normalized).min())
    assert minimum_eigenvalue > 0
    coherence = abs(normalized) ** 2
    coherence_q = np.percentile(coherence, [5, 50, 95], axis=(0, 1))
    np.testing.assert_allclose(
        coherence_q, physical["coherence_quantiles"], atol=2e-12, rtol=2e-12
    )
    del restored, draws, diagonals, normalized, coherence, residual
    reference = align_reference(
        source,
        basis,
        physical["fine_frequency"],
        Nb=provenance["config"]["Nb"],
        window=tuple(provenance["config"]["wishart_window"]),
    )
    expected = reference["expected_psd"]
    diag = np.diagonal(expected, axis1=-2, axis2=-1).real
    normalization = np.sqrt(diag[:, :, None] * diag[:, None, :])
    error = abs(q[1] - expected) / normalization
    contain_re = (q[0].real <= expected.real) & (expected.real <= q[2].real)
    contain_im = (q[0].imag <= expected.imag) & (expected.imag <= q[2].imag)
    report = {
        "case": provenance["generator"]["case"],
        "basis": basis,
        "reference": reference["metadata"],
        "analysis_source_sha256": sha256(__file__),
        "reference_source_sha256": sha256(HERE / "reference.py"),
        "validation": {
            **validate_direct_response(),
            "all_draw_restoration_max_relative_error": restore_error,
            "all_draw_quantile_max_relative_error": q_error,
            "minimum_correlation_matrix_eigenvalue_across_draws": minimum_eigenvalue,
            "coherence_unchanged": True,
        },
        "n_frequency": len(expected),
        "mean_normalized_absolute_matrix_error": error.mean(axis=0).tolist(),
        "median_normalized_absolute_matrix_error": np.median(error, axis=0).tolist(),
        "diagonal_median_fit_to_expected_ratio": np.median(
            np.diagonal(q[1], axis1=-2, axis2=-1).real / diag, axis=0
        ).tolist(),
        "diagonal_first_bin_fit_to_expected_ratio": (
            np.diagonal(q[1, 0]).real / diag[0]
        ).tolist(),
        "real_component_90pct_containment_fraction": contain_re.mean(axis=0).tolist(),
        "imag_component_90pct_containment_fraction": contain_im.mean(axis=0).tolist(),
        "interpretation": "Descriptive across correlated bins of one realization; not repeated-noise coverage calibration",
    }
    (output / "recovery.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output / "matched_reference.npz",
        **{key: value for key, value in reference.items() if key != "metadata"},
    )
    plot(physical, reference, output, provenance)
    print(json.dumps(report["validation"], indent=2))


def plot(physical, reference, output, provenance):
    f = physical["frequency"] * 1000
    q = physical["quantiles"]
    expected = reference["expected_psd"]
    diag = np.diagonal(expected, axis1=-2, axis2=-1).real
    norm = np.sqrt(diag[:, :, None] * diag[:, None, :])
    labels = [str(label).removesuffix("2") for label in physical["channel_names"]]
    # Avoid connecting curves across deliberately excluded null neighborhoods.
    gaps = (
        np.flatnonzero(
            physical["support_low"][1:] - physical["support_high"][:-1]
            > 2 * np.median(np.diff(physical["fine_frequency"][0]))
        )
        + 1
    )
    segments = np.split(np.arange(len(f)), gaps)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.2})
    fig, axes = plt.subplots(
        3, 3, figsize=(12, 9), sharex=True, constrained_layout=True
    )
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if i == j:
                summary = q[:, :, i, i].real
                target = expected[:, i, i].real
                empirical = physical["empirical_psd"][:, i, i].real
                ax.set_yscale("log")
                ax.set_title(f"{labels[i]} auto PSD [1/Hz]")
            else:
                # Always display upper-triangle S_ab with a<b to keep a single
                # sign convention for the imaginary CSD across all panels.
                a, b = min(i, j), max(i, j)
                part = np.real if i > j else np.imag
                summary = part(q[:, :, a, b]) / norm[:, a, b]
                target = part(expected[:, a, b]) / norm[:, a, b]
                empirical = part(physical["empirical_psd"][:, a, b]) / norm[:, a, b]
                ax.set_title(
                    f"{'Re' if i > j else 'Im'} S{labels[a]}{labels[b]} / √(S{labels[a]}{labels[a]} S{labels[b]}{labels[b]})"
                )
                ax.axhline(0, color="0.7", linewidth=0.7)
            ax.scatter(
                f,
                empirical,
                color="0.6",
                s=5,
                alpha=0.4,
                label="Observed pooled spectrum",
            )
            for k, selected in enumerate(segments):
                ax.fill_between(
                    f[selected],
                    summary[0, selected],
                    summary[2, selected],
                    color="#1764ab",
                    alpha=0.25,
                    label="90% posterior interval" if k == 0 else None,
                )
                ax.plot(
                    f[selected],
                    summary[1, selected],
                    color="#1764ab",
                    linewidth=1.2,
                    label="Posterior median" if k == 0 else None,
                )
                ax.plot(
                    f[selected],
                    target[selected],
                    color="black",
                    linewidth=1,
                    label="Matched expected spectrum" if k == 0 else None,
                )
            ax.set_xscale("log")
            if i == 2:
                ax.set_xlabel("Frequency [mHz]")
    handles, legends = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="outside lower center", ncol=4, frameon=False)
    noise = (
        "unequal-MOSA"
        if provenance["generator"]["case"] == "asymmetric"
        else "symmetric-MOSA"
    )
    days = provenance["generator"]["days"]
    fig.suptitle(
        f"Fixed-orbit {noise} LISA: restored physical spectrum\nOne {days:g}-day realization · global centering · known common response factored out",
        fontsize=14,
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"spectral_recovery.{extension}", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("asymmetric", "symmetric"), default="asymmetric"
    )
    parser.add_argument("--basis", choices=("aet", "xyz"), default="aet")
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or HERE / "output" / f"{args.case}_{args.basis}"
    analyze(args.data / f"{args.case}.h5", output, args.basis)


if __name__ == "__main__":
    main()
