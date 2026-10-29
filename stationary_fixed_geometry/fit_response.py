"""Fit the stationary LISA residual spectrum after removing a known response.

The package fits S/Gbar; this study restores every spectral draw as Gbar*S_res.
The residual spline prior differs from a spline prior directly on S. No true
spectral values are used by preprocessing, response construction or inference.
"""

from __future__ import annotations

# Configure JAX precision before importing the inference modules.
# ruff: noqa: E402

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import arviz_stats as azs
import h5py
import jax
import numpy as np
import pandas as pd

# The collapsed complex-Gaussian calculation requires float64.
jax.config.update("jax_enable_x64", True)

import log_psplines
from log_psplines import StationaryConfig, TimeSeries, fit
from log_psplines.diagnostics.sampling import sampling_diagnostics
from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig
from log_psplines.preprocessing.spectral import preprocess_to_freq_domain

HERE = Path(__file__).resolve().parent
TAU = 8.3
FMIN, FMAX = 1e-4, 0.1
NULL_HALF_WIDTH = 1e-4


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    def clean(item):
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        return item

    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def configuration(
    *,
    seed=20261029,
    chains=4,
    warmup=1000,
    samples=1000,
    blocks=8,
    bins=256,
    theta_knots=(12, 8),
    delta_knots=20,
):
    return StationaryConfig(
        method="nuts",
        smoothing_parameterization="centered",
        roughness_scale=1.28,
        Nb=blocks,
        coarse_grain_config=CoarseGrainConfig(enabled=True, Nc=bins),
        n_knots={
            "delta": delta_knots,
            "theta_re": theta_knots[0],
            "theta_im": theta_knots[1],
        },
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs={"method": "log"},
        fmin=FMIN,
        fmax=FMAX,
        exclude_freq_bands=(),
        wishart_window=("tukey", 0.1),
        wishart_floor_fraction=None,
        eta=1.0,
        n_warmup=warmup,
        n_samples=samples,
        num_chains=chains,
        chain_method="sequential",
        target_accept_prob=0.95,
        max_tree_depth=10,
        dense_mass=True,
        rng_key=seed,
        verbose=False,
    )


def transfer_power(frequency):
    x = 2 * np.pi * TAU * np.asarray(frequency)
    return 16 * np.sin(x) ** 2 * np.sin(2 * x) ** 2


def prepare_data(source, basis, config):
    """Globally center, block/taper, pool, mask full supports, then divide U.

    Returns physical data, residual data, fine supports, Gbar and metadata.
    Supports contain the actual fine Fourier ordinates used in each pooled bin.
    """
    names = ("X2", "Y2", "Z2") if basis == "xyz" else ("A2", "E2", "T2")
    with h5py.File(source) as file:
        raw = np.column_stack([file[name][:] for name in names])
        times = file["t"][:]
        metadata = json.loads(file.attrs["metadata_json"])
        if float(file.attrs["tau"]) != TAU:
            raise ValueError("This example's response requires fixed 8.3 s delays")
    physical = preprocess_to_freq_domain(TimeSeries(raw, times), config)
    # The library reports the lower-middle ordinate when Nh is even.
    indices = np.rint(physical.freq * physical.duration).astype(np.int64)
    offset = np.arange(physical.Nh) - (physical.Nh - 1) // 2
    fine = (indices[:, None] + offset[None, :]) / physical.duration
    np.testing.assert_allclose(
        fine[:, (physical.Nh - 1) // 2], physical.freq, rtol=0, atol=2e-17
    )
    nulls = np.arange(1, int(FMAX * 4 * TAU) + 1) / (4 * TAU)
    keep = np.ones(physical.N, dtype=bool)
    for null in nulls:
        keep &= ~(
            (fine[:, 0] <= null + NULL_HALF_WIDTH)
            & (fine[:, -1] >= null - NULL_HALF_WIDTH)
        )
    physical, fine = physical.apply_mask(keep), fine[keep]
    factor = np.mean(transfer_power(fine), axis=1)
    if not np.all(np.isfinite(factor) & (factor > 0)):
        raise ValueError("The retained response factor must be finite and positive")
    residual = replace(
        physical,
        u_re=physical.u_re / np.sqrt(factor)[:, None, None],
        u_im=physical.u_im / np.sqrt(factor)[:, None, None],
        raw_psd=None,
        raw_freq=None,
    )
    # Include complex cross-spectra in these scale and convention checks.
    scale = np.max(np.abs(physical.Y), axis=(1, 2))[:, None, None]
    np.testing.assert_allclose(
        residual.Y * factor[:, None, None] / scale,
        physical.Y / scale,
        rtol=2e-13,
        atol=2e-13,
    )
    empirical_scale = np.max(np.abs(physical.empirical_psd.psd), axis=(1, 2))[
        :, None, None
    ]
    np.testing.assert_allclose(
        residual.empirical_psd.psd * factor[:, None, None] / empirical_scale,
        physical.empirical_psd.psd / empirical_scale,
        rtol=2e-13,
        atol=2e-13,
    )
    return physical, residual, fine, factor, metadata


def package_provenance():
    package = Path(log_psplines.__file__).resolve().parent
    # Relative source hashes also identify editable installations with local changes.
    provenance = {
        "source_sha256": {
            str(path.relative_to(package)): sha256(path)
            for path in sorted(package.rglob("*.py"))
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("LogPSplinePSD", "jax", "numpyro", "numpy", "scipy", "h5py")
        },
        "python": platform.python_version(),
        "jax_enable_x64": jax.config.x64_enabled,
    }
    repository = package.parents[1]
    if (repository / ".git").exists():
        provenance["revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        provenance["tracked_source_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no", "--", "src"],
                cwd=repository,
                text=True,
            ).strip()
        )
    return provenance


def diagnostics(result):
    summary = sampling_diagnostics(result)
    # Aggregate extrema alone can hide undefined diagnostics for one coefficient.
    parameter_summary = azs.summary(
        result.posterior, kind="diagnostics", round_to="none"
    )
    all_parameters_finite = bool(
        np.isfinite(
            parameter_summary[["r_hat", "ess_bulk", "ess_tail"]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float, na_value=np.nan)
        ).all()
    )
    summary["all_parameter_diagnostics_finite"] = all_parameters_finite
    for row in summary["nuts"]:
        # Use numerical values for thresholds, not the library's display-rounded
        # extrema. Keep the same Cholesky-row variable grouping.
        channel = row["factor"]
        scalar_names = {
            f"sigma_delta_{channel}",
            f"weights_delta_{channel}",
            f"weights_delta_{channel}_raw",
        }
        prefixes = tuple(
            f"{name}_{channel}_"
            for name in (
                "sigma_theta_re",
                "weights_theta_re",
                "sigma_theta_im",
                "weights_theta_im",
            )
        )
        names = [
            name
            for name in result.posterior.data_vars
            if name in scalar_names or name.startswith(prefixes)
        ]
        numeric = azs.summary(
            result.posterior[names], kind="diagnostics", round_to="none"
        )
        row.update(
            rhat_max=float(np.max(numeric["r_hat"].to_numpy(dtype=float))),
            ess_bulk_min=float(np.min(numeric["ess_bulk"].to_numpy(dtype=float))),
            ess_tail_min=float(np.min(numeric["ess_tail"].to_numpy(dtype=float))),
        )
        energies = np.asarray(result.sample_stats[f"energy_channel_{row['factor']}"])
        with np.errstate(invalid="ignore", divide="ignore"):
            bfmi = np.mean(np.diff(energies, axis=1) ** 2, axis=1) / np.var(
                energies, axis=1, ddof=1
            )
        row["bfmi_per_chain"] = bfmi.tolist()
        row["diagnostic_pass"] = bool(
            result.posterior.sizes["chain"] >= 2
            and all_parameters_finite
            and row["divergences"] == 0
            and row["max_treedepth_hits"] == 0
            and np.isfinite(row["rhat_max"])
            and row["rhat_max"] < 1.01
            and row["ess_bulk_min"] >= 400
            and row["ess_tail_min"] >= 400
            and np.all(np.isfinite(bfmi) & (bfmi >= 0.3))
        )
    summary["all_factors_pass"] = all(row["diagnostic_pass"] for row in summary["nuts"])
    summary["diagnostic_thresholds"] = {
        "minimum_chains": 2,
        "rhat_max_exclusive": 1.01,
        "minimum_bulk_and_tail_ess": 400,
        "minimum_bfmi": 0.3,
        "maximum_divergences": 0,
        "maximum_depth_hits": 0,
    }
    return summary


def run(source, basis, output, config, *, prepare_only=False):
    output.mkdir(parents=True, exist_ok=True)
    physical, residual, fine, factor, generator = prepare_data(source, basis, config)
    np.savez_compressed(
        output / "preprocessed.npz",
        frequency=physical.freq,
        fine_frequency=fine,
        support_low=fine[:, 0],
        support_high=fine[:, -1],
        factor=factor,
        empirical_psd=physical.empirical_psd.psd,
        u_re=physical.u_re,
        u_im=physical.u_im,
        channel_stds=physical.channel_stds,
        Nb=physical.Nb,
        Nh=physical.Nh,
        duration=physical.duration,
        enbw=physical.enbw,
    )
    provenance = {
        "source_file": Path(source).name,
        "source_sha256": sha256(source),
        "generator": generator,
        "basis": basis,
        "config": asdict(config),
        "package": package_provenance(),
        "runner_sha256": sha256(__file__),
        "mean_removal": "global_per_channel",
        "n_frequency": physical.N,
        "response": "bin mean of 16*sin(2*pi*f*tau)^2*sin(4*pi*f*tau)^2; tau=8.3 s",
        "null_mask": "remove any pooled support intersecting n/(4*tau) +/- 0.0001 Hz",
        "spectrum_prior_target": "Residual coarse spectral matrix S/Gbar; log knots; changed prior and function space",
        "covariance_convention": "S_ij = E[d_i conjugate(d_j)]",
        "claim_scope": "One stationary synthetic realization; descriptive recovery, not coverage calibration or time-varying validation",
        "preparation_checks_pass": True,
    }
    write_json(output / "provenance.json", provenance)
    print(
        f"Prepared {physical.N} bins (Nb={physical.Nb}, Nh={physical.Nh})", flush=True
    )
    if prepare_only:
        return
    started = time.monotonic()
    result = fit(residual, config)
    seconds = time.monotonic() - started
    result.metadata.update(
        fit_seconds=seconds,
        spectrum_target="Residual coarse spectral matrix S/Gbar",
        known_transfer_factor=factor,
        physical_restoration="Multiply every residual spectrum draw by saved Gbar; channel units are already restored",
    )
    result.to_netcdf(output / "residual_inference_data.nc")
    # All chain/draw spectral samples are retained by the stationary fitter.
    draws = result.spectral_density * factor[None, None, :, None, None]
    quantiles = np.asarray(result.quantiles()) * factor[None, :, None, None]
    np.savez_compressed(
        output / "physical_spectrum.npz",
        frequency=physical.freq,
        draws=draws,
        quantiles=quantiles,
        percentiles=np.array([5.0, 50.0, 95.0]),
        coherence_quantiles=np.asarray(result.quantiles(kind="coherence")),
        magnitude_quantiles=np.asarray(result.quantiles(kind="magnitude"))
        * factor[None, :, None, None],
        empirical_psd=physical.empirical_psd.psd,
        factor=factor,
        fine_frequency=fine,
        support_low=fine[:, 0],
        support_high=fine[:, -1],
        channel_names=np.array(
            ("X2", "Y2", "Z2") if basis == "xyz" else ("A2", "E2", "T2")
        ),
    )
    summary = diagnostics(result)
    summary.update(
        fit_seconds=seconds,
        case=generator["case"],
        basis=basis,
        n_frequency=physical.N,
        Nb=physical.Nb,
        Nh=physical.Nh,
        finite_physical_draws=bool(np.isfinite(draws).all()),
        claim_scope=provenance["claim_scope"],
    )
    write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {"fit_seconds": seconds, "all_factors_pass": summary["all_factors_pass"]}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("asymmetric", "symmetric"), default="asymmetric"
    )
    parser.add_argument("--basis", choices=("aet", "xyz"), default="aet")
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--delta-knots", type=int, default=20)
    parser.add_argument(
        "--theta-knots", nargs=2, type=int, default=(12, 8), metavar=("REAL", "IMAG")
    )
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = configuration(
        seed=args.seed,
        chains=args.chains,
        warmup=args.warmup,
        samples=args.samples,
        blocks=args.blocks,
        bins=args.bins,
        theta_knots=args.theta_knots,
        delta_knots=args.delta_knots,
    )
    output = args.output or HERE / "output" / f"{args.case}_{args.basis}"
    run(
        args.data / f"{args.case}.h5",
        args.basis,
        output,
        config,
        prepare_only=args.prepare_only,
    )


if __name__ == "__main__":
    main()
