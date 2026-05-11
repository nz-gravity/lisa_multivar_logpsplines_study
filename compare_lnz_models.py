#!/usr/bin/env python3
"""Compare multivariate XYZ vs summed univariate AET models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import arviz_stats as azs
import numpy as np
from scipy.special import logsumexp
import xarray as xr


ROOT = Path(__file__).resolve().parent


def _duration_slug(days: float) -> str:
    presets = {
        "1m": 30.0,
        "3m": 90.0,
        "6m": 180.0,
        "1y": 365.0,
    }
    for label, value in presets.items():
        if math.isclose(days, value):
            return label
    if float(days).is_integer():
        return f"{int(days)}d"
    return f"{days:g}d".replace(".", "p")


def _eta_slug(eta: float) -> str:
    return f"eta{eta:g}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, bool | str | int | float) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return arr.tolist()
    except Exception:
        return str(value)


def _read_multivar_attrs(path: Path) -> dict[str, Any]:
    tree = xr.open_datatree(path, engine="h5netcdf")
    try:
        attrs = dict(tree.attrs)
    finally:
        tree.close()
    return attrs


def _read_univar_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_multivar_path(dataset_dir: Path, duration: str, eta: str) -> Path:
    candidates = (
        dataset_dir / "mcmc_output" / duration / eta / "inference_data.nc",
        dataset_dir / "mcmc_output" / duration / eta / "idata.nc",
        dataset_dir / "mcmc_output" / duration / "inference_data.nc",
        dataset_dir / "mcmc_output" / duration / "idata.nc",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate multivariate inference data. Checked:\n"
        + "\n".join(str(path) for path in candidates)
    )


def _resolve_univar_path(dataset_dir: Path, duration: str, eta: str) -> Path:
    candidate = (
        dataset_dir / "mcmc_output_univar" / duration / eta / "lnz_summary.json"
    )
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not locate univariate lnZ summary: {candidate}"
    )


def _read_tree(path: Path) -> xr.DataTree:
    return xr.open_datatree(path, engine="h5netcdf")


def _pointwise_loglik_from_tree(tree: xr.DataTree) -> xr.DataArray:
    dataset = tree["log_likelihood"].dataset
    if dataset is None or not dataset.data_vars:
        raise KeyError("Tree has no log_likelihood dataset.")
    if "log_likelihood" in dataset:
        return dataset["log_likelihood"]

    freq_vars = [
        name for name, var in dataset.data_vars.items() if "freq" in var.dims
    ]
    if not freq_vars:
        raise KeyError(
            "No pointwise log-likelihood variable with a freq dimension was found."
        )
    total = dataset[freq_vars[0]].copy(deep=True)
    for name in freq_vars[1:]:
        total = total + dataset[name]
    total.name = "log_likelihood"
    return total


def _combine_univar_aet_loglik(summary_path: Path) -> xr.DataTree:
    summary = _read_univar_summary(summary_path)
    channel_rows = summary.get("channels", [])
    if not channel_rows:
        raise ValueError(f"No per-channel entries found in {summary_path}")

    channel_arrays = []
    coords = None
    for row in channel_rows:
        outdir = Path(row["outdir"])
        candidate = outdir / "inference_data.nc"
        if not candidate.exists():
            candidate = outdir / "idata.nc"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Missing per-channel inference file in {outdir}"
            )
        tree = _read_tree(candidate)
        try:
            arr = _pointwise_loglik_from_tree(tree).load()
            if coords is None:
                coords = {dim: arr.coords[dim].values for dim in arr.dims}
            channel_arrays.append(arr.values)
        finally:
            tree.close()

    if coords is None:
        raise ValueError(
            "Could not resolve coordinates for combined univariate log-likelihood."
        )

    combined = sum(channel_arrays)
    ds = xr.Dataset(
        {
            "log_likelihood": xr.DataArray(
                combined,
                dims=("chain", "draw", "freq"),
                coords=coords,
            )
        }
    )
    return xr.DataTree(children={"log_likelihood": xr.DataTree(dataset=ds)})


def _elpd_to_summary(result: Any) -> dict[str, Any]:
    """Serialise numeric fields from an ELPDData object."""
    d: dict[str, Any] = {
        "elpd": _json_safe(result.elpd),
        "se": _json_safe(result.se),
        "p": _json_safe(result.p),
        "n_data_points": _json_safe(result.n_data_points),
        "n_samples": _json_safe(result.n_samples),
        "warning": bool(result.warning),
    }
    pk = getattr(result, "pareto_k", None)
    if pk is not None:
        arr = np.asarray(pk.values).flatten()
        d["pareto_k_max"] = float(arr.max())
        d["pareto_k_mean"] = float(arr.mean())
        d["pareto_k_frac_gt_07"] = float((arr > 0.7).mean())
        d["pareto_k_frac_gt_10"] = float((arr > 1.0).mean())
    return d


def run_loo_comparison(
    *,
    multivar_path: Path,
    univar_summary_path: Path,
) -> dict[str, Any]:
    """Run PSIS-LOO comparison for multivariate XYZ vs summed univariate AET.

    Per-frequency pointwise log-likelihoods are read from the stored
    log_likelihood group in each inference_data.nc file (written unconditionally
    by the pipeline after NUTS).

    Note: Pareto k > 0.7 is expected for smooth P-spline spectral models.  The
    paired ELPD difference (multivar minus univar) is more reliable than the
    individual ELPDs because correlated errors cancel.  Use WAIC (--method waic)
    as a more robust alternative when k is high.
    """
    multivar_tree = _read_tree(multivar_path)
    try:
        loo_multivar = azs.loo(
            multivar_tree,
            var_name="log_likelihood",
            pointwise=True,
            reff=1.0,
        )
    finally:
        multivar_tree.close()

    univar_tree = _combine_univar_aet_loglik(univar_summary_path)
    loo_univar = azs.loo(
        univar_tree,
        var_name="log_likelihood",
        pointwise=True,
        reff=1.0,
    )

    elpd_multi = float(loo_multivar.elpd)
    elpd_uni = float(loo_univar.elpd)
    delta_elpd = elpd_multi - elpd_uni

    # Paired SE uses per-bin differences (Vehtari et al. 2017).
    elpd_i_multi = getattr(loo_multivar, "elpd_i", None)
    elpd_i_uni = getattr(loo_univar, "elpd_i", None)
    if elpd_i_multi is not None and elpd_i_uni is not None:
        diff_i = np.asarray(elpd_i_multi.values).flatten() - np.asarray(
            elpd_i_uni.values
        ).flatten()
        n = diff_i.size
        delta_se = float(np.sqrt(n * np.var(diff_i)))
    else:
        se_multi = float(loo_multivar.se)
        se_uni = float(loo_univar.se)
        delta_se = math.sqrt(se_multi**2 + se_uni**2)

    return {
        "multivar_loo": _elpd_to_summary(loo_multivar),
        "univar_aet_sum_loo": _elpd_to_summary(loo_univar),
        "delta_elpd_loo_multivar_minus_univar": delta_elpd,
        "delta_elpd_loo_se": delta_se,
        "preferred_by_loo": (
            "multivar_xyz" if delta_elpd > 0.0 else "univar_aet_sum"
        ),
        "comparison_note": (
            "PSIS-LOO over frequency bins. "
            "The univariate AET model sums A/E/T pointwise log-likelihoods "
            "at each frequency. reff=1.0 is used for both models. "
            "Pareto k > 0.7 is expected for smooth P-spline models; "
            "see pareto_k_frac_gt_07 in each model's loo summary. "
            "Consider WAIC (--method waic) as a more stable alternative."
        ),
    }


def _compute_waic(tree: xr.DataTree) -> dict[str, Any]:
    """Compute WAIC from per-frequency pointwise log-likelihoods.

    No importance sampling — more stable than PSIS-LOO for smooth P-spline
    models with high Pareto k.

    Returns a dict with elpd, se, p_waic, lppd, n_data_points, and a private
    _elpd_i array used for paired comparisons (not JSON-serialised).
    """
    ll = tree["log_likelihood"].dataset["log_likelihood"].values  # (chain,draw,freq)
    ll_flat = ll.reshape(-1, ll.shape[-1])  # (S, F)
    S = ll_flat.shape[0]
    lppd_i = logsumexp(ll_flat, axis=0) - np.log(S)
    p_i = np.var(ll_flat, axis=0)
    elpd_i = lppd_i - p_i
    elpd = float(np.sum(elpd_i))
    se = float(np.sqrt(len(elpd_i) * np.var(elpd_i)))
    return {
        "elpd": elpd,
        "se": se,
        "p_waic": float(np.sum(p_i)),
        "lppd": float(np.sum(lppd_i)),
        "n_data_points": int(len(elpd_i)),
        "_elpd_i": elpd_i,
    }


def run_waic_comparison(
    *,
    multivar_path: Path,
    univar_summary_path: Path,
) -> dict[str, Any]:
    """Compute WAIC-based ELPD comparison for multivariate XYZ vs summed univariate AET."""
    multivar_tree = _read_tree(multivar_path)
    try:
        waic_multivar = _compute_waic(multivar_tree)
    finally:
        multivar_tree.close()

    # Build combined univariate lnl array and compute WAIC on it.
    univar_tree = _combine_univar_aet_loglik(univar_summary_path)
    waic_univar = _compute_waic(univar_tree)

    elpd_multi = waic_multivar["elpd"]
    elpd_uni = waic_univar["elpd"]
    delta_elpd = elpd_multi - elpd_uni

    # Paired SE.
    ei_m = waic_multivar["_elpd_i"]
    ei_u = waic_univar["_elpd_i"]
    diff_i = ei_m - ei_u
    n = diff_i.size
    delta_se = float(np.sqrt(n * np.var(diff_i)))

    def _public(d: dict[str, Any]) -> dict[str, Any]:
        return {k: _json_safe(v) for k, v in d.items() if not k.startswith("_")}

    return {
        "multivar_waic": _public(waic_multivar),
        "univar_aet_sum_waic": _public(waic_univar),
        "delta_elpd_waic_multivar_minus_univar": delta_elpd,
        "delta_elpd_waic_se": delta_se,
        "preferred_by_waic": (
            "multivar_xyz" if delta_elpd > 0.0 else "univar_aet_sum"
        ),
        "comparison_note": (
            "WAIC over frequency bins — no importance sampling. "
            "More stable than PSIS-LOO for smooth P-spline models. "
            "Paired SE computed from per-bin WAIC contributions."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset directory, e.g. noise_4a or noise_5a.",
    )
    parser.add_argument(
        "--duration-days",
        type=float,
        default=None,
        help="Duration in days. Used to build the standard output paths.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="Eta value. Used to build the standard output paths.",
    )
    parser.add_argument(
        "--multivar-path",
        type=Path,
        default=None,
        help="Explicit multivariate inference_data.nc/idata.nc path.",
    )
    parser.add_argument(
        "--univar-summary",
        type=Path,
        default=None,
        help="Explicit univariate lnz_summary.json path.",
    )
    parser.add_argument(
        "--favor",
        choices=("multivar", "univar"),
        default="multivar",
        help="Direction of the reported Bayes factor.",
    )
    parser.add_argument(
        "--method",
        choices=("lnz", "loo", "waic", "both"),
        default="lnz",
        help=(
            "Comparison method to run. "
            "'both' runs lnz + loo + waic. "
            "'loo' uses PSIS-LOO (may have high Pareto k for smooth models). "
            "'waic' uses WAIC (no IS step, recommended when k > 0.7)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.multivar_path is None or args.univar_summary is None:
        if args.dataset is None or args.duration_days is None or args.eta is None:
            raise ValueError(
                "Provide either both explicit paths, or --dataset with "
                "--duration-days and --eta."
            )
        dataset_dir = args.dataset
        if not dataset_dir.is_absolute():
            dataset_dir = ROOT / dataset_dir
        duration = _duration_slug(float(args.duration_days))
        eta = _eta_slug(float(args.eta))
        multivar_path = _resolve_multivar_path(dataset_dir, duration, eta)
        univar_path = _resolve_univar_path(dataset_dir, duration, eta)
    else:
        multivar_path = Path(args.multivar_path)
        univar_path = Path(args.univar_summary)

    result: dict[str, Any] = {
        "multivar_path": str(multivar_path),
        "univar_summary_path": str(univar_path),
    }

    if args.method in {"lnz", "both"}:
        multivar_attrs = _read_multivar_attrs(multivar_path)
        univar_summary = _read_univar_summary(univar_path)

        lnz_multi = float(multivar_attrs["lnz"])
        lnz_multi_err = float(multivar_attrs.get("lnz_err", math.nan))
        lnz_multi_valid = bool(multivar_attrs.get("lnz_valid", False))

        lnz_uni = float(univar_summary["combined_lnz"])
        lnz_uni_err = float(univar_summary.get("combined_lnz_err", math.nan))
        lnz_uni_valid = bool(univar_summary.get("all_channels_valid", False))

        log_bf_multi_over_uni = lnz_multi - lnz_uni
        log_bf_err = math.sqrt(lnz_multi_err**2 + lnz_uni_err**2)
        log10_bf_multi_over_uni = log_bf_multi_over_uni / math.log(10.0)

        if args.favor == "multivar":
            reported_log_bf = log_bf_multi_over_uni
            reported_log10_bf = log10_bf_multi_over_uni
            favored = "multivar_xyz"
            disfavored = "univar_aet_sum"
        else:
            reported_log_bf = -log_bf_multi_over_uni
            reported_log10_bf = -log10_bf_multi_over_uni
            favored = "univar_aet_sum"
            disfavored = "multivar_xyz"

        if reported_log_bf < 700.0:
            bayes_factor = math.exp(reported_log_bf)
            bayes_factor_repr = f"{bayes_factor:.6g}"
        else:
            bayes_factor_repr = "overflow"

        result["lnz"] = {
            "multivar_lnz": lnz_multi,
            "multivar_lnz_err": lnz_multi_err,
            "multivar_lnz_valid": lnz_multi_valid,
            "univar_summed_lnz": lnz_uni,
            "univar_summed_lnz_err": lnz_uni_err,
            "univar_summed_lnz_valid": lnz_uni_valid,
            "log_bf_multivar_over_univar": log_bf_multi_over_uni,
            "log10_bf_multivar_over_univar": log10_bf_multi_over_uni,
            "reported_favored_model": favored,
            "reported_disfavored_model": disfavored,
            "reported_log_bf": reported_log_bf,
            "reported_log_bf_err": log_bf_err,
            "reported_log10_bf": reported_log10_bf,
            "reported_bayes_factor": bayes_factor_repr,
            "comparison_note": (
                "This compares the full multivariate XYZ blocked-Wishart model "
                "against the summed independent univariate AET periodogram models."
            ),
        }

    if args.method in {"loo", "both"}:
        result["loo"] = run_loo_comparison(
            multivar_path=multivar_path,
            univar_summary_path=univar_path,
        )

    if args.method in {"waic", "both"}:
        result["waic"] = run_waic_comparison(
            multivar_path=multivar_path,
            univar_summary_path=univar_path,
        )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
