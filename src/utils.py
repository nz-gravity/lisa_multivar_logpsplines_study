"""Shared experiment utilities for LISA Log P-Spline runners."""

from __future__ import annotations

import numpy as np

DEFAULT_DURATION_PRESETS = {
    "1m": 30.0,
    "3m": 90.0,
    "6m": 180.0,
    "1y": 365.0,
}


def duration_slug(days: float, presets: dict[str, float] | None = None) -> str:
    """Return a compact folder/file label for a duration in days."""
    active_presets = presets or DEFAULT_DURATION_PRESETS
    for label, value in active_presets.items():
        if np.isclose(days, value):
            return label
    if float(days).is_integer():
        return f"{int(days)}d"
    return f"{days:g}d".replace(".", "p")


def choose_nb_for_duration(duration_days: float, block_days: float = 7.0) -> int:
    """Choose the Wishart block count from the requested analysis duration."""
    return max(1, int(np.floor(float(duration_days) / block_days)))


def compute_retained_frequency_count(
    n_trim: int,
    dt: float,
    nb: int,
    fmin: float,
    fmax: float,
) -> int:
    """Return the retained positive frequency count on the block FFT grid."""
    block_len = n_trim // nb
    if block_len <= 0:
        raise ValueError(
            f"Invalid block length: n_trim={n_trim}, nb={nb}, block_len={block_len}"
        )

    freq = np.fft.rfftfreq(block_len, d=dt)[1:]
    keep = (freq >= fmin) & (freq <= fmax)
    n_retained = int(np.count_nonzero(keep))
    if n_retained <= 0:
        raise ValueError(f"No positive frequencies retained in [{fmin}, {fmax}] Hz.")
    return n_retained


def choose_coarse_grain_nc(n_retained: int, nc_target: int) -> int:
    """Choose Nc <= target for coarse-graining."""
    if nc_target <= 0:
        raise ValueError(f"nc_target must be positive; got {nc_target}.")
    return min(nc_target, n_retained)


def light_travel_null_exclusion_bands(
    fmin: float,
    fmax: float,
    *,
    light_travel_time: float,
    halfwidth: float,
) -> tuple[tuple[float, float], ...]:
    """Build exclusion bands around transfer-function nulls in [fmin, fmax]."""
    base = 1.0 / (4.0 * light_travel_time)
    n_max = int(np.floor(fmax / base))
    bands: list[tuple[float, float]] = []
    for n in range(1, n_max + 1):
        center = n * base
        lo = max(fmin, center - halfwidth)
        hi = min(fmax, center + halfwidth)
        if lo < hi:
            bands.append((lo, hi))
    return tuple(bands)
