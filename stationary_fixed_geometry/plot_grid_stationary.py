"""Plot the stationary-data control of the multivariate time-varying fitter."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def plot(directory):
    with np.load(directory / "physical_summaries.npz") as data:
        f = data["frequency"] * 1000
        q = np.diagonal(data["quantiles"], axis1=-2, axis2=-1).real
        reference = np.diagonal(data["expected_psd"], axis1=-2, axis2=-1).real
        ratio = data["last_to_first_auto_psd_ratio_quantiles"]
        days = data["time"] / 86400
        fine = data["fine_frequency"]
        coherence_re = data["last_minus_first_coherence_re_quantiles"]
        coherence_im = data["last_minus_first_coherence_im_quantiles"]
    gaps = (
        np.flatnonzero(fine[1:, 0] - fine[:-1, -1] > 2 * np.median(np.diff(fine[0])))
        + 1
    )
    segments = np.split(np.arange(len(f)), gaps)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.2})
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 7), sharex=True, constrained_layout=True
    )
    for channel, name in enumerate("AET"):
        ax = axes[0, channel]
        for selected in segments:
            ax.plot(
                f[selected],
                reference[selected, channel],
                color="black",
                linewidth=1,
                label="Stationary expected PSD" if selected[0] == 0 else None,
            )
            for index, color in ((0, "#1764ab"), (-1, "#9e3f91")):
                ax.fill_between(
                    f[selected],
                    q[0, index, selected, channel],
                    q[2, index, selected, channel],
                    color=color,
                    alpha=0.18,
                )
                ax.plot(
                    f[selected],
                    q[1, index, selected, channel],
                    color=color,
                    label=f"Day {days[index]:.3g}: median and 90% interval"
                    if selected[0] == 0
                    else None,
                )
        ax.set_yscale("log")
        ax.set_title(f"{name} auto PSD [1/Hz]")
        below = axes[1, channel]
        below.axhline(1, color="black", linewidth=1)
        for selected in segments:
            below.fill_between(
                f[selected],
                ratio[0, selected, channel],
                ratio[2, selected, channel],
                color="#1764ab",
                alpha=0.25,
            )
            below.plot(
                f[selected], ratio[1, selected, channel], color="#1764ab", linewidth=1.2
            )
        below.set_title(f"{name}: last / first PSD")
        below.set_xlabel("Frequency [mHz]")
        below.set_xscale("log")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Stationary LISA data fitted with a flexible time-varying model\nFixed geometry · unequal MOSAs · one ten-day realization",
        fontsize=14,
    )
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"stationary_time_control.{extension}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(
        2, 3, figsize=(12, 6), sharex=True, constrained_layout=True
    )
    for column, pair in enumerate(("AE", "AT", "ET")):
        for row, (part, contrast) in enumerate(
            (("Re", coherence_re), ("Im", coherence_im))
        ):
            ax = axes[row, column]
            ax.axhline(0, color="black", linewidth=1)
            for selected in segments:
                ax.fill_between(
                    f[selected],
                    contrast[0, selected, column],
                    contrast[2, selected, column],
                    color="#1764ab",
                    alpha=0.25,
                )
                ax.plot(f[selected], contrast[1, selected, column], color="#1764ab")
            ax.set_title(f"{pair}: change in {part} complex coherency")
            ax.set_xscale("log")
            if row == 1:
                ax.set_xlabel("Frequency [mHz]")
    fig.suptitle(
        "Stationary LISA cross-spectrum contrasts\nLast minus first reference time · median and 90% interval",
        fontsize=14,
    )
    for extension in ("png", "pdf"):
        fig.savefig(
            directory / f"stationary_cross_spectrum_control.{extension}", dpi=170
        )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=HERE / "output" / "grid_stationary"
    )
    plot(parser.parse_args().output)
