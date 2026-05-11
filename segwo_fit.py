# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: sgwb-renate-noise-models (3.13.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Noise modeling for noise-5
#
# This notebook builds the covariance matrix for the fixed-armlength, asymetric
# noise of noise-5 datasets. We use SEGWO to build the covariance matrix, and plot
# them against the data as a simple sanity check.

# %%
import matplotlib
matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

NOISE5A = Path("noise_5a")
OUT = NOISE5A / "out_segwo"
OUT.mkdir(exist_ok=True)
import segwo
from lisaconstants import c
from lisaconstants.indexing import LINKS, MOSAS
from numpy.typing import NDArray
from pytdi import LISATDICombination
from pytdi.michelson import X2_ETA, Y2_ETA, Z2_ETA
from scipy.signal import csd, welch

# %% [markdown]
# ## First quicklook of data

# %%
fs = 0.5  # Hz
duration = 365.25 * 24 * 3600  # seconds
fmin = 1.0 / duration  # Hz
oms_kfnee = 2e-3  # Hz
tm_fknee = 0.4e-3  # Hz
laser_freq = 2.816e14  # Hz

# %%
# Define light travel times (single time point)
ltts = np.repeat(8.3, 6)[None]  # seconds, shape (1, 6)

# %%
with h5py.File(NOISE5A / "tdi.h5", "r") as f:
    assert np.allclose(fs, 1.0 / f.attrs["dt"])
    xyz = np.stack([f["X2"][:], f["Y2"][:], f["Z2"][:]], axis=-1)

print(xyz.shape)

# %%
freq, xyz_psd = welch(
    xyz, fs=fs, axis=0, window=("kaiser", 31), detrend=False, nperseg=2**18
)

freq = freq[3:]  # Remove DC and very low frequencies
xyz_psd = xyz_psd[3:]

# %%
plt.figure(figsize=(12, 6))
for i, tdi in enumerate("XYZ"):
    plt.loglog(freq, xyz_psd[:, i], alpha=0.7, label=f"TDI {tdi}")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [/Hz]")
plt.legend()
plt.title("Data PSD")
plt.savefig(OUT / "data_psd.png", dpi=150, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## Simulation parameters

# %%
tm_baseline_asd = 2.4e-15
oms_baseline_asd = 7.9e-12

tm_asds = {
    12: 3.2987207830707333e-15,
    23: 2.2068318236082305e-15,
    31: 4.799588250657558e-15,
    13: 3.711602738525377e-15,
    32: 4.271678793058393e-15,
    21: 4.467581058832328e-15,
}

oms_asds = {
    12: 1.3592816495174599e-11,
    23: 1.1705105384539884e-11,
    31: 1.1098305430366561e-11,
    13: 9.41502170206022e-12,
    32: 1.0832545004320958e-11,
    21: 7.6801489202984e-12,
}

# %%
# Plot noise levels for each MOSA

mosa_labels = [str(mosa) for mosa in MOSAS]
x = np.arange(6)

testmass_vals = np.array([tm_asds[m] for m in MOSAS])
oms_vals = np.array([oms_asds[m] for m in MOSAS])

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].scatter(x, testmass_vals, color="tab:blue")
axes[0].axhline(tm_baseline_asd, color="gray", ls="--")
axes[0].set_ylabel(r"TM ASD [m s$^{-2}$/$\sqrt{\mathrm{Hz}}$]")

axes[1].scatter(x, oms_vals, color="tab:orange")
axes[1].axhline(oms_baseline_asd, color="gray", ls="--")
axes[1].set_ylabel(r"OMS ASD [m/$\sqrt{\mathrm{Hz}}$]")
axes[1].set_xlabel("MOSA")
axes[1].set_xticks(x)
axes[1].set_xticklabels(mosa_labels)

fig.suptitle("Noise levels for each MOSA")
fig.savefig(OUT / "noise_levels.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Noise models

# %%
def tm_model_psd(asd: float, *, filter_approx: bool = True) -> NDArray:
    """Model for TM noise PSD in TMI carrier beatnote fluctuations."""

    # The model assumes the ASD is motion of the test mass, but what was given
    # is the ASD of the acceleration noise, so we need to divide by 2 to account
    # for the back-and-forth optical path
    asd /= 2.0

    if not filter_approx:
        psd_acc = asd**2 * (1 + (tm_fknee / freq) ** 2)
        psd_hertz = (2 * laser_freq / (2 * np.pi * c * freq)) ** 2 * psd_acc
    else:
        psd_highfreq = (
            (2 * asd * laser_freq / (2 * np.pi * c)) ** 2
            * np.abs(
                (2 * np.pi * fmin)
                / (1 - np.exp(-2 * np.pi * fmin / fs) * np.exp(-2j * np.pi * freq / fs))
            )
            ** 2
            * 1
            / (fs * fmin) ** 2
        )
        psd_lowfreq = (
            (2 * asd * laser_freq * tm_fknee / (2 * np.pi * c)) ** 2
            * np.abs(
                (2 * np.pi * fmin)
                / (1 - np.exp(-2 * np.pi * fmin / fs) * np.exp(-2j * np.pi * freq / fs))
            )
            ** 2
            * 1
            / (fs * fmin) ** 2
            * np.abs(1 / (1 - np.exp(-2j * np.pi * freq / fs))) ** 2
            * (2 * np.pi / fs) ** 2
        )
        psd_hertz = psd_lowfreq + psd_highfreq
    return psd_hertz


# %%
plt.figure(figsize=(10, 6))
for mosa in MOSAS:
    plt.loglog(freq, tm_model_psd(tm_asds[mosa]), label=f"MOSA {mosa}")
plt.loglog(
    freq,
    tm_model_psd(tm_baseline_asd),
    color="gray",
    ls="--",
    label="Baseline",
)
plt.xlabel("Frequency [Hz]")
plt.ylabel("ASD [/Hz]")
plt.title("TM noise PSD in TMI carrier beatnote fluctuations")
plt.legend()
plt.savefig(OUT / "tm_noise_psd.png", dpi=150, bbox_inches="tight")
plt.close()

# %%
def oms_model_psd(asd: float, *, filter_approx: bool = True) -> NDArray:
    """Model for OMS noise PSD in ISI carrier beatnote fluctuations."""

    if not filter_approx:
        psd_meters = asd**2 * (1 + (oms_kfnee / freq) ** 4)
        psd_hertz = (2 * np.pi * freq * laser_freq / c) ** 2 * psd_meters
    else:
        psd_highfreq = (asd * fs * laser_freq / c) ** 2 * np.sin(
            2 * np.pi * freq / fs
        ) ** 2
        psd_lowfreq = (
            (2 * np.pi * asd * laser_freq * oms_kfnee**2 / c) ** 2
            * np.abs(
                (2 * np.pi * fmin)
                / (1 - np.exp(-2 * np.pi * fmin / fs) * np.exp(-2j * np.pi * freq / fs))
            )
            ** 2
            * 1
            / (fs * fmin) ** 2
        )
        psd_hertz = psd_highfreq + psd_lowfreq
    return psd_hertz


# %%
plt.figure(figsize=(10, 6))
for mosa in MOSAS:
    plt.loglog(freq, oms_model_psd(oms_asds[mosa]), label=f"MOSA {mosa}")
plt.loglog(
    freq,
    oms_model_psd(oms_baseline_asd),
    color="gray",
    ls="--",
    label="Baseline",
)
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [/Hz]")
plt.title("OMS noise PSD in ISI carrier beatnote fluctuations")
plt.legend()
plt.savefig(OUT / "oms_noise_psd.png", dpi=150, bbox_inches="tight")
plt.close()

# %%
noise_cov = segwo.cov.construct_covariance_from_psds(
    [tm_model_psd(tm_asds[mosa]) for mosa in MOSAS]
    + [oms_model_psd(oms_asds[mosa]) for mosa in MOSAS]
)

print(noise_cov.shape)
print(np.diag(noise_cov[0]))

# %% [markdown]
# ## Transformation to $\eta$

# %%
# Define a dictionary of TDI combinations mapping our noises into each of the 6
# single links (aka, the 6 eta variables)
ETA_COMBS: dict[int, LISATDICombination] = {}

# We start by defining it for eta 12:
#
# A PyTDI combination is defined as a dictionary, where the keys are the names
# of the input variables (N_tm_12, N_oms_12, etc.), and the values are lists of
# tuples. Each tuple contains a scaling coefficient and a list of delays to be
# applied on the corresponding variable.
ETA_COMBS[12] = LISATDICombination(
    {
        "N_tm_21": [(1, ["D_12"])],
        "N_tm_12": [(1, [])],
        "N_oms_12": [(1, [])],
    }
)

# Use PyTDI symmetry operations to define the other links

# We can do cyclic permuations, rotating indices 1->2->3->1
ETA_COMBS[23] = ETA_COMBS[12].rotated()
ETA_COMBS[31] = ETA_COMBS[23].rotated()

# We can do reflections along the axis going through the respective spacecraft,
# ie., exchanging indices 2<->3, 3<->1, and 1<->2
ETA_COMBS[13] = ETA_COMBS[12].reflected(1)
ETA_COMBS[21] = ETA_COMBS[23].reflected(2)
ETA_COMBS[32] = ETA_COMBS[31].reflected(3)

# %%
# Define list of noise labels (our input variables)
noise_list = [f"N_tm_{mosa}" for mosa in LINKS] + [f"N_oms_{mosa}" for mosa in LINKS]

# Construct the mixing matrix for the noise covariance
noise2eta = segwo.cov.construct_mixing_from_pytdi(
    freq,
    measurements=noise_list,
    tdi_combinations=[ETA_COMBS[mosa] for mosa in LINKS],
    ltts=ltts,
)

# It's a 6x12 matrix (transforms 12 noise variables into 6 single link
# measurements), with two additional first axes for the time point (t=0.0) and
# frequencies
print(noise2eta.shape)

# %% [markdown]
# ## Transformation to XYZ

# %%
# We form the ordered list of input variables, i.e., the eta variables
eta_list = [f"eta_{mosa}" for mosa in LINKS]

# Then we construct our mixing matrix for X2, Y2, and Z2
eta2xyz = segwo.cov.construct_mixing_from_pytdi(
    freq, eta_list, [X2_ETA, Y2_ETA, Z2_ETA], ltts
)

# It's a 3x6 matrix (transform 6 eta variables into 3 XYZ variables), also given
# for each time point (here, just t=0.0) and frequencies
print(eta2xyz.shape)

# %%
noise_cov_xyz = segwo.cov.project_covariance(noise_cov, [noise2eta, eta2xyz])

print(noise_cov_xyz.shape)

# %%
plt.figure(figsize=(10, 6))

# Plot the noise PSD for each TDI channel
for i, tdi in enumerate("XYZ"):
    plt.loglog(freq, np.abs(noise_cov_xyz[0, :, i, i]), label=f"PSD TDI {tdi}")

# Plot the noise CSD between TDI channels
for i, tdi_i in enumerate("XYZ"):
    for j, tdi_j in enumerate("XYZ"):
        if i < j:
            plt.loglog(
                freq,
                np.abs(noise_cov_xyz[0, :, i, j]),
                "--",
                label=f"CSD TDI {tdi_i} with {tdi_j}",
            )

plt.xlabel("Frequency [Hz]")
plt.ylabel("Noise PSD/CSD [/Hz]")
plt.title("Noise PSD/CSD for each TDI channel")
plt.legend()
plt.savefig(OUT / "xyz_noise_psd_csd.png", dpi=150, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## Comparison with data

# %%
# Plot the noise PSD for each TDI channel
colors = ["tab:blue", "tab:orange", "tab:green"]
for i, tdi in enumerate("XYZ"):

    _, axes = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    axes[0].loglog(
        freq,
        np.abs(noise_cov_xyz[0, :, i, i]),
        c="black",
        lw=2,
        label="Noise model",
    )
    axes[0].loglog(freq, xyz_psd[:, i], c=colors[i], alpha=0.8, label="Data")
    axes[0].set_ylabel("Noise PSD [/Hz]")
    axes[0].set_title(f"TDI {tdi}")
    axes[0].legend()

    axes[1].semilogx(freq, np.zeros_like(freq), color="black", lw=2)
    axes[1].semilogx(
        freq,
        xyz_psd[:, i] - np.abs(noise_cov_xyz[0, :, i, i]),
        c=colors[i],
    )
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Residual")

    plt.subplots_adjust(hspace=0)
    plt.savefig(OUT / f"compare_{tdi}.png", dpi=150, bbox_inches="tight")
    plt.close()

# %% [markdown]
# ## PSD matrix triangle plot: SEGWO model vs data

# %%
import sys
sys.path.insert(0, str(NOISE5A))
from load_and_plot_data import plot_psd_coherence

_welch_kw = dict(fs=fs, window=("kaiser", 31), detrend=False, nperseg=2**18)
_, Sxy_w = csd(xyz[:, 0], xyz[:, 1], **_welch_kw)
_, Syz_w = csd(xyz[:, 1], xyz[:, 2], **_welch_kw)
_, Szx_w = csd(xyz[:, 2], xyz[:, 0], **_welch_kw)
Sxy_w = Sxy_w[3:]
Syz_w = Syz_w[3:]
Szx_w = Szx_w[3:]

S_true = noise_cov_xyz[0].copy()  # (nfreq, 3, 3)
# diagonal entries are real by definition; abs guards against small numerical imaginary parts
for _i in range(3):
    S_true[:, _i, _i] = np.abs(noise_cov_xyz[0, :, _i, _i])

# save full-range covariance for use as the noise_5a reference in run_mcmc.py
np.savez(NOISE5A / "segwo_true_cov.npz", freq=freq, cov=S_true)
print(f"Saved SEGWO reference covariance to {NOISE5A / 'segwo_true_cov.npz'}")

# restrict to LISA science band to avoid TDI null artefacts collapsing the y-axis
_band = freq >= 1e-4
_freq = freq[_band]
_S_true = S_true[_band]
S_emp = {
    "Sxx": xyz_psd[_band, 0].real,
    "Syy": xyz_psd[_band, 1].real,
    "Szz": xyz_psd[_band, 2].real,
    "Sxy": Sxy_w[_band],
    "Syz": Syz_w[_band],
    "Szx": Szx_w[_band],
}

plot_psd_coherence(
    _freq,
    _S_true,
    S_emp,
    fname=OUT / "segwo_psd_matrix.png",
    reference_label="SEGWO model",
    empirical_label="Welch",
    psd_unit_label="Hz",
)
print(f"Saved {OUT / 'segwo_psd_matrix.png'}")

# %%
