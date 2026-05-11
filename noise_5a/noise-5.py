# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: venv (3.9.25)
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.031856, "end_time": "2026-04-10T15:51:29.360204", "exception": false, "start_time": "2026-04-10T15:51:29.328348", "status": "completed"}
# # SGWB Data Generation – Noise 5
#
# Noise only dataset with LDC Spritz noise curves only, with no orbit file. We use constant and equal LTTs (no orbit files) of 8.3 seconds. However, if the noise shapes are identical on all MOSAs, the noise levels will have 20% variations across the MOSAs.
#
# As a result, the resulting noise covariance matrices at TDI level will be stationary but not symmetrical.
#
# 1. Generate the noise and compute the carrier beatnote frequency fluctuations
# 2. Compute the XYZ 2 time series
#
# Run this notebook with paperwill using
# ```shell
# source venv/bin/activate
# BASEDIR="/scratch/muck0/jb611z/sgwb-datasets/noise-5"
# mkdir -p $BASEDIR
# nohup papermill noise-5.ipynb $BASEDIR/noise-5.ipynb -p BASEDIR "$BASEDIR" > $BASEDIR/nohup.out 2>&1 &
# ```

# %% papermill={"duration": 0.898689, "end_time": "2026-04-10T15:51:30.274064", "exception": false, "start_time": "2026-04-10T15:51:29.375375", "status": "completed"}
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from h5py import File

from scipy.signal import welch

from lisaconstants import c
from lisainstrument import Instrument
from lisainstrument.containers import ForEachMOSA
from pytdi import Data, michelson

logging.basicConfig(level=logging.INFO)

# %% papermill={"duration": 0.023334, "end_time": "2026-04-10T15:51:30.312943", "exception": false, "start_time": "2026-04-10T15:51:30.289609", "status": "completed"} tags=["parameters"]
# Base directory for outputs
BASEDIR = "./data"

# Set to true for quick tests
QUICKTEST = False

# %% papermill={"duration": 0.01611, "end_time": "2026-04-10T15:51:30.342907", "exception": false, "start_time": "2026-04-10T15:51:30.326797", "status": "completed"} tags=["injected-parameters"]
# Parameters
BASEDIR = "/home/jb611z/sgwb-data-generation/data/noise-5a"


# %% papermill={"duration": 0.13952, "end_time": "2026-04-10T15:51:30.495856", "exception": false, "start_time": "2026-04-10T15:51:30.356336", "status": "completed"}
# Copy requirements.txt
# !cp requirements.txt $BASEDIR

# %% [markdown] papermill={"duration": 0.018589, "end_time": "2026-04-10T15:51:30.527879", "exception": false, "start_time": "2026-04-10T15:51:30.509290", "status": "completed"}
# ## Instrument

# %% papermill={"duration": 0.015148, "end_time": "2026-04-10T15:51:30.557054", "exception": false, "start_time": "2026-04-10T15:51:30.541906", "status": "completed"}
a_day = 24 * 3600  # s
a_year = 365.25 * a_day  # s

# %% papermill={"duration": 0.214169, "end_time": "2026-04-10T15:51:30.784361", "exception": false, "start_time": "2026-04-10T15:51:30.570192", "status": "completed"}
# Define instrument

# We offset the instrument t0 to have orbits information at emission time
instru_t0 = 0
instru_fs = 0.5
instru_dt = 1 / instru_fs
# Generate 1 year dataset
instru_duration = 10 * a_day if QUICKTEST else a_year
instru_size = int(instru_duration // instru_dt) + 1
instru_path = os.path.join(BASEDIR, "instrument.h5")

instru = Instrument(
    t0=instru_t0,
    dt=instru_dt,
    size=instru_size,
    physics_upsampling=1,
    aafilter=None,
    lock="six",
    orbits=8.3,
)

instru.disable_dopplers()
instru.disable_all_noises()

# %% papermill={"duration": 0.38935, "end_time": "2026-04-10T15:51:31.188268", "exception": false, "start_time": "2026-04-10T15:51:30.798918", "status": "completed"}
# Draw noise levels for each MOSA from uniform distributions around baselines

np.random.seed(1)
testmass_baseline = 2.4e-15
oms_isi_carrier_baseline = 7.9e-12

testmass_bounds = (
    testmass_baseline * 2.0,
    testmass_baseline * 0.5,
)
oms_isi_carrier_bounds = (
    oms_isi_carrier_baseline * 2.0,
    oms_isi_carrier_baseline * 0.5,
)

instru.testmass_asds = ForEachMOSA(
    dict(zip(instru.MOSAS, np.random.uniform(*testmass_bounds, size=6)))
)
instru.oms_isi_carrier_asds = ForEachMOSA(
    dict(zip(instru.MOSAS, np.random.uniform(*oms_isi_carrier_bounds, size=6)))
)

print(f"{instru.testmass_asds=}")
print(f"{instru.oms_isi_carrier_asds=}")

# %%
# Plot noise levels for each MOSA

mosa_labels = [str(mosa) for mosa in instru.MOSAS]
x = np.arange(6)

testmass_vals = np.array([instru.testmass_asds[m] for m in mosa_labels])
oms_vals = np.array([instru.oms_isi_carrier_asds[m] for m in mosa_labels])

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].scatter(x, testmass_vals, color="tab:blue")
axes[0].axhline(testmass_baseline, color="gray", ls="--")
axes[0].set_ylabel(r"TM ASD [m s$^{-2}$/$\sqrt{\mathrm{Hz}}$]")

axes[1].scatter(x, oms_vals, color="tab:orange")
axes[1].axhline(oms_isi_carrier_baseline, color="gray", ls="--")
axes[1].set_ylabel(r"OMS ASD [m/$\sqrt{\mathrm{Hz}}$]")
axes[1].set_xlabel("MOSA")
axes[1].set_xticks(x)
axes[1].set_xticklabels(mosa_labels)

fig.suptitle("Noise levels for each MOSA")

# %% papermill={"duration": 314.040064, "end_time": "2026-04-10T15:56:45.259743", "exception": false, "start_time": "2026-04-10T15:51:31.219679", "status": "completed"}
# Run simulation

if os.path.exists(instru_path):
    os.remove(instru_path)
instru.write(instru_path, mode="w")

logging.info("Writing complete")

# %% papermill={"duration": 19.158557, "end_time": "2026-04-10T15:57:04.453388", "exception": false, "start_time": "2026-04-10T15:56:45.294831", "status": "completed"}
# Read instrument file

with File(instru_path, "r") as hdf:

    assert instru.t0 == hdf.attrs["t0"]
    assert instru.dt == hdf.attrs["dt"]
    assert instru.size == hdf.attrs["size"]
    assert instru.duration == hdf.attrs["duration"]

    isi_carrier_fluctuations = hdf["isi_carrier_fluctuations"][:]
    tmi_carrier_fluctuations = hdf["tmi_carrier_fluctuations"][:]

    plt.figure(figsize=(12, 12))
    plt.subplot(311)
    for mosa in instru.MOSAS:
        plt.plot(instru.t[::100], isi_carrier_fluctuations[mosa][::100], label=mosa)
    plt.legend()
    plt.ylabel("ISI carrier fluctuations [Hz]")

    plt.subplot(312)
    for mosa in instru.MOSAS:
        plt.plot(instru.t[::100], tmi_carrier_fluctuations[mosa][::100], label=mosa)
    plt.legend()
    plt.ylabel("TMI carrier fluctuations [Hz]")
    plt.xlabel("Time [s]")


# %% [markdown] papermill={"duration": 0.044456, "end_time": "2026-04-10T15:57:04.548091", "exception": false, "start_time": "2026-04-10T15:57:04.503635", "status": "completed"}
# In the ISI carrier beatnotes, we should see the OMS noise.
#
# In terms of displacement, it's given by
# $$S_\text{oms}^\text{m}(f) = A^2 \Big[ 1 + \Big(\frac{f_\text{knee}}{f}\Big)^4 \Big].$$
#
# Multiplying by $(2\pi f / c)^2$ to express it as fractional frequency deviations, we can obtain the equivalent frequency shift by adding the $\nu_0^2$ factor,
# $$S_\text{oms}^\text{Hz}(f) = \Big(\frac{2\pi f \nu_0}{c}\Big)^2 S_\text{oms}^\text{m}(f).$$

# %% papermill={"duration": 0.055892, "end_time": "2026-04-10T15:57:04.653897", "exception": false, "start_time": "2026-04-10T15:57:04.598005", "status": "completed"}
def asd(x, fs, **kwargs):
    """Compute ASD from time series.

    Args:
        x (array): time series
        fs (float): sampling frequency [Hz]
        kwargs: keyword arguments passed to scipy.signal.welch
    """
    if "window" not in kwargs:
        kwargs["window"] = ("kaiser", 30)
    if "nperseg" not in kwargs:
        kwargs["nperseg"] = 2**18
    if "detrend" not in kwargs:
        kwargs["detrend"] = None
    freq, psd = welch(x, fs, **kwargs)
    return freq[5:], np.sqrt(psd[5:])


# %% papermill={"duration": 0.053413, "end_time": "2026-04-10T15:57:04.752589", "exception": false, "start_time": "2026-04-10T15:57:04.699176", "status": "completed"}
def oms_in_isi_carrier(freq, instru, mosa="12", filter_approx=True):
    """Model for OMS noise PSD in ISI carrier beatnote fluctuations.

    Args:
        freq (float): frequencies [Hz]
        instru (Instrument): LISA instrument object
        filter_approx (bool): if True, use the full filter transfer function
    """
    if mosa is None:
        asd = oms_isi_carrier_baseline
        fknee = instru.oms_fknees["12"]
    else:
        asd = instru.oms_isi_carrier_asds[mosa]
        fknee = instru.oms_fknees[mosa]
    fmin = 1.0 / instru.duration
    if not filter_approx:
        psd_meters = asd**2 * (1 + (fknee / freq) ** 4)
        psd_hertz = (2 * np.pi * freq * instru.central_freq / c) ** 2 * psd_meters
    else:
        psd_highfreq = (asd * instru.fs * instru.central_freq / c) ** 2 * np.sin(
            2 * np.pi * freq / instru.fs
        ) ** 2
        psd_lowfreq = (
            (2 * np.pi * asd * instru.central_freq * fknee**2 / c) ** 2
            * np.abs(
                (2 * np.pi * fmin)
                / (
                    1
                    - np.exp(-2 * np.pi * fmin / instru.fs)
                    * np.exp(-2j * np.pi * freq / instru.fs)
                )
            )
            ** 2
            * 1
            / (instru.fs * fmin) ** 2
        )
        psd_hertz = psd_highfreq + psd_lowfreq
    return np.sqrt(psd_hertz)


# %% papermill={"duration": 6.750895, "end_time": "2026-04-10T15:57:11.541187", "exception": false, "start_time": "2026-04-10T15:57:04.790292", "status": "completed"}
_, axes = plt.subplots(
    2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)

f, _ = asd(isi_carrier_fluctuations["12"], instru.fs)
model_simu = oms_in_isi_carrier(f, instru, mosa=None, filter_approx=True)
model = oms_in_isi_carrier(f, instru, mosa=None, filter_approx=False)

for mosa in instru.MOSAS:
    f, asd_mosa = asd(isi_carrier_fluctuations[mosa], instru.fs)
    axes[0].loglog(f, asd_mosa, label=mosa)
    axes[1].semilogx(f, (asd_mosa - model_simu) / model_simu, label=mosa)

axes[0].loglog(f, model_simu, color="black", lw=2, label="model (simulated)")
axes[0].loglog(f, model, color="black", ls="--", label="model")

axes[1].semilogx(f, np.zeros_like(f), color="black", lw=2)
axes[1].semilogx(f, (model - model_simu) / model_simu, color="black", ls="--")

axes[0].legend()
axes[1].set_xlabel("Frequency [Hz]")
axes[1].set_ylabel(r"ISI carrier fluctuations ASD [Hz/$\sqrt{\mathrm{Hz}}$]")
axes[1].set_ylabel("Relative error")
axes[1].set_ylim(-2, 4)

plt.subplots_adjust(hspace=0)


# %% [markdown] papermill={"duration": 0.051124, "end_time": "2026-04-10T15:57:11.642091", "exception": false, "start_time": "2026-04-10T15:57:11.590967", "status": "completed"}
# In the TMI carrier beatnotes, we should see the test-mass noise.
#
# In terms of acceleration,
# $$S_\delta^{\text{m}\,\text{s}^{-2}}(f) = A^2 [ 1 + \Big(\frac{f_\text{knee}}{f}\Big)^2 ].$$
#
# Multiplying by $1 / (2 \pi f)^2$ yields the noise as a velocity, and then dividing by $c$ yields the noise as a Doppler shift. One needs to scale this by $2 \nu_0$ to get the equivalent frequency shift (the factor 2 is for the bouncing on the test mass),
# $$S_\delta^\text{Hz}(f) = \Big(\frac{2 \nu_0}{2 \pi c f}\Big)^2 S_\delta^{\text{m}\,\text{s}^{-2}}(f).$$

# %% papermill={"duration": 0.053939, "end_time": "2026-04-10T15:57:11.751191", "exception": false, "start_time": "2026-04-10T15:57:11.697252", "status": "completed"}
def testmass_in_tmi_carrier(freq, instru, mosa="12", filter_approx=True):
    """Model for TM noise PSD in TMI carrier beatnote fluctuations.

    Args:
        freq (float): frequencies [Hz]
        instru (Instrument): LISA instrument object
        filter_approx (bool): if True, use the full filter transfer function
    """
    if mosa is None:
        asd = testmass_baseline
        fknee = instru.testmass_fknees["12"]
    else:
        asd = instru.testmass_asds[mosa]
        fknee = instru.testmass_fknees[mosa]
    fmin = 1.0 / instru.duration
    if not filter_approx:
        psd_acc = asd**2 * (1 + (fknee / freq) ** 2)
        psd_hertz = (2 * instru.central_freq / (2 * np.pi * c * freq)) ** 2 * psd_acc
    else:
        psd_highfreq = (
            (2 * asd * instru.central_freq / (2 * np.pi * c)) ** 2
            * np.abs(
                (2 * np.pi * fmin)
                / (
                    1
                    - np.exp(-2 * np.pi * fmin / instru.fs)
                    * np.exp(-2j * np.pi * freq / instru.fs)
                )
            )
            ** 2
            * 1
            / (instru.fs * fmin) ** 2
        )
        psd_lowfreq = (
            (2 * asd * instru.central_freq * fknee / (2 * np.pi * c)) ** 2
            * np.abs(
                (2 * np.pi * fmin)
                / (
                    1
                    - np.exp(-2 * np.pi * fmin / instru.fs)
                    * np.exp(-2j * np.pi * freq / instru.fs)
                )
            )
            ** 2
            * 1
            / (instru.fs * fmin) ** 2
            * np.abs(1 / (1 - np.exp(-2j * np.pi * freq / instru.fs))) ** 2
            * (2 * np.pi / instru.fs) ** 2
        )
        psd_hertz = psd_lowfreq + psd_highfreq
    return np.sqrt(psd_hertz)


# %% papermill={"duration": 12.683837, "end_time": "2026-04-10T15:57:24.484262", "exception": false, "start_time": "2026-04-10T15:57:11.800425", "status": "completed"}
_, axes = plt.subplots(
    2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)

f, _ = asd(tmi_carrier_fluctuations["12"], instru.fs, detrend="linear")
model_simu = testmass_in_tmi_carrier(f, instru, mosa="12", filter_approx=True)
model = testmass_in_tmi_carrier(f, instru, mosa="12", filter_approx=False)

for mosa in instru.MOSAS:
    f, asd_mosa = asd(tmi_carrier_fluctuations[mosa], instru.fs, detrend="linear")
    axes[0].loglog(f, asd_mosa, label=mosa)
    axes[1].semilogx(f, (asd_mosa - model_simu) / model_simu, label=mosa)

axes[0].loglog(f, model_simu, color="black", lw=2, label="model (simulated)")
axes[0].loglog(f, model, color="black", ls="--", label="model")

axes[1].semilogx(f, np.zeros_like(f), color="black", lw=2)
axes[1].semilogx(f, (model - model_simu) / model_simu, color="black", ls="--")

axes[0].legend()
axes[1].set_xlabel("Frequency [Hz]")
axes[1].set_ylabel(r"TMI carrier fluctuations ASD [Hz/$\sqrt{\mathrm{Hz}}$]")
axes[1].set_ylabel("Relative error")
axes[1].set_ylim(-2, 4)

plt.subplots_adjust(hspace=0)

# %% [markdown] papermill={"duration": 0.050095, "end_time": "2026-04-10T15:57:24.589938", "exception": false, "start_time": "2026-04-10T15:57:24.539843", "status": "completed"}
# ## TDI

# %% papermill={"duration": 0.052002, "end_time": "2026-04-10T15:57:24.688515", "exception": false, "start_time": "2026-04-10T15:57:24.636513", "status": "completed"}
# Same sampling as measurements
tdi_t0 = instru_t0
tdi_dt = instru_dt
tdi_size = instru_size
tdi_path = os.path.join(BASEDIR, "tdi.h5")

# %% papermill={"duration": 2151.502494, "end_time": "2026-04-10T16:33:16.243698", "exception": false, "start_time": "2026-04-10T15:57:24.741204", "status": "completed"}
# Compute time-delay interferometry data

data = Data.from_instrument(instru_path, signals="fluctuations")

# Write results

if os.path.exists(tdi_path):
    os.remove(tdi_path)

with File(tdi_path, "w") as hdf:

    hdf.attrs["t0"] = tdi_t0
    hdf.attrs["size"] = tdi_size
    hdf.attrs["dt"] = tdi_dt

    logging.info("Computing and writing time vector")
    hdf["t"] = instru.t
    logging.info("Computing and writing X2")
    hdf["X2"] = michelson.X2.build(**data.args)(data.measurements)
    logging.info("Computing and writing Y2")
    hdf["Y2"] = michelson.Y2.build(**data.args)(data.measurements)
    logging.info("Computing and writing Z2")
    hdf["Z2"] = michelson.Z2.build(**data.args)(data.measurements)

logging.info("Writing complete")


# %% [markdown] papermill={"duration": 0.0614, "end_time": "2026-04-10T16:33:16.362885", "exception": false, "start_time": "2026-04-10T16:33:16.301485", "status": "completed"}
# The transfer functions through TDI are taken from [https://arxiv.org/pdf/2211.02539.pdf].
#
# We use the common factor that depends on the average arm length $L$ (in the code, directly given in seconds by the PPRs),
# $$C_{XX}(f) = 16 \sin^2(2 \pi f L / c) \sin^2(4 \pi f L / c).$$
#
# The transfer function for the ISI OMS noise PSD is given by
# $$4 C_{XX}(f).$$
#
# The transfer function for the test-mass noise PSD reads
# $$C_{XX}(f) [3 + \cos(4 \pi f L / c)].$$
# Note that the factor 2 has been removed since it's already included in the noise PSD.

# %% papermill={"duration": 0.064309, "end_time": "2026-04-10T16:33:16.485615", "exception": false, "start_time": "2026-04-10T16:33:16.421306", "status": "completed"}
def tdi_common(freq, instru, mosa="12"):
    """TDI common factor.

    Args:
        freq (float): frequencies [Hz]
        instru (Instrument): LISA instrument object
    """
    armlength = np.mean(instru.pprs[mosa])
    return (
        16
        * np.sin(2 * np.pi * freq * armlength) ** 2
        * np.sin(4 * np.pi * freq * armlength) ** 2
    )


def tdi_tf_oms(freq, instru, mosa="12"):
    """TDI transfer function for ISI OMS noise.

    Args:
        freq (float): frequencies [Hz]
        instru (Instrument): LISA instrument object
    """
    psd = 4 * tdi_common(freq, instru, mosa)
    return np.sqrt(psd)


def tdi_tf_testmass(freq, instru, mosa="12"):
    """TDI transfer function for test mass noise.

    Args:
        freq (float): frequencies [Hz]
        instru (Instrument): LISA instrument object
    """
    armlength = np.mean(instru.pprs[mosa])
    psd = tdi_common(freq, instru, mosa) * (3 + np.cos(4 * np.pi * freq * armlength))
    return np.sqrt(psd)


# %% papermill={"duration": 7.895485, "end_time": "2026-04-10T16:33:24.441823", "exception": false, "start_time": "2026-04-10T16:33:16.546338", "status": "completed"}
# Read time-delay interferometry data

with File(tdi_path, "r") as hdf:
    assert tdi_t0 == hdf.attrs["t0"]
    assert tdi_dt == hdf.attrs["dt"]
    assert tdi_size == hdf.attrs["size"]

    X2 = hdf["X2"][:]
    Y2 = hdf["Y2"][:]
    Z2 = hdf["Z2"][:]

    plt.figure(figsize=(12, 4))
    plt.plot(instru.t[::1000], X2[::1000], label="X2")
    plt.plot(instru.t[::1000], Y2[::1000], label="Y2")
    plt.plot(instru.t[::1000], Z2[::1000], label="Z2")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("TDI [Hz]")

    f, asd_X2 = asd(X2[500:-500], instru.fs)
    f, asd_Y2 = asd(Y2[500:-500], instru.fs)
    f, asd_Z2 = asd(Z2[500:-500], instru.fs)

    testmass = tdi_tf_testmass(f, instru) * testmass_in_tmi_carrier(f, instru, mosa=None)
    oms = tdi_tf_oms(f, instru) * oms_in_isi_carrier(f, instru, mosa=None)
    model = np.sqrt(testmass**2 + oms**2)

    _, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    axes[0].loglog(f, asd_X2, label="X2")
    axes[0].loglog(f, asd_Y2, label="Y2")
    axes[0].loglog(f, asd_Z2, label="Z2")
    axes[0].loglog(f, testmass, "--", color="black", label="TM model (simulated)")
    axes[0].loglog(f, oms, "--", color="black", label="OMS model (simulated)")
    axes[0].loglog(f, model, color="black", lw=2, label="total model (simulated)")
    axes[0].set_ylim(1e-11, 1e-4)
    axes[0].set_ylabel(r"TDI ASD [Hz/$\sqrt{\mathrm{Hz}}$]")
    axes[0].legend()

    axes[1].semilogx(f, (asd_X2 - model) / model, label="X2")
    axes[1].semilogx(f, (asd_Y2 - model) / model, label="Y2")
    axes[1].semilogx(f, (asd_Z2 - model) / model, label="Z2")
    axes[1].semilogx(f, np.zeros_like(f), color="black", lw=2)
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Relative error")
    axes[1].legend()

    plt.subplots_adjust(hspace=0)

# %% papermill={"duration": 0.052859, "end_time": "2026-04-10T16:33:24.553171", "exception": false, "start_time": "2026-04-10T16:33:24.500312", "status": "completed"}
