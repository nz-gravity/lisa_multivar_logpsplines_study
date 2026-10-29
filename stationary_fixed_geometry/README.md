# Stationary LISA response-factoring demo

This directory keeps LISA-specific generation and response factoring outside
LogPSplinePSD. It supplies a reproducible stationary baseline before testing
multivariate time-varying inference. The historical scripts elsewhere in this
repository use older library APIs; this demo is independent of those scripts.

## Run

Use Python 3.12 and install the pinned library revision, which includes Gaussian
marginalization and global mean removal:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r stationary_fixed_geometry/requirements.txt
export JAX_ENABLE_X64=true
python stationary_fixed_geometry/generate_data.py
python stationary_fixed_geometry/validate_generator.py
python stationary_fixed_geometry/reference.py
python stationary_fixed_geometry/fit_response.py --case asymmetric --basis aet --theta-knots 50 20
python stationary_fixed_geometry/analyze.py --case asymmetric --basis aet
```

The generator defaults to ten days at a two-second cadence. It creates two HDF5
records and one orbit file in `data/`. Fits go to `output/`. These large files are
ignored by Git. The reference command checks its calculation against independent
finite Fourier response matrices. `validation/` contains compact results from a
run of this demo; see its README for the exact setup and diagnostic limits.

## Data and fixed geometry

The six directed links are ordered 12, 23, 31, 13, 32, 21. All delays are fixed at
8.3 s. The generator writes a `lisaorbits.StaticConstellation` orbit file, reloads
it, and verifies constant positions, zero velocity and acceleration, and constant
directed delays before using those delays in the transfer calculation. Orbit
roundoff below 1e-13 s is canonicalized to 8.3 s and recorded in the metadata.

Each record is a finite periodic, band-limited Gaussian realization of independent
MOSA test-mass and optical-metrology noises passed through the official PyTDI
second-generation Michelson delay polynomials. It is not an end-to-end
LISAInstrument simulation or a recreation of the private noise5a record. There
are no laser, clock, backlink or pointing noises, gravitational waves, or foreground.
The paired records share standard-normal source Fourier draws.

The symmetric ASDs are 3e-15 m/s²/√Hz for test-mass acceleration and
15e-12 m/√Hz for optical metrology. For the asymmetric record they are multiplied
by `[0.70, 1.30, 0.85, 1.40, 0.90, 1.15]` and
`[0.80, 1.20, 0.65, 1.40, 1.10, 0.90]`, respectively. Their source spectra include
the standard low-frequency knees and high-frequency test-mass rise, explicitly
defined in `generate_data.py`. Unequal levels produce nonzero AET cross spectra.

The convention is S_ij = E[d_i conjugate(d_j)], with
A=(2X−Y−Z)/√6, E=(Z−Y)/√2, T=(X+Y+Z)/√3. AET is synthesized directly from the
rotated transfer matrices to preserve quiet-T precision. DC and Nyquist are set
to zero and excluded from fits. The symmetric XYZ covariance and quiet-T power
are checked against independent equal-arm formulas.

## What factoring means

For these fixed equal arms, the common X2/Y2/Z2 transfer power is

```math
G(f)=16\sin^2(2\pi f\tau)\sin^2(4\pi f\tau),\qquad \tau=8.3\;\mathrm{s}.
```

The same scalar factor applies in AET. For a coarse bin b containing N_h fine
frequencies, we use their actual arithmetic mean of G:

```math
\bar G_b=\frac{1}{N_h}\sum_{f_k\in b}G(f_k),\quad
Y_b=U_bU_b^\dagger,\quad
\widetilde U_b=U_b/\sqrt{\bar G_b},\quad
\widetilde Y_b=Y_b/\bar G_b.
```

Fit the residual matrix R_b and restore every posterior draw using
S_b = Gbar_b R_b. For a p-channel coarse likelihood with effective count m,
the terms depending on the covariance are proportional to

```math
-m\log|S_b|-\operatorname{tr}(S_b^{-1}Y_b).
```

Substituting S_b=Gbar_b R_b gives

```math
-m\log|R_b|-\operatorname{tr}(R_b^{-1}\widetilde Y_b)
-mp\log\bar G_b.
```

The last term is known, so it does not affect inference for R_b. Package duration
and likelihood scaling factors do not change this argument. No true spectrum or
noise ASD is used to construct G or choose the logarithmic knots.

**The spline prior is now on R, so this changes the model/prior on S.** It is not
the same physical-spectrum spline prior evaluated more efficiently. Also,
mean(S)/mean(G) generally differs from mean(S/G). The fitted target is the mean
tapered, pooled spectrum divided by Gbar. Tapering and pooling still mix nearby
covariances and introduce dependence; factoring does not make the approximate
Wishart likelihood exact or remove the additional low-frequency suppression of T.

## Preprocessing and outputs

The library removes the global mean of each channel once, then uses eight blocks
with a Tukey(0.1) taper. The study fits 0.1–100 mHz, pools to at most 256 bins,
and excludes every bin whose full fine-frequency support intersects a ±0.1 mHz
neighborhood of a common transfer null. For ten days this gives 42 fine frequencies
per bin and 251 retained bins. Even-sized bins use the library's lower-middle
frequency, not their arithmetic midpoint. No eigenvalue floor is requested.

The demonstration uses exact Gaussian marginalization of cross-spectrum
coefficients and NUTS. Logarithmic knots are used for the residual spectrum;
the saved higher-resolution AET run uses 20 delta, 50 real-theta, and 20
imaginary-theta knots. Conditional Gaussian coefficient draws retain cross-spectrum
uncertainty. The response transformation is independent of sampler choice; this
saved demo validates the NUTS path.

`residual_inference_data.nc` consistently describes the residual model, including
coefficients, likelihood and spectra. `physical_spectrum.npz` separately stores
restored physical draws and summaries, the known factor, bin supports and empirical
spectra. Channel standard deviations have already been restored by the library;
only Gbar is multiplied back. Coherence is unchanged by a common positive scalar.
Provenance records the input hash, configuration and package source hashes.

`analyze.py` compares the physical posterior against the exact mean of the finite
periodic generator after the same global centering, block taper and pooling.
It keeps the underlying bin-averaged spectrum and midpoint spectrum as separate
references. Across-frequency interval containment is descriptive, not repeated-noise
coverage calibration. Marginal component medians need not themselves form a
positive-definite matrix; positive definiteness is checked on complete posterior draws.

## Next: time-varying tests

[LogPSplinePSD PR #53](https://github.com/nz-gravity/LogPSplinePSD/pull/53) is an
experimental draft based on an older library version. Its reported real-series
tests retained divergences, and it does not yet supply multivariate time-varying
VI. It should first incorporate the merged marginalization and centering changes,
then pass a stationary-limit comparison using this generator before introducing
time-dependent MOSA amplitudes with the orbit still fixed. Repeated realizations,
local-FFT controls and sampler diagnostics are needed before making recovery or
coverage claims. This directory does not implement or validate that extension.

## Model sources

- [PyTDI documentation](https://lisa-simulation.pages.in2p3.fr/pytdi/)
- [LISA Instrument noise models](https://lisa-simulation.pages.in2p3.fr/instrument/noises.html)
- [LDC noise formulas](https://lisa.pages.in2p3.fr/LDC/_modules/ldc/lisa/noise/noise.html)
- [Library Gaussian marginalization derivation](https://nz-gravity.github.io/LogPSplinePSD/examples/multivariate-example.html#gaussian-marginalization-integrate-the-cross-spectrum-coefficients)
