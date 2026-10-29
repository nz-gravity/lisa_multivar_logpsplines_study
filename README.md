# LISA Log P-Spline Runs

Multivariate Bayesian PSD estimation (Log P-Spline) on LISA TDI X2/Y2/Z2 data
for two simulated noise realisations.

## Reproducible stationary baseline

[The fixed-geometry demo](stationary_fixed_geometry/README.md) generates its own
paired symmetric and unequal-MOSA noise records, fits their matrix spectra with
the current Gaussian-marginalized library, and factors out the known common TDI
response in study code. It uses a validated static orbit file and removes each
channel's global mean once. Raw data and posterior files are generated locally.
This is the stationary baseline for subsequent multivariate time-varying tests.

## Datasets

- `noise_4a`: symmetric equal-arm equal-link LDC model.  
  All six MOSAs have identical OMS and test-mass noise levels.
- `noise_5a`: asymmetric noise — six MOSA OMS and test-mass ASDs are drawn
  independently from a uniform ±50% range around the LDC baselines (`np.random.seed(1)`).
  The original data and generation notebook are not included in this checkout.
  The reproducible demo above generates a new, explicitly specified unequal-MOSA
  realization; it does not reproduce the private `noise_5a` dataset.


