# LISA Log P-Spline Runs

Multivariate Bayesian PSD estimation (Log P-Spline) on LISA TDI X2/Y2/Z2 data
for two simulated noise realisations.

## Datasets

- `noise_4a`: symmetric equal-arm equal-link LDC model.  
  All six MOSAs have identical OMS and test-mass noise levels.
- `noise_5a`: asymmetric noise — six MOSA OMS and test-mass ASDs are drawn
  independently from a uniform ±50% range around the LDC baselines (`np.random.seed(1)`).
  See `noise_5a/noise-5.py` for the generation notebook.


