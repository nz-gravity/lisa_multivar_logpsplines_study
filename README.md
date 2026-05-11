# LISA Log P-Spline Runs

Multivariate Bayesian PSD estimation (Log P-Spline) on LISA TDI X2/Y2/Z2 data
for two simulated noise realisations.

## Datasets

### `noise_4a`
- **Noise model**: symmetric equal-arm equal-link LDC model.  
  All six MOSAs have identical OMS and test-mass noise levels.
- **True PSD reference**: analytic 3×3 covariance matrix computed from
  `lisa_link_noises_ldc()` + `tdi2_psd_and_csd()` (see `noise_4a/load_and_plot_data.py`).
  This is a smooth closed-form model and gives reliable RIAE/coverage diagnostics.
- **Expected diagnostics (1m)**: RIAE ≈ 0.12, Coverage ≈ 84%.

### `noise_5a`
- **Noise model**: asymmetric noise — six MOSA OMS and test-mass ASDs are drawn
  independently from a uniform ±50% range around the LDC baselines (`np.random.seed(1)`).
  See `noise_5a/noise-5.py` for the generation notebook.
- **No analytic 3×3 reference**: the generation notebook only provides a diagonal
  approximation using baseline (not per-MOSA) levels. The full cross-spectral matrix
  (SXY, SYZ, SZX) is not computed analytically. **Do not use the raw periodogram as
  truth** — it has ±100% chi-squared noise per bin.
- **Reference used**: heavily-averaged Welch estimate (`lisa.matrix`, DEFAULT_NPERSEG=65536,
  ~1000 averages from the full edge-trimmed dataset), stored in `LISAData.matrix`.
  After the fix, this is what is shown in plots and used for RIAE/coverage diagnostics.
- **Expected diagnostics (1m)**: RIAE ≈ 0.15–0.25 (comparing smooth posterior to Welch
  reference), Coverage ≈ 75–90%.

---

## Bugs Fixed (April 2026)

### 1. noise_5a: wrong reference PSD
`build_plot_inputs` and `ModelConfig.true_psd` previously used
`periodogram_covariance_from_timeseries()` (single-realization, ±100% noise per bin)
as the "true" reference. This caused RIAE = 0.73 and Coverage = 17% even when MCMC
converged perfectly. **Fix**: use `lisa.matrix` (smooth Welch ~1000 averages) as
reference throughout.

### 2. noise_5a: `choose_coarse_grain_nc` bug
The loop rejected `nc=1` as a valid divisor and fell back to returning `n_retained`
(no coarse-graining at all). Doesn't trigger for 1m/3m since a divisor >1 exists, but
would corrupt 6m/1y runs. **Fix**: now matches the correct noise_4a version.

### 3. Both: VI warm-start steps increased 20k → 50k
VI guide (`lowrank:max(20, 2*Nb)`) consistently produced k-hat > 2 with 20k steps.
NUTS corrects this during warmup (ESS > 1000, R-hat ≈ 1.0 in all runs), but more
VI steps reduce the recovery burden.

---

## Running Locally

From `/Users/avi/Documents/projects/lisa_logpspline`:

```bash
# 1m and 3m for both datasets (sequential, ~2–5 h total on a laptop)
.venv/bin/python run_duration_sweeps.py --datasets noise_4a noise_5a --durations 1m 3m
```

Individual runs:

```bash
.venv/bin/python run_mcmc.py --dataset noise_4a --model multivar --duration-days 30   # 1m
.venv/bin/python run_mcmc.py --dataset noise_4a --model multivar --duration-days 90   # 3m
.venv/bin/python run_mcmc.py --dataset noise_5a --model multivar --duration-days 30
.venv/bin/python run_mcmc.py --dataset noise_5a --model multivar --duration-days 90
```

Rough runtime per run on CPU (no GPU):

| Duration | Nb | Estimated time |
|----------|----|----------------|
| 1m       |  4 | ~1–2 h         |
| 3m       | 12 | ~2–4 h         |
| 6m       | 25 | ~4–8 h         |
| 1y       | 52 | ~8–16 h        |

> Runs > 30 min should use OzStar (see below).

---

## Running on OzStar

Sync the repo to OzStar, edit `REPO` in `run_1m_3m.slurm`, then:

```bash
sbatch run_1m_3m.slurm          # array of 4 jobs: noise_4a+noise_5a × 1m+3m
```

OzStar setup:
- Account: `oz303`
- Python: `/fred/oz303/avajpeyi/codes/LogPSplinePSD/.venv/bin/python`
- Code path: set `REPO` in `run_1m_3m.slurm` to wherever you clone this repo

---

## Outputs

Each run writes to `<dataset>/mcmc_output/<duration>/`:

| File | Description |
|------|-------------|
| `idata.nc` | ArviZ InferenceData (NUTS posterior samples) |
| `welch_psd_triangle.png` | X/Y/Z PSDs (diagonal) + coherences (lower triangle) with posterior CIs |
| `welch_psd_csd_matrix.png` | Full 3×3 PSD/CSD matrix (real/imag off-diagonal) |
| `summary_statistics.csv` | ESS, R-hat, posterior mean/SD per parameter |
| `diagnostics/` | NUTS diagnostics, RIAE, coverage, rank plots |
| `vi_diagnostics_summary.txt` | VI quality (PSIS k-hat, weight dispersion) |

### What to check

- **RIAE (matrix)** < 0.2: posterior median close to reference
- **Coverage** ≈ 90%: well-calibrated 90% CIs
- **R-hat** ≤ 1.01 and **ESS bulk** > 400: chains converged
- **Divergences** < 0.1%: no geometry issues
- **VI k-hat** > 0.7 is expected; NUTS corrects during warmup

---

## Lab Notebook — Model Comparison Investigation

### Scientific Question

Does the XYZ → AET transformation fully diagonalise the LISA noise covariance, making
the off-diagonal elements of S_AET negligible? Equivalently: is a multivariate model in
XYZ preferred over a simpler diagonal model in AET?

For **noise_4a** (equal-arm, symmetric noise) the answer should be "no preference" —
S_AET is analytically diagonal. For **noise_5a** (asymmetric noise, 6 MOSAs drawn
independently from ±50% of baseline) the AET diagonalisation is approximate, so
off-diagonals may genuinely matter.

---

### Attempt 1 — Univar(AET) vs Multivar(XYZ)

**What was run**: three independent univariate Whittle fits on A, E, T channels
(`--model univar`) versus the full multivariate Wishart fit in XYZ (`--model multivar`),
for both datasets at 30 / 180 / 365 days.

```bash
.venv/bin/python run_mcmc.py --dataset noise_4a --model univar  --compute-lnz --duration-days 30
.venv/bin/python run_mcmc.py --dataset noise_4a --model multivar --compute-lnz --duration-days 30
# … repeated for noise_5a, 180d, 365d
```

**Initial logBF table (lnZ_XYZ − lnZ_univar_AET, η = 0.03)**:

| Duration | noise_4a | noise_5a |
|----------|----------|----------|
| 30 d     | ~2244    | ~4118    |
| 180 d    | ~5437    | ~8130    |
| 365 d    | ~11 100  | ~14 400  |

**Problem — the comparison is not fair.**
The two models use different likelihoods with incompatible normalisation:

- `univar` uses a single-block Whittle likelihood: det term ∝ `−0.5 × Nh_cg` per coarse bin.
- `multivar` uses a Wishart likelihood summed over Nb blocks: det term ∝ `−Nb × Nh_cg` per bin.

The ratio is `2 × Nb` (≈ ×8 at 30 d/Nb=4, ×50 at 180 d, ×104 at 365 d). This inflates
|lnZ_multivar| regardless of fit quality. **logBF between these two models is not a valid
Bayes factor.** Both lnZ values are also on the η=0.03 tempered scale (lnZ_η), not lnZ_1,
but that affects both models equally and does not explain the scale mismatch.

A secondary extraction bug was also found: the raw diagnostic txt files
(`diagnostics/morphz/*/logz_morph_z_indep_silverman.txt`) store the pre-whitening-correction
lnZ, while `idata.attrs["lnz"]` stores the final corrected value. Mixing these inflated
the apparent logBF by hundreds of nats. Always use `idata.attrs["lnz"]` for multivar.

**Also checked — M_AET Jacobian**: the transformation matrix M_AET is orthonormal
(det = 1.000000000000000 exactly), so rotating data from XYZ to AET introduces zero
Jacobian correction in log-likelihood space.

---

### Attempt 2 — Multivar(AET, k=2) vs Multivar(XYZ)

**Motivation**: put both models on the same likelihood scale by using the multivariate
Wishart infrastructure for the "null" model too. In AET basis with very few off-diagonal
theta knots (k=2), the spline prior keeps S_AET near-diagonal — this tests the hypothesis
that off-diagonals are negligible, without changing the likelihood form or the Nb scaling.

**Key properties**:
- Same Wishart likelihood, same Nb, same Nh per coarse bin → lnZ values directly comparable.
- M_AET orthonormal → no Jacobian correction needed.
- `multivar_aet` uses `xyz_to_aet_timeseries()` and `xyz_to_aet_matrix()` from
  `run_univar_aet.py` to rotate data and the reference PSD before fitting.

```bash
.venv/bin/python run_mcmc.py --dataset noise_4a --model multivar_aet --compute-lnz --duration-days 30
# N_SPLINE_KNOTS_THETA_AET = 2 (in run_mcmc.py at the time)
```

**logBF (lnZ_XYZ − lnZ_AET, k=2, η = 0.03)**: large (~2000–3500), strongly preferring XYZ.

---

### Attempt 3 — Multivar(AET, k=10) vs Multivar(XYZ)

**Motivation**: with only 2 theta knots the AET model may be over-constrained — the prior
is too tight and not representing a fair "diagonal" model. Increased to 10 knots to give
the AET model more freedom before concluding that off-diagonals matter.

```bash
# N_SPLINE_KNOTS_THETA_AET = 10 (updated in run_mcmc.py)
.venv/bin/python run_mcmc.py --dataset noise_4a --model multivar_aet --compute-lnz --duration-days 30
.venv/bin/python run_mcmc.py --dataset noise_4a --model multivar_aet --compute-lnz --duration-days 180
.venv/bin/python run_mcmc.py --dataset noise_4a --model multivar_aet --compute-lnz --duration-days 365
# repeated for noise_5a
```

**logBF = lnZ(multivar_XYZ) − lnZ(multivar_AET, k=10), η = 0.03**

| Duration | noise_4a        | noise_5a        |
|----------|-----------------|-----------------|
| 30 d     |  997.80 ± 0.11  | 2173.15 ± 0.17  |
| 180 d    |  862.34 ± 0.21  | 1731.55 ± 0.17  |
| 365 d    | 1624.94 ± 0.21  | 2291.51 ± 0.15  |

logBF decreased versus k=2, confirming the 2-knot model was too constrained.
But logBF remains very large (>> 1) and does **not** shrink to zero for noise_4a
as duration increases — contrary to what would be expected if AET truly diagonalises
the noise.

---

### Diagnostics (for the runs above)

**Reference PSDs used for RIAE/L2/coverage**:
- `noise_4a`: analytic covariance (closed-form) → diagnostics are trustworthy.
- `noise_5a`: Welch estimate (~1000 block averages, `DEFAULT_NPERSEG=65536`) → **not the
  true PSD**. RIAE/L2/coverage for noise_5a reflect agreement with a noisy reference and
  should not be interpreted as absolute fit quality.

**NUTS diagnostics summary (η = 0.03, all durations)**:

| Model              | noise_4a RIAE | noise_4a coverage (30d / 365d) |
|--------------------|---------------|-------------------------------|
| multivar_XYZ       | ~0.005        | 0.51 / 0.31                   |
| multivar_AET k=10  | ~0.004        | 0.88 / 0.77                   |
| univar A/E/T       | 0.07–0.12     | 0.45–0.63 / 0.22–0.30         |

For noise_4a: multivar_AET has excellent RIAE (comparable to XYZ) but *overcovering*
posteriors (88% at 30d vs 51% for XYZ) — the model's posterior is too wide, suggesting
the restricted off-diagonal prior is compensating incorrectly. All NUTS runs had R̂ = 1.0,
0 divergences, and ESS > 3000.

For noise_5a: RIAE and coverage are not reliable absolute metrics (Welch reference only).
The AET model and XYZ model have nearly identical RIAE values against the Welch reference,
so diagnostics alone cannot explain the large lnZ gap.

---

### Summary and Open Questions

**What we know**:
1. Univar(AET) vs Multivar(XYZ) is not a valid comparison — incompatible likelihood
   normalisation (factor 2×Nb).
2. The fair comparison is Multivar(AET, k=10) vs Multivar(XYZ): same Wishart likelihood,
   same Nb, lnZ directly comparable.
3. XYZ is strongly preferred in both datasets across all durations (logBF ≫ 1).
4. For noise_4a (analytically diagonal AET noise), XYZ still wins — logBF does not shrink
   to zero with increasing duration, which is unexpected.
5. noise_5a logBF is consistently ~2–2.3× larger than noise_4a, suggesting genuine
   additional off-diagonal structure in the asymmetric noise case.

**Open questions**:
- Is k=10 still too few AET theta knots? Logically, if we pushed k → (same as XYZ), logBF
  should converge to 0 for noise_4a. Testing k=20–50 would clarify whether the large
  logBF is a model-capacity artefact or reflects real finite-sample off-diagonal power.
- Why is logBF non-monotone with duration for noise_4a (dips at 180d)? Could be
  Nb-dependent prior volume effects in the bridge sampling estimator.
- Is there a better "null" model for the diagonal hypothesis — e.g., an exact hard-zero
  off-diagonal model rather than a spline prior that only softly penalises off-diagonals?

**Next step to pursue**: run `multivar_aet` with a larger theta knot count (e.g., k=30 or
matching the XYZ knot count) to test whether logBF(noise_4a) asymptotes toward zero.
