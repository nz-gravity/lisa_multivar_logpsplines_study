#!/usr/bin/env python3
"""Block K-fold ELPD comparison between H1 (multivar_aet) and H3 (multivar_xyz).

Each fold holds out a contiguous block of coarse frequency bins.  The model is
re-fit on the remaining K-1 blocks, and the Wishart log-likelihood is evaluated
at the held-out frequencies using the posterior spline weights.  This gives a
valid predictive score even for global spline models where PSIS-LOO breaks down.

Usage
-----
  python kfold_compare.py --dataset noise4a --duration-days 14 --n-folds 5
  python kfold_compare.py --dataset noise4a --duration-days 14 --n-folds 3 --eta 0.5

  ============================================================
K-fold ELPD: model=H1  dataset=noise4a  dur=14.0d  K=3  eta=0.5
============================================================
Loaded 15778801 samples from data/noise4a.h5 (fs=0.500000 Hz); using 15777801 after edge trim. reference='analytic'.
Traceback (most recent call last):
  File "/Users/avi/Documents/projects/lisa_logpspline/kfold_compare.py", line 472, in <module>
    main()
  File "/Users/avi/Documents/projects/lisa_logpspline/kfold_compare.py", line 433, in main
    results[model] = run_kfold(
                     ^^^^^^^^^^
  File "/Users/avi/Documents/projects/lisa_logpspline/kfold_compare.py", line 301, in run_kfold
    class _Args:
  File "/Users/avi/Documents/projects/lisa_logpspline/kfold_compare.py", line 302, in _Args
    duration_days = duration_days
                    ^^^^^^^^^^^^^
NameError: name 'duration_days' is not defined. Did you mean: 'duration_slug'?
➜  lisa_logpspline git:(main) ✗ python kfold_compare.py --dataset noise4a --duration-days 14 --n-folds 3 --eta 0.5

============================================================
K-fold ELPD: model=H1  dataset=noise4a  dur=14.0d  K=3  eta=0.5
============================================================
Loaded 15778801 samples from data/noise4a.h5 (fs=0.500000 Hz); using 15777801 after edge trim. reference='analytic'.
|00:02| LogPSpline | INFO | Wishart averaging (blocks=2): n=604800 -> Lb=302400, N=60420, p=3
|00:02| LogPSpline | INFO | Standardized data: scale ~2.73e-11
|00:02| LogPSpline | INFO | Inferred sampler type: multivar_blocked_nuts
|00:02| LogPSpline | INFO | Nl trimmed to 60416 (dropped 4 trailing bins) to make Nc=1024 an exact divisor.
|00:02| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=1024, Nh=59, Nl=60416 (kept 1.7%, decimated 98.3%).
|00:02| LogPSpline | INFO | Null-band excision: removing 6 bins across 3 band(s). 1018 bins retained.

[H1] n_freq=1018, Nb=2, eta=0.5, K=3
  Fold 1/3: train=679 bins, held=339 bins
|00:03| LogPSpline | INFO | Saved preprocessing eigenvalue plot to out/kfold/noise4a/14d/eta0.5/K3/H1/fold_00/diagnostics/preprocessing_eigenvalue_ratios.png
|00:03| LogPSpline | INFO | Adjusting explicit coarse VI bins: N_full=679 not divisible by Nc=256; using Nc=97.
|00:03| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=97, Nh=7, Nl=679 (kept 14.3%, decimated 85.7%).
|00:19| LogPSpline | INFO | Spline model: MultivariateLogPSplines(channels=3, knots=mixed[2-50], degree=2, penaltyOrder=2, N=679, basis_shapes=[(679, 51), (679, 51), (679, 51), (679, 3), (679, 3), (679, 3), (679, 3), (679, 3), (679, 3)])
Running chain 1: 100%|███████████████████| 3500/3500 [00:01<00:00, 2924.93it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:01<00:00, 2904.06it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:01<00:00, 2885.28it/s]
Running chain 0: 100%|███████████████████| 3500/3500 [00:01<00:00, 2488.61it/s]
Running chain 0: 100%|███████████████████| 3500/3500 [00:02<00:00, 1300.84it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:02<00:00, 1249.96it/s]
Running chain 1: 100%|███████████████████| 3500/3500 [00:02<00:00, 1236.95it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:02<00:00, 1191.06it/s]
Running chain 0: 100%|████████████████████| 3500/3500 [00:05<00:00, 641.82it/s]
Running chain 3: 100%|████████████████████| 3500/3500 [00:05<00:00, 626.64it/s]
Running chain 2: 100%|████████████████████| 3500/3500 [00:05<00:00, 622.72it/s]
Running chain 1: 100%|████████████████████| 3500/3500 [00:05<00:00, 592.37it/s]
    elpd = -21080.43%|██████████████████ | 3325/3500 [00:05<00:00, 1782.32it/s]
  Fold 2/3: train=679 bins, held=339 bins
|00:32| LogPSpline | INFO | Saved preprocessing eigenvalue plot to out/kfold/noise4a/14d/eta0.5/K3/H1/fold_01/diagnostics/preprocessing_eigenvalue_ratios.png
|00:32| LogPSpline | INFO | Adjusting explicit coarse VI bins: N_full=679 not divisible by Nc=256; using Nc=97.
|00:32| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=97, Nh=7, Nl=679 (kept 14.3%, decimated 85.7%).
|00:45| LogPSpline | INFO | Spline model: MultivariateLogPSplines(channels=3, knots=mixed[2-50], degree=2, penaltyOrder=2, N=679, basis_shapes=[(679, 51), (679, 51), (679, 51), (679, 3), (679, 3), (679, 3), (679, 3), (679, 3), (679, 3)])
Running chain 0: 100%|███████████████████| 3500/3500 [00:01<00:00, 2955.52it/s]
Running chain 1: 100%|███████████████████| 3500/3500 [00:01<00:00, 2901.18it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:01<00:00, 2763.97it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:01<00:00, 2485.18it/s]
Running chain 1: 100%|███████████████████| 3500/3500 [00:02<00:00, 1208.02it/s]
Running chain 0: 100%|███████████████████| 3500/3500 [00:02<00:00, 1207.02it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:02<00:00, 1195.78it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:03<00:00, 1151.70it/s]
Running chain 1: 100%|█████████████████████| 3500/3500 [00:59<00:00, 59.07it/s]
Running chain 2: 100%|█████████████████████| 3500/3500 [00:59<00:00, 58.46it/s]
Running chain 3: 100%|█████████████████████| 3500/3500 [00:59<00:00, 58.45it/s]
Running chain 0: 100%|█████████████████████| 3500/3500 [01:00<00:00, 58.07it/s]
    elpd = -7226380.76
  Fold 3/3: train=678 bins, held=340 bins█| 3500/3500 [00:59<00:00, 209.16it/s]
|01:54| LogPSpline | INFO | Saved preprocessing eigenvalue plot to out/kfold/noise4a/14d/eta0.5/K3/H1/fold_02/diagnostics/preprocessing_eigenvalue_ratios.png
|01:54| LogPSpline | INFO | Adjusting explicit coarse VI bins: N_full=678 not divisible by Nc=256; using Nc=226.
|01:54| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=226, Nh=3, Nl=678 (kept 33.3%, decimated 66.7%).
|02:08| LogPSpline | INFO | Spline model: MultivariateLogPSplines(channels=3, knots=mixed[2-50], degree=2, penaltyOrder=2, N=678, basis_shapes=[(678, 51), (678, 51), (678, 51), (678, 3), (678, 3), (678, 3), (678, 3), (678, 3), (678, 3)])
Running chain 1: 100%|███████████████████| 3500/3500 [00:01<00:00, 2969.45it/s]
Running chain 0: 100%|███████████████████| 3500/3500 [00:01<00:00, 2123.39it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:01<00:00, 2112.89it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:01<00:00, 1994.25it/s]
Running chain 1: 100%|███████████████████| 3500/3500 [00:02<00:00, 1209.88it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:02<00:00, 1197.77it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:02<00:00, 1166.90it/s]
Running chain 0: 100%|███████████████████| 3500/3500 [00:03<00:00, 1161.58it/s]
Running chain 3: 100%|█████████████████████| 3500/3500 [00:53<00:00, 65.58it/s]
Running chain 0: 100%|█████████████████████| 3500/3500 [00:53<00:00, 65.27it/s]
Running chain 1: 100%|█████████████████████| 3500/3500 [00:53<00:00, 65.12it/s]
Running chain 2: 100%|█████████████████████| 3500/3500 [00:54<00:00, 64.35it/s]
    elpd = -1466643.15███████████████████ | 3325/3500 [00:53<00:00, 235.68it/s]
Running chain 2: 100%|████████████████████| 3500/3500 [00:54<00:00, 244.07it/s]
============================================================
K-fold ELPD: model=H3  dataset=noise4a  dur=14.0d  K=3  eta=0.5
============================================================
Loaded 15778801 samples from data/noise4a.h5 (fs=0.500000 Hz); using 15777801 after edge trim. reference='analytic'.
|03:12| LogPSpline | INFO | Wishart averaging (blocks=2): n=604800 -> Lb=302400, N=60420, p=3
|03:12| LogPSpline | INFO | Standardized data: scale ~2.73e-11
|03:12| LogPSpline | INFO | Inferred sampler type: multivar_blocked_nuts
|03:12| LogPSpline | INFO | Nl trimmed to 60416 (dropped 4 trailing bins) to make Nc=1024 an exact divisor.
|03:12| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=1024, Nh=59, Nl=60416 (kept 1.7%, decimated 98.3%).
|03:12| LogPSpline | INFO | Null-band excision: removing 6 bins across 3 band(s). 1018 bins retained.

[H3] n_freq=1018, Nb=2, eta=0.5, K=3
  Fold 1/3: train=679 bins, held=339 bins
|03:13| LogPSpline | INFO | Saved preprocessing eigenvalue plot to out/kfold/noise4a/14d/eta0.5/K3/H3/fold_00/diagnostics/preprocessing_eigenvalue_ratios.png
|03:13| LogPSpline | INFO | Adjusting explicit coarse VI bins: N_full=679 not divisible by Nc=256; using Nc=97.
|03:13| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=97, Nh=7, Nl=679 (kept 14.3%, decimated 85.7%).
|03:30| LogPSpline | INFO | Spline model: MultivariateLogPSplines(channels=3, knots=mixed[2-50], degree=2, penaltyOrder=2, N=679, basis_shapes=[(679, 51), (679, 51), (679, 51), (679, 51), (679, 51), (679, 51), (679, 3), (679, 3), (679, 3)])
Running chain 1: 100%|███████████████████| 3500/3500 [00:01<00:00, 3097.09it/s]
Running chain 0: 100%|███████████████████| 3500/3500 [00:01<00:00, 2996.48it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:01<00:00, 2884.71it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:01<00:00, 2514.98it/s]
Running chain 2: 100%|████████████████████| 3500/3500 [00:11<00:00, 317.71it/s]
Running chain 1: 100%|████████████████████| 3500/3500 [00:11<00:00, 311.76it/s]
Running chain 3: 100%|████████████████████| 3500/3500 [00:11<00:00, 307.56it/s]
Running chain 0: 100%|████████████████████| 3500/3500 [00:12<00:00, 280.87it/s]
Running chain 3: 100%|█████████████████████| 3500/3500 [00:48<00:00, 71.95it/s]
Running chain 1: 100%|█████████████████████| 3500/3500 [00:48<00:00, 71.91it/s]
Running chain 2: 100%|█████████████████████| 3500/3500 [00:50<00:00, 69.96it/s]
Running chain 0: 100%|█████████████████████| 3500/3500 [00:50<00:00, 69.02it/s]
    elpd = -21083.13%|████████████████████| 3500/3500 [00:50<00:00, 347.20it/s]
  Fold 2/3: train=679 bins, held=339 bins
|04:37| LogPSpline | INFO | Saved preprocessing eigenvalue plot to out/kfold/noise4a/14d/eta0.5/K3/H3/fold_01/diagnostics/preprocessing_eigenvalue_ratios.png
|04:37| LogPSpline | INFO | Adjusting explicit coarse VI bins: N_full=679 not divisible by Nc=256; using Nc=97.
|04:37| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=97, Nh=7, Nl=679 (kept 14.3%, decimated 85.7%).
|04:55| LogPSpline | INFO | Spline model: MultivariateLogPSplines(channels=3, knots=mixed[2-50], degree=2, penaltyOrder=2, N=679, basis_shapes=[(679, 51), (679, 51), (679, 51), (679, 51), (679, 51), (679, 51), (679, 3), (679, 3), (679, 3)])
Running chain 3: 100%|███████████████████| 3500/3500 [00:01<00:00, 3001.36it/s]
Running chain 1: 100%|███████████████████| 3500/3500 [00:01<00:00, 2965.40it/s]
Running chain 0: 100%|███████████████████| 3500/3500 [00:01<00:00, 2323.30it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:01<00:00, 2322.14it/s]
Running chain 2: 100%|████████████████████| 3500/3500 [00:13<00:00, 255.43it/s]
Running chain 0: 100%|████████████████████| 3500/3500 [00:13<00:00, 252.10it/s]
Running chain 1: 100%|████████████████████| 3500/3500 [00:14<00:00, 235.69it/s]
Running chain 3: 100%|████████████████████| 3500/3500 [00:14<00:00, 234.22it/s]
Running chain 2: 100%|█████████████████████| 3500/3500 [00:56<00:00, 61.93it/s]
Running chain 3: 100%|█████████████████████| 3500/3500 [00:56<00:00, 61.52it/s]
Running chain 1: 100%|█████████████████████| 3500/3500 [00:58<00:00, 59.68it/s]
Running chain 0: 100%|█████████████████████| 3500/3500 [01:14<00:00, 46.86it/s]
    elpd = -5765610.83████████████████████| 3500/3500 [00:56<00:00, 339.52it/s]
  Fold 3/3: train=678 bins, held=340 bins
|06:30| LogPSpline | INFO | Saved preprocessing eigenvalue plot to out/kfold/noise4a/14d/eta0.5/K3/H3/fold_02/diagnostics/preprocessing_eigenvalue_ratios.png
|06:30| LogPSpline | INFO | Adjusting explicit coarse VI bins: N_full=678 not divisible by Nc=256; using Nc=226.
|06:30| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=226, Nh=3, Nl=678 (kept 33.3%, decimated 66.7%).
|06:46| LogPSpline | INFO | Spline model: MultivariateLogPSplines(channels=3, knots=mixed[2-50], degree=2, penaltyOrder=2, N=678, basis_shapes=[(678, 51), (678, 51), (678, 51), (678, 51), (678, 51), (678, 51), (678, 3), (678, 3), (678, 3)])
Running chain 0: 100%|███████████████████| 3500/3500 [00:01<00:00, 3178.99it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:01<00:00, 3121.18it/s]
Running chain 1: 100%|███████████████████| 3500/3500 [00:01<00:00, 3088.20it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:01<00:00, 2502.97it/s]
Running chain 2: 100%|████████████████████| 3500/3500 [00:10<00:00, 322.66it/s]
Running chain 1: 100%|████████████████████| 3500/3500 [00:10<00:00, 321.69it/s]
Running chain 3: 100%|████████████████████| 3500/3500 [00:11<00:00, 315.45it/s]
Running chain 0: 100%|████████████████████| 3500/3500 [00:11<00:00, 312.66it/s]
Running chain 2: 100%|█████████████████████| 3500/3500 [00:55<00:00, 62.70it/s]
Running chain 0: 100%|█████████████████████| 3500/3500 [00:55<00:00, 62.65it/s]
Running chain 1: 100%|█████████████████████| 3500/3500 [00:56<00:00, 61.58it/s]
Running chain 3: 100%|█████████████████████| 3500/3500 [00:59<00:00, 59.19it/s]
    elpd = -1476215.46
Running chain 3:  85%|█████████████████   | 2975/3500 [00:56<00:02, 195.71it/s]
============================================================<00:00, 210.58it/s]
SUMMARY  dataset=noise4a  dur=14.0d  K=3  eta=0.5
============================================================
  H1: elpd = -8714104.34 ± 6602294.37
  H3: elpd = -7262909.41 ± 5172809.74

  Δelpd (H3 - H1) = 1451194.93 ± 1465580.87  (paired)
  → Not significant  (|Δ|/SE = 1.0)

Results saved to /Users/avi/Documents/projects/lisa_logpspline/out/kfold/noise4a/14d/eta0.5/K3
➜  lisa_logpspline git:(main) ✗ python kfold_compare.py --dataset noise5a --duration-days 14 --n-folds 3 --eta 0.5

============================================================
K-fold ELPD: model=H1  dataset=noise5a  dur=14.0d  K=3  eta=0.5
============================================================
Loaded 15778801 samples from data/noise5a.h5 (fs=0.500000 Hz); using 15777801 after edge trim. reference='segwo'.
|00:02| LogPSpline | INFO | Wishart averaging (blocks=2): n=604800 -> Lb=302400, N=60420, p=3
|00:02| LogPSpline | INFO | Standardized data: scale ~5.18e-11
|00:02| LogPSpline | INFO | Inferred sampler type: multivar_blocked_nuts
|00:02| LogPSpline | INFO | Nl trimmed to 60416 (dropped 4 trailing bins) to make Nc=1024 an exact divisor.
|00:02| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=1024, Nh=59, Nl=60416 (kept 1.7%, decimated 98.3%).
|00:02| LogPSpline | INFO | Null-band excision: removing 6 bins across 3 band(s). 1018 bins retained.

[H1] n_freq=1018, Nb=2, eta=0.5, K=3
  Fold 1/3: train=679 bins, held=339 bins
|00:03| LogPSpline | INFO | Saved preprocessing eigenvalue plot to out/kfold/noise5a/14d/eta0.5/K3/H1/fold_00/diagnostics/preprocessing_eigenvalue_ratios.png
|00:03| LogPSpline | INFO | Adjusting explicit coarse VI bins: N_full=679 not divisible by Nc=256; using Nc=97.
|00:03| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=97, Nh=7, Nl=679 (kept 14.3%, decimated 85.7%).
|00:18| LogPSpline | INFO | Spline model: MultivariateLogPSplines(channels=3, knots=mixed[2-50], degree=2, penaltyOrder=2, N=679, basis_shapes=[(679, 51), (679, 51), (679, 51), (679, 3), (679, 3), (679, 3), (679, 3), (679, 3), (679, 3)])
Running chain 1: 100%|███████████████████| 3500/3500 [00:01<00:00, 3295.20it/s]
Running chain 0: 100%|███████████████████| 3500/3500 [00:01<00:00, 3256.02it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:01<00:00, 3218.02it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:01<00:00, 3081.38it/s]
Running chain 1: 100%|███████████████████| 3500/3500 [00:02<00:00, 1331.17it/s]
Running chain 0: 100%|███████████████████| 3500/3500 [00:02<00:00, 1288.96it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:02<00:00, 1267.22it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:02<00:00, 1222.35it/s]
Running chain 1: 100%|████████████████████| 3500/3500 [00:05<00:00, 656.17it/s]
Running chain 3: 100%|████████████████████| 3500/3500 [00:05<00:00, 652.09it/s]
Running chain 2: 100%|████████████████████| 3500/3500 [00:05<00:00, 637.76it/s]
Running chain 0: 100%|████████████████████| 3500/3500 [00:05<00:00, 636.95it/s]
    elpd = -22682.80%|██████████████████ | 3325/3500 [00:05<00:00, 1788.00it/s]
  Fold 2/3: train=679 bins, held=339 bins
|00:31| LogPSpline | INFO | Saved preprocessing eigenvalue plot to out/kfold/noise5a/14d/eta0.5/K3/H1/fold_01/diagnostics/preprocessing_eigenvalue_ratios.png
|00:31| LogPSpline | INFO | Adjusting explicit coarse VI bins: N_full=679 not divisible by Nc=256; using Nc=97.
|00:31| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=97, Nh=7, Nl=679 (kept 14.3%, decimated 85.7%).
|00:43| LogPSpline | INFO | Spline model: MultivariateLogPSplines(channels=3, knots=mixed[2-50], degree=2, penaltyOrder=2, N=679, basis_shapes=[(679, 51), (679, 51), (679, 51), (679, 3), (679, 3), (679, 3), (679, 3), (679, 3), (679, 3)])
Running chain 0: 100%|███████████████████| 3500/3500 [00:01<00:00, 2746.39it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:01<00:00, 1979.37it/s]
Running chain 1: 100%|███████████████████| 3500/3500 [00:01<00:00, 1943.86it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:01<00:00, 1878.15it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:02<00:00, 1169.53it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:03<00:00, 1153.81it/s]
Running chain 1: 100%|███████████████████| 3500/3500 [00:03<00:00, 1116.50it/s]
Running chain 0: 100%|███████████████████| 3500/3500 [00:03<00:00, 1077.65it/s]
Running chain 3: 100%|█████████████████████| 3500/3500 [01:02<00:00, 56.08it/s]
Running chain 1: 100%|█████████████████████| 3500/3500 [01:02<00:00, 56.01it/s]
Running chain 0: 100%|█████████████████████| 3500/3500 [01:02<00:00, 55.86it/s]
Running chain 2: 100%|█████████████████████| 3500/3500 [01:03<00:00, 55.33it/s]
    elpd = -3993001.04
  Fold 3/3: train=678 bins, held=340 bins█| 3500/3500 [01:03<00:00, 212.19it/s]
|01:55| LogPSpline | INFO | Saved preprocessing eigenvalue plot to out/kfold/noise5a/14d/eta0.5/K3/H1/fold_02/diagnostics/preprocessing_eigenvalue_ratios.png
|01:55| LogPSpline | INFO | Adjusting explicit coarse VI bins: N_full=678 not divisible by Nc=256; using Nc=226.
|01:55| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=226, Nh=3, Nl=678 (kept 33.3%, decimated 66.7%).
|02:10| LogPSpline | INFO | Spline model: MultivariateLogPSplines(channels=3, knots=mixed[2-50], degree=2, penaltyOrder=2, N=678, basis_shapes=[(678, 51), (678, 51), (678, 51), (678, 3), (678, 3), (678, 3), (678, 3), (678, 3), (678, 3)])
Running chain 1: 100%|███████████████████| 3500/3500 [00:01<00:00, 2961.80it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:01<00:00, 2957.63it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:01<00:00, 2952.07it/s]
Running chain 0: 100%|███████████████████| 3500/3500 [00:01<00:00, 2477.47it/s]
Running chain 0: 100%|███████████████████| 3500/3500 [00:02<00:00, 1351.39it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:02<00:00, 1307.48it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:02<00:00, 1304.76it/s]
Running chain 1: 100%|███████████████████| 3500/3500 [00:02<00:00, 1260.35it/s]
Running chain 0: 100%|█████████████████████| 3500/3500 [00:54<00:00, 64.80it/s]
Running chain 2: 100%|█████████████████████| 3500/3500 [00:54<00:00, 64.10it/s]
Running chain 1: 100%|█████████████████████| 3500/3500 [00:54<00:00, 64.03it/s]
Running chain 3: 100%|█████████████████████| 3500/3500 [00:55<00:00, 63.55it/s]
    elpd = -1439981.81███████████████████ | 3325/3500 [00:54<00:00, 232.99it/s]

============================================================<00:00, 239.52it/s]
K-fold ELPD: model=H3  dataset=noise5a  dur=14.0d  K=3  eta=0.5
============================================================
Loaded 15778801 samples from data/noise5a.h5 (fs=0.500000 Hz); using 15777801 after edge trim. reference='segwo'.
|03:13| LogPSpline | INFO | Wishart averaging (blocks=2): n=604800 -> Lb=302400, N=60420, p=3
|03:13| LogPSpline | INFO | Standardized data: scale ~5.18e-11
|03:13| LogPSpline | INFO | Inferred sampler type: multivar_blocked_nuts
|03:13| LogPSpline | INFO | Nl trimmed to 60416 (dropped 4 trailing bins) to make Nc=1024 an exact divisor.
|03:13| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=1024, Nh=59, Nl=60416 (kept 1.7%, decimated 98.3%).
|03:13| LogPSpline | INFO | Null-band excision: removing 6 bins across 3 band(s). 1018 bins retained.

[H3] n_freq=1018, Nb=2, eta=0.5, K=3
  Fold 1/3: train=679 bins, held=339 bins
|03:14| LogPSpline | INFO | Saved preprocessing eigenvalue plot to out/kfold/noise5a/14d/eta0.5/K3/H3/fold_00/diagnostics/preprocessing_eigenvalue_ratios.png
|03:14| LogPSpline | INFO | Adjusting explicit coarse VI bins: N_full=679 not divisible by Nc=256; using Nc=97.
|03:14| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=97, Nh=7, Nl=679 (kept 14.3%, decimated 85.7%).
|03:34| LogPSpline | INFO | Spline model: MultivariateLogPSplines(channels=3, knots=50, degree=2, penaltyOrder=2, N=679, basis_shapes=[(679, 51), (679, 51), (679, 51), (679, 51), (679, 51), (679, 51), (679, 51), (679, 51), (679, 51)])
Running chain 0: 100%|███████████████████| 3500/3500 [00:01<00:00, 3184.26it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:01<00:00, 3089.96it/s]
Running chain 1: 100%|███████████████████| 3500/3500 [00:01<00:00, 2972.84it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:01<00:00, 2492.51it/s]
Running chain 1: 100%|████████████████████| 3500/3500 [00:22<00:00, 158.10it/s]
Running chain 3: 100%|████████████████████| 3500/3500 [00:22<00:00, 153.40it/s]
Running chain 0: 100%|████████████████████| 3500/3500 [00:22<00:00, 153.15it/s]
Running chain 2: 100%|████████████████████| 3500/3500 [00:23<00:00, 147.83it/s]
Running chain 3: 100%|█████████████████████| 3500/3500 [01:00<00:00, 57.62it/s]
Running chain 2: 100%|█████████████████████| 3500/3500 [01:00<00:00, 57.47it/s]
Running chain 0: 100%|█████████████████████| 3500/3500 [01:01<00:00, 57.15it/s]
Running chain 1: 100%|█████████████████████| 3500/3500 [01:02<00:00, 55.68it/s]
    elpd = -20080.76%|████████████████████| 3500/3500 [01:02<00:00, 272.70it/s]
  Fold 2/3: train=679 bins, held=339 bins
|05:06| LogPSpline | INFO | Saved preprocessing eigenvalue plot to out/kfold/noise5a/14d/eta0.5/K3/H3/fold_01/diagnostics/preprocessing_eigenvalue_ratios.png
|05:06| LogPSpline | INFO | Adjusting explicit coarse VI bins: N_full=679 not divisible by Nc=256; using Nc=97.
|05:06| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=97, Nh=7, Nl=679 (kept 14.3%, decimated 85.7%).
|05:26| LogPSpline | INFO | Spline model: MultivariateLogPSplines(channels=3, knots=50, degree=2, penaltyOrder=2, N=679, basis_shapes=[(679, 51), (679, 51), (679, 51), (679, 51), (679, 51), (679, 51), (679, 51), (679, 51), (679, 51)])
Running chain 1: 100%|███████████████████| 3500/3500 [00:01<00:00, 2986.98it/s]
Running chain 0: 100%|███████████████████| 3500/3500 [00:01<00:00, 2960.22it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:01<00:00, 2954.65it/s]
Running chain 3: 100%|███████████████████| 3500/3500 [00:01<00:00, 2637.11it/s]
Running chain 2: 100%|████████████████████| 3500/3500 [00:23<00:00, 150.71it/s]
Running chain 3: 100%|████████████████████| 3500/3500 [00:23<00:00, 149.96it/s]
Running chain 1: 100%|████████████████████| 3500/3500 [00:23<00:00, 147.27it/s]
Running chain 0: 100%|████████████████████| 3500/3500 [00:24<00:00, 140.52it/s]
Running chain 0: 100%|█████████████████████| 3500/3500 [01:29<00:00, 39.19it/s]
Running chain 3: 100%|█████████████████████| 3500/3500 [01:29<00:00, 38.93it/s]
Running chain 1: 100%|█████████████████████| 3500/3500 [01:31<00:00, 38.31it/s]
Running chain 2: 100%|█████████████████████| 3500/3500 [01:51<00:00, 31.44it/s]
    elpd = -7806081.57███████▎             | 1225/3500 [01:30<01:22, 27.47it/s]
  Fold 3/3: train=678 bins, held=340 bins█| 3500/3500 [01:51<00:00, 124.52it/s]
|07:47| LogPSpline | INFO | Saved preprocessing eigenvalue plot to out/kfold/noise5a/14d/eta0.5/K3/H3/fold_02/diagnostics/preprocessing_eigenvalue_ratios.png
|07:47| LogPSpline | INFO | Adjusting explicit coarse VI bins: N_full=678 not divisible by Nc=256; using Nc=226.
|07:47| LogPSpline | INFO | Coarse-grained multivariate FFT: CoarseGrainSpec(Nc=226, Nh=3, Nl=678 (kept 33.3%, decimated 66.7%).
|08:05| LogPSpline | INFO | Spline model: MultivariateLogPSplines(channels=3, knots=50, degree=2, penaltyOrder=2, N=678, basis_shapes=[(678, 51), (678, 51), (678, 51), (678, 51), (678, 51), (678, 51), (678, 51), (678, 51), (678, 51)])
Running chain 3: 100%|███████████████████| 3500/3500 [00:01<00:00, 3278.34it/s]
Running chain 1: 100%|███████████████████| 3500/3500 [00:01<00:00, 3118.71it/s]
Running chain 0: 100%|███████████████████| 3500/3500 [00:01<00:00, 3088.49it/s]
Running chain 2: 100%|███████████████████| 3500/3500 [00:01<00:00, 3027.71it/s]
Running chain 0: 100%|████████████████████| 3500/3500 [00:19<00:00, 181.34it/s]
Running chain 1: 100%|████████████████████| 3500/3500 [00:19<00:00, 177.70it/s]
Running chain 2: 100%|████████████████████| 3500/3500 [00:20<00:00, 174.73it/s]
Running chain 3: 100%|████████████████████| 3500/3500 [00:20<00:00, 168.68it/s]
Running chain 1: 100%|█████████████████████| 3500/3500 [01:27<00:00, 40.20it/s]
Running chain 0: 100%|█████████████████████| 3500/3500 [01:28<00:00, 39.44it/s]
Running chain 2: 100%|█████████████████████| 3500/3500 [01:34<00:00, 37.05it/s]
Running chain 3: 100%|█████████████████████| 3500/3500 [01:34<00:00, 36.86it/s]
    elpd = -1425359.20████████████████████| 3500/3500 [01:34<00:00, 142.77it/s]
Running chain 3:  95%|███████████████████ | 3325/3500 [01:33<00:01, 144.51it/s]
============================================================<00:00, 149.36it/s]
SUMMARY  dataset=noise5a  dur=14.0d  K=3  eta=0.5
============================================================
  H1: elpd = -5455665.65 ± 3484972.78
  H3: elpd = -9251521.53 ± 7187149.43

  Δelpd (H3 - H1) = -3795855.88 ± 3821707.03  (paired)
  → Not significant  (|Δ|/SE = 1.0)

Results saved to /Users/avi/Documents/projects/lisa_logpspline/out/kfold/noise5a/14d/eta0.5/K3
  
  
  """

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpyro
import xarray as xr

from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.datatypes.multivar import MultivarFFT
from log_psplines.pipeline.make_pipeline import make_pipeline
from log_psplines.pipeline.preprocessing import preprocess_to_freq_domain
from log_psplines.pipeline.evidence import _pointwise_multivar_log_likelihood
from log_psplines.psplines.initialisation import init_basis_and_penalty

from src.load_data import LISAData
from src.aet import xyz_to_aet_timeseries, xyz_to_aet_matrix
from main import (
    DATASET_CONFIGS,
    MODEL_SPECS,
    NUM_CHAINS,
    N_SAMPLES,
    N_WARMUP,
    BLOCK_DAYS,
    ETA,
    NC_TARGET,
    VI_NC_TARGET,
    build_multivar_config,
    _prepare_run,
)
from src.utils import (
    choose_coarse_grain_nc,
    choose_nb_for_duration,
    compute_retained_frequency_count,
    duration_slug,
    light_travel_null_exclusion_bands,
)


# ---------------------------------------------------------------------------
# Data slicing helpers
# ---------------------------------------------------------------------------

def make_freq_blocks(n_freq: int, k: int) -> list[np.ndarray]:
    """K contiguous frequency blocks of approximately equal size."""
    block_size = n_freq // k
    blocks = []
    for i in range(k):
        start = i * block_size
        end = (i + 1) * block_size if i < k - 1 else n_freq
        blocks.append(np.arange(start, end))
    return blocks


def subset_multivar_fft(data: MultivarFFT, idx: np.ndarray) -> MultivarFFT:
    """Return a new MultivarFFT restricted to the given frequency indices."""
    raw_psd = data.raw_psd[idx] if data.raw_psd is not None else None
    return MultivarFFT(
        u_re=data.u_re[idx],
        u_im=data.u_im[idx],
        freq=data.freq[idx],
        N=len(idx),
        p=data.p,
        Nb=data.Nb,
        scaling_factor=data.scaling_factor,
        channel_stds=data.channel_stds,
        fs=data.fs,
        duration=data.duration,
        raw_psd=raw_psd,
        Nh=data.Nh,
        enbw=data.enbw,
    )


# ---------------------------------------------------------------------------
# Held-out log-likelihood
# ---------------------------------------------------------------------------

def _build_held_out_bases(
    spline_train,
    held_freq: np.ndarray,
    train_freq: np.ndarray,
    degree: int,
    diff_order: int,
    p: int,
) -> tuple[list, list[list], list[list]]:
    """Evaluate spline bases from the training run at held-out frequencies.

    The [0,1] normalization is anchored to the TRAINING frequency range so
    that knot positions stay consistent between the model fit and evaluation.
    Held-out frequencies outside [train_min, train_max] are clipped to [0,1].
    """
    train_min = float(train_freq.min())
    train_max = float(train_freq.max())
    denom = train_max - train_min if train_max > train_min else 1.0
    held_norm = np.clip((held_freq - train_min) / denom, 0.0, 1.0)
    n_held = len(held_freq)

    bases_delta = []
    for j in range(p):
        m = spline_train.component_specs[spline_train.delta_key(j)].model
        basis_h, _ = init_basis_and_penalty(
            knots=np.asarray(m.knots),
            degree=degree,
            n_grid_points=n_held,
            diff_matrix_order=diff_order,
            grid_points=held_norm,
        )
        bases_delta.append(np.asarray(basis_h, dtype=np.float32))

    bases_theta_re: list[list] = []
    bases_theta_im: list[list] = []
    for j in range(p):
        br, bi = [], []
        for l in range(j):
            m_re = spline_train.get_theta_model("re", j, l)
            m_im = spline_train.get_theta_model("im", j, l)
            b_re, _ = init_basis_and_penalty(
                knots=np.asarray(m_re.knots),
                degree=degree,
                n_grid_points=n_held,
                diff_matrix_order=diff_order,
                grid_points=held_norm,
            )
            b_im, _ = init_basis_and_penalty(
                knots=np.asarray(m_im.knots),
                degree=degree,
                n_grid_points=n_held,
                diff_matrix_order=diff_order,
                grid_points=held_norm,
            )
            br.append(np.asarray(b_re, dtype=np.float32))
            bi.append(np.asarray(b_im, dtype=np.float32))
        bases_theta_re.append(br)
        bases_theta_im.append(bi)

    return bases_delta, bases_theta_re, bases_theta_im


def eval_held_out_lnl(
    posterior: xr.Dataset,
    spline_train,
    train_data: MultivarFFT,
    held_data: MultivarFFT,
    degree: int,
    diff_order: int,
) -> np.ndarray:
    """Evaluate untempered Wishart log-likelihood at held-out frequencies.

    Parameters
    ----------
    posterior : xr.Dataset
        Posterior samples from the training-data MCMC run.
    spline_train : MultivariateLogPSplines
        Spline model built from training data (provides knot positions).
    train_data, held_data : MultivarFFT
        Training and held-out frequency-domain data.
    degree, diff_order : int
        Spline degree and penalty order (must match the training config).

    Returns
    -------
    np.ndarray, shape (S, n_held_freq)
        Log-likelihood per posterior draw per held-out frequency bin.
    """
    import jax.numpy as jnp

    p = train_data.p
    bases_delta, bases_theta_re, bases_theta_im = _build_held_out_bases(
        spline_train,
        held_freq=held_data.freq,
        train_freq=train_data.freq,
        degree=degree,
        diff_order=diff_order,
        p=p,
    )

    held_kwargs: dict[str, Any] = {
        "u_re": jnp.asarray(held_data.u_re, dtype=jnp.float32),
        "u_im": jnp.asarray(held_data.u_im, dtype=jnp.float32),
        "n_channels": p,
        "bases_delta": bases_delta,
        "bases_theta_re": bases_theta_re,
        "bases_theta_im": bases_theta_im,
        "Nb": int(held_data.Nb),
        "Nh": int(held_data.Nh),
        "duration": float(held_data.duration),
        "enbw": float(held_data.enbw),
    }

    ll_ds = _pointwise_multivar_log_likelihood(posterior, held_data, held_kwargs)
    total_ll = np.asarray(ll_ds["log_likelihood"].values)  # (chain, draw, n_held)
    n_chain, n_draw, _ = total_ll.shape
    return total_ll.reshape(n_chain * n_draw, -1)


# ---------------------------------------------------------------------------
# Full-data MultivarFFT builder
# ---------------------------------------------------------------------------

def build_full_freq_data(
    args: argparse.Namespace,
    cfg,
    lisa: LISAData,
    runner: str,
    k_theta: int | None,
    eta: float,
) -> tuple[MultivarFFT, Any, int, int, int, float]:
    """Preprocess time-domain data to a single full MultivarFFT.

    Returns (full_data, config, nb, main_nc, vi_nc, eta).
    The config is built only for spline/pipeline settings; outdir is /tmp.
    """
    duration_days = float(args.duration_days)
    nb = choose_nb_for_duration(duration_days, block_days=BLOCK_DAYS)
    n_raw = min(
        lisa.data.shape[0],
        int(duration_days * 86400.0 / lisa.delta_t),
    )
    n_trim = (n_raw // (2 * nb)) * (2 * nb)
    n_band_raw = compute_retained_frequency_count(
        n_trim=n_trim, dt=lisa.delta_t, nb=nb,
        fmin=1e-4, fmax=1e-1,
    )
    main_nc = choose_coarse_grain_nc(n_retained=n_band_raw, nc_target=NC_TARGET)
    vi_nc   = choose_coarse_grain_nc(n_retained=n_band_raw, nc_target=VI_NC_TARGET)

    y = lisa.data[:n_trim].astype(np.float64)
    t = lisa.time[:n_trim].astype(np.float64)
    true_psd_matrix = lisa.true_matrix

    if runner == "multivar_aet":
        y = xyz_to_aet_timeseries(y)
        true_psd_matrix = xyz_to_aet_matrix(true_psd_matrix)

    ts = MultivariateTimeseries(y=y, t=t)

    theta_re_knots = k_theta if k_theta is not None else None
    theta_im_knots = k_theta if k_theta is not None else None

    config = build_multivar_config(
        cfg=cfg,
        true_psd_freq=lisa.freq,
        true_psd_matrix=true_psd_matrix,
        nb=nb,
        main_nc=main_nc,
        vi_nc=vi_nc,
        eta=eta,
        outdir=Path("/tmp/kfold_scratch"),
        compute_lnz=False,
        lnz_n_resamples=512,
        lnz_n_estimations=1,
        lnz_max_iter=5000,
        lnz_tol=1e-2,
        lnz_kde_bw="silverman",
        theta_re_knots=theta_re_knots,
        theta_im_knots=theta_im_knots,
    )

    full_data = preprocess_to_freq_domain(ts, config)
    return full_data, config, nb, main_nc, vi_nc, eta


# ---------------------------------------------------------------------------
# K-fold runner
# ---------------------------------------------------------------------------

def run_kfold(
    model: str,
    dataset: str,
    duration_days: float,
    n_folds: int,
    outdir: Path,
    eta: float,
) -> dict:
    """Run K-fold ELPD for one (model, dataset) pair.

    Returns a dict with keys: elpd, se, fold_elpds, n_folds, model, dataset.
    """
    cfg  = DATASET_CONFIGS[dataset]
    spec = MODEL_SPECS[model]
    lisa = LISAData.load(cfg.data_path, reference=cfg.reference)

    import types
    args = types.SimpleNamespace(duration_days=duration_days)
    full_data, config, nb, main_nc, vi_nc, _eta = build_full_freq_data(
        args, cfg, lisa, runner=spec.runner, k_theta=spec.k_theta, eta=eta,
    )
    n_freq = full_data.N
    print(f"\n[{model}] n_freq={n_freq}, Nb={nb}, eta={eta:.3g}, K={n_folds}")

    blocks   = make_freq_blocks(n_freq, n_folds)
    numpyro.set_host_device_count(NUM_CHAINS)

    fold_elpds: list[float] = []

    for fold_idx, held_idx in enumerate(blocks):
        train_idx = np.concatenate(
            [blocks[k] for k in range(n_folds) if k != fold_idx]
        )
        train_data = subset_multivar_fft(full_data, train_idx)
        held_data  = subset_multivar_fft(full_data, held_idx)

        fold_nc    = choose_coarse_grain_nc(train_data.N, NC_TARGET)
        fold_vi_nc = choose_coarse_grain_nc(train_data.N, VI_NC_TARGET)

        fold_outdir = outdir / f"fold_{fold_idx:02d}"
        fold_outdir.mkdir(parents=True, exist_ok=True)

        print(
            f"  Fold {fold_idx+1}/{n_folds}: "
            f"train={len(train_idx)} bins, held={len(held_idx)} bins"
        )

        fold_config = build_multivar_config(
            cfg=cfg,
            true_psd_freq=lisa.freq,
            true_psd_matrix=(
                xyz_to_aet_matrix(lisa.true_matrix)
                if spec.runner == "multivar_aet"
                else lisa.true_matrix
            ),
            nb=nb,
            main_nc=fold_nc,
            vi_nc=fold_vi_nc,
            eta=eta,
            outdir=fold_outdir,
            compute_lnz=False,
            lnz_n_resamples=512,
            lnz_n_estimations=1,
            lnz_max_iter=5000,
            lnz_tol=1e-2,
            lnz_kde_bw="silverman",
            theta_re_knots=spec.k_theta,
            theta_im_knots=spec.k_theta,
        )

        pipeline = make_pipeline(train_data, fold_config)
        result   = pipeline.run()

        posterior    = result.idata["posterior"].dataset
        spline_train = pipeline.spline_model

        held_ll = eval_held_out_lnl(
            posterior=posterior,
            spline_train=spline_train,
            train_data=train_data,
            held_data=held_data,
            degree=fold_config.degree,
            diff_order=fold_config.diffMatrixOrder,
        )  # shape (S, n_held)

        # elpd for this fold = E_posterior[sum_h log L(h|theta)]
        fold_elpd = float(np.mean(np.sum(held_ll, axis=1)))
        fold_elpds.append(fold_elpd)
        print(f"    elpd = {fold_elpd:.2f}")

        np.save(fold_outdir / "held_out_lnl.npy", held_ll)

    total_elpd = float(sum(fold_elpds))
    se = (
        float(np.sqrt(n_folds * np.var(fold_elpds, ddof=1)))
        if n_folds > 1
        else float("nan")
    )

    result_dict = {
        "model": model,
        "dataset": dataset,
        "duration_days": duration_days,
        "n_folds": n_folds,
        "eta": eta,
        "elpd": total_elpd,
        "se": se,
        "fold_elpds": fold_elpds,
    }
    (outdir / "kfold_result.json").write_text(json.dumps(result_dict, indent=2))
    return result_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Block K-fold ELPD comparison.")
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    p.add_argument("--duration-days", type=float, default=14.0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--eta", type=float, default=1.0,
                   help="Tempering for MCMC (1.0 = untempered, recommended).")
    p.add_argument("--models", nargs="+", default=["H1", "H3"],
                   choices=list(MODEL_SPECS))
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    duration_label = f"{int(args.duration_days)}d"
    base_outdir = args.outdir or Path(
        f"out/kfold/{args.dataset}/{duration_label}/eta{args.eta:g}/K{args.n_folds}"
    )

    results: dict[str, dict] = {}
    for model in args.models:
        model_outdir = base_outdir / model
        model_outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"K-fold ELPD: model={model}  dataset={args.dataset}  "
              f"dur={args.duration_days}d  K={args.n_folds}  eta={args.eta}")
        print(f"{'='*60}")
        results[model] = run_kfold(
            model=model,
            dataset=args.dataset,
            duration_days=args.duration_days,
            n_folds=args.n_folds,
            outdir=model_outdir,
            eta=args.eta,
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY  dataset={args.dataset}  dur={args.duration_days}d  "
          f"K={args.n_folds}  eta={args.eta}")
    print(f"{'='*60}")
    for model, r in results.items():
        print(f"  {model}: elpd = {r['elpd']:.2f} ± {r['se']:.2f}")

    if len(results) == 2:
        keys = list(results.keys())
        a, b = results[keys[0]], results[keys[1]]
        # Paired SE: use per-fold differences to account for correlation
        fold_deltas = np.array(b["fold_elpds"]) - np.array(a["fold_elpds"])
        delta = float(fold_deltas.sum())
        n = len(fold_deltas)
        se_diff = float(np.sqrt(n * np.var(fold_deltas, ddof=1))) if n > 1 else float("nan")
        print(f"\n  Δelpd ({keys[1]} - {keys[0]}) = {delta:.2f} ± {se_diff:.2f}  (paired)")
        if np.isfinite(se_diff) and abs(delta) > 2 * se_diff:
            winner = keys[1] if delta > 0 else keys[0]
            print(f"  → {winner} preferred  (|Δ|/SE = {abs(delta)/se_diff:.1f})")
        else:
            ratio = abs(delta) / se_diff if np.isfinite(se_diff) else float("nan")
            print(f"  → Not significant  (|Δ|/SE = {ratio:.1f})")

    summary_path = base_outdir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {base_outdir.resolve()}")


if __name__ == "__main__":
    main()
