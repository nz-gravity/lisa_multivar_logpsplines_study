# Stationary data through the time-varying model

This is an **incomplete, cancelled control**, prepared for the updated
[LogPSplinePSD PR #53](https://github.com/nz-gravity/LogPSplinePSD/pull/53).
Controlled simulation studies take priority before another long LISA fit.
This record concerns the former ANOVA implementation. PR #53 now uses plain
tensor Cholesky splines; the pinned old driver is retained for historical
reproduction, not as a second supported model.
The input is the saved ten-day,
unequal-MOSA AET simulation with fixed 8.3-second orbit delays. The same stationary
record was submitted to an ANOVA spectral matrix model that permits a linear time
contribution. The model integrates cross-spectrum coefficients by Gaussian
marginalization and restores them with joint conditional draws after sampling.

## What is matched

The preprocessing comparison uses the same globally centered record, channel
normalization, eight nonoverlapping Tukey(0.1) blocks, fine-frequency support,
coarse-bin response factor and null exclusions. Pooling adjacent time blocks gives
four observed times; the original eight block centers remain the reference grid
used for ANOVA centering and reconstruction. There are 124 retained frequency
bins, 84 fine frequencies per bin and 672 complex observations per frequency when
summed over time. ENBW is 1.0387811634349031.

The time-local sufficient statistics summed over time agree with stationary
preprocessing to **3.1423e-15**, after normalizing each matrix entry by the geometric
mean of its auto powers. This normalization also checks the quiet T channel.
Counts and ENBW match exactly to numerical precision.

The separate library tests set the time design to zero and match the Gaussian
frequency priors. They verify the marginal likelihood, conditional coefficient
mean/covariance, and likelihood gradients against the stationary implementation.
This is the exact stationary-limit check. The proposed flexible ANOVA fit has a
different prior from the earlier stationary demo and therefore is a recovery
control, not an assertion that their posterior distributions must be identical.

## Cancelled sampling attempt

The attempted fit requested four sequential chains, each with 500 warmup and 500 retained draws;
target acceptance is 0.95, maximum tree depth 10 and the mass matrix is dense.
The frequency design has 16 coefficients and the centered linear time design has
one coefficient. Each Cholesky row samples 32 delta coefficients plus smoothing
scales: 34, 38 and 42 variables respectively. The third row would additionally recover
128 integrated theta coefficients per posterior draw.

The process was stopped after **1 h 02 min 34 s** of
observed wall time (245 min 54 s cumulative CPU time), to prioritize controlled
simulation studies. An interrupt was followed by termination; the process exited
with code 143. **No completed posterior was retained.** Divergence counts, R-hat,
ESS, BFMI, tree-depth statistics and recovery results are unavailable. This
cancellation is not evidence that any diagnostic gate failed.

The intended sampler gate requires zero divergences and tree-depth hits, finite parameter
diagnostics, maximum R-hat below 1.01, minimum bulk/tail ESS of 400, and BFMI of at
least 0.3 in every chain. The saved driver also reports worst-parameter diagnostics,
step counts, step sizes and per-row sampling/recovery time. Its proposed PSD ratios
and complex-coherency differences compare the
first and last **reference-grid** times. These endpoints bracket the four pooled
observation times. On a completed future run, contrasts and finite-value and
positive-definiteness checks will use all posterior draws, independently of the
stored spectrum preview. None of those posterior checks ran for this attempt.

The compact receipts are [preparation checks](preparation_checks.json),
[original fit-start provenance](provenance.json), [cancellation record](run_status.json)
and [captured run log](run.log). Large input data remain local. There are no
posterior, recovery or figure artifacts for this cancelled attempt.

The process loaded inference revision `d0b3aab6a615ef5ff47468d7e92be850466203a9`.
Later revision `61be849` corrected a documentation heading, and `393ed6a` added
row-start/end logging only. Those later changes did not alter this running
process or its inference logic. The original provenance retains its fit-start
source hashes, driver hash and `progress_bar=false`. The current driver enables
progress bars and adds postprocessing diagnostics for future use; it therefore
has a different hash. These edits did not produce a new fit.

## Reproduce and interpret

The [demo README](../../README.md#next-time-varying-tests) has preparation commands
and instructions for a future full run. Full LISA sampling is deferred until the
controlled simulation checks are satisfactory.
`provenance.json` records the data hash, configuration, inference source hashes and
the driver hash at fit start. A completed future run will save the final analysis
source hash in `recovery.json`, so postprocessing improvements can be distinguished
from the fitted model.
Large posterior arrays and raw data are generated locally and ignored by Git.

The physical reference includes the finite periodic generator, global centering,
tapering and frequency pooling. The known response factor remains study code.
Marginal component quantiles need not themselves form a positive-definite matrix;
the driver checks complete draws instead. Frequency-by-frequency interval containment
is descriptive for one realization, not repeated-noise coverage calibration.
This control does not validate genuinely time-varying recovery, overlapping-window
likelihood independence, or multivariate time-varying VI.
