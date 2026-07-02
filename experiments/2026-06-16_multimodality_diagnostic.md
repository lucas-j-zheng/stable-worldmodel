# Multimodality diagnostic — where does the multimodality actually live? — 2026-06-16

**Why this experiment.** Both negative results so far (TwoRoom 2026-06-14, and
PushT/E6 trending the same way) compare a diffusion *dynamics* model against a
deterministic predictor and find the deterministic one wins. The standing
hypothesis was "the domain just isn't multimodal." But that was an assumption, not
a measurement, and the whole D-MPC premise rests on it. So before sinking more
compute into PushT tuning, **measure** it: is `P(z_{t+1} | state, action)`
actually multimodal? Run the same metric on TwoRoom (the near-deterministic
reference) and PushT (the supposed multimodal win) so the comparison is anchored.

The tool (`scripts/data/multimodality_diagnostic.py`): k-NN in conditioning space
to sample the conditional next-latent distribution, then **detrend** — fit a
local-linear map and measure the *residual*, which removes deterministic local
sensitivity and isolates true conditional stochasticity. The residual's Sarle
bimodality coefficient says whether that stochasticity is *multimodal* (the
diffusion edge) or just unimodal Gaussian noise (diffusion gains nothing). A
linear fit can't erase multimodality (it passes between branches), so residual
bimodality genuinely detects split modes.

---

## Results

### A methodological correction worth recording
The **first pass used raw within-neighborhood spread** and gave a result that
*contradicted* the closed-loop experiments: TwoRoom looked *more* multimodal than
PushT (ratio 0.81 vs 0.55; bimodal-fraction 0.79 vs 0.74). That was a **confound,
not a contradiction.** Raw spread mixes (a) deterministic local sensitivity, which
scales with how wide the k-NN neighborhood is — and TwoRoom has 37× fewer
transitions (5.4k vs 200k) in a 4-D vs 9-D conditioning space, so its
neighborhoods are far wider — with (b) true stochasticity. The fix (detrend +
density-matched control) made the metric comparable. **Lesson: never compare raw
conditional spread across datasets of different size/dimension; detrend first.**

### Dynamics multimodality — `P(z_{t+1} | state, action)`
Detrended (the trustworthy signal), with a density-matched PushT control:

| domain | n | `det_R2` (deterministic share) | `residual_bimodal_frac` (diffusion edge) | residual median BC |
|---|---|---|---|---|
| TwoRoom | 5.4k | 0.50 | **0.27** | 0.52 |
| PushT (full) | 200k | **0.86** | **0.07** | 0.38 (≈Gaussian) |
| PushT (matched 5.4k) | 5.4k | 0.73 | **0.09** | 0.43 |

- **PushT forward dynamics are *more* deterministic than TwoRoom** (`det_R2`
  0.86 vs 0.50): a local-linear `(state, action) → z'` map explains 86% of
  next-latent variance.
- **PushT's residual stochasticity is essentially Gaussian, not multimodal**
  (7–9% bimodal; median BC 0.38 vs a Gaussian's 0.33). There is **no multimodal
  next-latent structure for a diffusion dynamics model to exploit.**
- **The fix held:** `residual_bimodality_fraction` is stable across the 37× sample
  swing (0.07 → 0.09), where the raw fraction was a pure density artifact. The
  shape metric is robust; the magnitude metrics (`residual_ratio`) remain partly
  density-sensitive (wider neighborhoods leave more curvature residual) — so read
  the **fraction**, not the magnitude, across domains.

### Full picture — dynamics vs policy vs chunked policy
Detrended `residual_bimodal_frac` (the diffusion edge) across all probes:

| | dynamics `P(z'\|s,a)` | policy `P(a\|s)` | policy_chunk `P(a_{t:t+8}\|z)` |
|---|---|---|---|
| **TwoRoom** | det_R2 0.50 · **0.27** | det_R2 0.03 · **0.32** | det_R2 0.51 · **0.0003** |
| **PushT** | det_R2 0.86 · **0.07** | det_R2 0.51 · **0.05** | det_R2 0.86 · **0.0025** |

### Instrument validation (synthetic positive control, `--self-test`)
| synthetic | det_R2 | residual_bimodal_frac |
|---|---|---|
| unimodal (linear+Gaussian) | 0.95 | **0.000** |
| bimodal (±sep) | 0.44 | **0.978** |
| trimodal | 0.53 | **0.477** |

The metric reads ~0 on truly unimodal data and ~1 on truly bimodal data, so the
domain readings are trustworthy: the ~0s are real, not a detrend artifact.

**The surprise:** I expected PushT *policy* multimodality to be high (the
canonical Diffusion Policy win). It is **not** — `P(a|s)` and even the faithful
`P(action-chunk | latent-obs)` are near-deterministic and unimodal on PushT
(det_R2 0.86, residual ≈ unimodal). The multimodality DP exploits is **not in
this dataset**: `quentinll/lewm-pusht` is a single scripted/near-optimal expert,
whereas DP's multimodality comes from **diverse multi-human demonstrations**
(different people solve the task different ways). Multimodality is a property of
the **demonstration source**, not the task.

---

## What it means

- **The bottleneck is the DATA, not the parametrization or the dynamics-vs-policy
  choice.** Across every probe — dynamics, single-step policy, chunked
  latent-conditioned policy — PushT is near-deterministic and unimodal. There is
  no model-accessible multimodality for *any* generative model (dynamics or
  policy) to exploit on this dataset. A diffusion model can only beat a
  deterministic/Gaussian one where the target is multimodal; here it isn't, so the
  whole D-MPC-vs-MPC comparison on these datasets is **uninformative by
  construction**.
- **Why PushT — the DP benchmark — reads unimodal here:** the multimodality that
  makes Diffusion Policy win is in **diverse multi-human demonstrations**, not in
  the task. `quentinll/lewm-pusht` is a single near-optimal expert, so its policy
  is an (almost) deterministic function of the observation. Same task, different
  data source, completely different multimodality.
- **This makes a falsifiable prediction for the still-running E6 closed-loop:**
  diffusion ≤ deterministic on PushT, same as TwoRoom. The diagnostic and the
  closed-loop are mutually validating; if confirmed, the question is settled *for
  these datasets*.
- **TwoRoom has real (moderate) multimodality and STILL lost** (0.27/0.32 bimodal,
  yet deterministic won 58% vs 46/30%). So even where some multimodality exists,
  the diffusion sampler's overhead/variance outweighs it at this scale. The
  multimodality has to be *substantial* (and exploitable, not encoder aliasing) to
  pay for the sampler — a higher bar than "nonzero".
- **Selection rule, upgraded:** this is now a *mandatory pre-experiment screen*.
  Never run a diffusion-vs-deterministic comparison on a dataset whose target
  `residual_bimodal_frac` is near the unimodal floor — there is nothing to win.

---

## UPDATE 2026-06-16 (pm) — multi-human PushT tested; hypothesis NOT supported

Sourced the **multi-human** PushT data (LeRobot `lerobot/pusht_image`, 206 human
teleop episodes; + the DP zarr `pusht_cchi_v7_replay` for the ground-truth 5-D
state agent+block) to test "data source, not task". Built an isolated 3.12
`.venv312` (lerobot 0.5.1) for ingestion, kept the 3.10 pipeline untouched.

**Caught + fixed a methodology bug:** the latent-conditioned `policy_chunk`
detrend fit 33 params (32 PCA dims + bias) from k=64 neighbors -> overfit ->
det_R2 inflated, residual whitened, `residual_bimodal -> 0` spuriously. Added an
overfit guard (k auto-bumped to >=10*(params); records `overfit_risk`). The
earlier expert `policy_chunk`-latent numbers were contaminated by this; the
low-dim **dynamics**/**policy** results (the headline) were not.

**Clean, overfit-guarded, matched results:**

| screen | cond | det_R2 | residual_bimodal_frac |
|---|---|---|---|
| human state(5-D) -> chunk | 5-D | 0.67 | **0.071** |
| expert state(7-D) -> chunk | 7-D | 0.53 | **0.101** |
| human latent -> chunk (k=330) | 32-D | 0.88 | **0.000** |
| expert latent -> chunk (k=330) | 32-D | 0.67 | **0.002** |

**Human PushT is NOT more multimodal than the scripted expert** (slightly less on
the clean state screen). Across *every* probe and *both* datasets, action/dynamics
distributions read near-unimodal. "Data source, not task" is unsupported by the
structural screen.

**Bounding caveat:** Sarle's BC detects HARD separated modes (validated 0.98 on
3-sigma synthetic modes). It under-detects soft/overlapping, minority
decision-point, and long-horizon multimodality -- plausibly where DP's PushT
advantage lives. So this is strong evidence against *gross* multimodality, not
proof a diffusion policy can't help. **The structural screen has reached its
limit; the behavioral test (diffusion policy vs deterministic, closed-loop) is the
only definitive arbiter.**

## What's next

1. **Test the data-source hypothesis directly (the next experiment):** take a
   **multi-human PushT demonstration set** (e.g. the original Diffusion Policy /
   `lerobot/pusht` data), cache it through the *same* `lewm_pusht` encoder, and
   run `--mode policy_chunk` on it. Prediction: high `residual_bimodal_frac` where
   the scripted expert reads ~0. If so, this *proves* "data source, not task" and
   hands us the dataset on which diffusion could actually win.
2. **Only then** build the diffusion policy (revive the dropped proposal ρ) and
   test it closed-loop on that multimodal data — the real experiment the whole
   arc points at.
3. **Let E6 closed-loop finish** (queued behind rmax/concurrent) as the empirical
   check; expect diffusion ≤ deterministic.
4. **Deprioritize** all further diffusion tuning on the current TwoRoom/PushT
   datasets — the screen says there's no signal there to recover.

---

## Artifacts
- Script: `scripts/data/multimodality_diagnostic.py` (`--mode dynamics|policy`)
- Job: `scripts/slurm/multimodality_diagnostic.sbatch` (2×2, gpu-debug)
- Results: `logs/multimodality_{tworoom,pusht}_{dynamics,policy}.json` on Oscar
