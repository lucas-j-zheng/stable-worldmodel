# Policy attribution: is the +10 test-time conditional multimodality? — 2026-06-30

**Why.** The audit (2026-06-29) flagged that the diffusion-policy win (+10 DP over
same-backbone TransformerMSE, architecture-controlled) tracks the `door_prob` dose
**only when multimodality is measured under `policy_target`** (condition on the
episode's final destination). Under `policy_goal` (condition on the +8 future
position, = what the policy actually uses) the screen reads flat ~0.14 with no dose
gradient. The two modes differ in BOTH goal-type and timestep-range, so neither alone
attributes the win. This entry isolates the axes.

Build: added `--frac-lo/--frac-hi` within-episode timestep windows to the screen
(`multimodality_diagnostic.py`), and `policy_goal --cond-from latent` (condition on
`(latent_t, latent_{t+8})`, the conditioning the closed-loop policy literally uses).
Encoder-free of new compute (latents cached). Jobs 3575246 (state goal), 3575527
(latent goal).

## Step 1 — timestep axis, raw +8 STATE goal (job 3575246, door_prob 0.5 data)

| `policy_goal` window | det_R² | residual_ratio | residual_bimodal |
|---|---|---|---|
| early [0,0.33) | 0.933 | 0.052 | 0.116 |
| mid   [0.33,0.66) | 0.801 | 0.159 | 0.161 |
| late  [0.66,1.0) | 0.825 | 0.126 | 0.159 |
| full | 0.880 | 0.090 | 0.155 |
| **`policy_target` early (dest goal, ref)** | **0.304** | **0.465** | **0.572** |
| **`policy_target` full (dest goal, ref)** | **0.312** | **0.493** | **0.578** |

Unimodal contrast (door_prob 1.0), `policy_goal` early: bimodal **0.097** ≈ mm0's
0.116. **The timestep axis does not rescue it** — under the +8 state goal the
conditional is ~unimodal at every bucket, and multimodal data ≈ unimodal data. The
0.57 dose is entirely a goal-TYPE effect (destination vs +8-future), not timestep.

## Step 2 — conditioning-MATCHED, +8 LATENT goal (job 3575527)

The policy conditions on the +8 LATENT (encoded image), lossier than the raw state —
so it might preserve route ambiguity the state resolves. The decisive screen:

| data | window | det_R² | residual_ratio | residual_bimodal |
|---|---|---|---|---|
| mm0 (multimodal) | full | 0.910 | 0.116 | **0.0083** |
| mm0 (multimodal) | early | 0.941 | 0.112 | **0.0063** |
| p10 (unimodal) | full | 0.958 | 0.134 | **0.0053** |
| p10 (unimodal) | early | 0.782 | 0.332 | **0.0073** |

**residual_bimodal ≈ 0.006 everywhere — the unimodal floor (self-test unimodal =
0.000), and mm0 indistinguishable from p10 (0.008 vs 0.005).** The lossy latent goal
does NOT preserve the dose; it kills the multimodality just like the raw state. The
policy conditions on even more (full history, not just `latent_t`), so its conditional
is at least this resolved.

## VERDICT — the conditional-multimodality attribution does NOT hold

Under the conditioning the diffusion/TMSE policies actually use (history + +8 latent
goal), `p(action chunk | history, goal)` is **unimodal**, and multimodal-demonstrator
data (door_prob 0.5) reads the same as unimodal (door_prob 1.0). The `door_prob` dose
is real but lives only in the **destination-conditioned / demonstrator-marginal** view
(`policy_target` 0.578 → 0.166), which the policy never conditions on (train AND eval
use the +8-future goal: `diffusion_policy.py:164`, `diffusion_policy_solver.py:64`).

So: **the +10 DP-over-TMSE win is real and tracks the demonstrator's multimodality,
but it is NOT test-time conditional multimodality** — the clean "the policy faces a
multimodal conditional and diffusion samples a mode" story is falsified for this setup.

## What this means for the thesis

The strict thesis ("diffusion beats MSE iff the conditional *it models* is multimodal")
is NOT supported on the policy side as operationalized here: the modeled conditional is
unimodal, yet diffusion wins +10. The surviving, defensible statement is weaker and
about the **training distribution**: *diffusion beats MSE on data from a multimodal
demonstrator, and the gap tracks the demonstrator's multimodality (door_prob, measured
via destination-conditioning).* The mechanism is most likely the MSE objective's
corruption by training on multimodal demonstrations (mode-averaging in the supervised
target / train-eval goal interplay), NOT test-time mode sampling — but that mechanism
is NOT yet isolated and must not be claimed.

## OPEN (the new question this surfaces)

Why does +10 persist when the matched conditional is unimodal? Candidate mechanisms,
none yet tested: (a) the supervised MSE target averages route-modes for (history,goal)
bins where the +8 goal doesn't fully separate them during *training*, biasing TMSE
toward the wall; (b) train/eval goal mismatch makes the test conditional more ambiguous
than the demo-latent screen shows; (c) calibration/generalization difference unrelated
to multimodality. Isolating this is the next cheap step before any stronger claim.

## Artifacts
- Code: `multimodality_diagnostic.py` (`--frac-lo/--frac-hi`, `policy_goal --cond-from latent`)
- `scripts/slurm/attribution_timestep_screen.sbatch` (job 3575246),
  `scripts/slurm/attribution_latentgoal_screen.sbatch` (job 3575527)
- JSON: `logs/multimodality_attrib_*.json`, `logs/multimodality_attriblat_*.json`
