# Diffusion Policy — build + experiment plan — 2026-06-22

**Why.** The whole D-MPC investigation concluded diffusion-for-*dynamics* doesn't
pay off on deterministic-physics benchmarks; the multimodality that exists is in
the *policy*. So the decisive experiment is a diffusion *policy* (the dropped
proposal ρ) vs a mode-averaging MSE baseline, on multimodal demos. This entry
documents the build (the reusable deliverable) and the experiment to run.

## What was built (committed + pushed to origin)
- `stable_worldmodel/wm/diffusion_policy/`:
  - `module.py` `ActionTrajectoryDenoiser` — transformer denoiser; tokens
    [history latents | goal latent | noisy action chunk], output = action tokens.
  - `diffusion_policy.py` `DiffusionPolicy` — conditional diffusion over
    (latent history, goal latent) -> action chunk. Reuses the cosine schedule,
    q_sample, eps/x0/v parametrization, DDIM from the dynamics model. Internal
    action normalization (z-score buffers).
  - `mlp_policy.py` `MLPPolicy` — MSE mode-averaging baseline, same interface.
- `stable_worldmodel/solver/diffusion_policy_solver.py` `DiffusionPolicySolver`
  — drop-in for CEMSolver; encodes latent history + goal, calls
  `model.sample_actions`. Reuses WorldModelPolicy's history/goal machinery (no CEM).
- `scripts/train/train_diffusion_policy.py` (+ `config/diffusion_policy.yaml`);
  `scripts/plan/config/{diffusion_policy_eval.yaml,solver/diffusion_policy.yaml}`.

## Layout bugs caught by the smoke (recorded)
- **action_dim = 10, not 2.** The cached latent dataset's `action` column is
  blocked by frameskip: raw 2 * action_block 5 = 10 per latent step. This matches
  the CEM solver's `action_dim` (raw*action_block), so the policy output plugs
  straight into the eval harness.
- **action stats from batches, not get_col_data.** `get_col_data('action')`
  returns the raw 2-dim column; the DataLoader blocks it to 10-dim. Compute
  NaN-masked stats from actual batches so the normalizer dim matches.

## Validated / pending
- ✅ DiffusionPolicy training runs (loss decreasing, checkpoint saved).
- ⏳ MLP-baseline training (override fix) + closed-loop eval integration
  (`diffusion_policy_eval`) — submitted; results unread (VPN outage 06-22).

## The experiment to run (decisive test)
1. **Retrain LeWM encoder on the 2-door mm data** (`tworoom_mm0.lance`) ->
   `lewm_tworoom_mm`. REQUIRED: the earlier 0%/0% mm-dynamics eval failed because
   the 1-door encoder was OOD on 2-door scenes. The eval scene must be in-dist.
2. Cache mm latents with that encoder.
3. Train **DiffusionPolicy** and **MLPPolicy** on those latents.
4. Closed-loop **DiffusionPolicy vs MLP** on the 2-door env (eval.options =
   2-door geometry, dataset = tworoom_mm0). 
   - PREDICTION: on the multimodal (random-door) data, the MSE/MLP policy
     mode-averages the two routes -> drives into the wall between doors -> low
     success; the diffusion policy commits to one mode -> higher success. If so,
     this is the first POSITIVE result: diffusion wins where multimodality is in
     the policy and provably present. If they tie, the multimodality still isn't
     being exploited (investigate goal-conditioning / action horizon).

## RESULT (2026-06-23) — FIRST POSITIVE RESULT FOR DIFFUSION

Trained DiffusionPolicy + MLP baseline (250 ep) on 2-door mm latents (encoder
`lewm_tworoom_mm` epoch 297, in-distribution). Closed-loop on the 2-door env:

| goal_offset | DiffusionPolicy | MLP baseline | n |
|---|---|---|---|
| **8**  | **42%** | **18%** | 50 |  <- headline (matched goal horizon)
| 12 | 8%  | 0%  | 50 |  (harder; DP still > MLP, MLP solves nothing)
| 25 | 0%  | 0%  | 20 |  (goal too far for the ~8-step training window)

**The diffusion policy beats the mode-averaging MLP ~2.3x (offset 8, n=50:
42% vs 18%); at offset 12 it still wins (8% vs 0%).** Stable from n=20 (45/15) to
n=50 (42/18). Exactly the predicted mechanism: on the random-door (multimodal)
demos the MLP regresses to the mean of the two routes -> drives into the wall
between the doors -> fails; the diffusion policy samples ONE mode -> reaches the
goal. n=50 confirmation: job 3388265.

**Two debugging lessons baked in:**
- *Goal-horizon mismatch:* eval `goal_offset_steps` must roughly match the policy's
  training goal window (end-of-window latent ~ history+horizon). At 25 both
  policies scored 0% (the earlier 0%/0% scare); at 8/12 (matching) they work.
- *Encoder must be in-distribution:* the 2-door encoder (`lewm_tworoom_mm`) was
  required; the 1-door `lewm_tworoom` gave 0% on 2-door scenes (OOD).

**Significance for the whole arc.** Every prior negative (TwoRoom/PushT diffusion
*dynamics* losing) was diffusion on the wrong side of the problem — deterministic
dynamics have no multimodality to exploit. Put the diffusion on the POLICY, on
provably-multimodal data, and it wins decisively. The thesis holds: *diffusion
helps where the conditional distribution is multimodal; in these robotics tasks
that is the policy, not the dynamics.*

## THESIS TEST (2026-06-23) — matched negative control confirms it

Ran the SAME pipeline on the GREEDY single-door (UNIMODAL) demos as a negative
control (encoder `lewm_tworoom`, 1-door eval, goal_offset 8, n=50):

| data | DiffusionPolicy | MLP baseline | gap |
|---|---|---|---|
| **multimodal** (random door) | 42% | 18% | **DP +24 (2.3x)** |
| **unimodal** (greedy 1 door) | 82% | 84% | **tie (-2)** |

This is the clean controlled test. The diffusion advantage is a multimodality x
method INTERACTION: **large on multimodal data, zero on unimodal data.** It rules
out the boring alternative ("diffusion is just a better policy class") -- it ties
the MLP when there is nothing multimodal to model. The win is specifically about
capturing multimodality, exactly as the thesis predicts. (The unimodal task is
also easier -- both ~83% -- which is fine; the KEY is the GAP, not the level.)

## DOSE-RESPONSE (2026-06-26) — the clean story DID NOT hold; needs a control

Tuned the multimodality "dose" via `door_prob` (P(door 0) on the same 2-door env;
0.5 = bimodal, ->1.0 = unimodal), reusing the 2-door encoder. Eval offset 8, n=50:

| door_prob | nominal multimodality | DiffusionPolicy | MLP | gap |
|---|---|---|---|---|
| 0.5 | maximal | 42% | 18% | +24 |
| 0.7 | partial | 42% | 14% | +28 |
| 0.9 | weak    | 40% | 16% | +24 |
| 1.0 | none (always door 0) | 38% | 14% | +24 |

**The gap did NOT shrink toward zero.** Even at door_prob=1.0 (nominally unimodal)
DP beats MLP by +24. This breaks the simple "gap proportional to multimodality"
reading and exposes a CONFOUND in the earlier 2x2: its unimodal cell was the
*1-door* env (DP 82 ~ MLP 84), which differs from the multimodal cell in TWO ways
-- multimodality AND door-count/difficulty. The 1-door tie may just be a ceiling
effect (easy task), not removal of a multimodality advantage.

**BUT door_prob=1.0 is a pathological "unimodal": "always take door 0" forces long
detours to the far door** even when the near door is right there -- so the low MLP
score there could be trajectory-complexity, not (lack of) multimodality. The FAIR
unimodal control is greedy closest-of-2-doors on the SAME 2-door env (natural,
efficient, unimodal). Running now (`greedy2door_control.sbatch`, job 3466532):
- DP ~= MLP there -> thesis holds (door_prob=1.0 was a bad operationalization).
- DP >> MLP there -> the 2-door task favors diffusion for reasons OTHER than
  multimodality; the headline thesis is not cleanly supported and needs rework.

## MEASURED DOSE + ARCHITECTURE CONFOUND (2026-06-28) — thesis NOT yet supported

Two results that undercut the clean story and reframe what's left to test:

**1. The door_prob knob never varied actual multimodality.** Added `policy_goal`
mode: condition on (state_t, state_{t+8}) = (current, synthesized goal, as eval
forms it), target = 8-step action chunk; `residual_bimodality_fraction` = the
literal P(action|state,goal) multimodality. Measured across the dose datasets:

| door_prob | measured multimodality | DP-MLP gap |
|---|---|---|
| 0.5 | 0.142 | +24 |
| 0.7 | 0.142 | +28 |
| 0.9 | 0.141 | +24 |
| 1.0 | 0.130 | +24 |
| greedy(2door) | 0.168 | (n/a) |

Flat ~0.14 everywhere -- toggling door_prob did NOT change measured multimodality.
Cause: the goal = agent's *future position*, which LEAKS which door was taken, so
the door-choice multimodality is conditioned away (in the screen AND in training/
eval). So (a) the dose-response is uninformative (no dose gradient), and (b) the
data is only weakly multimodal (~0.14, near the 0.00 unimodal floor; synthetic
bimodal = 0.98) -- yet DP still beats MLP +24. That is NOT what "diffusion wins
because of multimodality" predicts.

**2. The DP-vs-MLP baseline confounded architecture with objective.**
DiffusionPolicy = 6-layer transformer denoiser; MLPPolicy = 3-layer MLP. The +24
could be "transformer > MLP" with nothing to do with diffusion/multimodality. The
missing control: TransformerMSE (same denoiser backbone, MSE objective) on the
same data. Running (`transformer_mse_control.sbatch`, 3466553):
- TransformerMSE ~= 42 -> the win is ARCHITECTURE; diffusion/multimodality adds
  nothing here. Thesis NOT supported by this experiment.
- TransformerMSE ~= 18 -> the win is the OBJECTIVE (mode-averaging hurts) ->
  thesis holds even at low measured multimodality.

**Whole-investigation verdict (HONEST, current).** The diffusion-*dynamics*
negatives stand. The diffusion-*policy* "win" (42 vs 18) is NOT yet a clean win
for multimodality: the dose knob didn't move measured multimodality, the data is
only weakly multimodal, and the baseline confounded architecture with objective.
The TransformerMSE control is what decides whether any multimodality claim
survives. Do NOT over-claim. (greedy-2door control abandoned: greedy paths are
shorter -> empty training windows -> also an episode-length confound vs the
random-door data; door_prob=1.0 is the length-matched unimodal point and it
already shows +24.)

## E1 VERDICT (2026-06-29) — thesis SURVIVES the two missing controls

> **Label correction (audit 2026-06-29):** the "3-seed" / "seeded" replicates below
> ran with `seed: None` (logs print "runs won't be exactly reproducible"). They are
> **3 independent unseeded reruns**, not fixed-seed replicates — the +10 mean is real
> but not reproducible-as-labeled, and the error bars are run-to-run (±~5/cell), not a
> true CI (eval seed was fixed at 42, so episode-sampling noise is excluded).

Closed the two open confounds with the correct measure + the architecture-matched
baseline + seed replicates.

**(1) door_prob IS a real multimodality dose** (the "leaky knob" was a screen
artifact). The 06-28 measure used `policy_goal`, which leaks the route. Re-screened
with `policy_target` (conditions on the episode's final destination, door-agnostic;
job 3505337):

| door_prob | det_R² | residual_ratio | residual_bimodal_frac |
|---|---|---|---|
| 0.5 | 0.312 | 0.493 | 0.578 |
| 0.7 | 0.320 | 0.467 | 0.596 |
| 0.9 | 0.400 | 0.341 | 0.595 |
| 1.0 | 0.762 | 0.160 | 0.166 |

Multimodality genuinely collapses toward door_prob=1.0. So the flat gap-vs-MLP was
a real tension, not "the dose never moved."

**(2) Architecture confound RESOLVED — and it explained most of the flat gap.**
DiffusionPolicy = transformer denoiser; the right baseline is TransformerMSE (same
backbone, MSE objective). 3-seed eval at the two endpoints (jobs 3466737/3466738),
offset 8, n=50:

| door_prob | mm | DP (3-seed) | TransformerMSE (3-seed) | MLP | **DP−TMSE (pure diffusion)** |
|---|---|---|---|---|---|
| 0.5 (multimodal) | high | 38,38,36 → **37.3** | 28,26,28 → **27.3** | 18 | **+10** |
| 1.0 (unimodal) | low | 32,40,40 → **37.3** | 34,44,30 → **36.0** | 14 | **+1.3 (≈0)** |

**Controlling for architecture, the pure diffusion advantage tracks multimodality:
+10 at door_prob 0.5, ~0 at 1.0.** The original +24-over-MLP was a MIXTURE: at the
pathological unimodal point the transformer *backbone* alone buys ~+22 (TMSE 36 ≈
DP 37 ≫ MLP 14 — a transformer regresses the long-detour data far better than an
MLP), and only ~+1 there is diffusion. At the multimodal point, ~+10 is genuinely
the diffusion OBJECTIVE (mode-averaging fails regardless of backbone: TMSE 27 ≈
MLP 18 ≪ DP 37).

**HONEST VERDICT.** The diffusion-policy win is REAL but smaller than headline:
a clean multimodality×OBJECTIVE interaction of ~+10 (seed-averaged,
architecture-controlled), not +24. The MLP baseline overstated it by conflating
architecture; single-seed n=50 was noisy. Intermediate dose points (p07,p09) are
single-seed and unreliable (p09 TMSE=56 is a tail) — only the seeded endpoints are
trustworthy. greedy-2door control stays abandoned (episode-length confound); the
door_prob dose + TransformerMSE + seeds settle it without it.
SEEDED DOSE CURVE COMPLETE (2026-06-29, jobs 3524802/3524803) — fill-in of 0.7/0.9:

| door_prob | measured mm (res_ratio) | DP (3-seed) | TMSE (3-seed) | DP−TMSE |
|---|---|---|---|---|
| 0.5 | 0.49 | 37.3 | 27.3 | **+10** |
| 0.7 | 0.47 | 38.7 (34,38,44) | 30.7 (28,34,30) | **+8** |
| 0.9 | 0.34 | 37.3 (42,36,34) | 37.3 (36,32,44) | **0** |
| 1.0 | 0.16 | 37.3 | 36.0 | **+1** |

**Clean monotone-ish dose-response: diffusion effect +8..+10 at high multimodality
(0.5,0.7), ~0 at low (0.9,1.0).** The transition sits where measured mm falls off.
Noisy (±~5/cell at n=50) but the multimodal-vs-unimodal separation is consistent.
POLICY HALF OF THE THESIS = SUPPORTED: a real multimodality×OBJECTIVE interaction
of ~+9, architecture-controlled. Optional firm-up: n=100 / more seeds to tighten.

## Artifacts
- Build: `stable_worldmodel/wm/diffusion_policy/`, `solver/diffusion_policy_solver.py`
- Pipeline: `scripts/slurm/policy_mm_pipeline.sbatch`; configs `diffusion_policy{,_eval,_eval_mm}.yaml`
- Checkpoints: `dp_mm/weights_epoch_250.pt`, `mlp_mm/weights_epoch_250.pt`, `lewm_tworoom_mm/weights_epoch_297.pt`
- Result job: re-eval `3387441` (goal-offset sweep).
