# Constructed multimodal TwoRoom — behavioral test (ran aground) — 2026-06-21

**Why.** Structural screens found neither PushT (scripted/human) nor TwoRoom
strongly multimodal. To test the diffusion-vs-deterministic apparatus where
multimodality *provably* exists, I built a designed-multimodal TwoRoom: 2 symmetric
doors + an expert that commits to a RANDOM door per episode (independent of
state). Then trained a diffusion dynamics model on it and ran closed-loop vs the
deterministic LeWM predictor.

## Results

**Screen (after fixing action_noise=2.0 → 0):** with weak conditioning (agent-2D,
or PCA-32 latent, det_R2≈0.1) the screen only measures the *marginal* action
multimodality (high for both mm and greedy) — it cannot isolate the door-choice
multimodality because the dataset stores **no target column** (only agent pos +
latent), so I can't condition on (agent, target). The construction is multimodal
*by design* (verifiable from code); the screen just can't see it here.

**Behavioral closed-loop (2-door env): diffusion 0% vs deterministic 0%.**
A broken eval, not a finding. The deterministic predictor solves *standard*
TwoRoom at 58%, so 0% means the setup is broken: **`lewm_tworoom` was trained on
1-door data (`tworoom_expert.lance`, options=null), so the 2-door eval scenes are
out-of-distribution for the encoder** → uninformative latents → the goal-distance
score is noise → the planner can't reach the goal. Both models share the encoder,
so both fail.

## What it means

- **The mm-TwoRoom tests POLICY multimodality, but the D-MPC pipeline is a
  DYNAMICS model.** TwoRoom dynamics are deterministic given the action, so a
  diffusion *dynamics* model gains nothing from multimodal *demonstrations* — the
  experiment was conceptually mismatched from the start (expected null even if the
  eval worked).
- **Encoder OOD lesson:** any constructed env variation (extra doors, etc.) must
  be reflected in the *encoder's* training data, not just the diffusion/eval. A
  valid mm-TwoRoom test needs the encoder retrained on 2-door data first
  (~300 ep), then re-cache → re-train diffusion → re-eval. High cost for an
  expected-null, conceptually-mismatched test → not pursued.
- **The screen needs strong, low-dim conditioning** that captures the
  deterministic structure to isolate residual (e.g. door-choice) multimodality.
  Agent-only / weak-latent conditioning measures the marginal and can't
  discriminate. Recorded as a method limitation.

## Bottom line for the whole investigation
This doesn't change the conclusion; it sharpens it. Diffusion-for-**dynamics**
needs multimodal **dynamics** (stochastic / partially-observed envs), which none
of these deterministic-physics benchmarks have — so the D-MPC dynamics model
loses (TwoRoom 58 vs 46/30; PushT 16 vs 8). The multimodality that *does* exist
(PushT human demos; constructed mm-TwoRoom) lives in the **policy**, where a
diffusion **policy** (the dropped proposal ρ) — not the dynamics pipeline — is the
right tool. The genuinely decisive next experiments are larger builds: (a) a
diffusion **policy** on multimodal demos, or (b) a **stochastic-dynamics** domain
for the dynamics question.

## Artifacts
- `stable_worldmodel/envs/two_room/expert_policy.py` (`stochastic_door`)
- `scripts/data/collect_tworooms_multimodal.py`, `scripts/plan/config/tworoom_mm_diffusion.yaml`
- `scripts/slurm/{train_diffusion_tworoom_mm,plan_tworoom_mm}.sbatch`
- Datasets: `tworoom_mm0.lance`, `tworoom_mm0_latent.lance` (1-door encoder — OOD for 2-door eval)
- `scripts/data/multimodality_diagnostic.py` now supports `--cond-cols`, overfit guard, `--self-test`
