# Ideas — all research tracks (outline)

One section per idea: what it is, why, current status. **Per-idea task lists live
in `todos/<idea>.md`** — this doc is the map, not the checklist. Living doc; edit
freely. Priority: ⭐⭐ now / ⭐ high / ◐ medium / ○ someday.

## The thesis (through-line of the whole project)

Diffusion beats a deterministic/MSE model *iff* the conditional distribution it
models has hard, separated, frequent modes that are present in the data — given
what the model actually conditions on. In these robotics tasks that multimodality
has so far lived in the **policy** `p(a|s,g)`, not the **dynamics** `p(s'|s,a)`.

**Status snapshot (2026-07-01):**
- **Banked:** diffusion-*dynamics* loses on deterministic physics (TwoRoom 58% det
  vs 46/30% diff; PushT 16% vs 8%), explained by the validated multimodality screen
  (near-unimodal everywhere, incl. multi-human PushT). Diffusion-*policy* beats a
  same-backbone TransformerMSE **+10** on multimodal-demonstrator TwoRoom,
  dose-dependent (collapses to ~+1 at door_prob 1.0), architecture-controlled.
- **Open liabilities:** the +10 is NOT test-time conditional multimodality (the
  matched conditional is unimodal, ~0.006 — 2026-06-30 attribution screen), so the
  mechanism is unisolated; the dynamics half of the thesis has no domain yet
  (bounded-nav POMDP route exhausted 2026-06-29); prior "3-seed" runs were
  unseeded; intermediate dose points noisy at n=50.
- Full history: `experiments/` logs + `experiments/PROGRESS_ONEPAGER.md`.

## Standing methodology rules (earned the hard way)
- Select world models / policies on **closed-loop reward** — never eps-loss or
  open-loop MSE (they anti-rank).
- **Run the multimodality screen before any diffusion-vs-deterministic run.**
  Near-unimodal target => nothing to win.
- Encoder must be **in-distribution** for the eval env (OOD latents -> 0%).
- Eval `goal_offset_steps` must match the policy's training goal window (~8).
- "Multimodal" is a property of (dataset x conditioning x measurement), not of
  the env name. Verify with the screen, don't assume.

---

## ⭐⭐ Diffusion vs deterministic — the thesis project → `todos/diffusion.md`
ALL the diffusion work, one track, staged:
- **P0a — isolate the +10 mechanism (current work).** The one positive result: DP
  beats same-backbone TransformerMSE +10 on multimodal-demonstrator data,
  dose-dependent. But the clean "policy samples one mode of a multimodal
  conditional at test time" story is **falsified** — the matched conditional is
  unimodal. Cheap probes (failure-position clustering, mm0-vs-p10 fit error,
  eval-time conditioning screen) + seeded n=100 endpoints decide the paper's
  mechanism claim.
- **P0 — real measured-multimodality dose axis** (door count, screen-gated) to
  turn the policy result from illustration into proof.
- **P1 — multimodal-DYNAMICS domain (the missing half).** Bounded-nav POMDP route
  is DEAD (hidden doors det_R² 0.966; hidden drift self-defeats via boundary
  clamping). Surviving routes: intrinsic stochastic transitions (bimodal "slip"
  with a probability dose knob mirroring door_prob) and K-step temporal
  abstraction (shared rung with H-JEPA below). If diffusion-dynamics wins on a
  screened domain, the thesis becomes symmetric — the highest-value missing cell.
- **P2 — JEDI-variant end-to-end** (unfreeze the encoder, joint multi-step window
  denoising — the original research idea). Blocked on P1.
- **Backlog** — D-MPC fairness controls (training parity), co-trained-encoder
  track, parallel decoder, planners (iCEM/MPPI), sampling sweeps, domains to port
  (OGBench-Cube, Reacher), and why open-loop MSE anti-ranks closed-loop.

## ⭐ H-JEPA — stacked JEPAs / temporal hierarchy → `todos/hjepa.md`
Train a second JEPA over frozen LeWM latents predicting K steps ahead → 2-level
temporal hierarchy (LeCun's H-JEPA), hierarchical CEM planning. Doubles as the
temporal-abstraction route to multimodal dynamics: `p(Z_{t+K}|Z_t,a)` can be
multimodal where 1-step is deterministic (bimodal_frac ~ K/T). Open science
question: does SIGReg/isotropy compose up an abstraction hierarchy? Rung-1 gate
(K-step screen) is built and cheap. **Full design:
`experiments/2026-07-01_hjepa_design.md`; plan review (corrections, extensions,
paper skeleton, revised ladder): `experiments/2026-07-01_hjepa_plan_review.md`.**
Key review corrections: it's the *first clean shot at the strict thesis* (not "the
proven +10 lifted" — 06-30 falsified that mechanism at level 1); rung 1 gets a
three-outcome table (Σa may be near-sufficient in an integrator env); the
action-free subgoal-proposer cell (`coarse=none`) is the planning-relevant
det-vs-generative contrast. Future paper / could fold into the JEDI-variant track.

## ◐ Encoder / LeWM improvements → `todos/encoder-lewm.md`
Fix the early-training val-loss blow-up (~35 wasted epochs; SIGReg early dynamics
/ BatchNorm projector suspects); encoder-quality → planning ablation.

## ⭐ Adversarial goal-hijacking of world models → `todos/goal-hijacking.md`
Future paper (collaborative, needs people + GPUs): small adversarial perturbation
on the object makes the planner solve the WRONG goal (PushT at wrong rotation).
The interesting math is the bi-level optimization — attacking *through* a planning
loop. Reuses our latent-MPC harness; pairs with H-JEPA (attack at different
abstraction levels).

## ◐ Infra / tooling → `todos/infra.md`
Login-node guardrail hook, h5→lance conversion, one-command domain porting,
generalize the checkpoint converter.
