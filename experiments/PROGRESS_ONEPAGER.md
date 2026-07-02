# When Do Generative Models Beat Deterministic Ones in Latent Planning?

**Lucas Zheng — progress writeup**

## Question
Diffusion models are increasingly used for both **world models** (predict dynamics)
and **policies** (predict actions). But they are slower and harder to train than a
deterministic regressor. *When is that cost actually worth paying?* This project
isolates one answer: **generative models help exactly when the distribution they
model is multimodal — and not otherwise.** We test this on both sides of a
model-based pipeline (the dynamics and the policy) in a controlled way.

## Setup
All experiments operate in a frozen self-supervised latent space (LeWM/SIGReg
encoder, 224×224 image → 192-d latent). Two model families are compared in that
space, each against a deterministic counterpart:
- **Diffusion *dynamics* world model** + CEM planning (latent MPC), vs a
  deterministic next-latent predictor.
- **Diffusion *policy*** (samples an action chunk from latent history + goal), vs
  MSE-regression baselines.
Domains: **TwoRoom** (2D visual navigation; a wall with doors) and **PushT**
(contact manipulation). TwoRoom is instrumented with a *controllable multimodality
knob*: a 2-door wall + an expert that commits to a random door per episode
(`door_prob`), so the demonstration multimodality can be dialed 0→max while the
task is held fixed.

## Key findings

**1. A diffusion *world model* does NOT beat a deterministic one here.**
Closed-loop success (latent MPC): TwoRoom **58%** (deterministic) vs 46%/30%
(diffusion); PushT **16%** vs 8%. Reason, confirmed by a k-NN multimodality
diagnostic: these dynamics are near-**deterministic given the action**
(`p(z'|z,a)` residual ≈ Gaussian) — there is no multimodality for a generative
model to capture, so it only adds sampling variance.

**2. On the policy side, diffusion does NOT improve mean success — the earlier
"+10" did not survive proper seeding (2026-07-01 final).** With real seeds
(`seed_everything`; the prior "3-seed" runs were unseeded), 8 training seeds per
cell, and 2 eval seeds × n=50 per run (episode-sampling variance included), the
architecture-controlled 2×2 is:

| 8 seeds × n=100, mean (cross-seed sd) | Multimodal (door_prob 0.5) | Unimodal (door_prob 1.0) |
|---|---|---|
| Diffusion Policy | 34.9 **(3.6)** | 32.5 **(2.8)** |
| Transformer-MSE  | 33.5 **(7.3)** | 31.3 **(2.7)** |
| mean gap | +1.4 (≈0) | +1.25 (≈0) |

The banked +10 was three unseeded TMSE runs landing low on a baseline whose true
cross-run sd is ~7, read through a fixed-eval-seed design that hid both variance
terms. Corroborating null: failed episodes show **no mode-averaging signature**
(TMSE failures at the wall between doors 10–14% ≈ DP 9–11%; both mostly exhaust
the budget elsewhere).

**3. The real multimodality×objective interaction is in the VARIANCE, not the
mean.** The MSE objective's cross-training-run sd nearly triples on
multimodal-demonstrator data (2.7 → 7.3) while diffusion's barely moves
(2.8 → 3.6). This coheres with a direct fit measurement: multimodal-demonstrator
action targets are ~1.6× harder to regress (TMSE val MSE 0.39–0.43 vs 0.22–0.27,
train≈val) — the heterogeneity corrupts the MSE loss landscape enough to
destabilize training runs, but closed-loop replanning (receding horizon) forgives
the averaged chunks, so the mean doesn't move. Honest claim: *on
multimodal-demonstrator data, a diffusion policy is not better on average — it is
far more reliable across training runs.* (Sampling-vs-training attribution of the
stabilization: probe in flight.)

**4. Temporal/aleatoric multimodal DYNAMICS exist and are now instrumented
(2026-07-01).** Two screened routes where `p(next|state,action)` is genuinely
multimodal: (a) an **intrinsic per-step "slip"** TwoRoom variant — residual
bimodality 0.11 (slip 0) → 0.98 (slip ≥ 2), pervasive and dose-gated — with the
diffusion-vs-same-backbone-MSE dynamics comparison + closed-loop D-MPC verdict
running; (b) **K-step temporal abstraction** — `p(Z_{t+K}|Z_t, Σa)` bimodality
rises monotonically with K (0.06 → 0.135 at K=8; action-free proposer cell ~0.93),
greenlighting the H-JEPA level-2 track.

## Takeaway
On the **dynamics** side the negative half of the law holds: diffusion loses where
the modeled conditional `p(z'|z,a)` is near-deterministic (TwoRoom, PushT) — *right
method, wrong side of the problem*. On the **policy** side the once-headline "+10"
did **not** survive statistical rigor (real seeds, 8 runs/cell, eval-seed variance):
diffusion and same-backbone MSE tie on mean success at both dose endpoints. What
survives — and is arguably more interesting — is a **variance law**: *the MSE
objective becomes unreliable across training runs exactly when the demonstrator is
multimodal (sd 2.7 → 7.3), while the diffusion objective stays stable (≈3)*,
consistent with directly-measured fit-error corruption that closed-loop replanning
otherwise forgives. The strict mean-level "generative beats deterministic iff the
modeled conditional is multimodal" now rests entirely on the **dynamics** half —
which finally has screened-multimodal domains to test on.

## Status & next step
**Closed:** policy side (P0a) — no mean effect, variance interaction confirmed at
both endpoints; dynamics-on-deterministic-physics negatives — explained.
**RESOLVED (2026-07-02): the dynamics advantage is real, multimodality-
independent, and fully attributed.** Fair test (same backbone, same post-hoc
budget, 3 seeds × n=100): diffusion beats MSE dynamics +6 on screened-bimodal
AND +10 on the deterministic control — not mode-capture. Attribution: the gap
survives 1-step inference (not iterative-refinement compute), and an x0 cell
shows **noise-curriculum regression @1 step is the best model of the program
(78.3/75.3 vs clean-MSE 64.3/65.7)** while DDIM sampling actively hurts it —
also resolving the old open-loop-anti-ranking puzzle (a sampling penalty).
**Final claim: every surviving diffusion advantage here is a training-time
property of the denoising objective — better one-shot dynamics regressors and
variance-stabilized policies; sampling and multimodality contribute nothing.**
Full chain: `experiments/2026-07-01_p0a_mechanism_loop.md` (15 iterations).
**Next arcs (open):** cross-domain test of the noise-curriculum recipe
(PushT); H-JEPA level-2 proposer cell (last venue for a true multimodality
effect); P2/JEDI re-rationalized as denoising-as-latent-shaping.
