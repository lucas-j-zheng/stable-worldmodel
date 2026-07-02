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

**2. The multimodality lives in the *policy*, not the dynamics.**
`p(action | obs)` is multimodal (many valid behaviors); `p(next-state | s,a)` is
not (physics). So the generative model belongs on the policy side. A diffusion
*policy* on multimodal (random-door) TwoRoom beats both MSE baselines:
**Diffusion ~37%** vs **Transformer-MSE ~27%** vs **MLP-MSE 18%** (n=50, 3 unseeded reruns).

**3. The advantage is the *objective*, not the architecture.**
Holding the transformer backbone fixed and swapping diffusion→MSE drops 37→27 in the
multimodal cell — that **+10 is the diffusion objective**. The backbone is *not*
free, though: at the unimodal endpoint the transformer alone lifts MLP 14→TMSE 36,
while the objective there adds ~0. So the diffusion objective is the entire lift
*where multimodality is present*, and ~nothing where it isn't. Mechanism: MSE
policies average the two routes → drive into the wall between the doors → fail; the
diffusion policy commits to one mode → succeeds.

**4. The advantage tracks the *demonstrator's* multimodality — but it is NOT
test-time conditional multimodality (2026-06-30 reframe).** Holding the transformer
backbone fixed and varying only the data via the `door_prob` dose (success %, offset
8, n=50, mean over 3 unseeded reruns; ±~5/cell run-to-run noise; architecture-
controlled, DP vs same-backbone TransformerMSE):

| | Multimodal (door_prob 0.5) | Unimodal (door_prob 1.0) |
|---|---|---|
| Diffusion Policy | **37.3** | 37.3 |
| Transformer-MSE  | **27.3** | 36.0 |
| Diffusion − MSE  | **+10** | **+1 (tie)** |

The gap tracks the dose **measured under *destination* conditioning** (`policy_target`
residual_bimodal 0.58 → 0.17 across door_prob 0.5 → 1.0). BUT under the conditioning
the policy actually uses — history + the +8 *latent* goal — the action conditional is
**unimodal**, and multimodal data is indistinguishable from unimodal (residual_bimodal
~0.006 both; jobs 3575246/3575527). So the win is real and dose-dependent, yet the
clean "the policy samples one of several modes *at test time*" story is **falsified
here**: the relevant multimodality is in the training *demonstrations*, not the
conditional the policy models. Mechanism not yet isolated — see
`experiments/2026-06-30_attribution_screen.md`.

## Takeaway
On the **dynamics** side the law holds cleanly: diffusion loses where the modeled
conditional `p(z'|z,a)` is near-deterministic (TwoRoom, PushT) — *right method, wrong
side of the problem*. On the **policy** side a diffusion policy genuinely beats a
same-backbone MSE policy (+10) on multimodal-demonstrator data, and the gap scales
with the `door_prob` dose — BUT the matched-conditioning screen shows the policy's own
conditional is unimodal, so the strict "generative beats deterministic iff the modeled
conditional is multimodal" statement is **not** what's operating here. The honest claim
is weaker and about the training distribution: *a diffusion policy beats MSE on data
from a multimodal demonstrator, with a dose-dependent gap*; the mechanism (training-time
mode-averaging vs test-time sampling) is open.

## Status & next step
**Confirmed (architecture-controlled + 3 unseeded reruns, ±~5/cell):** the policy side
is positive and dose-dependent; the dynamics side is negative and explained.
**Open on the policy side:** isolate WHY +10 persists when the matched conditional is
unimodal (training-time mode-averaging? train/eval goal mismatch? calibration?) — cheap
next step before any stronger multimodality claim. **Missing cell — highest-value
experiment:** a domain with genuinely *multimodal dynamics* (intrinsic stochastic
transitions, not bounded-nav POMDP — that track is exhausted) where a diffusion **world
model** wins, with the same dose knob — the symmetric half of the law.
