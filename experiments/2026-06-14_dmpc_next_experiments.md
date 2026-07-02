# D-MPC latent — next experiments (2026-06-14)

Driven by the findings in `2026-06-14_dmpc_results.md`. The governing metric is
**open-loop rollout MSE vs. the LeWM deterministic predictor (0.053)**, then
closed-loop TwoRoom success rate. eps-loss is deprecated as a selection metric.

---

## E1 — Parametrization sweep ✅ DONE — hypothesis confirmed

**Result:** open-loop MSE eps 4.13 → **v 0.647 → x0 0.449**; sample variance
8.5 → ~0.4. The eps→√ᾱ amplification was the cause. **x0 is the winner.** Both x0/v
are sane but still ~8–12× above the LeWM deterministic baseline (0.053), so the
open-loop crown stays with the deterministic predictor — making E3 (closed-loop, does
stochastic sampling help *planning*?) the decisive test. Closed-loop on v/x0/LeWM is
running now. Original design preserved below.

---

### E1 (original design) — Parametrization sweep (array 3222443, plan 3222444)

**Hypothesis.** The eps→x0 division by √ᾱ (amplification up to 2029×) is what breaks
rollout. `x0`- and `v`-prediction remove that division and should drop open-loop MSE
toward the LeWM baseline *without* any other change.

**Design.** One variable. Three matched runs on `tworoom_latent_e300.lance`, 300
epochs, lr 3e-4, 20 DDIM steps, depth 6, clamp ±6:
| task | prediction_type | checkpoint |
|---|---|---|
| 0 | `v` | `latent_diffusion_tworoom_v300/` |
| 1 | `x0` | `latent_diffusion_tworoom_x0300/` |
| 2 | `eps` (control) | `latent_diffusion_tworoom_eps300/` |

Each run does open-loop eval; the closed-loop TwoRoom D-MPC eval (CEM N=64, K=4, 50
episodes) is chained on the `v` checkpoint.

**Decision rules.**
- If `v`/`x0` open-loop MSE ≪ 2.85 (ideally < ~0.3, approaching 0.053) and the
  per-step profile **rises** like LeWM's → parametrization confirmed as the fix.
  Promote the winner; proceed to E3 (closed-loop is the headline).
- If `v` and `x0` are *also* ≈2.8 and flat → parametrization is **not** the (whole)
  cause; the conditioning pathway is suspect. Go to E2 immediately.
- eps control should reproduce ≈2.85, anchoring the comparison.

**Status:** queued behind the `rmax_v12` CPU-QOS limit; ~10 min/task once it starts.
Results will be appended here and to the results doc.

---

## E2 — Conditioning diagnostic (CONDITIONAL on E1 not fixing it)

**Question.** Does the denoiser actually use history + actions, or has it collapsed
to an unconditional prior?

**Design (cheap, no training; eval-only on an existing checkpoint).** At a fixed low
noise level (e.g. t≈10, ᾱ≈0.95 so reconstruction is well-posed), compare x0
reconstruction MSE under three conditioning regimes:
1. true history + true actions,
2. true history + **shuffled** actions (break action–outcome correspondence),
3. **zeroed** history + true actions.
If (1) ≈ (2) ≈ (3), the model ignores conditioning → architectural bug (e.g. action
embedding scale, type/positional embeddings, or history tokens not attended). If (1)
≪ (2),(3), conditioning works and the problem is purely the sampler/parametrization.

**Why it matters.** Distinguishes "fix the sampler" from "fix the model wiring" — two
very different follow-ups. Implement as a small `scripts/diagnostics/` eval.

---

## E3 — Closed-loop D-MPC headline ✅ DONE — negative result

Ran v300, x0300, and the LeWM deterministic predictor through `eval_wm.py`
(`tworoom_diffusion`, CEM N=64, 30 iters, horizon 8, 50 episodes).

| policy | success rate | eval time |
|---|---|---|
| **LeWM deterministic predictor** | **58.0%** | 78 s |
| diffusion v300 | 46.0% | 655 s |
| diffusion x0300 | 30.0% | 666 s |

**Outcome:** the deterministic predictor wins and is 8× faster. Open-loop MSE
anti-ranked closed-loop (x0 best open-loop, worst planner). On TwoRoom, latent D-MPC
with diffusion is not worth it. See results doc §3c. This **redirects** the plan: the
remaining TwoRoom knobs (E4) are low-value polish; the real next step is E6 (a
multimodal domain where a sampler can actually help). E2 stays relevant only as a
sanity check — but the sample-variance collapse (8.5→0.4) already shows conditioning
works, so E2 is largely answered and **deprioritized**.

---

## E4 — Sampler / schedule robustness (only if E1 helps but doesn't reach baseline)

Secondary knobs, swept **after** parametrization is fixed, one at a time:
- **Inference steps**: {10, 20, 50}. More steps help most once x0/v is stable
  (they barely helped under eps: 2.67→1.94 from 10→20). 
- **eta (DDIM→DDPM)**: {0.0, 0.25, 0.5}. Slightly stochastic sampling can improve
  sample calibration; interacts with K-averaging in planning.
- **Schedule terminal**: cap ᾱ_min (offset cosine / "zero-terminal-SNR"-style floor)
  so √ᾱ never gets pathological — an alternative/complement to changing
  parametrization. Lower priority if v-pred already works.
- **K (dynamics samples in scoring)**: {1, 4, 8}. Only meaningful once individual
  samples are informative; with var≈8.5 today it's noise-averaging garbage.

---

## E5 — Longer encoder ablation (parked — moot for now)

Whether a worse encoder (epoch-100) changes planning only matters if we keep pursuing
diffusion on TwoRoom, which E3 says we shouldn't. Parked.

---

## E6 — **NEW PRIORITY:** test the D-MPC premise on multimodal dynamics

E3 showed diffusion loses where dynamics are near-deterministic — the worst case for a
sampler. The honest test of the whole approach is a domain where a deterministic
predictor is *forced* to fail by averaging over modes:

- **Candidate domains** (action_dim/goal-conditioned, already in repo or close): PushT
  (contact dynamics → multimodal push outcomes — this was the *original* D-MPC target
  in the project plan), or a stochastic/branching maze.
- **Hypothesis:** where true next-latent is multimodal, the deterministic predictor's
  MSE-optimal output is a blurred mode-average that plans poorly, while diffusion
  samples commit to individual modes → diffusion should now *win* closed-loop.
- **Design:** reuse the entire fixed pipeline (cache_latents → latent_diffusion with
  `prediction_type=v` or `x0` → eval_wm closed-loop) against the deterministic-predictor
  baseline. The infrastructure is done and validated; only the dataset/env changes.
- **Diagnostic to include:** measure next-latent multimodality directly (e.g. variance
  of expert next-latents conditioned on (state, action) bins) to confirm the domain
  actually exercises the diffusion advantage before reading too much into the result.

This is the experiment that decides whether latent D-MPC is worth pursuing at all.

---

## Sequencing (updated after E1/E3)
```
E1 parametrization ✅ (x0/v fix the bug)  ──→  E3 closed-loop ✅ (deterministic wins on TwoRoom)
                                                      │
                                                      ▼
                              E6 multimodal domain (PushT) ── the decisive test
                                                      │
                                   win? ──yes──→ E4 polish (steps/eta/K) on that domain
                                        └──no───→ latent D-MPC with diffusion is not worth it; stop
```
TwoRoom-only knobs (E2 diagnostic, E4 polish, E5 encoder) are deprioritized — they
refine a setting we've shown the simpler baseline already wins. Compute is cheap
(≤15 min/training run on an L40S); GPU-partition queue time is the bottleneck.
