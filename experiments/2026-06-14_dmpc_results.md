# D-MPC in LeWM latent space — results & implications (2026-06-14)

Pipeline under test: frozen LeWM/SIGReg encoder (d=192) → cache latents → train a
latent-trajectory **diffusion** dynamics model → plan with CEM (D-MPC). Domain:
TwoRoom. The headline question is whether the diffusion world model is good enough
to plan with. Short answer so far: **not yet — and we found why.**

---

## 1. LeWM encoder: longer training helped, and it converged

Re-ran the TwoRoom LeWM with 300 epochs + 5% warmup (was 100 epochs / 1% warmup).

| Validation metric | 100 epochs | 300 epochs |
|---|---|---|
| pred_loss | 0.0288 | **0.0133** |
| total loss | 0.372 | **0.244** |
| sigreg_loss | 3.81 | **2.57** |

- ~2× better on every metric; validation pred_loss is flat over the last ~25
  epochs (0.0177 → 0.0133), so this genuinely **converged** — 600 epochs would buy
  little. 300 was the right length.
- The 5% warmup did **not** fix the early-training loss blow-up (val loss spikes to
  ~10³ around epoch 12, recovers by ~epoch 36 in both runs). The instability is not
  a warmup problem — likely SIGReg early dynamics or the BatchNorm projector. Costs
  ~35 epochs but recovers; left alone for now.
- Checkpoint: `lewm_tworoom/weights_epoch_300.pt`. This is the encoder used for all
  latent caching below (`tworoom_latent_e300.lance`).

**Cached-latent sanity:** mean 0.012, std 1.025, |max| 5.09, per-dim std ∈ [0.72,
1.39]. SIGReg isotropy holds — latents are ≈N(0, I), so no normalization is applied
and the cosine-schedule terminal distribution matches the data marginal *by design*.
(This turns out to matter — see §3.)

---

## 2. The diffusion world model does not roll out usefully

Open-loop eval compares, per future step, the MSE of predicted latents vs. the true
encoded latents, for the deterministic LeWM predictor (baseline) and the diffusion
model (mean of K=8 samples).

| Model / variant | train eps-loss | open-loop rollout MSE | vs. baseline |
|---|---|---|---|
| **LeWM deterministic predictor** | — | **0.053** | 1× (reference) |
| diffusion, 25 ep (base, eps) | 0.30 | 2.67 | 50× worse |
| diffusion, 25 ep (lr3e-4, eps) | ~0.30 | 2.51 | 47× worse |
| diffusion, **500 ep** (eps) | **0.10** | **2.85** | 54× worse |

Random chance for N(0,1) latents is MSE ≈ 2.0 (E‖x−y‖² for two independent draws).
The diffusion model sits **at or above chance.**

### The decisive observation
**20× more training (650 → 13k steps) cut eps-loss 3× but left rollout MSE
unchanged (and slightly worse).** This rejects the undertraining hypothesis outright.
The eps-loss and the thing we care about (rollout quality) are decoupled.

### Per-step profile — the smoking gun
```
step:        1     2     3     4     5     6     7     8
LeWM MSE:  0.033 0.042 0.046 0.049 0.059 0.062 0.067 0.068   ← rises (errors compound)
diff MSE:  2.95  2.80  2.84  2.92  2.80  2.92  2.73  2.83    ← FLAT from step 1
diff var:  8.51  8.55  8.62  8.48  8.61  8.46  8.70  8.49    ← ~8.5, >> data var 1.0
```
- LeWM error **rises monotonically** over the horizon — correct world-model
  behavior (one-step errors accumulate).
- Diffusion error is **flat from the very first step** and the per-sample variance
  (~8.5) is far wider than the data (~1.0). The samples are over-dispersed,
  condition-independent noise. The model has learned a marginal-ish distribution
  and ignores the history + actions it is conditioned on at sampling time.

---

## 3. Root cause: eps-parametrization is numerically broken on this schedule

This is not a capacity or training-time problem; it is a sampling/parametrization
problem baked into the cosine schedule.

The DDIM sampler recovers the clean latent from predicted noise via
`x0 = (x_t − √(1−ᾱ)·ε̂) / √ᾱ`. On our cosine schedule the terminal ᾱ is tiny, so
that `1/√ᾱ` is enormous:

| t | ᾱ(t) | √ᾱ(t) | eps→x0 error amplification √((1−ᾱ)/ᾱ) |
|---|---|---|---|
| 99 | 2.4e-07 | 0.0005 | **2029×** |
| 90 | 1.95e-02 | 0.140 | 7.1× |
| 80 | 8.5e-02 | 0.292 | 3.3× |
| 50 | 0.478 | 0.692 | 1.04× |
| 5 | 0.989 | 0.995 | 0.11× |

With the model's eps RMS error ≈ √0.10 ≈ 0.32, the implied x0 estimate at the first
DDIM step has error ≈ **640** — saturating the ±6 clamp. The first sampling step,
which seeds the entire trajectory, is therefore pure clamp-bounded garbage, and the
few remaining DDIM steps cannot recover. That is exactly the observed signature:
over-dispersed (var ≈ 8.5, i.e. many dims pinned near the ±6 clamp), flat across the
horizon. The ironic part: the very SIGReg property we exploited to justify the
cosine terminal (marginal ≈ N(0, I)) is what pushes ᾱ→0 and breaks eps.

**Why eps-loss looked fine anyway:** the eps target is ≈N(0,1) at every timestep, so
a model can drive the average eps-MSE down mostly on high-noise timesteps (where
predicting ε is nearly trivial) while contributing almost nothing usable to the
low-ᾱ reconstruction that sampling actually depends on. eps-loss is a poor proxy
for rollout quality on this schedule.

---

## 3b. Fix confirmed: x0/v parametrization (E1 sweep)

Re-trained three matched runs (300 epochs, lr 3e-4, 20 DDIM steps) changing only
`prediction_type`. The fix is dramatic and exactly as predicted:

| variant | open-loop rollout MSE | sample variance | vs. eps |
|---|---|---|---|
| eps (control) | 4.13 | 14.5 | — (broken) |
| **v** | 0.647 | 0.50 | **6.4× better** |
| **x0** | **0.449** | **0.34** | **9.2× better** |
| LeWM baseline | 0.053 | — | reference |

- Sample variance collapsed **8.5 → ~0.4** (samples went from over-dispersed noise to
  informative draws). This is the direct signature of the eps→x0 amplification being
  removed.
- The eps control at this length is actually *worse* (4.13) than the earlier eps runs
  — more eps training drives the model further into the degenerate regime, reinforcing
  that eps-loss is anti-correlated with rollout quality here.
- Per-step profiles (now sane magnitude, roughly flat rather than steeply rising):
  ```
  v   : 0.77 0.62 0.58 0.61 0.61 0.59 0.64 0.76
  x0  : 0.50 0.43 0.41 0.48 0.44 0.41 0.38 0.54
  ```
  The slight U-shape (first/last steps worst) is a mild edge effect, not error
  compounding — the model is conditioning but not yet matching the deterministic
  predictor's accuracy.

**Caveat that now drives the headline question:** x0/v are fixed and sane, but still
**~8–12× above the LeWM deterministic predictor (0.053)** open-loop. So diffusion does
*not* beat the deterministic model at one-step latent prediction. Whether its
stochasticity nonetheless helps *planning* is exactly what the closed-loop comparison
(diffusion vs. deterministic predictor vs. random) decides — see §3c.

## 3c. Closed-loop headline (E3): the deterministic predictor wins

Closed-loop TwoRoom goal-reaching, 50 episodes, identical harness (CEM N=64, 30 CEM
iters, horizon 8, receding 5). This is the metric that actually matters.

| policy (dynamics model) | **success rate** | open-loop MSE | eval time |
|---|---|---|---|
| **LeWM deterministic predictor** | **58.0%** | 0.053 | 78 s |
| diffusion v300 | 46.0% | 0.647 | 655 s |
| diffusion x0300 | 30.0% | 0.449 | 666 s |

Two hard findings:

1. **D-MPC's stochastic-sampling premise does not pay off on TwoRoom.** The
   deterministic latent predictor plans *better* (58% vs 46/30%) and runs **~8×
   faster** (78 s vs ~660 s — diffusion does 20 DDIM steps × K samples per CEM
   candidate). The whole point of the diffusion world model is to capture multimodal
   dynamics; TwoRoom dynamics are near-deterministic in latent space, so sampling adds
   cost and variance without benefit.

2. **Open-loop MSE anti-ranks closed-loop performance.** x0 was the *best* open-loop
   (0.449) but the *worst* planner (30%); v was worse open-loop (0.647) but a better
   planner (46%). Mean-sample MSE rewards conservative, mean-reverting predictions —
   x0's tighter samples (var 0.34 vs v's 0.50) minimize average error but give CEM too
   little diversity to discriminate good action sequences, and a low-variance biased
   predictor can systematically mislead the planner. **Lesson: select world models on
   closed-loop reward, never on open-loop reconstruction MSE.** (This mirrors §2's
   eps-loss lesson one level up: every cheaper proxy we tried — eps-loss, then
   open-loop MSE — failed to rank the thing we care about.)

**Verdict.** On TwoRoom, latent D-MPC with a diffusion forward model is **not worth
it** versus the deterministic LeWM predictor. The diffusion path is now *correct*
(the parametrization bug is fixed, samples are informative) but does not beat the
simpler, faster baseline. A fair test of the diffusion premise needs a domain with
genuinely **multimodal / stochastic** dynamics, where a deterministic predictor must
blur and a sampler can commit — TwoRoom is not that domain.

## 4. Implications

1. **Select on closed-loop reward only.** Two cheaper proxies both failed to rank the
   real metric: eps-loss was *anti*-correlated with rollout quality (§2), and open-loop
   MSE was *anti*-correlated with planning success (§3c). Each layer of proxy lied.
2. **The parametrization fix was necessary but not sufficient.** `x0`/`v`-prediction
   (Salimans & Ho 2022) made the diffusion model *correct* — informative samples, sane
   variance — a real ~6–9× open-loop gain. But correct ≠ competitive: it still loses
   closed-loop to the deterministic predictor.
3. **On TwoRoom, prefer the deterministic LeWM predictor.** Higher success (58%),
   8× cheaper, far simpler. Latent D-MPC with diffusion is not justified here.
4. **The diffusion premise is untested, not refuted.** TwoRoom dynamics are
   near-deterministic in latent space — the worst case for a sampler. The experiment
   that would actually test D-MPC is a domain with multimodal/stochastic dynamics. Until
   then, "diffusion didn't help" means "didn't help *where a sampler can't help anyone*."
5. **The clamp was a correct but insufficient band-aid.** It stopped the MSE from
   being 10⁵, but ±6 garbage is still garbage. With x0/v the clamp rarely binds.

---

## 5. Bugs fixed along the way (committed)

- **`eval_latent_diffusion.py`** crashed at config dump on an unresolved
  `${output_model_name}` wandb interpolation — defined the key in the eval config.
- **`eval_wm.py`** (closed-loop) crashed twice on lance datasets: (a) episode-column
  detection fell through to `ep_idx`; (b) `get_row_data(...)['episode_idx']` —
  `get_row_data` only returns *loaded* keys and lance hides the index columns. Both
  now go through `get_col_data`, which loads index columns regardless.
- **`tworoom_diffusion.yaml`** referenced legacy `pos_agent`/`goal_pos_agent` state
  keys absent from the lance dataset; `_apply_callables` **silently skips** missing
  keys, so every episode would have evaluated from a default start/goal and produced
  meaningless "results." Switched to `state`/`goal_state` and the lance dataset.

Without these fixes the closed-loop numbers in §3c would have been artifacts.

---

## Artifacts
- Encoder: `lewm_tworoom/weights_epoch_300.pt`
- Latents: `tworoom_latent_e300.lance`
- Diffusion (eps, undertrained/long): `latent_diffusion_tworoom_{base,lr3e4,long}/`
- Open-loop eval JSON: `<ckpt_dir>/latent_diffusion_stage0.json`
- Code: `stable_worldmodel/wm/latent_diffusion/latent_diffusion.py` (prediction_type),
  schedule math reproduced in §3.
