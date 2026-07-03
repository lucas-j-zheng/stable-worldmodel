# P0a mechanism isolation + diffusion research loop — 2026-07-01 →

**What this is.** Running log of the autonomous experiment loop over the diffusion
track (`todos/diffusion.md`): P0a mechanism isolation → P0 dose axis → P1
multimodal-dynamics domain → P2 JEDI end-to-end, with H-JEPA rung 1 as P1 Route 2.
Each entry: results read, reasoning, what was launched next.

**The question driving P0a.** The +10 DP-over-TMSE win (architecture-controlled,
dose-dependent) survives even though the conditional the policy models is
UNIMODAL under matched conditioning (attribution screen 2026-06-30). Candidates:
(a) training-time mode-averaging — the MSE net can't separate neighboring
(history, goal) bins whose demos took different doors, so its function class
smooths across them; (b) train/eval goal mismatch; (c) calibration/generalization
unrelated to multimodality.

---

## Iteration 1 — 2026-07-01 evening: launches

**In flight (submitted earlier today):**
- **3611839 (mm05)** / **3611840 (p10)** — `p0a_seeded_cell.sbatch`: REAL-seeded
  endpoint reruns, 3 seeds × {DP, TMSE} × 250 ep, eval at eval-seeds {42,123}
  n=50 each (n=100/cell), positions npz per eval (failure probe data).
  3611839 RUNNING (first training already through 250 epochs), 3611840 pending
  on QOSMaxCpuPerUserLimit.

**Launched this iteration:**
1. **Smoothing test / fit-error screen (`p0a_fit_error.sbatch`)** on the EXISTING
   unseeded endpoint checkpoints ({dp,tmse}×{mm05,p10}×s{1,2,3}, split seeds
   matched): per-sample action-chunk error vs the demo chunk in normalized action
   space, train+val splits (`scripts/data/policy_fit_error.py`).
   *Prediction if candidate (a) is live:* TMSE fit error mm05 ≫ TMSE p10, and the
   gap exceeds DP's. If TMSE fits mm05 as well as p10, training-time
   mode-averaging is NOT the mechanism → weight shifts to (b)/(c).
   *Why now:* zero dependency on the running jobs — reuses old checkpoints;
   answers the mechanism question fastest.
2. **H-JEPA rung-1 K-step screen (`hjepa_kstep_screen.sbatch`)** — the built,
   cheap gate for P1 Route 2 (temporal abstraction as the multimodal-dynamics
   domain). K∈{1,2,4,8} × coarse∈{full,sum,none}, encoder-free.
   *Greenlight:* `full` flat/low for all K; `sum`/`none` rise with K;
   K=8/sum ≈ 0.10. *Kill:* flat & low everywhere → the doors domain lacks
   K-scale branching → P1 falls back to Route 1 (slip env).
   *Why now:* independent of P0a; decides whether P1 Route 2 exists before any
   level-2 build. (Required pushing the uncommitted `dynamics_k` mode of
   `multimodality_diagnostic.py`.)

> **Workflow note (Lucas, iteration 1):** no `git push` during the loop — commit
> locally, rsync changed files to the Oscar checkout directly, single push when
> the loop concludes.

**Queued next (design, not yet built):**
- Failure-position analysis script for the `positions_off8_seed*.npz` files —
  runs as soon as 3611839 finishes. Metric: distance of failed episodes' final
  position to the wall segment BETWEEN the doors, DP vs TMSE, mm05.
- Candidate (b) screen: conditional multimodality at EVAL-time conditioning
  (rolled-out history + dataset goal) — needs rollout latents; the positions npz
  + videos inform whether this is even distinguishable from (a).
- P1 Route 1 slip-env build — starts if/when the K-step screen kills Route 2, or
  in parallel once P0a analysis is done.

---

## Iteration 2 — 2026-07-01 ~19:50: mm05 seeded cell landed — THE +10 GAP DOES NOT REPLICATE

**Result 1 — seeded mm05 endpoint (job 3611839, 3 real seeds × 2 eval seeds × n=50):**

| train seed | DP (es42/es123) | TMSE (es42/es123) |
|---|---|---|
| 1 | 38 / 38 | 42 / 46 |
| 2 | 36 / 34 | 44 / 36 |
| 3 | 36 / 32 | 22 / 26 |
| **mean** | **35.7** | **36.0** |

**DP−TMSE at the multimodal endpoint ≈ 0 (35.7 vs 36.0).** TMSE is wildly
training-run-variant (per-seed means 44 / 40 / 24; cross-run sd ~9–10). Pooling
the old unseeded runs (TMSE 28,26,28; DP 38,38,36) with these: DP ~36.5 vs TMSE
~32.7 over 6 runs each → gap ≈ +4 ± big. **The banked "+10 objective effect" was
most plausibly sampling noise on a high-variance baseline — 3 unseeded TMSE runs
that happened to land low.**

**Result 2 — failure-position probe (positions npz, 300 eps/model pooled):**
DP fails 193/300, TMSE 192/300. Final positions of failures in the
wall-between-doors band: **TMSE 10–14%, DP 9–11%** (both axis conventions) —
no mode-averaging pile-up at the wall; both models' failures are mostly
"elsewhere" (mean final dist-to-goal ~50 px, i.e. ran out of budget short of
goal). **Candidate (a) — training-time mode-averaging — has NO behavioral
support.**

**Reasoning.** The two results cohere: if the MSE objective were being corrupted
by multimodal demos, TMSE should (i) underperform DP and (ii) die at the wall.
Neither holds under seeding. The P0a question flips from "why does +10 persist
with a unimodal conditional?" to "**was there ever an objective effect at all?**"
The attribution screen's 'unimodal matched conditional' (06-30) now reads as
consistent with the null: no test-time multimodality AND no gap.

**Launched:**
- **3611999** — mm05 seeds 4–8 extension (same cell script, `SEEDS="4 5 6 7 8"`):
  brings each model to 8 seeded runs × n=100. With TMSE sd ~10, 8 runs give a
  standard error ~3.5 — enough to distinguish gap=0 from gap=+10, which is the
  decisive P0a comparison now.
- p10 cell (3611840) + fit-error (3611948) still queued on CPU quota; K-step
  screen (3611937) still running.

> **Iteration 3 (~20:15): Oscar unreachable** — ssh key auth falling back to
> keyboard-interactive (VPN outage pattern, same as 06-22). Jobs unaffected,
> visibility lost; loop idles on heartbeat until connectivity returns.

**Decision rule for next iteration:** if seeds 1–8 confirm DP≈TMSE at mm05 AND
p10 shows the same, the policy half of the thesis is DEAD as an objective effect
(architecture explains everything) → write the honest negative, close P0a/P0,
and move the loop's weight to P1 (multimodal DYNAMICS: K-step screen verdict →
Route 2 build, else Route 1 slip env). If the extension REOPENS a gap (seeds 1–3
unlucky), P0a continues with the fit-error + eval-conditioning screens.

---

## Iteration 4 — 2026-07-01 ~21:30: p10 + fit-error + K-step landed (during a
VPN outage ~20:00–21:20); slip env built + screening

**Result 3 — seeded p10 (unimodal) endpoint (job 3611840):**

| train seed | DP (es42/es123) | TMSE (es42/es123) |
|---|---|---|
| 1 | 44 / 30 | 38 / 24 |
| 2 | 38 / 26 | 34 / 18 |
| 3 | 40 / 30 | 40 / 24 |
| **mean** | **34.7** | **29.7** |

**DP−TMSE = +5 at the UNIMODAL endpoint** vs ≈0 at the multimodal one — the
dose-response is not just gone, it points the wrong way. Taken with mm05, the
seeded 2×2 shows **no multimodality×objective interaction**; any residual DP
edge (~+2–3 pooled) is within noise. Side finding: eval-seed 123 is ~12 pts
harder than 42 for every model — episode-sampling variance is as big as the
claimed effect, vindicating the two-eval-seed design (the old fixed-seed-42
evals hid this term entirely).

**Result 4 — fit-error / smoothing screen (job 3611948, old checkpoints):**
TMSE val error mm05 0.39–0.43 vs p10 0.22–0.27 (~1.6×); DP sample-vs-demo
distance mm05 ~0.95–1.03 vs p10 ~0.47–0.56 (~2×); train≈val everywhere (no
overfit gap). **Multimodal-demonstrator data IS genuinely harder to regress**
(residual heterogeneity in the targets is real, consistent with the
policy_target screen's 0.49 residual_ratio) — **but this fit-level corruption
does not surface as a closed-loop success gap.** Plausible reconciliation:
receding-horizon replanning (every 5 steps) forgives locally-averaged action
chunks; the task's failure modes (budget exhaustion far from goal, per the
position probe) are not mode-confusion failures.

**Result 5 — H-JEPA K-step screen (job 3611937): GREENLIGHT.**

| K | coarse=full (ctrl) | coarse=sum | coarse=none (proposer) |
|---|---|---|---|
| 1 | 0.266 | 0.266 | 0.956 |
| 2 | 0.047 | 0.061 | 0.941 |
| 4 | 0.048 | 0.095 | 0.932 |
| 8 | 0.059 | **0.135** | 0.918 |

Exactly the greenlight signature from the review doc: `full` flat/low (K≥2),
`sum` rises monotonically to 0.135 at K=8 (predicted ~0.10), proposer cell
massively bimodal. **Temporal abstraction DOES create measurable multimodal
dynamics where 1-step had none** — P1 Route 2 is open. (K=1 full/sum at 0.266
is an oddity worth a look — likely the K=1 conditioning quirk or small-sample;
does not affect the verdict.) H-JEPA execution stays in Lucas's lane (his
hjepa-dose-latent / hjepa-pusht-kstep jobs ran tonight); this loop feeds it.

**Launched:**
- **P1 Route 1 slip env, built + screening (job 3612481).** `TwoRoomEnv` gains
  `slip_scale`: every step, a fair coin displaces the agent ±slip_scale along
  the along-wall axis before collisions → p(next|state,action) is a two-point
  mixture on every step, per-step coin so NO conditioning resolves it (unlike
  drift), interior-axis choice to dodge the drift experiment's clamp-absorption
  failure. `collect_tworooms_slip.py` + `slip_dose_screen.sbatch`: collect+screen
  S ∈ {0,2,4,8}, hidden vs observed(slip_state) contrast. PASS = hidden bimodal
  rises with S while observed stays ~0.
- mm05 seeds 4–8 extension (3611999) still running.

**Addendum (~21:15) — two more launches:**
- **3612576** — p10 seeds 4–8 extension: the +5-at-unimodal anomaly gets the
  same 8-seed power as mm05; both endpoints then support a clean final verdict.
- **3612577** — slip latent caching + LATENT-space dynamics screen, chained
  `--dependency=afterok:3612481` with an IN-JOB gate (aborts unless slip8
  hidden bimodal > 0.3, preserving the screen-before-train rule). Key shortcut
  identified: slip changes dynamics only, not appearance, and uses default
  1-door geometry → the existing `lewm_tworoom` encoder is in-distribution —
  **no encoder retrain needed for Route 1.** If both screens pass, the next
  stage is diffusion-vs-deterministic dynamics training on slip latents.

**Addendum 2 (~21:45) — P1 Route 1 full pipeline queued + dynamics MSE control built:**
- **`TransformerMSEDynamics`** (`wm/latent_diffusion/transformer_mse_dynamics.py`):
  the dynamics-side analogue of the policy TransformerMSE control — same
  LatentTrajectoryDenoiser backbone, MSE objective, deterministic forward. This
  is what makes the slip test a clean OBJECTIVE contrast (diffusion vs
  same-backbone MSE), and it also fixes the old training-budget-parity confound
  (both post-hoc, same budget). Also patched `latent_diffusion.py` with
  `seed_everything` (same unseeded-split bug the policy script had).
- **Gotcha found & handled:** `wm.latent_diffusion` is name-shadowed by its inner
  `latent_diffusion.py` (a pre-existing `import *` collision), so
  `_target_=...latent_diffusion.TransformerMSEDynamics` would resolve to the
  wrong module. Reference the class by full submodule path
  (`...latent_diffusion.transformer_mse_dynamics.TransformerMSEDynamics`) and
  left the package `__init__` untouched.
- **Queued chain (all `afterok`):** slip screen 3612481 → latent cache 3612577 →
  dynamics cells **3612892 (slip8)** / **3612893 (slip0 control)**, each 3 seeds
  × {diffusion, TMSE} with an in-job bimodality gate. Slip changes dynamics only
  (default 1-door geometry) so `lewm_tworoom` is in-distribution — no encoder
  retrain. NEXT after these: closed-loop D-MPC eval (diff vs det) on the slip env
  — the actual dynamics-half verdict.

**Addendum 3 (~22:10) — Route 1 chain closed end-to-end:**
Built `tworoom_slip_diffusion.yaml` + `slip_dmpc_eval.sbatch` and chained the
closed-loop verdict evals: **3612972** (slip8 cell, afterok:3612892) and
**3612973** (slip0 control, afterok:3612893). Each: 3 seeds × {diffusion, TMSE
dynamics} × 2 eval seeds × n=50, CEM D-MPC, offset 25 (matches the original
TwoRoom D-MPC setting), positions npz per eval. The whole Route 1 ladder now
runs unattended: screen → cache (gated) → train (gated) → closed-loop eval.
Thesis prediction: diffusion > TMSE at slip8 (deterministic averages the two
slip branches), tie at slip0. If BOTH tie, the dynamics half fails even on
screened-multimodal dynamics — a much stronger negative than the old one.

---

## Iteration 5 — 2026-07-01 ~22:20: SLIP SCREEN PASSES (with a timeout + a control flaw)

**Result 6 — slip dose screen (3612481, TIMEOUT at 1h during slip8 collection):**

| slip_scale | hidden residual_bimodal |
|---|---|
| 0 | 0.114 (unimodal floor) |
| 2 | 0.975 |
| 4 | 0.987 |
| 8 | 0.984 (recollected, job 3613072) |

**The intrinsic slip env delivers exactly the designed multimodality: pervasive,
hard, dose-gated bimodal dynamics** (0.11 → 0.98 the moment the coin exists;
saturates immediately since the ±S separation dwarfs the kNN floor — a graded
dose would need sub-pixel S, not needed for the verdict cells). This is the
multimodal-DYNAMICS domain the whole program lacked — where hidden-doors
(det_R² 0.966) and hidden-drift (~1%, shrinking) failed.

**Two flaws found & handled:**
1. *Timeout:* 4×4000-episode collections don't fit gpu-debug's 1h. The chained
   dependents were auto-cancelled (afterok on TIMEOUT). Rebuilt.
2. *Observed control off-by-one:* `slip_state` on row t is the slip that
   produced state_t; the screened transition (state_t, a_t)→state_{t+1} needs
   row t+1's slip, so "observed" read the same as "hidden" (0.99). Not fixed
   tonight — the slip0-vs-slipN dose is the unambiguous control.

**Relaunched chain:** finish/recollect slip8 **3613072** → cache **3613073**
(gated) → dyncells **3613074** (slip8) / **3613075** (slip0) → closed-loop
D-MPC verdicts **3613076** / **3613077**. Runs unattended overnight.

**P0a verdict taking shape:** no reliable policy-side objective effect; the
banked +10 was noise on a high-variance baseline measured with a
variance-hiding eval design. Final call when seeds 4–8 land — then P0/P0a close
and the program's weight moves to P1 (slip + K-step routes, both now gated
open/1 screening).

---

## Iteration 6 — 2026-07-01 ~22:40: P0a FINAL VERDICT (mm05, 8 seeds × n=100)

| model | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | mean | cross-seed sd |
|---|---|---|---|---|---|---|---|---|---|---|
| DiffusionPolicy | 38 | 35 | 34 | 35 | 40 | 36 | 33 | 28 | **34.9** | **3.6** |
| TransformerMSE | 44 | 40 | 24 | 29 | 27 | 29 | 40 | 35 | **33.5** | **7.3** |

**VERDICT: DP−TMSE = +1.4 ± ~2.9 at the multimodal endpoint — the diffusion
objective effect on mean success is ZERO.** (8 real seeds, 2 eval seeds × n=50
each, architecture-matched.) The banked "+10 multimodality×objective
interaction" was sampling noise: 3 unseeded TMSE runs that landed low, read
through a fixed-eval-seed design that hid both variance terms. Chain of
evidence: seeded tie (iter 2) → no wall-clustering (iter 2) → inverted p10
"dose" (iter 4) → fit-error real but behaviorally inert (iter 4) → 8-seed null
(this). **The policy half of the thesis, as operationalized on TwoRoom, closes
as an honest negative.**

**Genuine secondary finding worth keeping: the diffusion objective HALVES
cross-training-run variance (sd 3.6 vs 7.3, same backbone, same data).**
Diffusion as a *stabilizer*, not a mean-improver, on weak/ambiguous-multimodal
data — consistent with the fit-error picture (the MSE loss surface on
heterogeneous targets is nastier run-to-run). If p10's 8-seed cell reproduces
the variance ratio, this is a defensible, novel-ish observation for the writeup.

**Program state after this verdict:**
- P0 (door-count dose axis) is MOOT — there is no policy-side effect to dose.
  Mark dead pending only the p10 8-seed symmetry check (3612576, queued).
- The program's live thesis bet is now entirely on the DYNAMICS side:
  the slip chain (3613072→…→3613076/77) and the K-step/H-JEPA route.
- PROGRESS_ONEPAGER rewrite queued for when p10 lands (both endpoints final).

---

## Iteration 7 — 2026-07-02 ~00:00: p10 8-seed lands — P0a COMPLETE, and the real finding is in the VARIANCE

**p10 (unimodal), 8 seeds × n=100:** DP 32.5 (sd 2.8) vs TMSE 31.3 (sd 2.7) —
mean gap +1.25 ≈ 0, matching mm05's +1.4 ≈ 0. Final seeded 2×2 (8 seeds/cell):

| | mm05 (multimodal demos) | p10 (unimodal demos) |
|---|---|---|
| DP mean (sd) | 34.9 **(3.6)** | 32.5 **(2.8)** |
| TMSE mean (sd) | 33.5 **(7.3)** | 31.3 **(2.7)** |
| mean gap | +1.4 (≈0) | +1.25 (≈0) |

**FINAL P0a SYNTHESIS.** (1) *Mean* success: NO diffusion-objective effect at
either endpoint — the original +10 is conclusively noise. (2) *Variance*: a
clean multimodality×objective interaction — **the MSE objective's cross-run sd
nearly triples on multimodal-demonstrator data (2.7 → 7.3) while diffusion's
barely moves (2.8 → 3.6).** This coheres exactly with the fit-error screen
(multimodal targets are ~1.6× harder to regress → nastier MSE loss landscape →
run-to-run instability) and reframes the policy half honestly: *diffusion
doesn't make the policy better here; it makes it reliable when the demos are
multimodal.* The variance probe (3613638, queued) tests whether that
stabilization is sampling-time or training-time.

**Program status:** P0a/P0 CLOSED. One-pager rewritten (both endpoints final).
Dynamics half is the live bet: slip chain now RUNNING (3613072 finish stage),
sub-pixel dose running locally, K-step route in Lucas's lane.

**Follow-ups launched (~23:20):**
- **3613638 variance-mechanism probe:** re-eval the 8 mm05 DP checkpoints under
  sampling variations (eta 1.0, steps 5/50 vs base) via a new
  `+model_overrides` hook in eval_wm.py. Flat cross-seed sd across configs ⇒
  the stabilization is training-time (objective/loss landscape); sd moving with
  the sampler ⇒ sampling-averaging. Decides how to frame the keeper finding.
- **3613639 sub-pixel slip dose:** S ∈ {0.25, 0.5, 1} — resolves the saturated
  dose curve (0.11→0.98 between S=0 and 2) into a graded transition + locates
  the screen's detection threshold.
- Deferred: the observed-control shift-fix in the diagnostic (next iteration;
  the S=0-vs-S>0 dose remains the control meanwhile).

---

## Iteration 8 — 2026-07-02 ~00:40: sub-pixel slip dose complete (ran LOCALLY on the Mac)

| slip_scale S | residual_ratio | residual_bimodal |
|---|---|---|
| 0    | —      | 0.114 |
| 0.25 | 0.0063 | 0.896 |
| 0.5  | 0.0100 | 0.905 |
| 1    | 0.0179 | 0.926 |
| 2    | —      | 0.975 |
| 4    | —      | 0.987 |
| 8    | —      | 0.984 |

**Reading:** the bimodality FLAG is effectively binary — it switches on at a
quarter-pixel slip (0.11 → 0.90) because Sarle BC detects any hard separation
above the kNN floor. The genuinely graded axis is the residual MAGNITUDE
(`residual_ratio` ≈ linear in S: 0.006 → 0.010 → 0.018). For the paper: dose =
residual scale (2S) with bimodality as a qualifier, not the x-axis. Also sets up
a sharper prediction for the verdict cells: the diffusion-vs-MSE planning gap
should track residual MAGNITUDE (how much the averaged prediction is off), not
the flag — testable later by adding slip1/slip2 verdict cells if slip8 wins.

Ops notes: ran on the Mac per Lucas's local-runs-OK rule (Oscar quota full).
Local venv needed scikit-learn (uv). The diagnostic resolves datasets via
STABLEWM_HOME (LOCAL_DATASET_DIR alone insufficient). One truncated dataset
from a killed background task recollected under a fresh name (rm was denied;
scratchpad self-cleans).

**Iteration 8b (~00:55) — LATENT-space gate PASSES (job 3613073):** slip
bimodality survives the encoder: latdyn residual_bimodal slip0 0.004 / slip4
0.758 / slip8 0.935. The dynamics models condition on latents, so this was the
last could-kill-it check for Route 1 — the branch structure is fully visible in
the conditioning space the models actually use. Dyncells queued (behind the
variance probe, now running); verdict evals follow.

---

## Iteration 9 — 2026-07-02 ~01:10: variance probe — stabilization is TRAINING-TIME (job 3613638)

Re-eval of the 8 seeded mm05 DP checkpoints under sampling variations
(single eval seed 42, n=50):

| config | mean | cross-seed sd |
|---|---|---|
| base (eta 0, steps 20) | 36.3 | 5.4 |
| eta 1.0 (ancestral)    | 39.5 | 3.2 |
| steps 5                | 35.8 | 4.7 |
| steps 50               | 37.3 | 4.4 |

**Read:** cross-seed sd stays 3–5 under EVERY sampling regime — ancestral noise
and coarse DDIM included — never approaching TMSE's 7.3. Caveat: with n=50
single-eval binomial noise (~±7/cell), differences BETWEEN configs are not
resolvable; the defensible statement is that the DP stability is insensitive to
the sampler. **Framing for the writeup: the variance stabilization on
multimodal demos is a property of the denoising TRAINING OBJECTIVE (loss
landscape), not of test-time sampling.** Coheres with fit-error (mm targets
corrupt the MSE landscape; the denoising objective spreads the target over
noise levels and escapes that). Probe CLOSED.

**Iteration 10 (~01:45) — budget fix:** slip dynamics trainings pace at ~90
min/model at EP=250 (627 steps/epoch — the slip data is ~20× the policy latent
sets), so 6 models ≫ the 6h limit. Cancelled 3613074/75/76/77 preemptively and
rechained with **EP=100** (63k steps/model — the seeded policy models that
produced clean results saw 500 steps total): dyncells **3614760** (slip8) /
**3614761** (slip0) → verdicts **3614762** / **3614763**. ~3.6h/cell, fits.

---

## Iteration 11 — 2026-07-02 ~10:55: SLIP8 VERDICT — diffusion dynamics +6, consistent across seeds

Closed-loop D-MPC (CEM) on the slip8 env (job 3614762; offset 25, 2 eval seeds
× n=50 per seed):

| train seed | Diffusion | TMSE-dynamics |
|---|---|---|
| 1 | 69 (68/70) | 61 (58/64) |
| 2 | 69 (68/70) | 64 (58/70) |
| 3 | 73 (68/78) | 68 (68/68) |
| **mean** | **70.3** | **64.3** |

**Diffusion beats the same-backbone MSE dynamics by +6 on screened-bimodal
transitions — direction consistent in ALL seeds, spreads tight (unlike the
policy-side pseudo-effect, which flapped seed-to-seed).** Gap ≈ +6 ± ~2.4.
HOLD the headline until the slip0 CONTROL (3614763, running) reads: tie there
⇒ first confirmed multimodality×objective interaction on the DYNAMICS side
(the thesis's missing half, on its fair architecture-matched, budget-matched
test); +6 there ⇒ artifact, back to the drawing board.

---

## Iteration 12 — 2026-07-02 ~12:30: SLIP0 CONTROL BREAKS THE INTERACTION — the gap is multimodality-INDEPENDENT

slip0 control (job 3614763), same protocol as slip8:

| | slip8 (bimodal dynamics) | slip0 (deterministic control) |
|---|---|---|
| Diffusion dynamics | 70.3 (69/69/73) | **75.7** (76/77/74) |
| TMSE dynamics | 64.3 (61/64/68) | **65.7** (66/66/65) |
| **gap** | **+6** | **+10** |

**The registered prediction (win at slip8, tie at slip0) is FALSIFIED.** The
diffusion-dynamics advantage is real, consistent, and *larger on deterministic
dynamics* — it cannot be mode-capture. Had we only run slip8 (the "obvious"
experiment), we'd have banked a spurious dynamics-side multimodality win —
the same trap the policy side fell into pre-seeding. The control saved the
program twice now.

**What the whole 2×2×2 (policy/dynamics × mm/uni × diff/MSE) says:** the strict
thesis — *generative beats deterministic iff the modeled conditional is
multimodal* — is dead on BOTH sides of this benchmark family. What survives:
1. **Policy:** no mean effect; diffusion = cross-run VARIANCE stabilizer on
   multimodal demos (training-time property).
2. **Dynamics:** a consistent +6..+10 diffusion advantage UNCORRELATED with
   multimodality. Prime suspect: **inference-time compute** — 20 DDIM
   refinement passes per prediction vs TMSE's single forward — or CEM
   benefiting from sampling diversity.
3. **Methods:** unseeded, single-condition comparisons manufacture
   "multimodality interactions" in either direction at will. Controls +
   seeds + eval-seed variance are not optional.

**Launched: 3618231 — 1-step inference probe.** Same 6 diffusion checkpoints,
`num_inference_steps=1` (v-param single-step readout ≈ one forward), both
cells. Gap collapses ⇒ iterative-refinement compute is the advantage; gap
survives ⇒ the denoising objective trains a genuinely better one-shot
predictor (training-time, mirroring the policy variance story).

---

## Iteration 13 — 2026-07-02 ~14:20: 1-STEP PROBE — the dynamics gap is the TRAINING OBJECTIVE (job 3618231)

| | slip8 | slip0 |
|---|---|---|
| Diffusion @20 steps | 70.3 | 75.7 |
| **Diffusion @1 step** | **74.0** | **73.7** |
| TMSE (1 forward) | 64.3 | 65.7 |

**The gap does NOT collapse at one inference step — a single forward pass of
the diffusion-trained model matches its own 20-step sampling and beats TMSE by
+8–10.** Iterative-refinement compute is ruled out. The denoising objective
trains a genuinely better ONE-SHOT dynamics predictor than direct MSE
regression (same backbone/data/budget).

**UNIFIED PICTURE (both halves of the program):** every surviving diffusion
advantage is a TRAINING-TIME property of the denoising objective —
(policy) cross-run variance stabilization on heterogeneous targets;
(dynamics) better one-shot predictors — and NONE of it is test-time multimodal
sampling. The strict multimodality law is dead; the honest replacement:
*"denoising is a better training signal for regression-hard targets; sampling
per se buys nothing here."*

**Launched (mechanism rung 2): 3618840/3618841** — `prediction_type=x0` cells
(same denoiser, denoising regression of the clean target: keeps the noise
curriculum, drops the v-target), 3 seeds × both slip levels, eval @1 & @20.
x0 ≈ v ⇒ the NOISE CURRICULUM is the ingredient; x0 ≈ TMSE ⇒ the v-target is.
(NB the old paramtype sweep found x0 best open-loop / worst planner AT 20-STEP
SAMPLING on deterministic data — this retests at 1 step on the slip cells.)

---

## Iteration 14 — 2026-07-02 ~15:50: x0 slip8 cell — THE NOISE CURRICULUM IS THE INGREDIENT (job 3618840)

| slip8, 3 seeds × n=100 | @1 step | @20 steps |
|---|---|---|
| x0 (denoising regression) | **78.3** (73/81/81) | 67.7 |
| v-param (iter 13) | 74.0 | 70.3 |
| TMSE (clean-input regression) | 64.3 | — |

**x0@1 is the best dynamics model of the whole program: +14 over clean-input
MSE, same backbone/data/budget.** The v-target is not the ingredient (x0 ≥ v);
**training on noise-corrupted inputs across the diffusion schedule is.** Bonus
resolution: x0@20 (67.7) < x0@1 (78.3) — DDIM sampling actively hurts the x0
model, which retroactively explains the old "x0 best open-loop / worst planner"
anti-ranking (that was a SAMPLING penalty at 20 steps, not a bad model).
slip0 x0 cell (3618841) running — expect the same pattern (mechanism is
multimodality-independent).

**The arc's final shape (pending slip0-x0 confirmation):** "diffusion world
models beat deterministic ones" on this benchmark family reduces to *denoising
regression is a better training recipe for one-shot latent dynamics than clean
MSE* — no sampling, no multimodality, no iterative refinement required. Paper
framing: a deflationary mechanism study with two constructive artifacts (the
noise-curriculum regressor recipe; the variance-stabilized diffusion policy)
plus the methodology lesson (controls + seeds kill both naive interactions).

---

## Iteration 15 — 2026-07-02 ~17:30: slip0-x0 confirms — ARC COMPLETE, LOOP CONCLUDES

slip0 x0 cell (job 3618841): @1 step **75.3** (80/73/73), @20 steps 66.7.
Final dynamics table (3 seeds × n=100 per cell):

| | slip8 (bimodal) | slip0 (deterministic) |
|---|---|---|
| **x0 @1 (noise-curriculum regression)** | **78.3** | **75.3** |
| v @1 | 74.0 | 73.7 |
| v @20 ("diffusion world model") | 70.3 | 75.7 |
| x0 @20 | 67.7 | 66.7 |
| TMSE (clean-input regression) | 64.3 | 65.7 |

## FINAL SYNTHESIS (the whole 2026-06→07 diffusion program)

1. **The strict multimodality law is dead on both sides.** Policy: no mean
   effect at either dose endpoint (8 seeds × n=100). Dynamics: a real gap that
   is multimodality-INDEPENDENT (+6 bimodal / +10 deterministic).
2. **Every surviving diffusion advantage is a TRAINING-TIME property of the
   denoising objective.** Policy: cross-run variance stabilization on
   multimodal demos (2.7→7.3 MSE sd vs ~3 DP; sampler-insensitive). Dynamics:
   noise-curriculum regression trains one-shot predictors +10–14 over
   clean-input MSE; the v-target is unnecessary (x0 ≥ v); DDIM sampling is
   neutral-to-HARMFUL (x0: 78→68 going 1→20 steps) — which also resolves the
   old open-loop-MSE-anti-ranking puzzle (a sampling penalty, not model
   quality).
3. **Methodology:** unseeded, control-free comparisons manufactured
   multimodality "interactions" in BOTH directions during this program; seeds
   + eval-seed variance + matched controls killed each one. Non-optional.
4. **Constructive artifacts:** the noise-curriculum one-shot regressor recipe
   (best dynamics model of the program, deployable without any sampling
   machinery); the variance-stabilized diffusion policy; the slip env + dose
   instrumentation; TransformerMSE{Policy,Dynamics} controls; seeded eval
   harness with position logging.

**Loop concluded** — no defensible next experiment in scope. New-arc
candidates (Lucas's call): (a) cross-domain test of the noise-curriculum
recipe (PushT) — turns the recipe into a general claim; (b) H-JEPA rung 2
(the level-2 `coarse=none` proposer cell remains the one place a genuine
multimodality effect could still appear); (c) P2/JEDI needs a NEW rationale —
its premise ("diffusion wins because multimodal dynamics") is invalidated, but
"denoising as latent-shaping training signal" is a coherent replacement.

---

# ARC 2 — 2026-07-03: does the noise-curriculum recipe transfer? (PushT)

**Why.** Arc 1's headline is a training-recipe claim (*denoising regression >
clean MSE for one-shot latent dynamics, multimodality-independent*). One env is
an observation; two is a recipe. PushT (contact manipulation, deterministic
physics — so the multimodality-independence prediction applies directly) has
all assets in place: `pusht_latent.lance`, `lewm_pusht`, the E6 eval harness.
Bonus stake: the old "diffusion loses on PushT" negative (det 16% vs diff 8%)
carried BOTH confounds arc 1 exposed (joint-vs-post-hoc budget; 20-step
sampling penalty) — x0@1step may overturn it.

**Launched (2026-07-03): jobs 3634351 (x0) / 3634352 (v) / 3634353 (tmse)** —
3 real seeds each, matched budget (EP=5, the proven PushT budget), closed-loop
eval @1 and @20 steps × 2 eval seeds × n=50. Registered predictions:
(1) x0@1 > tmse (recipe transfers); (2) x0@1 > x0@20 (sampling penalty
reproduces); (3) if x0@1 also clears the old jointly-trained deterministic
16%, the historical PushT negative is overturned as an artifact.

## ARC 2, iteration 1 — 2026-07-03: x0 cell lands (job 3634374, after an encoder-path fix resubmit)

| dyn_x0 PushT, 3 seeds × n=100 | mean (per-seed) |
|---|---|
| @1 step | **20.0** (18/19/23) |
| @20 steps | **13.0** (11/12/16) |

Predictions 2 and 3 already CONFIRMED on the second domain: the sampling
penalty reproduces (20.0 @1 vs 13.0 @20, every seed), and x0@1 clears the old
jointly-trained deterministic 16% while the old "diffusion loses at 8%" now
reads as artifact (budget confound + sampling penalty compounded). Prediction 1
(x0@1 > post-hoc TMSE, the recipe claim proper) awaits the tmse cell (3634376,
queued); v cell (3634375) mid-run. NB: eval-seed spread is large on PushT
(±6-12 at n=50) but per-seed means are tight.
