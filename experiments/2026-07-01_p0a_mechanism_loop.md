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

**Decision rule for next iteration:** if seeds 1–8 confirm DP≈TMSE at mm05 AND
p10 shows the same, the policy half of the thesis is DEAD as an objective effect
(architecture explains everything) → write the honest negative, close P0a/P0,
and move the loop's weight to P1 (multimodal DYNAMICS: K-step screen verdict →
Route 2 build, else Route 1 slip env). If the extension REOPENS a gap (seeds 1–3
unlucky), P0a continues with the fit-error + eval-conditioning screens.
