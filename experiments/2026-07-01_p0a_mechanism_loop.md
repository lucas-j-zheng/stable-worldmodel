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

**Queued next (design, not yet built):**
- Failure-position analysis script for the `positions_off8_seed*.npz` files —
  runs as soon as 3611839 finishes. Metric: distance of failed episodes' final
  position to the wall segment BETWEEN the doors, DP vs TMSE, mm05.
- Candidate (b) screen: conditional multimodality at EVAL-time conditioning
  (rolled-out history + dataset goal) — needs rollout latents; the positions npz
  + videos inform whether this is even distinguishable from (a).
- P1 Route 1 slip-env build — starts if/when the K-step screen kills Route 2, or
  in parallel once P0a analysis is done.
