# H-JEPA autonomous experiment loop — results log

Living doc; every completed job's numbers land here with a verdict against the
pre-registered predictions in `HJEPA_REASONING.md`.

## Job index

| job | script | what | status |
|---|---|---|---|
| 3611937 | `hjepa_kstep_screen.sbatch` | E1 rung-1 K×coarse gate (fresh dp05 collection + 12-cell grid) | RUNNING |
| — | `hjepa_dose_latent_screen.sbatch` | E2 dose grid (mm0/p07/p09/p10) + rung-2 latent confirmation | pending submit |
| — | `hjepa_pusht_kstep_screen.sbatch` | E3 PushT K-step re-screen (scripted + multi-human) | pending submit |

## Results

### R0 — instrument extension self-test (local, 2026-07-01)

New metrics added (commit local-only): `--stride`, coarse actions
`sumhalf`/`dir`, and per-anchor GMM(k=2) stats. Synthetic controls
(600 anchors, k=96):

| control | det_R² | BC bimodal_frac | gmm2_frac | gap_ratio | separation |
|---|---|---|---|---|---|
| unimodal | 0.953 | 0.000 | 0.002 | 0.845 | 2.8 |
| bimodal | 0.473 | 0.998 | 1.000 | **0.000** | 9.9 |
| trimodal | 0.555 | 0.730 | **0.952** | 0.357 | 4.2 |

**Verdict: VALID, with two upgrades over Sarle BC.** (1) GMM-BIC keeps power
where BC fades (trimodal 0.952 vs 0.730) — also the soft-modes tool for the
06-30 policy puzzle. (2) `gap_ratio` measures the mechanism itself: 0.000 on
bimodal = the conditional mean (what an MSE model predicts) sits in a zero-density
gap; 0.357 on trimodal correctly reads "a mode exists AT the mean" (average is
feasible when a center mode exists) — exactly the planning-relevant distinction
(review C1). Not yet deployed to Oscar (waiting for in-flight jobs to finish so
the sweep stays internally consistent).

### R1 — E1 rung-1 gate (job 3611937, COMPLETED 36 min)

Fresh `tworoom_mm_dp05` (door_prob 0.5). residual_bimodal_frac (BC), with
det_R² / residual_ratio for context:

| K | full | sum | none |
|---|---|---|---|
| 1 | 0.266 (R²=.993, res=.001) | = full | 0.956 (R²=.118, res=.083) |
| 2 | 0.047 (.992/.002) | 0.061 (.989/.002) | 0.941 (.062/.164) |
| 4 | 0.048 (.993/.002) | 0.095 (.982/.004) | 0.932 (.046/.326) |
| 8 | 0.059 (.994/.004) | 0.135 (.970/.009) | 0.918 (.040/.613) |

**Verdict: decision-table OUTCOME 3** (predicted as the live risk in E1 §3).
- `full`: composed determinism confirmed (det_R² .99 at all K; K=1 bimodal 0.27
  is the documented tiny-residual artifact — residual is 0.1% of marginal).
- `sum`: K=8 bimodal 0.135 ≈ the design doc's ~0.10, AND the K-trend is
  monotone (.061→.095→.135 ≈ linear in K) — but it sits on a ~1% residual with
  det_R² 0.97: **Σa is near-sufficient for the endpoint (integrator env,
  review B3)**. An MSE level-2 model conditioned on Σa would be fine → no
  generative edge at THIS abstraction.
- `none`: huge and hard-branching (res_ratio 0.61, bimodal 0.92 at K=8), and
  residual_ratio RISES with K (.08→.61) — the branching magnitude grows with
  horizon even though bimodal_frac is already saturated at K=1.

### R2 — E2a dose grid (job 3611956): PREDICTION FALSIFIED → confound found

K=8 cells by door_prob (bimodal_frac):

| dose | full | sum | none |
|---|---|---|---|
| mm0 (0.5) | 0.072 | 0.155 | 0.923 |
| p07 | 0.077 | 0.147 | 0.905 |
| p09 | 0.069 | 0.137 | 0.911 |
| p10 (1.0) | 0.065 | 0.124 | 0.929 |

**NO dose effect — p10 (expert always takes the same door) reads the same as
mm0.** The `none` cell's multimodality is therefore NOT door-branching: it is
dominated by the per-episode GOAL variation (cond = state only; the expert's
destination is not conditioned on, so p(s_{t+K}|s_t) branches by goal at every
dose — visible already at K=1, bimodal 0.95). **The proposer screen must be
goal-conditioned:** p(s_{t+K} | s_t, goal). This is the 06-30 lesson inverted —
under-conditioning manufactures spurious (planner-irrelevant) multimodality just
as over-conditioning destroys the real kind. The sum-cell dose gradient
(0.155→0.124, weak but correctly ordered) hints the door signal exists
underneath; the goal-conditioned re-screen (E6) isolates it.

### R3 — E2b latent-target rung 2 (job 3611956)

mm0_latent K=8: none 0.820 (state-target: 0.923), sum 0.229 (state: 0.155),
full 0.054 low. **Branching survives the SIGReg encoder** (within the
pre-registered 2×) ✓ — but the same goal confound applies; re-confirm
goal-conditioned before relying on it.

### R4 — E3 PushT K-step (job 3611957): NEGATIVE (pre-registered either-way)

Multi-human state target: none-cell det_R² falls with K (.945→.640) and
residual_ratio rises (.038→.207) — the reachable set WIDENS — but bimodal_frac
stays at floor (0.047→0.071): **smooth growth, no branching. PushT lacks hard
K-scale route structure even with multi-human demos** (consistent with the
06-16 screen). Scripted expert: same shape. (pusht_human_latent shows bimodal
~0.55 even under FULL conditioning with det_R² 0.2–0.3 — encoder/domain-gap
artifact on human data, not dynamics; noted, not chased.) **PushT is out as
the second domain; maze family is the fallback if TwoRoom survives E6.**
