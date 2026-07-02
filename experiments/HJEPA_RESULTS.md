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

### R5 — empirical T and the K/T law (derived from pair counts)

Uncapped pair counts give T ≈ 25–26 (not the design doc's assumed 40), N_ep ≈
5.5k (mm0) / 8.8k (dp05). Law prediction at K=8: 0.5·K/T ≈ **0.155** — the
observed sum-cell bimodal_frac is 0.155 (mm0) / 0.135 (dp05). BUT read
skeptically: p10's sum cell reads 0.124 where the law (p_choice=0) predicts ~0,
so ~0.12 of the sum-cell fraction is dose-independent background (tiny-residual
noise + non-door structure) and only the **dose-differenced increment ~0.03**
is door-attributable. The numerical match is partly coincidence. The law gets
its fair test in E6 (goal-conditioned), where the background should drop out.

### R6 — E7 maze attempt 1 (job 3612584): FAILED (harness, not science)

`World(...)` requires `image_shape` unless `add_pixels=False`; collection died
at construction, all screen cells skipped on the missing dataset. Fixed
(`add_pixels=False`, state-only) and resubmitted as **3612832**.

### R7 — E6 goal-conditioned, PARTIAL (mm0 + dp05 landed; p10 contrast pending)

Goal conditioning works as diagnosed in R2: at K=8/none the residual drops
0.61→0.22 and det_R² rises 0.04→0.59 — most of the R1 `none`-cell signal was
indeed goal variation. What REMAINS in the goal-conditioned proposer cell
p(s_{t+K}|s_t,goal) on dp05:

| K | res_ratio | bimodal | gmm2 | gap_ratio | sep(σ) |
|---|---|---|---|---|---|
| 1 | 0.027 | 0.385 | 0.668 | 0.692 | 4.5 |
| 2 | 0.052 | 0.419 | 0.682 | 0.646 | 4.8 |
| 4 | 0.103 | 0.419 | 0.668 | 0.590 | 4.8 |
| 8 | 0.222 | 0.484 | 0.668 | 0.487 | 5.7 |

Residual magnitude ~doubles per K-doubling; modes separate further with K
(gap_ratio 0.69→0.49, sep 4.5→5.7σ). mm0 matches dp05 (K=8/none: 0.451/0.679/
0.514). Also: `dir` ≈ `sum` (det_R² 0.97–0.99) — ANY executed-action statistic
that retains direction leaks the route; the action-free proposer is the only
non-leaking abstraction in a nav domain. **Verdict awaits the p10 cells** (dose
attribution: door-branching vs residual goal-independent structure).

### R8 — E6 COMPLETE (job 3612480): GREENLIGHT, dose restored

The p10 contrast landed. K=8/`none` (goal-conditioned proposer conditional):

| | mm0 (multimodal demos) | p10 (unimodal demos) | ratio/Δ |
|---|---|---|---|
| residual_ratio | **0.225** | 0.050 | 4.5× (≥2× req. ✓) |
| bimodal_frac | **0.451** | 0.126 | Δ0.33 (>0.10 req. ✓) |
| gmm2_frac | 0.679 | 0.383 | |
| gap_ratio | 0.514 | 0.856 | deep-vs-none ✓ dir. |
| separation | 5.2σ | 2.8σ | |

**p(s_{t+K}|s_t,goal) is the project's first conditional that is multimodal
under matched conditioning, dose-dependent, hard-separated, and
planner-relevant.** Gap_ratio median 0.51 (not <0.3 as registered — anchors are
a mixture of deep-gap and shallow; per-anchor gap distribution worth plotting
for the paper). K-scaling on dp05: residual doubles per K-doubling, separation
grows 4.5→5.7σ. RUNG 3 UNLOCKED → E9 (job 3612901).

### R9 — E7 maze attempt 2 (3612832): collection OK, screens skipped

3000 eps collected in ~3 min (state-only). Column is `observation`, not
`state` — all cells skipped. Attempt 3 (3612881) runs screens with
`--target-col observation`; collection skipped (dataset cached).

### R10 — E7 maze (job 3612881): NEGATIVE for random walk (pre-registered risk 4)

`none` cell: bimodal ≈ 0.00 at ALL K≤16, res_ratio only 0.013→0.050 — a
per-step random walk diffuses ~√K and never straddles junctions. (full/sum
bimodal 0.4–0.9 on 0.1–0.7% residuals = the tiny-residual artifact, ignore.)
NOT a domain kill: data-policy limitation → E11 persistent-walk retry
(hold-4 actions, T=128, K≤32; job 3613349).

### R11 — E9 PROPOSER BENCH (job 3612901): **THE STRICT THESIS CONFIRMS**

p(s_{t+8}|s_t,goal), episode-split, 1000 anchors, M=32 samples/head:

| head | mm0 precision | mm0 coverage | mm0 energy | p10 precision | p10 coverage |
|---|---|---|---|---|---|
| mse | 0.608 | **0.058** | 0.393 | 0.857 | 0.076 |
| gauss | 0.629 | 0.270 | 0.205 | 0.832 | 0.182 |
| mdn | **0.850** | 0.469 | 0.185 | 0.881 | 0.224 |
| knn | **0.895** | **0.594** | **0.107** | 0.916 | 0.467 |
| diff | 0.776 | 0.524 | 0.145 | 0.805 | 0.258 |

(dp05 replicates mm0 almost exactly.) Pre-registered predictions: (1) ✓ MSE
precision collapses on multimodal data (−0.24 to −0.29 vs mdn/knn) and its
mean covers essentially NO mode (0.058); (2) ✓ p10 parity (MSE −0.02…−0.06,
within head noise); (3) ✓ mdn ≈ knn ≥ diff — **diffusion is NOT special on 2
hard modes; k-NN retrieval, the cheapest possible sampler, wins energy
everywhere**; (4) ✓ gauss ≈ mse precision (calibration ≠ mode-splitting);
(5) partial — generative coverage 0.47–0.61 (< the registered 0.8; the 2δ
criterion is strict), MSE 0.06 (worse than the 0.5 registered — the mean sits
near NEITHER mode). **First positive, dose-controlled det-vs-generative result
of the project** — at the goal-conditioned K-step proposer, exactly where the
screen said it must live. Caveats: state-space toy (E10 = latent version
running), single seed (E12 seeds running), no closed-loop yet (rung 5).

### R12 — E8a policy retro-test: 06-30 soft-mode hypothesis is DEAD

Conditioning-matched policy conditional: BC ~0.006 everywhere (replicates
06-30); gmm2_frac shows NO dose direction (mm0 0.37–0.51 vs p10 0.79–0.80 —
higher on the UNIMODAL set) and gap_ratio ≈ 1.00 everywhere: whatever soft
structure exists, the conditional mean sits at FULL density. Combined with the
+10 seeding collapse (P0a): the policy side is now fully law-consistent —
no modes ⇒ no gap. The 2×2 is coherent: policy (no modes, no gap) vs proposer
(modes, gap).

### R13 — E8b seeds + E5 stride/coarseness: instrument robustness banked

Seed σ(bimodal_frac) ≈ 0.005 (dp05 K8/none: .916/.918/.923; gap_ratio .033–
.035 — the unconditioned deep-gap number is rock solid). Stride=8
(decorrelated): none 0.859 vs 0.918 (−6%), sum 0.164 vs 0.135 (+21%) — inside
the registered ±30%. Coarseness dial at K=8 perfectly monotone:
full .059 < sum .135 ≈ sumhalf .138 < dir .233 < none .918.

### R14 — E13 K-sweep + dose curve (job 3613634): **the central figure holds**

MSE precision (deficit vs MDN in parens), mm0 K-sweep:
K=4 0.83 (−0.01) → K=8 0.61 (−0.24) → K=16 0.48 (−0.35). Deficit GROWS with K
exactly as the screen's residual/separation growth predicts ✓ (registered).
Dose curve at K=8, MSE precision: mm0 0.61 < p07 0.67 < p09 0.76 < p10 0.86 —
perfectly monotone in door_prob; deficit vs MDN: 0.24 → 0.13 → 0.07 → 0.02 ✓.
knn/mdn flat (0.80–0.92) across all cells ✓. **The screen quantitatively
predicts where and how much the MSE proposer loses.**

### R15 — E12 seeds (job 3613348): the gap REPLICATES

mm0 K=8 MSE-vs-MDN precision gap across torch seeds {0,1,2}: 0.24/0.22/0.26
(±0.02, registered ±0.05 ✓); p10 parity stable (0.02/0.01/−0.04). Unlike the
+10, this result is seed-stable.

### R16 — E10 latent bench: FAILED on missing goal column (harness)

Latent-cached datasets carry no goal_state. Fix: `--goal-from final`
(destination conditioning, validated by policy_target). Retry queued (E15).

### R17 — E11 persistent-walk maze (job 3613349): still no hard modes

Hold-4 walk, T=128: `none` res_ratio grows 0.042→0.121 (K=4→32) but
bimodal ≤0.06, gmm2 ≤0.31, gap ~0.55 — occupancy fills corridors smoothly
rather than splitting into separated route modes. Random exploration (any
persistence) fails the registered threshold ⇒ per pre-registration, the maze
route requires GOAL-DIRECTED expert data. Given two strikes, second-domain
effort pivots to the 3-door TwoRoom geometry (E14) + optional OGBench expert
data later.
