# H-JEPA autonomous experiment loop — reasoning log

Living doc. One entry per experiment *decision*, written BEFORE results land, with
pre-registered predictions and decision rules. Results go to `HJEPA_RESULTS.md`.
Plan: `2026-07-01_hjepa_design.md`; corrections/extensions:
`2026-07-01_hjepa_plan_review.md`; ladder: `../todos/hjepa.md`.

---

## Cycle 1 — 2026-07-01

### E1: Rung-1 K×coarse screen (job 3611937, gpu-debug)

**What.** `hjepa_kstep_screen.sbatch`: self-test → collect fresh
`tworoom_mm_dp05.lance` (door_prob=0.5) → screen
`p(state_{t+K} | state_t, coarse_a)` over K∈{1,2,4,8} × coarse∈{full,sum,none},
encoder-free, 4000 anchors, k=96.

**Why first.** The pre-registered go/no-go gate for the whole H-JEPA program: the
composed-determinism trap (full-action conditioning stays deterministic) and the
temporal-abstraction rescue (marginalized fine actions branch) are both directly
observable here, at ~1 GPU-hour, before any model is trained.

**Pre-registered predictions.**
1. `full`: residual_bimodal_frac stays low (≲0.03) at every K — composed
   determinism (the control).
2. `none`: rises with K; largest cell in the grid at K=8; rough magnitude
   ~p_choice·K/T ≈ 0.10 at K=8 for T≈40 (measure T from data).
3. `sum`: BETWEEN full and none, with a live risk it hugs `full` — TwoRoom is a
   near-integrator (`next = pos + a·speed` except walls), so Σa is close to
   sufficient for the endpoint in free space (review B3). K=8/sum ≈ 0.10 is the
   design doc's number; treat a low `sum` WITH a rising `none` as
   "Σa-sufficiency", not as a kill.
4. det_R² decreases with K for sum/none (reachable set widens).

**Decision table (three outcomes, review B3).**
- `full` low + `sum`/`none` rise → GREENLIGHT as designed; proceed to rung 2/3
  with Σa as the coarse action.
- flat & low everywhere (incl. `none` at K=8) → domain lacks K-scale branching →
  kill TwoRoom-2-door for this purpose; escalate to mazes (collectors exist).
- `full`+`sum` low, `none` rises → branching real, Σa indexes the mode → switch
  coarse action (quantized direction / first-half sum — instrument extension) and
  prioritize the action-free proposer arm (review B5), which only needs `none`.

### E2: Dose grid + rung-2 latent confirmation (hjepa_dose_latent_screen.sbatch)

**What.** Zero-collection re-screens on datasets already cached from the policy
arc: dose grid `tworoom_{mm0,p07,p09,p10}` (door_prob 0.5/0.7/0.9/1.0), K∈{1,8} ×
coarse∈{full,sum,none}, state target; plus latent-target screens on
`tworoom_mm0_latent` and `tworoom_p10_latent` (rung 2: does branching survive the
SIGReg encoder?).

**Why now (not gated).** Pure instrument re-runs — the marginal cost of running
them concurrently with rung 1 is ~zero and they build the paper's central
dose-response axis (review C2) and the encoder-survival check (design rung 2)
in one batch. Interpretation stays gated: if rung 1 kills, these are archived as
negative-control documentation.

**Pre-registered predictions.**
1. Dose: at K=8/`none`, bimodal_frac decreases monotonically in door_prob and
   ≈ instrument floor (≤0.02) at p10 (unimodal expert). At K=1 all doses ≈ floor.
2. Latent: mm0_latent K=8 sum/none within ~2× of the matched state-target cell
   (SIGReg preserves separation — LeJEPA isotropy is a marginal constraint, the
   two rooms should stay separated in latent space); p10_latent at floor.
3. If latent ≪ state at matched cells → the encoder is blurring route modes →
   level-2 must be built on states or the encoder re-examined (this would be a
   major, reportable finding on its own).

### E3: PushT K-step re-screen (hjepa_pusht_kstep_screen.sbatch)

**What.** dynamics_k on cached PushT: scripted expert (`pusht_latent`, state
target), multi-human (`pusht_human_state`, state target; `pusht_human_latent`,
latent target), K∈{1,4,8} × coarse∈{full,sum,none}.

**Why.** Review C6: PushT read unimodal at 1-step with a scripted expert; contact
manipulation could branch at K-step (left-vs-right approach = contact-mode route
choice). Free second domain if positive; confirms the behavioral-reachability
coverage caveat (review B7) if scripted stays flat while human rises.

**Pre-registered predictions.**
1. Scripted expert: flat at all K (single demonstrated route — coverage caveat).
2. Multi-human: the live cell — if it rises with K under sum/none, PushT is
   rescued at K-step like the doors domain and becomes the paper's manipulation
   domain. Genuinely uncertain; no confident magnitude prediction.
3. `full` control stays low everywhere (composed determinism is domain-agnostic).

**Note on Σa in PushT.** PushT state includes the T-block pose; the agent's Σa
does NOT determine the block motion (contact nonlinearity), so the B3
Σa-sufficiency concern is weaker here than in TwoRoom — `sum` is a genuinely
coarse action for the block DOF.

### Standing rules for this loop
- All compute via sbatch (login node = orchestration only).
- Predictions and decision rules logged here BEFORE reading results.
- No model-selection on open-loop losses at later rungs (project rule).
- experiments/*.md stay local-only; only code + sbatch runners are pushed.

### E4: Instrument extensions (built locally, deploy after E1–E3 finish)

**What.** `--stride` (non-overlapping-window robustness); coarse actions
`sumhalf` (first-half Σa — coarser in time) and `dir` (unit net direction —
coarser in space); per-anchor GMM(k=2) metrics: `residual_gmm2_fraction`
(BIC k2<k1 — soft-mode power where Sarle BC is blind), `gmm2_mean_gap_ratio`
(density at the conditional mean ÷ density at the modes — the direct
infeasible-average-mechanism number), `gmm2_mode_separation`.

**Why.** Review B3 (Σa-sufficiency risk needs coarser hand abstractions to
disambiguate outcome 3), B9 (window correlation; soft-mode power), C1 (measure
the mechanism, not just bimodality). Validated on synthetic controls before any
deployment (see RESULTS R0) — the gap_ratio's trimodal behavior (center mode ⇒
mean feasible) is a bonus: the metric distinguishes "multimodal but MSE-OK"
from "multimodal and MSE-broken".

**Deployment rule.** NOT rsynced to Oscar while E1–E3 cells are mid-sweep (each
cell re-invokes the script; mixing versions inside one sweep would make cells
non-comparable). Deploy + run the extended screen (E5) as soon as the three
jobs reach terminal states.

**Pre-registered predictions for the extended screen (E5, dp05 data).**
1. stride=K at K=8/none: bimodal_frac within ±30% of stride=1 (window overlap
   inflates N but not the signal); if it collapses, the stride=1 numbers were
   correlation artifacts.
2. Coarseness ordering at K=8: full ≤ sum ≤ sumhalf ≤ dir ≤ none in
   bimodal_frac (monotone in information dropped).
3. gap_ratio at K=8/none ≲ 0.2 on dp05 (hard-separated route modes ⇒ mean in a
   deep gap ⇒ MSE level-2 proposals infeasible); ~1.0 wherever bimodal_frac ~0.

---

## Cycle 2 — 2026-07-01 (after R1–R4)

**Where we are.** Rung 1 landed on decision-table outcome 3 (Σa near-sufficient;
branching lives in the action-free cell), BUT R2 exposed a confound the plan
missed: the `none` cell's branching is per-episode GOAL variation, not route
choice — no dose effect (p10 ≈ mm0). New correction (call it B10, the mirror of
06-30/B2): **the screen must condition on everything the planner will KNOW
(state + goal), and marginalize only what it will CHOOSE (the route/actions).**
Under-conditioning manufactures planner-irrelevant multimodality; the goal is
known at plan time, so goal-variation modes are not a menu the planner picks
from — they're a nuisance the screen must remove.

### E5: extended dp05 screen (upgraded instrument) — submitted with E6

Coarseness dose (full/sum/sumhalf/dir/none) × K, stride=8 decorrelation cells,
GMM2 + gap_ratio everywhere. Predictions already registered under E4.
Note E5 is NOT goal-conditioned — its none/dir cells inherit the goal confound;
read them only for the coarseness ordering and the stride check, and use E6 for
any thesis-relevant magnitude.

### E6: goal-conditioned K-step screen — THE decisive cell (hjepa_goalcond_screen.sbatch)

**What.** p(state_{t+K} | state_t, goal, coarse_a) on dp05 (K∈{1,2,4,8} ×
{none,dir,sum}) and the dose endpoints mm0 vs p10 (K∈{1,8} × {none,sum}).
Goal column discovered in-job (name varies by dataset generation).

**Pre-registered predictions.**
1. Dose RESTORED: dp05/mm0 K=8/none bimodal exceeds p10 by >0.10 AND mm0
   residual_ratio ≥ 2× p10 (route branching is real spread, not floor noise).
2. dp05 rises with K (windows straddling the door commitment scale ~K/T).
3. gap_ratio < 0.3 in dp05 K=8/none (MSE-mean-in-gap mechanism); ~1 on p10.
4. Kill branch: if mm0 ≈ p10 ≈ floor goal-conditioned, TwoRoom has no
   controllable K-scale route branching → escalate to maze domain.

**Why this is now the thesis cell.** Goal-conditioned, action-free K-step
prediction IS the subgoal proposer a hierarchical planner uses (Director-style).
If its conditional is dose-dependently bimodal with a deep mean-gap, the MSE
proposer demonstrably proposes infeasible subgoals exactly when the demonstrator
was multimodal — the strict thesis, measured at the conditional the level-2
model will actually be trained on (conditioning-matched, per B2).
