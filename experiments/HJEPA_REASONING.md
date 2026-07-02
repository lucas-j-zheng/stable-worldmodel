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

### E7: second domain — point-maze random-walk K-step screen (hjepa_maze_screen.sbatch)

**What.** Collect 3000 eps × T=64 state-only random-walk point-maze (medium),
screen K∈{1,2,4,8,16} × {full,sum,none}.

**Why now, not gated on E6.** Needed under EITHER E6 outcome: it is the
escalation path if TwoRoom dies, and the paper's second domain if TwoRoom
lives (PushT died in R4). Random-walk maze data is the cleanest reachability
screen: no goal confound (no goal-directed policy), no demonstrator-preference
confound (uniform coverage), branching = pure wall geometry.

**Pre-registered.** (1) `full` composed-determinism control holds. (2) `none`
rises with K, ≥0.2 by K=8–16 if junction spacing is within random-walk reach.
(3) gap_ratio small in branched cells. (4) Known risk: random walk too slow to
reach junctions in K≤16 → unimodal local diffusion; remedy is longer K /
action-repeat, not a kill.

### E8: aux screens (hjepa_aux_screens.sbatch)

**(a) Policy-puzzle retro-test.** The 06-30 "matched conditional is unimodal"
verdict used Sarle BC — blind to soft modes (the design doc's own §6
hypothesis). gmm2_frac has soft-mode power (R0: trimodal 0.952 vs BC 0.730).
Re-run the conditioning-matched policy_goal screen on mm0_latent vs p10_latent.
Pre-registered: soft modes real ⇒ gmm2(mm0) − gmm2(p10) > 0.1 while BC stays
~0.006 for both; gmm2 also equal/floor ⇒ hypothesis (a) of the puzzle is dead
and the +10 mechanism is training-time corruption or calibration, not hidden
conditional multimodality of any softness.

**(b) Seed calibration.** K=8 decision cells at seeds 1,2 → run-to-run σ of
bimodal_frac, needed to judge dose deltas (0.155 vs 0.124) against noise.

---

## Cross-track event — 2026-07-01 ~21:00 (parallel P0a loop, same repo)

The P0a loop (2026-07-01_p0a_mechanism_loop.md) landed two thesis-level results:
**(1) the +10 policy win does NOT replicate under real seeding** (mm05: DP 35.7
vs TMSE 36.0, 3 seeds × n=100; TMSE cross-run sd ~10 — the banked +10 was most
plausibly noise on a high-variance baseline); **(2) the seeded p10 endpoint
shows DP +5 at the UNIMODAL dose** — no multimodality×objective interaction.

**Implications for this loop.**
- The project now has ZERO robust positive diffusion-vs-MSE cells. The design
  doc's "proven policy win lifted to level 2" framing is dead empirically, not
  just methodologically (review B2 anticipated this direction).
- E6 is now THE load-bearing experiment of the whole program: a dose-dependent,
  hard-separated, conditioning-matched multimodal conditional at level 2 would
  be the FIRST such cell in the project — and rung 3 on it the first fair test
  of the law. If E6 fails: TwoRoom out → maze (E7, running) → paper pivots
  toward "rigorous negative + predictive screen methodology".
- Note their "K-step greenlit" reading (iteration 4) predates R2's goal-confound
  discovery — the correct current verdict is outcome 3 + B10 (goal-conditioned
  re-screen pending), logged in HJEPA_RESULTS.md R1/R2.
- **Caveat for the other track's slip env (P1 Route 1):** slip modes are
  per-step ALEATORIC (nature's coin, not the planner's choice). Review B1 /
  design §2: even if the slip screen passes, a diffusion dynamics model gets no
  MODE-SELECTION edge there — only a sample-hungry expectation-recovery edge,
  plus possibly nonlinear-reward bias effects. A screen PASS on slip does not
  predict a planning win; route-structure multimodality (this track) does.

---

## Cycle 3 — 2026-07-01 (after E6 verdict)

### E6 VERDICT: GREENLIGHT — dose restored under goal conditioning

K=8/none, mm0 vs p10: res_ratio 0.225 vs 0.050 (4.5×, ≥2× required ✓),
bimodal 0.451 vs 0.126 (Δ0.33 > 0.10 required ✓), gap 0.51 vs 0.86, sep 5.2σ
vs 2.8σ. Predictions 1–2 met; 3 partially (median gap 0.49 vs predicted <0.3 —
a mixture of deep-gap and shallow anchors, not uniformly deep). The
goal-conditioned action-free proposer conditional is the FIRST conditional in
this project that is (i) multimodal under matched conditioning, (ii)
dose-dependent, (iii) hard-separated, (iv) planner-relevant. Also learned:
`dir`≈`sum` — any executed-action statistic retaining direction leaks the
route; in nav domains the only non-leaking abstraction is action-free.

### E9: RUNG 3 — proposer bench (scripts/train/proposer_bench.py)

**What.** Small heads, same data, same conditioning as the screen:
MSE / Gaussian-NLL / MDN(5) / kNN-retrieval / conditional-DDPM on
p(s_{t+8}|s_t,goal); episode-level split; per-anchor distributional eval vs the
held-out empirical conditional (energy distance, precision, 2-mode coverage,
mean-gap in delta units). Datasets mm0, p10 (dose contrast), dp05 (replication).

**Why state-space first.** Mirrors the screened conditional exactly (no encoder
confound), runs in minutes, and after the +10 collapse the project needs its
first clean det-vs-generative gap measurement at MINIMUM cost before any
latent/level-2 build. Latent-space version (E10) follows only if E9 shows the
gap.

**Pre-registered predictions.**
1. mm0: MSE precision LOW / meangap HIGH; mdn/knn/diff precision HIGH,
   meangap LOW (the mechanism, now at model level).
2. p10: parity across heads — the unimodal control (seeded-2×2 lesson).
3. mdn ≈ knn ≈ diff on mm0 (2 hard modes — any multimodal head suffices;
   diffusion not special). If diff ≫ mdn/knn something is off — investigate
   before believing it.
4. gauss ≈ mse (calibration alone doesn't fix mode-averaging).
5. mode_coverage: generative > 0.8; mse ≈ 0.5.

**Decision rule.** Predictions 1+2 confirmed ⇒ the strict thesis holds at the
proposer level ⇒ E10 latent bench + rung-4/5 build. 1 fails (MSE precision
fine) ⇒ the modes, though present, are too shallow to hurt the mean — measure
gap-depth vs precision relation, rethink. 2 fails (gap on p10 too) ⇒ eval bug
or leakage — audit before anything else.

### E13: bench K-sweep + dose curve (hjepa_bench_ksweep_dose.sbatch)

The "screen predicts the win" central figure: MSE precision deficit vs the
screen's reading, across K∈{4,8,16} (mm0) and door_prob∈{0.5,0.7,0.9,1.0}
(K=8). Pre-registered: deficit grows with K (saturation allowed, T≈25);
deficit monotone in dose; knn/mdn flat. Follow-ups queued behind results:
mini rung-5 closed-loop proposer planning (+level-1 competence gate first),
rung-4 SIGReg Pareto, trimodal (3-door) falsification test of gap_ratio,
per-anchor gap-vs-precision mechanism figure.

---

## Cycle 4 — 2026-07-02

### E14: 3-door trimodal falsification test (hjepa_3door_latent.sbatch)

The sharpest test the mechanism claim allows. 3 doors {40, 112(center), 184},
expert uniform among fitting doors. By symmetry, at route-ambiguous anchors the
conditional mean ≈ the center-route mode — a REAL mode.

**Pre-registered.** (1) screen: gmm2 ≥ 0.6 (≥2-door multimodality) but
gap_ratio median > 0.7 (vs 0.49 on 2-door); (2) bench: MSE precision recovers
to ≥ 0.75 (vs 0.61) despite the multimodality; (3) MSE mode_coverage stays
~1/3 — precision and coverage DISSOCIATE (feasible mean, no menu).
**Falsified if** MSE precision stays ~0.6 → the deficit is about multimodality
generally and gap_ratio is not the operative mechanism.

### E15: latent bench retry with --goal-from final

Latent datasets lack goal columns; destination conditioning (episode final
state) is the conditioning-matched substitute (policy_target-validated).
Predictions unchanged from E10: E9's signature should reproduce in the frozen
LeWM latent space; if it VANISHES there, that is an encoder finding
(route-mode blurring) that redirects the level-2 build to state/goal-relative
features.

### E16: latent bench round 2 (hjepa_latent_bench2.sbatch)

Fix R19's two gaps: run the p10_latent CONTROL (required before any latent
dose-contrast claim) and give the parametric heads capacity/time at 192-d
(hidden 512, 75 epochs, gpu partition, 3h). Pre-registered: (1) p10_latent
mse≈knn parity; (2) tuned mdn/diff ≥ 0.75 on mm0_latent; if diff stays ~0 the
small-MLP DDPM is inadequate at 192-d and the paper's latent-space generative
representatives are kNN/MDN (the law is about objective family, not diffusion
specifically — already established in R11/R14).

### Rung-5 design (build starts when E16 lands; recorded now)

Harness: `scripts/plan/eval_wm.py` (latent MPC, CEM, tworoom.yaml) — extend
rather than replace.
1. **Level-1 competence gate (C8):** run the standard eval with goals at
   graded distances (near→far); success-vs-distance curve sets the subgoal
   step K* the hierarchy can rely on (expect K*≈8 given F=8 CEM).
2. **Proposer arm:** subgoal generator p(z_{t+K}|z_t, goal) — kNN retrieval
   (proven in both spaces) vs MSE MLP (the mean baseline) vs MDN if E16
   rehabilitates it. Space (latent vs state+encode) decided by E16.
3. **Hierarchical loop:** every K steps propose M=16 subgoals, score by
   (i) level-1 reachability proxy (CEM cost after level-1 planning toward the
   subgoal) + (ii) progress-to-final-goal; commit to best; level-1 CEM plans
   toward it with the SAME budget as flat baseline per compute-matched rule.
4. **Baselines:** flat CEM F=8 (stock), flat CEM compute-matched (more
   samples/iters), MSE-proposer + CEM jitter (B5 fairness arm).
5. **Cells:** mm0 dose (0.5) + p10 control × {flat, flat-matched, hier-MSE,
   hier-kNN}; n=50 × ≥3 eval seeds; report thesis (kNN>MSE at level 2) and
   capability (hier>flat) SEPARATELY (they can dissociate).
**Pre-registered:** hier-kNN > hier-MSE on mm0 long-horizon goals (the R11
precision gap must surface closed-loop as wall-crashes/wrong-room commits for
MSE proposals); hier-kNN ≈ hier-MSE on p10; hier-vs-flat reported without
prejudice.

---

## Cycle 6 — 2026-07-02

### E16 verdict (R20): latent dose contrast CONFIRMED; MDN rehabilitated;
DDPM inadequate at 192-d (as pre-registered fallback); NEW symmetric finding —
wrong-head cost runs both directions (MDN pays on unimodal latent data), so
the screen is a head-SELECTION tool. Rung 5 is a GO with kNN/MDN proposers.

### E17: level-1 competence gate (hjepa_competence_gate.sbatch)

Flat deterministic-LeWM CEM on mm TwoRoom, offset ∈ {4,8,12,16,20} (T≈25 caps
offsets). Sets K* (largest offset with ≥40% success) and the flat baseline.
Pre-registered: monotone decrease; K* ∈ [8,16]; if offset-20 still ≥40%, flat
has no long-horizon deficit at data-reachable offsets → rung-5 capability
payoff needs cross-episode goals (thesis payoff unaffected).

### E18: closed-loop subgoal-proposer eval (scripts/plan/eval_hier.py)

Minimal rung-5: stock level-1 CEM untouched; wrapper swaps the goal IMAGE for
a K=8 subgoal frame every 25 steps, from a kNN bank over (state_t, goal_state)
— the E9-validated conditional. The kNN-retrieval framing makes the subgoal a
REAL dataset frame, so no cost plumbing changes and no generative decoding.
Arms isolate mean-vs-sample on the SAME bank, zero training: `sample` (random
neighbor's future frame) vs `mean` (frame nearest the neighbors' mean future
state — the conservative-STRONG MSE analog; snapping can only help the mean)
vs `off` (flat). Deviation from the recorded design, logged: single proposal
per replan (no M-candidate scoring) — v1 tests whether the bench gap surfaces
at all; candidate scoring is an upgrade arm.

**Pre-registered:** (1) sample > mean at offsets {12,16}; (2) sample > flat
(28%/10%); (3) deltas count only if > ~14 (±7/cell noise) pending seed
replicates. Failure reads: sample ≈ mean ≈ flat ⇒ subgoal guidance doesn't
bind (replan cadence/K mismatch — diagnose videos); mean ≈ sample > flat ⇒
guidance helps but snapping rescues the mean (then the raw-mean arm needs a
latent-cost variant to expose infeasibility closed-loop).
