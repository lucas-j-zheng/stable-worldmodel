# H-JEPA plan: full reasoning pass, corrections, and extensions — 2026-07-01

**What this is.** A pressure-test of `2026-07-01_hjepa_design.md` against every prior
result in the project (PROGRESS_ONEPAGER, 06-28 POMDP verdict, 06-30 attribution
screen, the dynamics_k instrument as implemented), plus the extensions needed to turn
the ladder into a defensible paper. Structure: §A what holds up, §B corrections the
design doc needs, §C extensions (prioritized), §D paper skeleton, §E revised ladder
and order of operations.

---

## A. What holds up (and why, precisely)

**A1. The loss-not-architecture framing is exactly right.**
`argmin_f E‖y−f(x)‖² = E[y|x]` is a theorem; "MSE baseline, not deterministic
baseline" is the correct terminology fix and should propagate to every writeup. The
+10 policy decomposition (same transformer backbone, swap objective) already isolates
this empirically — the strongest asset the project owns.

**A2. The aleatoric vs controllable split is the right taxonomy — it is the paper's
conceptual core.** The POMDP arc (doors: det_R² 0.966, ambiguity too sparse; drift:
signal shrinks with magnitude via boundary clamping) plus the §2 argument explains
*why the whole dynamics track kept losing* and predicts *where a win must live*:
multimodality whose mode index is set by the agent's own (marginalized) actions, so a
planner selects among modes instead of averaging over them. Temporal abstraction is
the only mechanism on the table that manufactures this from near-deterministic 1-step
physics. (But see B1 for a needed weakening of the "aleatoric gives NO edge" claim.)

**A3. The K/T rescue argument is sound and falsifiable — extend it to the full
curve.** The linear rise `bimodal_frac(K) ≈ p_choice·K/T` holds for K ≪ T; as K
approaches the commitment span every window straddles the decision and the curve
saturates at ≈ p_choice. Pre-register the *shape* — linear rise then saturation, with
the fraction-of-windows-containing-a-uniform-decision-time form roughly
`p_choice·min(K/(T−K), 1)` — not just the K=8 point. A measured curve matching a
dated, pre-registered functional form is far stronger paper evidence than one cell.

**A4. The ladder discipline is right.** Frozen level-1 (single controlled variable);
hand-defined coarse action before any learned abstraction (a learned macro-action can
collapse toward the fine actions and un-coarsen away its own multimodality); cheap
encoder-free gate before any training; kill criteria written before the run. Keep all
of it. Add one more standing rule from the project's own history: **never
model-select on eps-loss or open-loop MSE** — both anti-correlated with closed-loop
success in the D-MPC arc. Selection at rungs 4–5 must be on the pre-registered screen
metrics (rungs 1–3) and closed-loop success (rung 5) only.

**A5. The metric flip (§5) and the BC-power argument (§6) are correct.** Sarle BC has
power on hard-separated modes (0.978 synthetic) and none on soft ones; route modes at
K-step are hard-separated by construction. Reading residual_bimodal_frac here while
reading det_R²/residual_ratio on the POMDP screen is principled, not cherry-picking —
the question flips (shape of the reachable set vs magnitude of uncontrolled noise).

---

## B. Corrections the design doc needs

### B1. §2 "aleatoric multimodality gives NO planning edge" is too strong as stated

Certainty equivalence is exact for linear dynamics + quadratic cost; outside LQ, a
mean model is *biased*: `E[R(s_{t+K})] ≠ R(E[s_{t+K}])` whenever reward is nonlinear
over the residual spread (reward cliffs, collision constraints — the mean trajectory
of two door-branches passes through the wall even when the branching is aleatoric).
The precise, defensible statement distinguishes **two kinds of edge**:

- **Expectation-recovery edge** (aleatoric): a generative model can de-bias
  `E[R|actions]` by sampling, but needs many samples per candidate action sequence,
  adds variance, and the planner *cannot* steer the mode. Weak, sample-hungry, often
  eaten by sampler overhead — and empirically untested here because the POMDP domains
  never produced substantial aleatoric multimodality in the first place.
- **Mode-selection edge** (controllable): the planner optimizes *over* modes; one
  sample per mode suffices because sampled outcomes are decision alternatives, not
  Monte-Carlo estimates. Strong, cheap, and exactly what temporal abstraction creates.

For the paper: claim the dichotomy and that H-JEPA targets the strong kind; do not
claim the weak kind is exactly zero (a nonlinear-reward counterexample kills that
sentence in review).

### B2. §3 "this is the proven policy win lifted to the subgoal level" — overstated,
and fixing it makes H-JEPA *more* important, not less

The 06-30 attribution screen **falsified** the clean mechanism story at level 1: under
the conditioning the diffusion policy actually uses (history + the +8 latent goal),
the modeled conditional screens *unimodal* (residual_bimodal ~0.006, multimodal data
indistinguishable from unimodal), yet the +10 persists. So the project has **never
demonstrated** the strict thesis "generative beats MSE *because* the modeled
conditional is multimodal at test time" — the policy win tracks the demonstrator's
multimodality through a mechanism that is still open.

Consequences for the H-JEPA plan:

1. **Reframe:** H-JEPA is not "the proven win lifted." It is **the first clean shot
   at proving the strict thesis at all**, because at level 2 the modeled conditional
   is hard-multimodal *by construction and by measurement* (rungs 1–2 verify it under
   the exact conditioning the predictor will use). If diffusion wins at rung 4, the
   project finally has a cell where the win and the measured conditional
   multimodality coincide — closing the 2×2 that the policy side left open.
2. **Inherit the 06-30 lesson as a hard rule:** the +8-latent-goal case looked
   lossy but resolved the modes anyway. Therefore: **screen the exact conditional the
   level-2 predictor will model** — same conditioning variables, same
   representation (g-encoded, if rung 3 changes the input), same history length —
   before training it. A conditioning-matched screen is a *prerequisite* for rung 4,
   not an optional robustness check.

### B3. The Σa coarse action may be nearly sufficient in an integrator env — add an
intermediate outcome branch to rung 1

TwoRoom dynamics are `next = pos + action·speed` except at walls (06-28 doc). Absent
wall contact, `state_{t+K} = state_t + speed·Σa` **exactly** — conditioning on the
recorded action sum determines the endpoint, and door-A vs door-B windows have
*different* Σa (they head toward different doors). So the residual multimodality in
the `sum` cells comes only from wall-clamped windows (commanded ≠ realized) — the
K=8/sum ≈ 0.10 prediction implicitly assumes Σa carries ~no route information, which
is false in free space and only partially true near the wall.

This creates a **third rung-1 outcome** the design doc doesn't handle:

| outcome | reading | action |
|---|---|---|
| `full` low, `sum` & `none` rise with K | greenlight as written | proceed |
| flat & low everywhere | domain lacks K-scale branching | kill (as written) |
| **`full` & `sum` low, `none` rises** | branching is real but Σa indexes the mode (integrator sufficiency) — **not a kill** | switch coarse action (B4) and/or take the proposer path (B5) |

Cheap additions that disambiguate: report `det_R²` of the `sum` cell (if ≈1, Σa is
sufficient — mechanical, not thesis-relevant); add coarser hand abstractions —
**quantized net direction** (e.g. 8-way), **first-half-only sum**, **duration only**
— as extra `--coarse-action` choices. A monotone dose–response of bimodal_frac in
*coarseness* (bits of action information retained) is itself a strong paper figure.

### B4. Make explicit the logical link: *if the coarse action fully indexes the modes,
deterministic level-2 suffices and the thesis evaporates at that abstraction*

If `p(Z_{t+K}|Z_t, coarse_a)` is unimodal for the coarse-action space the level-2
planner optimizes over, then an MSE level-2 model is fine and there is no generative
edge — hierarchy may still help (capability), but the thesis half dies *at that
abstraction choice*. The thesis needs abstract actions coarse enough to leave
residual multimodality yet still useful for planning. This is a real, reportable
tension — name it in the paper ("the abstraction dial": full → composed determinism,
none → maximal branching) rather than letting a reviewer discover it.

### B5. `coarse=none` is not just an upper bound — it is the planning-relevant cell
for the cleanest instantiation

A subgoal **proposer** `p(Z_{t+K}|Z_t)` conditions on no actions at all. Under
`coarse=none`, a deterministic "model" degenerates to the conditional mean — the
average subgoal, mid-wall, infeasible — while a generative model emits the reachable
menu. This is the sharpest possible det-vs-generative contrast, and it is what
Director-style hierarchical planners actually use (sample candidate subgoals, score
them, hand the winner to the low level). Concretely, rung 4 should have two arms:

- **4a (proposer arm, primary):** MSE vs diffusion `p(z_{t+K}|z_t)`; the planner
  scores sampled subgoals by task value + level-1 reachability. The deterministic
  baseline must be given CEM-noise perturbations around its mean (fairness: CEM
  jitter could stumble onto modes).
- **4b (action-conditioned arm):** MSE vs diffusion `p(z_{t+K}|z_t, coarse_a)` with
  the coarse action chosen by the B3 screen to retain multimodality.

### B6. Rung-3 windowing has a train/plan asymmetry — fix the segment convention

The design doc's `g: z_{t:t+K} → Z_t` encodes the *future* window starting at t; at
plan time that window hasn't happened. Convention that works: level-2 tokens are
**non-overlapping past segments** — `Z_t = g(z_{t−K:t})`, predictor maps segment
(t−K..t] → segment (t..t+K]. Train and plan then use the same map. (Alternative that
sidesteps g entirely at first: see C3.)

### B7. Modes are *behaviorally* reachable, not truly reachable

`p(Z_{t+K}|Z_t, coarse)` branches only over routes the **data-collection policy**
took. Feature: proposals are demonstrated-feasible (an offline-RL-style behavior
prior for free — the proposer never suggests a subgoal no trajectory achieved).
Limitation: no novel-route proposals, and a coverage requirement on the dataset
(door_prob=0.5 guarantees both modes are demonstrated; a real dataset might not).
State both in the paper; the asymmetric-dose experiment (C7) probes the boundary.

### B8. SIGReg §7 — sharpen the geometry, and note TwoRoom is near the worst case

The marginal constraint aggregates over anchors: per-anchor bimodal conditionals
whose mode axes *vary* across anchors can mix to a near-isotropic marginal — the
conflict is severe only when the branching direction is **globally aligned** across
the dataset. In TwoRoom it *is* globally aligned (one wall, one door axis), so this
domain is close to the adversarial case for SIGReg — which makes the rung-3 Pareto a
strong test, not a soft one. Additions: (i) measure the marginal *projected onto the
known door axis*, not only the global statistic — that is where distortion will show
first; (ii) run a **VICReg-style variance–covariance arm** (decorrelation without
full Gaussianity) and an **EMA/stop-grad arm** (no marginal constraint) as
anti-collapse alternatives — turning "does LeJEPA isotropy compose?" into "which
anti-collapse mechanism composes best up a temporal hierarchy?", a more useful
result under either outcome.

### B9. Instrument nits (cheap, do before the big runs)

- **Overlapping windows:** sliding windows share K−1 steps; effective N ≪ row count.
  Add a `--stride` option and confirm conclusions hold at stride=K (non-overlapping).
  (`overfit_risk` flags density, not correlation, so this is a distinct check.)
- **sum-vs-mean "identical" claim:** true for the local-linear *fit*, but k-NN
  neighbor selection is scale-sensitive — verify the conditioning dims are
  standardized before the k-NN, else sum (scale K) vs mean (scale 1) weight the
  action block differently.
- **Soft-mode power (§6 upgrade):** add Hartigan's dip and/or GMM BIC(k=2 vs 1)
  alongside Sarle BC — needed anyway for the policy-puzzle follow-up, and here it
  guards against under-reading partially-overlapping route modes at small K.
- Episode-boundary handling is already correct in `dynamics_k` (windows never cross
  episodes; `T ≤ K` episodes skipped) — verified in code.

---

## C. Extensions (prioritized; each with the claim it buys)

**C1. Mean-infeasibility metric — measure the *mechanism*, not just bimodality.**
(S, do at rung 1.) Bimodality alone doesn't prove the MSE model's prediction is
*bad*; the paper's mechanism is "the conditional mean is an infeasible subgoal." Add
to the instrument, per anchor: fit GMM k=2 on the residual, report (a) density at
the modes' midpoint ÷ density at modes ("mean-gap ratio"), and — encoder-free in
state space — (b) **mean-in-wall fraction**: is the conditional-mean endpoint inside
an obstacle / below an occupancy threshold of the data? This is the single most
direct piece of evidence "deterministic level-2 proposes unreachable subgoals" and
costs nothing extra. It also gives the money visual: anchors on the map with the two
mode endpoints and the MSE-mean endpoint sitting inside the wall.

**C2. Dose–response as the central figure — "the screen predicts the win."**
(M, spans rungs 1→5.) Run the K×coarse screen at door_prob ∈ {0.5, 0.7, 0.9, 1.0}
and, at rung 4/5, the det-vs-diffusion gap at the same doses. Plot planning gap vs
screened bimodal_frac across doses. If the gap tracks the screen (as the policy +10
tracked its dose), the paper's claim upgrades from "diffusion won once" to **"a
cheap, encoder-free, pre-training data screen predicts when a generative level-2
model pays off"** — the actually useful result for practitioners deciding whether to
pay the diffusion cost.

**C3. Minimal instantiation first: temporal-skip predictor, no level-2 encoder.**
(M, restructures rungs 3–4.) The thesis comparison needs only
`p(z_{t+K} | z_t, coarse)` **in the frozen level-1 latent space** — a "K-step-skip
JEPA" with no g, no SIGReg question, no new representation. Diffusion machinery for
latents already exists in the D-MPC harness (retarget from z_{t+1} to z_{t+K},
swap conditioning). Subgoal handoff to level-1 is trivial (the prediction *is* a
level-1 latent). This decouples the paper's contribution #1 (generative edge from
controllable multimodality) from contribution #2 (does window abstraction + SIGReg
compose — the g of rung 3), so a SIGReg-side failure cannot sink the thesis result.
Rung 3 then becomes an *upgrade arm*: does g's abstraction improve over raw z_{t+K}
targets (compression, planning cost, robustness), and how does SIGReg behave.

**C4. Baseline family beyond MSE-vs-diffusion.** (M, rung 4.) Reviewers will ask
"is diffusion necessary, or any multimodal head?" The thesis is about the objective,
so the informative ladder is: MSE → **Gaussian NLL** (calibrated but unimodal) →
**MDN(k)** (minimal multimodal fix) → **CVAE** → **VQ/BeT-style discrete** (route
modes are discrete!) → **diffusion** → optional **flow matching**. Plus the honesty
baseline: **k-NN retrieval sampler** (return an actual observed z_{t+K} from a
similar (z_t, coarse) — is learning needed at all?). Prediction worth
pre-registering: MDN/VQ capture most of the gap on 2-door TwoRoom (2 clean modes);
diffusion's edge should appear only where modes are many/continuous — that
refinement (which generative model, when) is itself paper-worthy.

**C5. Distributional evaluation for rung 4 (diffusion NLL is intractable).**
(S.) Per-anchor, sample-based: energy distance / MMD between model samples and
held-out empirical K-step outcomes; **precision** (samples land near data modes →
feasibility) and **recall** (all demonstrated modes covered → menu completeness);
mean-gap density from C1 for the MSE arm. Planner needs precision AND recall — an
MSE model has ~zero precision at bimodal anchors (mean in the gap), an
over-conservative generative model has low recall (drops the minority door). Report
both; they dissociate.

**C6. PushT K-step re-screen — free, and potentially the second domain.**
(S, immediately after rung 1.) PushT read unimodal at 1-step (det_R² 0.86) with a
single scripted expert. But contact manipulation at K-step may branch (approach the
T from the left vs right — a contact-mode choice). The data and encoder are already
cached; `--mode dynamics_k` runs on them today. If PushT branches at K=8, the paper
gets a manipulation domain for generality at ~zero collection cost; if it doesn't
(single scripted expert → one route), that *confirms* B7's coverage requirement —
informative either way. Same trick for the pointmaze/antmaze collectors if a third
domain is wanted (mazes are the canonical hard-branching-at-K-scale family).

**C7. Asymmetric doses: mode coverage vs mode averaging.** (S, one more collection
knob.) door_prob = 0.7/0.3: the MSE mean shifts toward the majority door but stays
infeasible; an under-fit generative model may drop the minority mode entirely
(recall failure); a good one keeps both with correct weights. Distinguishes "avoids
averaging" from "covers the menu" — and probes B7's coverage boundary continuously.

**C8. Level-1 competence gate before rung 5.** (S.) Measure level-1 CEM's
goal-reaching success as a function of subgoal distance *first*; choose K so
subgoals sit inside the competence radius. Otherwise rung 5 confounds level-2
proposal quality with level-1 reachability, and a true level-2 win can read as a
failure. This is a diagnostic, not a result — but skipping it is the most likely way
rung 5 produces an uninterpretable negative.

**C9. Fair rung-5 baselines.** (M.) (i) flat CEM at longer horizon; (ii)
**compute-matched** flat CEM (same wall-clock / model calls as the hierarchical
stack — hierarchy must beat *equal compute*, not just F=8); (iii) deterministic
subgoal + CEM jitter (B5); (iv) iCEM/MPPI variants if planner choice looks
load-bearing. Report the two payoffs separately as the design doc already says:
diffusion>MSE at level 2 (thesis) and hierarchical>flat (capability) — they can
dissociate and both dissociation outcomes are publishable.

**C10. Learned coarse actions with an information bottleneck.** (L, post-win arm.)
The doc's warning (learned abstractions un-coarsen to minimize prediction loss) has
a principled fix: constrain I(coarse_a; a_{1:K}) — a rate penalty (VIB) or a small
VQ codebook of macro-actions. The dial from B4 becomes learnable: find the coarsest
abstraction that still plans. Also the natural bridge to the options/HRL literature.

**C11. Variable K / multi-scale.** (L, future work.) Geometric-horizon conditioning
(γ-model-flavored) or K ∈ {4,8,16} heads; needed eventually for tasks whose decision
points aren't at a fixed scale. Mention as future work; don't build now.

**C12. Stitching probe.** (M, future work.) Temporal abstraction enables trajectory
stitching (compose route segments never demonstrated end-to-end) — a benefit
orthogonal to multimodality. A goal-pair split that requires stitching would isolate
it. Scope creep for this paper; one paragraph in discussion.

---

## D. Paper skeleton

**Working title.** *When Do Generative World Models Pay Off? Controllable
Multimodality from Temporal Abstraction.*

**Contributions.**
1. **Taxonomy + theory (C-level):** aleatoric vs controllable multimodality in world
   models; expectation-recovery vs mode-selection edges (B1); composed-determinism
   corollary (1-step-deterministic physics stays deterministic under full-action
   conditioning, and branches only when fine actions are marginalized); the
   abstraction dial (B4).
2. **A predictive screen:** encoder-free K×coarseness diagnostic with a
   pre-registered K/T law; dose–response showing the screen predicts the
   det-vs-generative planning gap (C2). *The practitioner-facing deliverable.*
3. **Controlled comparison at the subgoal level:** same backbone, same frozen
   latent space, objective swapped — MSE vs (MDN/VQ/diffusion) K-step predictors
   (C4), evaluated distributionally (C5) and closed-loop (rung 5), with the
   mean-infeasibility mechanism metric (C1).
4. **Hierarchy/representation result:** does SIGReg/isotropy compose up a temporal
   hierarchy (rung 3 Pareto, B8) — independently interesting to the LeJEPA line
   whichever way it goes.

**Central figures.** (1) Screen heatmap K × coarseness × door_prob with the
pre-registered K/T curve overlaid; (2) map visual: mode endpoints vs MSE-mean
endpoint in the wall (C1); (3) planning gap vs screened bimodal_frac across doses
(C2) — the thesis in one plot; (4) generative-family ladder bar chart (C4);
(5) SIGReg Pareto (rung 3).

**Related work to position against (name+year level).**
- *JEPA line:* LeCun 2022 (H-JEPA position paper — this is its first controlled
  level-2 dynamics test); I-JEPA (Assran 2023); V-JEPA / V-JEPA-2 (action-conditioned
  latent planning); LeJEPA/SIGReg (arXiv:2511.08544).
- *Hierarchical planning/RL:* Director (Hafner 2022 — closest system: latent
  hierarchy + generative goal proposer; we add the controlled det-vs-generative
  question + the screen); options (Sutton–Precup–Singh 1999); HIRO (Nachum 2018);
  LEAP (Nasiriany 2019); HIQL (Park 2023).
- *Diffusion planning:* Diffuser (Janner 2022); Decision Diffuser (Ajay 2023);
  Hierarchical/jumpy diffusion planners (e.g. "Simple Hierarchical Planning with
  Diffusion," ICLR 2024 — differentiate: they build the system, we isolate *when and
  why* the generative component is necessary); D-MPC; UniPi.
- *Mode-averaging in BC / multimodal heads:* Diffusion Policy (Chi 2023); IBC
  (Florence 2021); BeT / VQ-BeT; MDN (Bishop 1994).
- *K-step / temporally-abstract models:* γ-models (Janner 2020); Clockwork VAE
  (Saxena 2021); successor features.
- *Control theory:* certainty equivalence (LQG, standard); risk-sensitive control
  (for the B1 nuance).

**Threats to validity (pre-empt in the paper).** Single domain family → C6/mazes;
hand-picked coarse action → B3 coarseness dose + C10; Sarle BC idiosyncrasies → B9
dip/GMM cross-check; overlapping-window correlation → B9 stride check; sampler/CEM
hyperparameters favoring one arm → compute-matched baselines (C9) + no tuning on
open-loop losses (A4); frozen level-1 quality ceiling → acknowledged, co-training
explicitly out of scope (separate arm); seeds/CIs → ≥3 unseeded reruns per cell as in
the policy work (±~5/cell noise there sets the bar for claiming gaps).

---

## E. Revised ladder and order of operations

Rungs renumbered; changes vs the design doc in **bold**.

1. **Rung 1 — K×coarse screen** (as built, ~1h gpu-debug) **plus**: extra coarse
   actions (quantized direction, first-half sum) [B3], `--stride` robustness [B9],
   mean-infeasibility metric [C1], **three-outcome decision table** [B3] instead of
   binary go/kill. Hold K=8/sum against ~0.10 but do **not** kill on that cell alone
   if `none` rises.
2. **Rung 1.5 — free generality screens**: dynamics_k on cached PushT [C6]; dose
   grid door_prob ∈ {0.5,0.7,0.9,1.0} [C2, C7]. Pure re-runs of the instrument.
3. **Rung 2 — latent-space confirmation** (as designed), now including the
   **conditioning-matched rule** [B2]: screen the exact conditional the predictor
   will model.
4. **Rung 3 — temporal-skip predictor comparison** [C3, was rung 4]:
   `p(z_{t+K}|z_t, ·)` in frozen level-1 space; **proposer arm (coarse=none) primary,
   action-conditioned arm secondary** [B5]; baseline ladder MSE / NLL / MDN / VQ /
   kNN / diffusion [C4]; distributional eval [C5].
5. **Rung 4 — level-2 encoder g + SIGReg Pareto** [was rung 3; now an upgrade arm,
   not a prerequisite]: past-segment convention [B6]; door-axis-projected marginal +
   VICReg and EMA arms [B8]. Runs in parallel with rung 5 prep; its outcome cannot
   sink the thesis result.
6. **Rung 5 — hierarchical closed-loop**: level-1 competence gate first [C8];
   compute-matched flat baselines + det-proposer-with-jitter [C9]; two payoffs
   reported separately (as designed); ≥3 reruns per cell.

**Do now (this week, in order):** commit the uncommitted dynamics_k + sbatch + design
doc; add the B9/C1/B3 instrument extensions (small diffs to one file); submit rung 1
+ rung 1.5 (PushT + dose grid) as one debug-partition batch; write the rung-1
decision using the three-outcome table. Everything after that is gated on what the
grid says.

**Standing risks not otherwise covered.** (i) Diffusion training in latent space was
finicky in the D-MPC arc (x0/v-param, terminal SNR) — rung 3 inherits those fixes,
and marginal drift at level 2 re-raises the terminal-matching caveat the design doc
already flags. (ii) The 1000-line story requires the policy-side puzzle (06-30) to be
acknowledged in the paper — the H-JEPA result, if positive, becomes the clean cell
the policy side couldn't provide, and the dip/GMM instrument upgrade [B9] can
retroactively test the soft-modes hypothesis there. (iii) If *everything* screens
unimodal even at K=16 under `none`, the honest conclusion is that bounded 2-door nav
lacks K-scale richness — go to mazes (collectors exist) before abandoning the idea.
