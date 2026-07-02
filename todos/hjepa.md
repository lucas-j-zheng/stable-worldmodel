# Stacked JEPAs / temporal hierarchy (H-JEPA over LeWM latents) · TODOs

> Outline: `../IDEAS.md` § H-JEPA. **Full design + logic/math check + experiment
> ladder: `experiments/2026-07-01_hjepa_design.md`. Plan review, corrections +
> extensions + paper skeleton: `experiments/2026-07-01_hjepa_plan_review.md`.**
> Cross-link: rung 1 doubles as Route 2 of `todos/diffusion.md` § P1.

## Plan-review corrections (2026-07-01, details in the review doc §B)

1. **Reframe, don't inherit:** "proven policy +10 lifted to subgoal level" is
   overstated — the 06-30 attribution screen falsified the test-time-conditional
   mechanism at level 1. H-JEPA is the *first clean shot at proving the strict
   thesis*, not a lift of a proven one. Standing rule inherited from 06-30:
   **screen the exact conditional the level-2 predictor will model** before
   training it.
2. **Σa may be nearly sufficient in an integrator env** (free-space endpoint is
   deterministic given the action sum) → rung 1 uses a **three-outcome table**:
   greenlight / kill (flat everywhere) / **switch coarse action** (`sum` flat but
   `none` rises — NOT a kill).
3. **`coarse=none` is the planning-relevant cell**, not just an upper bound: a
   subgoal *proposer* `p(z_{t+K}|z_t)` is where MSE degenerates to the mid-wall
   average by definition → proposer arm is primary in the predictor comparison.
4. If the coarse action fully indexes the modes, deterministic level-2 suffices
   and the thesis dies *at that abstraction* — "the abstraction dial" is a named
   axis (full → composed determinism, none → maximal branching), report it.

## Background (moved from IDEAS.md, 2026-07-01)

Two arguments that came out of the math and aren't in the bullets below: (1) the
multimodality here is *controllable reachability* (planner picks the mode), the
exploitable kind the aleatoric POMDP route structurally couldn't produce — and the
win is the proven *policy* +10 lifted to the subgoal level; (2) the doors domain
that died at 1-step (det_R² 0.966) is *rescued* at K-step, bimodal_frac ~ K/T
(≈0.10 at K=8, door_prob=0.5). Two real constraints: domain needs hard branching
structure at K-scale; SIGReg-at-level-2 is open, not free.

LeWM plans with CEM over **short** horizons in latent space (the predictor is
1-step; F=8 windows). The idea: train a **second JEPA on top of LeWM's frozen
latents** that predicts **K steps ahead** instead of 1, coarse-graining time → a
2-level temporal hierarchy. This is exactly LeCun's **H-JEPA** ("A Path Towards
Autonomous Machine Intelligence": stacked JEPAs, level-1 = short-horizon/
low-abstraction, level-2 = long-horizon/high-abstraction with pooling/
coarse-graining between levels), and the V-JEPA-2-style "encoder +
action-conditioned latent predictor + plan in embedding space" recipe we already run.

- **Concrete build (reuses our stack):** level-1 = frozen LeWM (192-d latents,
  SIGReg-isotropic). Level-2 = a small encoder `g` over a **window** of level-1
  latents `z_{t:t+K}` → a coarse latent `Z`, plus a level-2 action-conditioned
  predictor `p(Z_{t+K} | Z_t, a_{t:t+K})`. Train level-2 with the same JEPA recipe
  (latent-prediction loss + **SIGReg** anti-collapse), then **plan hierarchically**:
  CEM at level-2 over a few coarse steps to pick a sub-goal corridor, level-1 CEM
  refines inside it. Cheaper long-horizon planning than flat F=8 CEM.
- **THE OPEN QUESTION (the interesting bit): is SIGReg still well-defined /
  motivated at the higher level?** Decompose it — these are *different*
  distributions and conflating them is the trap:
  - **Marginal (what SIGReg actually constrains).** SIGReg is a regularizer we
    *impose* on level-2 embeddings, so we can still force the level-2 marginal to
    N(0,I) by construction. LeJEPA's "isotropic Gaussian minimizes downstream risk"
    argument is **source-agnostic**, so the *motivation* survives at level-2. Real
    question: is isotropy **achievable without collapse** when the *inputs* to `g`
    are already N(0,I)-marginal but **temporally correlated** level-1 latents? A
    pooled/aggregated function of correlated near-Gaussians is generally **not**
    isotropic — so forcing it may fight the predictive signal (the level-2 encoder
    has to whiten away exactly the temporal structure it needs to predict). Measure:
    the SIGReg statistic itself on the level-2 marginal pre/post, and whether
    isotropy trades off against level-2 predictive loss.
  - **Conditional `p(Z_{t+K} | Z_t, a)` (what the level-2 *predictor* models — NOT
    what SIGReg constrains).** This is where "different structure" really bites and
    it ties **directly into our multimodality thesis**: at K-step horizons the
    conditional future is **more multimodal / branching** than at 1 step (the exact
    thing the bounded-nav POMDP couldn't manufacture). So level-2 dynamics are a
    *natural candidate* for genuinely multimodal dynamics — meaning a **deterministic
    level-2 predictor may blur across futures and a generative (diffusion) level-2
    predictor could finally win**, closing the dynamics half of the thesis that the
    POMDP track left open.
- **Why it's worth it:** (1) hierarchical/long-horizon planning over LeWM (a real
  capability gain), (2) a principled probe of whether SIGReg/isotropy composes up an
  abstraction hierarchy (open question in the LeJEPA line, arXiv:2511.08544), and
  (3) a clean shot at multimodal **dynamics** via *temporal abstraction* rather than
  partial observability — reusing every screen/diagnostic we built.
- **Watch-outs:** marginal drift breaks the cosine-diffusion terminal trick (re-apply
  SIGReg at level-2 or renormalize, same caveat as the co-trained-encoder track);
  K-step windowing shrinks effective sample count (level-2 has K× fewer transitions —
  same density confound the diagnostic detrend was built to handle); start with a
  **frozen** level-1 (clean controlled comparison) before any joint co-training.

## TODOs (revised ladder, 2026-07-01 review §E — full version in the review doc)

### Now (this week)

- [ ] **Commit + push the uncommitted rung-1 work**: `dynamics_k` mode in
      `multimodality_diagnostic.py`, `scripts/slurm/hjepa_kstep_screen.sbatch`,
      the design doc + review doc (all currently local-only; Oscar can't run them).
- [ ] **Instrument extensions (small diffs, one file, do before the big runs):**
  - [ ] extra `--coarse-action` choices: quantized net direction (8-way),
        first-half sum, duration-only [review B3]
  - [ ] `--stride` option; confirm conclusions hold at stride=K
        (non-overlapping windows) [B9]
  - [ ] **mean-infeasibility metric**: per-anchor GMM(k=2) mean-gap density ratio
        + encoder-free mean-in-wall fraction — the mechanism metric and the money
        visual [C1]
  - [ ] Hartigan dip / GMM-BIC alongside Sarle BC (soft-mode power; also
        retroactively tests the 06-30 policy-puzzle soft-modes hypothesis) [B9]
  - [ ] check conditioning dims are standardized before the k-NN (sum-vs-mean
        scale sensitivity) [B9]
- [ ] **Rung 1 — K×coarse screen (built, ~1h gpu-debug):**
      `hjepa_kstep_screen.sbatch`. Hold K=8/sum against ~0.10 but decide via the
      **three-outcome table**, and check the pre-registered curve shape
      (linear ~p·K/T then saturation), plus det_R² of the `sum` cell
      (≈1 ⇒ Σa sufficiency, mechanical not thesis-relevant).
- [ ] **Rung 1.5 — free re-screens (same batch):**
  - [ ] `dynamics_k` on **cached PushT** — contact-mode branching at K-step?
        Second domain for ~free, or confirms the coverage caveat [C6]
  - [ ] dose grid door_prob ∈ {0.5, 0.7, 0.9, 1.0} × K — feeds the central
        "screen predicts the win" figure [C2]; asymmetric 0.7/0.3 cell for
        mode-coverage-vs-averaging [C7]

### Gated on rung 1

- [ ] **Rung 2 — latent-space confirmation** (`--target-col latent` on cached
      LeWM latents), under the **conditioning-matched rule** [B2].
- [ ] **Rung 3 — temporal-skip predictor comparison** (no level-2 encoder;
      thesis result decoupled from SIGReg): `p(z_{t+K}|z_t,·)` in frozen level-1
      space, retargeted D-MPC diffusion harness. **Proposer arm (coarse=none)
      primary**, action-conditioned arm secondary [B5/C3]. Baseline ladder:
      MSE / Gaussian-NLL / MDN / VQ-BeT / k-NN retrieval / diffusion [C4].
      Distributional eval: per-anchor energy distance/MMD + precision (feasibility)
      / recall (mode coverage) [C5]. Never select on open-loop loss.
- [ ] **Rung 4 — level-2 encoder `g` + SIGReg Pareto** (upgrade arm, can't sink
      the thesis): **past-segment convention** `Z_t = g(z_{t−K:t})` [B6]; SIGReg
      on/off × weight, plus VICReg-style var-cov arm and EMA/stop-grad arm;
      measure the marginal **projected on the door axis** (TwoRoom is near the
      SIGReg worst case — globally aligned branching) [B8].
- [ ] **Rung 5 — hierarchical closed-loop:**
  - [ ] level-1 competence gate first (goal-reaching success vs subgoal distance;
        pick K inside the competence radius) [C8]
  - [ ] baselines: longer-horizon flat CEM, **compute-matched** flat CEM,
        deterministic-proposer + CEM jitter [C9]
  - [ ] report diffusion>MSE-at-level-2 (thesis) and hierarchical>flat
        (capability) **separately**; ≥3 unseeded reruns per cell (±~5 noise bar).

### Later / separate arms

- [ ] Learned coarse actions with an information bottleneck (rate-constrained
      I(coarse; a_{1:K}) or small VQ codebook) — the principled fix for
      un-coarsening collapse [C10].
- [ ] Third domain if needed: pointmaze/antmaze via existing collectors
      (canonical hard-branching-at-K-scale family).
- [ ] Variable K / multi-scale (γ-model-flavored) [C11]; stitching probe [C12] —
      discussion-section future work, don't build now.
- [ ] Paper assembly: contributions C1–C4, five central figures, related-work
      positioning (Director, hierarchical diffusion planners, γ-models, BeT/MDN)
      — skeleton in review doc §D.
