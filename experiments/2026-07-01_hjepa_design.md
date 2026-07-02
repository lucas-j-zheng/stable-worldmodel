# H-JEPA over LeWM latents: design + logic/math check — 2026-07-01

**What this is.** A design doc for the stacked-JEPA (H-JEPA) idea (IDEAS.md
"Stacked JEPAs for temporal hierarchy"), written *after* pressure-testing the logic
and math against every prior result in this project. It (1) records the reasoning
that makes or breaks the idea, (2) lays out an experiment ladder with explicit kill
criteria, and (3) specifies the cheap rung-1 gate that must pass before any level-2
encoder is built. The headline: the idea is coherent, and the math *strengthens* it
in two ways not in the original backlog entry — but it carries two real constraints.

---

## 0. Where this sits in the arc

The project's law: **an MSE-trained model learns the conditional mean
`E[y|x]`, so it mode-averages; a generative model samples a mode. Generative beats
deterministic exactly where the modeled conditional is multimodal.** Proven on the
POLICY side (diffusion +10 vs same-backbone MSE on multimodal-demonstrator data).
The DYNAMICS side was never fairly tested: `p(z'|z,a)` is near-deterministic on every
domain (det_R² 0.86 PushT, 0.50 TwoRoom), and the bounded-nav POMDP tracks (hidden
doors, hidden drift) failed to manufacture multimodal dynamics
(`2026-06-28_pomdp_dynamics.md`). **H-JEPA is a candidate for the missing
multimodal-dynamics cell — via temporal abstraction rather than partial
observability.**

---

## 1. The core law is about the LOSS, not "deterministic"

`argmin_f E[‖y − f(x)‖²] = E[y|x]` exactly. On a bimodal target with modes at ±a,
`E[y|x]=0` — a low/zero-density point (drive into the wall between two doors). The
whole edge is the L2 objective picking the mean; it has nothing to do with
architecture (your +10 decomposition already isolates this). **Precision fix for all
writeups:** say "MSE baseline," not "deterministic." The baselines are MSE-trained so
the identification holds in practice, but the mechanism lives in the loss.

## 2. Why the dynamics half kept losing — deeper than "sampler overhead"

Split dynamics multimodality:

- **Aleatoric** (POMDP drift, env stochasticity): *nature* picks the mode; the planner
  **cannot** control it.
- **Controllable / reachability**: *actions* pick the mode.

For **aleatoric** multimodality a planner wants `E[reward | actions]` (it must
marginalize over nature). But the MSE model hands you the certainty-equivalent mean
for free (exactly optimal for LQG), while a diffusion model needs *many samples per
action sequence* to recover that same expectation — adding variance and cost for no
gain. **⇒ Aleatoric dynamics multimodality gives a generative model no planning edge,
even when it is real and substantial.** This is a stronger statement than "overhead
outweighs it": it means the POMDP dynamics track was doomed at the *planning* stage
regardless of the screen, because drift multimodality is uncontrollable. No amount of
aleatoric multimodality completes the dynamics half.

## 3. Why H-JEPA escapes §2 — and it is the same win as the policy result

Level-2 models `p(Z_{t+K} | Z_t, coarse_a)`, where `coarse_a` **marginalizes out the
fine actions** `a_{t:t+K}`. But fine actions are the *control variable*, so the
resulting modes are **controllable reachability**: the set of coarse outcomes reachable
from `Z_t` depending on how fine control is used. That is a **menu the planner chooses
from**, not noise it must average over. And the mode-averaging failure now bites in a
planning-relevant way: a deterministic level-2 model proposes the **average subgoal**
— between the reachable modes, i.e. possibly *unreachable* — sending the low-level
planner toward an infeasible target; the generative level-2 model proposes reachable
subgoals. **This is the proven policy win (+10) lifted to the subgoal level.** The
level-2 "dynamics" model is really a *proposal/policy over abstract actions* — a more
honest and stronger framing than "diffusion dynamics."

**Composed-determinism corollary (the make-or-break):** condition on the *full*
`a_{t:t+K}` and you compose K near-deterministic maps → deterministic → no modes → no
win (the PushT trap at longer horizon). Drop the fine actions → controllable modes →
win. The multimodality *is* exactly the information you throw away, and it is
throw-away-able precisely because it is controllable.

## 4. The doors domain that died at 1-step is RESCUED at K-step — with a slope

POMDP doors read det_R² 0.966 ("ambiguity bites only in a thin sliver at the wall — a
rounding error across 200k transitions"). That is a **1-step** count. At K-step:

- fraction of *transitions* where the door matters ≈ `1/T` (T = episode length);
- fraction of *K-windows* that straddle the door choice ≈ `K/T`.

Temporal abstraction converts a **sparse per-step bifurcation into a pervasive
per-window one, scaling ~K×.** Quantitatively, for door_prob=0.5 (half the episodes
make a genuine door commitment) and `T≈40`, at `K=8`:

    residual_bimodal_frac(K=8) ≈ 0.5 · K/T ≈ 0.5 · 8/40 ≈ 0.10

vs ≈ 0.01 at K=1 (the rounding error that read det_R² 0.966). **This is a falsifiable
prediction, not a hope:** bimodal_frac should rise ~linearly in K with slope ≈
(fraction of genuine-choice episodes)/T. (Measure T from the data; the prediction
scales as K/T.)

**Domain requirement that falls out:** the win needs **hard branching route
structure** at the K-scale — doors, bottlenecks, obstacles, contact-mode switches.
Free-space navigation gives a *smooth* reachable disk → higher residual variance but
**unimodal** → det_R² drops, bimodal_frac stays ~0 → no win. "Bigger K" alone is not
sufficient; the domain must have discrete route choice. TwoRoom-with-doors qualifies
at K-step for the same reason it was useless at 1-step.

## 5. Which metric to read flips vs the POMDP screen

POMDP screen: read det_R²/residual_ratio (aleatoric *magnitude*), **not** bimodal
fraction. H-JEPA screen: the multimodality is the **signal** and is **hard-separated**,
so **read residual_bimodal_frac** (Sarle BC has full power on separated modes — 0.978
on the synthetic control). The metric-to-read flips because the *question* flips
(magnitude of uncontrolled noise vs shape of the reachable set). The detrend
cooperates: the local-linear fit absorbs `coarse_a`'s deterministic effect and passes
*between* the two door-branches, so even a small residual gets correctly flagged
bimodal.

## 6. The +10 policy puzzle is probably an instrument false-negative (and doesn't threaten this)

The open puzzle (`PROGRESS_ONEPAGER.md`): diffusion policy beats MSE +10 yet the
matched conditional screens *unimodal* (residual_bimodal ~0.006). Sarle's BC only
detects **hard, separated** modes; "many humans solving slightly differently" produces
**soft, overlapping** multimodality — BC's blind spot. So ~0.006 is strong evidence
against *gross* modes and near-zero against *soft* ones; the parsimonious reading is a
softly-multimodal conditional BC can't see. **Testable methodological upgrade:** swap
BC for **Hartigan's dip test** or a **GMM BIC(k>1) vs BIC(k=1)** comparison (real power
against soft modes). Crucially this does **not** threaten H-JEPA: BC is weak exactly
where the policy puzzle lives (soft modes) and strong exactly where H-JEPA needs it
(hard route modes).

## 7. SIGReg at level-2 is a genuine open question — NOT a freebie

An isotropic marginal and a multimodal conditional *can* coexist, but not
automatically: `p(Z)=∫ p(Z|C)p(C)dC = N(0,I)` is a **global constraint on how the
conditional modes are arranged** — they must average to isotropic Gaussian. Two
symmetric clusters at ±μ marginalize to a bimodal/shell marginal, *not* N(0,I), so
SIGReg would **distort the modes** to satisfy the marginal, potentially suppressing the
branching the level-2 predictor needs. (Correction to an earlier over-optimistic
take: coexistence is *possible*, not *free*.) Make it a measurement, not a debate —
train `g` with SIGReg **on vs off** and plot the level-2 SIGReg statistic against
level-2 predictive loss. That Pareto curve is the actual open scientific result
(whether LeJEPA isotropy composes up an abstraction hierarchy, arXiv:2511.08544); do
not pre-judge it.

---

## Experiment ladder (each rung gates the next)

1. **Coarseness × K screen** (encoder-free, no retrain). Does `p(state_{t+K} |
   state_t, coarse_a)` branch as conditioning coarsens? **Greenlight:** `coarse=full`
   flat/low across K; `coarse=sum/none` rises with K; K=8/sum ≈ 0.10 on door_prob=0.5.
   **Kill:** flat & low under all conditioning (reachable sets smooth, wrong domain).
   *Tool ready:* `--mode dynamics_k --horizon K --coarse-action {full,sum,none}`;
   runner `scripts/slurm/hjepa_kstep_screen.sbatch`. ~1h, gpu-debug.
2. **Confirm in latent space.** Repeat rung 1 with `--target-col latent` on the cached
   LeWM latents (drop `--cond-cols state`, let it use the latent base). Confirms the
   branching survives the encoder (SIGReg should preserve separation). Cheap.
3. **Level-2 encoder `g` + SIGReg Pareto.** Small encoder over a window `z_{t:t+K}`→`Z`
   (~64-d), JEPA loss + SIGReg. Train SIGReg on/off; plot isotropy vs predictive loss
   (§7). Yields the composition result regardless of outcome.
4. **Deterministic vs diffusion level-2 predictor** (open-loop, in level-2 space),
   `p(Z_{t+K} | Z_t, coarse_a)`, same controlled comparison as level-1 D-MPC-vs-MPC.
   *This is the symmetric half of the thesis:* does the diffusion edge finally appear
   where the screen said it should?
5. **Hierarchical closed-loop planning** vs flat F=8 CEM on a long-horizon task.
   Level-2 CEM picks a subgoal corridor; level-1 CEM refines inside it. Track two
   independent payoffs: (i) diffusion > deterministic *at level 2* (thesis), (ii)
   hierarchical > flat CEM (capability). They can dissociate — report separately.

**Discipline notes.**
- Keep level-1 **frozen** for rungs 1–5 (clean controlled comparison); co-training is
  a later, separate arm (changes two variables at once).
- Use a **hand-defined** coarse action (net displacement `Σa`) for the screen and the
  first predictor comparison — multimodality guaranteed present by construction. A
  *learned* coarse action (macro-action/subgoal) can collapse toward the fine actions
  to minimize prediction loss and thereby *un*-coarsen away its own multimodality;
  only learn the abstraction after the win is confirmed with a fixed one.
- `sum` vs `mean` coarse actions are identical under the local-linear detrend (scale
  only), so only `sum` is offered.
- K-step windowing shrinks the *effective* (independent) sample count; sliding windows
  still yield ~N−K rows but they are correlated — the same density confound the
  detrend/overfit-guard were built for. Watch `overfit_risk` in the output.
- Marginal drift at level-2 breaks the cosine-diffusion terminal trick — re-apply
  SIGReg at level-2 or renormalize (same caveat as the co-trained-encoder track).

## Verdict

Coherent, and the math strengthens it: (§3) H-JEPA's multimodality is *controllable
reachability*, the planning-exploitable kind the POMDP route structurally could not
produce, and its win is the proven policy win lifted to the subgoal level; (§4) the
doors domain is rescued at K-step with a predicted ~K/T slope. Two real constraints:
the domain must have **hard branching structure** at K-scale (§4), and **SIGReg
composition is an open question, not free** (§7). No fatal flaw. The one thing that
would kill the dynamics half — composed determinism under full-action conditioning —
is exactly what rung 1 detects, cheaply. **Run rung 1 first; hold the K=8/sum /
door_prob=0.5 cell against ~0.10 as the go/no-go number.**

## Artifacts
- Instrument: `scripts/data/multimodality_diagnostic.py --mode dynamics_k`
  (`--horizon`, `--coarse-action {full,sum,none}`; encoder-free via `--target-col
  state`). Self-test unchanged (unimodal 0.000 / bimodal 0.978 / trimodal 0.477).
- Runner: `scripts/slurm/hjepa_kstep_screen.sbatch` (collect door_prob=0.5 + 4×3 grid).
