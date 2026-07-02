# Diffusion vs deterministic — the thesis project · TODOs

> Outline & thesis status: `../IDEAS.md` § Diffusion vs deterministic.
> Everything diffusion lives here: the policy story (P0a/P0), the missing
> multimodal-DYNAMICS domain (P1), the JEDI end-to-end endgame (P2), and the
> D-MPC method/domain backlog. Key notes:
> `experiments/2026-06-22_diffusion_policy_build.md`,
> `experiments/2026-06-30_attribution_screen.md`,
> `experiments/2026-06-28_pomdp_dynamics.md`, `experiments/PROGRESS_ONEPAGER.md`.

---

## P0a — Isolate the +10 mechanism + firm up stats (2026-07-01, do first — cheap)
*Goal: decide whether the paper's mechanism claim is "training-time mode-averaging"
(strong) or "empirical, mechanism open" (weak). All items reuse existing
checkpoints/data; no retraining except the seeded reruns.*

- [x] **Failure-mode probe.** DONE 2026-07-01 (`failure_positions.py` on the
      seeded-cell npz's): **NO mode-averaging signature** — TMSE failures at the
      wall-between-doors 10–14% ≈ DP 9–11%; both mostly fail "elsewhere"
      (~50 px short of goal). Candidate (a) unsupported behaviorally.
- [x] **Smoothing test (candidate a).** DONE 2026-07-01 (`policy_fit_error.py`,
      job 3611948): TMSE regresses mm05 ~1.6× worse than p10 (real target
      heterogeneity, train≈val) — but with the seeded closed-loop gap ≈ 0 the
      fit-level corruption is behaviorally inconsequential. Candidate (a) exists
      at the regression level, explains nothing closed-loop.
- [ ] **Train/eval goal-mismatch check (candidate b).** Re-run the matched screen
      with EVAL-time conditioning — goals synthesized exactly as the solver forms
      them (`diffusion_policy_solver.py:64`), not demo latents. If the eval-time
      conditional is more ambiguous than the demo-latent screen showed, mismatch is
      live; if still ~0.006, rule it out.
- [ ] **Seeded endpoint reruns.** DP + TMSE at door_prob {0.5, 1.0}, REAL fixed
      seeds (audit flag: prior "3-seed" runs were `seed: None`), n=100. 4 jobs.
      Converts the +10 from "probably" (±~5/cell at n=50) to defensible.
      → *2026-07-01: mm05 LANDED (3611839) — **THE +10 DOES NOT REPLICATE:
      DP 35.7 vs TMSE 36.0** (3 seeds × n=100); TMSE cross-run sd ~10. Extension
      to seeds 4–8 running (3611999) for power; p10 cell (3611840) still queued.
      See `experiments/2026-07-01_p0a_mechanism_loop.md` iteration 2 — if the
      null holds at 8 seeds + p10, the policy-side objective effect is DEAD and
      the loop moves to P1.*
- [ ] **Done =** wall-clustering + higher mm0 fit error ⇒ claim training-time
      mode-averaging as the mechanism; otherwise write "empirical, mechanism open"
      and do NOT over-claim. Either way update PROGRESS_ONEPAGER.md.

## P0 — Real measured-multimodality dose axis (cheap, frozen encoder)
*Goal: turn the diffusion-policy result from "illustration" into "proof" with a
real measured-multimodality axis. No encoder retraining.*

- [ ] **Fix the empty-loader crash** (`train_diffusion_policy.py:90`,
      `torch.cat()` on empty list). Likely episode-length vs history+horizon
      windowing drops every sample on `tworoom_greedy2_latent.lance`. Verify the
      loader yields batches before training. (commit `acf492b` only makes it
      error gracefully — root cause still open.)
- [ ] **Build a REAL dose axis = number of doors / mode separation**, not
      `door_prob`. Use `door.number` (1->2->3 fitting doors) with the
      `stochastic_door` expert committing among them. The variation space already
      supports up to 3 doors.
- [ ] **Gate every dose level through the screen** (`policy_goal` mode): record
      measured `residual_bimodal_frac` as the x-axis. Only keep levels that span
      a real low->high range.
- [ ] **DP vs MLP across that screened axis** (offset 8, n=50, in-dist encoder).
- [ ] **Done =** monotone gap-vs-measured-multimodality curve (thesis confirmed)
      OR a documented non-monotonicity (thesis needs rework). Either way, write it
      up and stop over-claiming.

---

## P1 — Multimodal-DYNAMICS domain: the missing half of the thesis
*Goal: a domain where `p(s'|s,a)` is genuinely multimodal, so a diffusion WORLD
MODEL can finally win, with the same dose-knob design as door_prob. If it does,
the thesis becomes symmetric (wins on BOTH multimodal policy AND multimodal
dynamics).*

### Route 1 — intrinsic stochastic transitions (bimodal "slip")
- [x] **Design the slip env.** DONE 2026-07-01: `TwoRoomEnv(slip_scale=...)` —
      per-STEP fair coin, ±slip_scale along the along-wall axis before
      collisions (no conditioning resolves it; axis chosen to dodge the drift
      clamp-absorption failure). `slip_state` recorded for the observed control.
- [ ] **Screen FIRST** (`multimodality_diagnostic.py --mode dynamics`): confirm
      `residual_bimodal_frac` of `p(z'|z,a)` actually rises with the knob before
      training anything. (Standing rule: no diffusion run without a passing screen.)
      → *RUNNING 2026-07-01: `slip_dose_screen.sbatch` job 3612481, S∈{0,2,4,8},
      hidden vs observed contrast.*
- [ ] **Diffusion-dynamics vs deterministic D-MPC** (CEM) across the dose.
      Prediction: deterministic predictor smears across the two outcomes ->
      diffusion finally wins closed-loop, gap tracks the knob.
- [ ] **Done =** diffusion-dynamics > deterministic on a screened-multimodal
      domain, dose-dependent — the symmetric half of the law.

### Route 2 — temporal abstraction (K-step / H-JEPA level-2)
`p(Z_{t+K}|Z_t,a)` may be multimodal even where 1-step is deterministic
(branching futures; bimodal_frac ~ K/T ≈ 0.10 at K=8, door_prob=0.5). The rung-1
gate is built and cheap — see `todos/hjepa.md` (this route doubles as H-JEPA's
first rung; run it there, feed the result back here).

### DEAD — bounded-nav POMDP (kept for the record, do NOT resurrect)
> **Verdict 2026-06-29:** both operationalizations fail. Hidden DOORS: collision
> ambiguity too sparse, det_R² 0.966. Hidden DRIFT: weak ~1% residual that
> SHRINKS with drift magnitude (boundary clamping absorbs the variance). In a
> bounded deterministic-physics nav env, partial observability does NOT create
> substantial multimodal dynamics. The 8h encoder retrain was not justified.
> See `experiments/2026-06-28_pomdp_dynamics.md`.

---

## P2 — JEDI-variant: end-to-end encoder + joint-window denoising
*The original research idea — let the denoising loss shape the latent.*
**BLOCKED on P1** — end-to-end has nothing to exploit on deterministic dynamics,
and it costs the clean frozen-encoder control + opens collapse risk.

- [ ] Prereq: a stochastic/multimodal-dynamics domain (from P1) where
      diffusion-dynamics already wins frozen.
- [ ] Unfreeze the encoder: let the denoising loss train it (JEDI). Inherit the
      collapse stack: stop-grad target, ~0.3x encoder LR, tanh clamp, random
      switching.
- [ ] The novel cell: replace JEDI's single-step AR denoising with **joint
      multi-step window** denoising (we already have F=8 cosine+DDIM windows).
- [ ] Add an end-to-end **decoder** for grounding/visualization/pixel goals
      (note: in tension with JEDI's no-reconstruction ablation — use stop-grad
      readout to preserve the thesis, or co-train as an anti-collapse anchor).
- [ ] **Done =** end-to-end JEDI-variant matches/beats the frozen two-stage
      pipeline on the P1 domain, with the latent shaped by the denoising loss.

---

## Backlog — D-MPC method iterations, fairness & domains
*Context: latent D-MPC = MPC with a **diffusion** dynamics model over frozen
LeWM/SIGReg latents, vs **regular MPC** = same CEM loop with the deterministic
LeWM predictor. Priority: ⭐ high / ◐ medium / ○ someday, effort in parens.*

### Making the comparison fair before concluding
- [ ] ⭐ **Training-budget parity** (S). Diffusion gets few post-hoc epochs; the
      deterministic predictor is trained jointly with the encoder. Train diffusion to
      convergence (watch the loss curve flatten) before any "diffusion loses" verdict.
- [x] ⭐ **Multimodality diagnostic** (S). Built + validated
      (`scripts/data/multimodality_diagnostic.py`; 0.00 unimodal / 0.98 synthetic
      control). Now a standing gate — see methodology rules in IDEAS.md.
- [ ] ◐ **Joint / fine-tuned encoder** (M). Train (or fine-tune) the diffusion
      dynamics jointly with the encoder instead of frozen post-hoc. Directly removes
      the joint-vs-post-hoc confound; may also just work better. → track below.

### Track: jointly-learned latent world model (Dreamer-flavored)
Co-train the encoder with the dynamics instead of freezing it — a distinct, larger
project, not a tweak to the current run. Keep the **frozen-space comparison as the
controlled experiment** (does diffusion beat deterministic in the *same* latent
space); co-training changes two variables at once, so it's a separate arm.
- [ ] ⭐ **Decoder first, as the anchor** (M). The parallel decoder (pixel
      reconstruction loss) is itself an **anti-collapse signal** — you can't collapse
      the latent if you must decode back to pixels. Add it before unfreezing
      anything; it makes encoder co-training safe and is the Dreamer recipe
      (encoder + dynamics + decoder trained together).
- [ ] ◐ **Light fine-tune, not full unfreeze** (M). Low-LR encoder updates with
      SIGReg kept on. Tests whether task-shaping the latents (easier-to-predict,
      more control-relevant) helps, without the worst non-stationarity.
- ⚠️ **Caveats that make this non-trivial** (design around, not ignore):
  - **Collapse.** A latent-space prediction loss *alone* rewards uninformative
    latents (constant latent → perfectly predictable → zero loss). MUST keep an
    anti-collapse term (SIGReg/VICReg/contrastive) and/or the decoder on the moving
    encoder.
  - **Lose the N(0,I) marginal.** SIGReg makes the latent marginal ≈ N(0,I), which
    the cosine diffusion schedule's terminal was chosen to match (no normalization
    needed). A moving encoder drifts that marginal → re-handle normalization / keep
    SIGReg.
  - **Non-stationary target.** Diffusion chases a latent distribution that's
    shifting under it during training; already-finicky diffusion training gets
    harder. Stage it (e.g. warm up encoder, then co-train) and watch for instability.

### Method iterations
- [ ] ⭐ **Train a decoder in parallel** (M). Current setup predicts purely in frozen
      latent space (JEPA — no decoder by design). Add a latent→pixels decoder trained
      *alongside* the diffusion dynamics model (same run), without backprop into the
      frozen encoder. Payoffs: (1) **visualize** what the world model imagines —
      decode predicted/rolled-out latents and the CEM-selected plan back to images,
      instead of staring at MSE numbers; (2) a sanity check on whether good-looking
      latents correspond to sensible states (helps explain the open-loop-MSE-vs-
      closed-loop mismatch); (3) optional auxiliary reconstruction signal. Keep it
      strictly diagnostic at first — decoder is *not* in the planning loop, just an
      observer trained in parallel.
- [x] ◐ **Revisit the dropped action proposal ρ** (M). DONE — became the diffusion
      policy (P0/P0a above), the project's first positive result.
- [ ] ◐ **Better planners** (S–M). iCEM (colored-noise CEM) vs vanilla CEM; MPPI.
      Temporal smoothing may matter more on continuous-control domains than TwoRoom.
- [ ] ◐ **Sampling-knob sweeps** (S). Once a decent checkpoint exists: inference
      steps {10,20,50}, eta {0,0.25,0.5}, K dynamics-samples {1,4,8}. Cheap; only
      worth it if diffusion is already close to the deterministic baseline.
- [ ] ○ **Zero-terminal-SNR / offset-cosine schedule** (M). Cap ᾱ_min so √ᾱ never
      gets pathological — a principled alternative/complement to the ±6 x0-clamp.
      Lower priority now that x0/v fixed the worst of it.
- [ ] ○ **Longer horizon F / receding-horizon tuning** (S). F=8 was sized to
      TwoRoom; multimodal/contact tasks may benefit from longer lookahead or a
      different replanning rate.
- [ ] ○ **Latent value function / learned reward** (L). Was explicitly out of scope.
      Could replace the hand-coded goal-distance score with a learned terminal value.

### Domains to test (all in the quentinll/lewm collection)
- [x] ⭐ **PushT** — done (E6): deterministic 16% vs diffusion 8%; screen reads
      near-unimodal (incl. multi-human data). Negative, explained.
- [ ] ◐ **OGBench-Cube** (M). Manipulation; another multimodal candidate — screen
      it first.
- [ ] ○ **Reacher** (S). Likely near-deterministic — useful as a *negative control*
      (expect diffusion to lose, like TwoRoom). Each new domain is ~1h port now
      (convert encoder → cache → sweep → plan); generalize the converter for
      cube/reacher (same version skew).

### Science question: why does open-loop MSE anti-rank closed-loop?
- [ ] ◐ Investigate (M). x0 was best open-loop but worst planner; v the reverse.
      Hypothesis: mean-sample MSE rewards conservative mode-averaging that gives CEM
      too little useful spread; calibration matters more than point accuracy for
      planning. Worth understanding — it generalizes beyond this project.

---

## Done / superseded (for the record)
- [x] Build: DiffusionPolicy + solver + MLP/TransformerMSE baselines (2026-06-22/23).
- [x] Multi-human PushT screen — reads near-unimodal too; "data source, not task"
      answered, but PushT is not the multimodal-policy domain (2026-06-16..).
- [x] Architecture control (TransformerMSE) + door_prob dose curve: +10 at 0.5,
      ~0 at 0.9/1.0, 3 unseeded reruns (2026-06-29).
- [x] Attribution screen — falsified the test-time conditional-multimodality story;
      matched conditional is unimodal (~0.006) (2026-06-30).
- [x] greedy-closest-of-2 control — ABANDONED (episode-length confound +
      empty-loader crash); door_prob=1.0 is the length-matched unimodal point.
- [x] Bounded-nav POMDP route — DEAD (see P1 record above).
