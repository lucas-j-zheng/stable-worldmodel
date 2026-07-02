# POMDP dynamics: can partial observability make diffusion-dynamics win? — 2026-06-28

**Why.** Every diffusion-*dynamics* result lost because, given full state,
`p(next | state, action)` is a delta (deterministic physics). The policy half of
the thesis is proven (diffusion-policy 42 vs 18); the dynamics half was never
*fairly* tested — there was never multimodal dynamics to test on. A POMDP makes
`p(next | obs, action)` multimodal **without faking physics**: hide part of the
state, randomize it across episodes, and the same observation+action yields
different futures. This is LeCun's JEPA latent `z` / the regime JEDI beats
DreamerV3 on (partially-observed Atari). If diffusion-dynamics wins here, the
thesis becomes symmetric.

Discipline: a cheap, ENCODER-FREE screen gates every operationalization BEFORE
any 8h encoder retrain. Screen `p(next_state | obs, action)` [hidden] vs
`+hidden_var` [observed]; hidden ≫ observed greenlights the GPU pipeline.

Build (committed+pushed): `door_state`/`drift_state` columns in
`envs/two_room/env.py:_get_info`; `--target-col` in `multimodality_diagnostic.py`
(encoder-free next-state dynamics); `collect_tworooms_pomdp.py`;
`scripts/slurm/pomdp_dynamics_screen.sbatch`.

## Operationalization #1: hidden DOORS — NEGATIVE (gate failed, job 3471611)

TwoRoom motion is deterministic except at the wall, where pass-vs-bounce depends
on the door layout. Doors aren't in the observation (`state`=agent pos). Randomize
the hidden doors per episode + high-noise exploratory expert; screen next-agent-pos
dynamics with doors hidden vs observed (200k pairs, encoder-free):

| conditioning | det_R² | residual_ratio | residual_bimodal_frac |
|---|---|---|---|
| **HIDDEN** (agent + action) | **0.966** | **0.0076** | 0.27 |
| **OBSERVED** (agent + door_state + action) | 0.994 | 0.0116 | 0.49 |

**Read det_R² and residual_ratio, not the bimodal fraction.** Even with doors
hidden a deterministic local-linear map explains **96.6%** of next-state variance;
leftover stochastic spread is **0.76%** of signal. Partial observability did NOT
make the aggregate dynamics meaningfully multimodal. (residual_bimodal being
*higher* for observed is noise on a ~1% residual — the screen's documented caveat;
the meaningful fact is det_R² 0.966→0.994 when doors are added, confirming the
collision mechanism is real but negligible.)

**Mechanism of failure (same lesson, new costume):** agent dynamics are
`next = pos + action·speed` — deterministic *almost everywhere*. The door
ambiguity bites only in a thin sliver at the wall; across 200k transitions that's
a rounding error. The multimodality is too SPARSE to matter — a diffusion-dynamics
model would have ~nothing to exploit, just like TwoRoom/PushT before. Gate killed
a GPU-days pipeline in 15 min. Self-test passed (unimodal 0.000, bimodal 0.978),
so the instrument is sound.

## Operationalization #2: hidden DRIFT — pervasive (building, gate next)

Fix: make the hidden variable bite EVERYWHERE, not just at the wall. Per-episode
constant unobserved drift ("wind"): `next = pos + action·speed + drift`,
drift ∈ {−d, +d} along x, resampled per episode, hidden from the observation. Now
the same (obs, action) splits into two separated next-state branches on EVERY
step — pervasive bimodality. Bonus: history *resolves* the drift (the Dreamer/JEDI
belief-state story) → directly tests whether the diffusion WM exploits residual
uncertainty a deterministic predictor can't.

Build: `drift_scale` env kwarg (per-episode sign via env RNG in reset; drift added
in step; recorded as `drift_state`); `collect_tworooms_drift.py`;
`pomdp_drift_screen.sbatch`. Same encoder-free gate: screen `p(next | agent, a)`
[drift hidden] vs `+drift_state` [observed].
- PREDICTION: hidden det_R² drops hard (~0–0.3) + residual_bimodal HIGH; observed
  det_R² ~0.99, residual_bimodal ~0. A clean PASS → then (and only then) retrain
  encoder on drift-POMDP renders and run diffusion-dynamics vs deterministic.

### Drift result — QUALIFIED then FAILS TO SCALE (jobs 3505007, 3524804)

drift_scale=3: hidden det_R² 0.956 / residual_ratio 0.0126 vs observed 0.989 /
0.0043. A real, correctly-directed 3× contrast (unlike doors, which had none) —
but modest (~1% residual). Swept larger drift to check it SCALES:

| drift_scale | hidden det_R² | hidden residual_ratio | observed residual_ratio |
|---|---|---|---|
| 3 | 0.956 | 0.0126 | 0.0043 |
| 6 | 0.967 | 0.0071 | 0.0018 |
| 10 | 0.970 | 0.0055 | 0.0016 |

**Bigger drift gives LESS multimodality, not more** (residual_ratio 0.0126 → 0.0055).
Opposite of a real scalable signal. Mechanism: in a BOUNDED arena, strong drift
pins the agent against walls/borders where motion is CLAMPED — and clamping is
deterministic, absorbing the drift variance. The drift POMDP self-defeats at
magnitude.

## WHOLE POMDP VERDICT (2026-06-29) — bounded-nav POMDP can't make dynamics multimodal

Both operationalizations fail: hidden DOORS (collision ambiguity too sparse,
det_R² 0.966) and hidden DRIFT (weak ~1% signal that shrinks with magnitude via
boundary clamping). **In a bounded deterministic-physics navigation env, partial
observability does NOT create substantial multimodal dynamics.** The 8h encoder
retrain is NOT justified. To fairly test diffusion-DYNAMICS, need a domain with
INTRINSIC stochastic/multimodal transitions (random branching, or stochastic
Atari à la JEDI) — not a bounded maze. The dynamics half of the thesis stays
untested for lack of a multimodal-dynamics domain; the POLICY half is supported
(see 2026-06-22 writeup, E1 verdict).
