# Experiment log

Dated logs of the latent D-MPC work (diffusion world model over frozen LeWM/SIGReg
latents, planned with CEM). Each entry: **Results / What it means / What's next**.
Local-only (gitignored via `.git/info/exclude`) — not pushed.

| Date | Entry | Headline |
|---|---|---|
| 2026-06-14 | [dmpc_results](2026-06-14_dmpc_results.md) · [next](2026-06-14_dmpc_next_experiments.md) | TwoRoom: eps-parametrization bug fixed (x0/v); deterministic predictor **beats** diffusion closed-loop (58% vs 46/30%) — negative result on near-deterministic dynamics |
| 2026-06-15 | [pusht_e6](2026-06-15_dmpc_pusht_e6.md) | E6: bring the (now-correct) diffusion pipeline to PushT's *multimodal* dynamics — the domain that can actually test the D-MPC premise. Pipeline built; running. |
| 2026-06-16 | [multimodality](2026-06-16_multimodality_diagnostic.md) | Diagnostic (validated vs synthetic control): **PushT is near-deterministic/unimodal everywhere** — dynamics, policy, chunked policy. Multi-human PushT also NOT more multimodal. Blocker is the *data*, not the method. |
| 2026-06-21 | [mm_tworoom](2026-06-21_mm_tworoom_behavioral.md) | Built a designed-multimodal TwoRoom (random-door expert) + behavioral test. Ran aground: eval 0%/0% (encoder OOD on 2-door). Sharpens the conclusion: mm is in the **policy**; the D-MPC **dynamics** pipeline can't use it — need a diffusion *policy* or a *stochastic-dynamics* domain. |
| 2026-06-22 | [diffusion_policy](2026-06-22_diffusion_policy_build.md) | Built a latent **Diffusion Policy** + MSE baseline + closed-loop eval. **FIRST POSITIVE RESULT:** on 2-door multimodal TwoRoom, DiffusionPolicy beats the mode-averaging MLP **42% vs 18%** (offset 8, n=50). Diffusion wins where multimodality lives (the policy). |
| 2026-06-28 | [pomdp_dynamics](2026-06-28_pomdp_dynamics.md) | Test the DYNAMICS half: make `p(next\|obs,a)` multimodal via partial observability. Cheap encoder-free gate. Op#1 hidden DOORS = **NEGATIVE** (det_R² 0.966, collisions too sparse); Op#2 hidden DRIFT = **NEGATIVE** (weak ~1% signal, shrinks with magnitude via boundary clamping). **Whole POMDP verdict: bounded-nav can't make dynamics multimodal** — dynamics half stays untested. |
| 2026-06-30 | [attribution_screen](2026-06-30_attribution_screen.md) | Close the policy attribution (audit D2). Conditioning-MATCHED screen (history + +8 *latent* goal, jobs 3575246/3575527): `p(action\|history,goal)` is **unimodal** (residual_bimodal ~0.006) and multimodal data ≈ unimodal. The +10 is real & dose-dependent but **NOT test-time conditional multimodality** — it lives in the training *demonstrations*. Strict "wins iff modeled conditional is multimodal" falsified on the policy side; mechanism now open. |
