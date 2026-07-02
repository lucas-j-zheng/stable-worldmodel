# E6 — Latent D-MPC on PushT (multimodal test) — 2026-06-15

**Why this experiment.** TwoRoom (2026-06-14) gave a negative result: with the
eps→x0 parametrization bug fixed, the diffusion world model became *correct* but still
lost closed-loop to the deterministic LeWM predictor (46/30% vs 58%). The likely
reason: TwoRoom latent dynamics are near-deterministic — the worst case for a sampler,
since a deterministic predictor loses nothing by not modeling multiple modes. **E6
moves the same fixed pipeline to PushT**, whose contact/pushing dynamics are genuinely
multimodal, to test whether diffusion's stochasticity finally pays off where a
deterministic predictor is forced to blur across modes.

This entry is mostly **infrastructure + setup results** — the scientific closed-loop
numbers are still pending (jobs queued). Recorded now because the setup surfaced
several real findings and dead-ends worth not repeating.

> **UPDATE 2026-06-16 — closed-loop DONE, and it's another negative.** PushT, 50 ep,
> CEM N=64: **deterministic (regular MPC) 16% vs diffusion-v (D-MPC) 8%** — the
> deterministic predictor wins ~2×, same direction as TwoRoom. This *confirms the
> prediction* made by the 2026-06-16 multimodality diagnostic (see that entry):
> PushT next-latents are near-deterministic and unimodal (det_R2 0.86), so a
> diffusion dynamics model has nothing to exploit. Caveat: diffusion got only 5
> epochs vs the fully-trained deterministic predictor (budget confound) — but the
> diagnostic gives the mechanistic reason it loses *regardless* of budget (no
> multimodal target), so training longer is not expected to flip it. **The real
> blocker is the data (single scripted expert); the next move is multimodal data,
> not more diffusion epochs.** Both success rates are low in absolute terms (PushT
> goal-conditioned is hard at this eval budget); the *ranking* is the result.

---

## Results (setup stage)

**Dataset sourcing — resolved a gated-access dead-end.**
- The README's link `galilai-group/lewm-pusht` is **gated and inaccessible** to this
  account (404 even authenticated; not a member of that HF org).
- The paper (le-wm.github.io) actually points to the author's collection
  `quentinll/lewm`. **`quentinll/lewm-pusht` is accessible** with the user's token —
  both as a dataset and as a pretrained model. Use `quentinll`, not `galilai-group`.
- Dataset is a single 13 GB `pusht_expert_train.h5.zst`; the repo's HF loader only
  auto-handles raw `.h5`/`.lance`, so it needs manual download + `zstd -d`
  (→ 46 GB `.h5`).

**Dataset shape (verified via SLURM inspect job):**
- ~**1.98M frames** (~360× TwoRoom's 5.5k).
- Columns: `pixels(224,224,3)`, `action(2)`, `proprio(4)`, `state(7)`, `episode_idx`,
  `step_idx`. No `goal_state` column — but that's fine: the eval harness *synthesizes*
  `goal_state` from `state` at the goal offset, so the existing `pusht_diffusion` plan
  callables work unchanged.

**Pretrained encoder — skips encoder training, after a version-skew fix.**
- `quentinll/lewm-pusht` (model repo) is `config.json` + `weights.pt` (72.3 MB) — the
  exact LeWM checkpoint format. So **no 300-epoch encoder training needed** (which on
  2M frames would have been many GPU-hours).
- But it was saved with an **older `transformers`**: ViT attention named
  `encoder.layer.N.attention.attention.{query,key,value}` / `intermediate.dense` /
  `output.dense`. Current code builds the flattened `encoder.layers.N.attention.
  {q,k,v,o}_proj` / `mlp.fc1` / `mlp.fc2`. Only the 12 ViT blocks differ — everything
  else (embeddings, predictor, projector) matched.
- `scripts/data/convert_lewm_checkpoint.py` remaps those keys. **Strict load passed
  (0 missing, 0 unexpected)** → the remap is exact (same architecture + shapes, just
  renamed). Converter job COMPLETED; `checkpoints/lewm_pusht/weights.pt` written.

**Pipeline (all SLURM, chained):** convert encoder → cache latents (pretrained
encoder, 2M frames) → v/x0 diffusion sweep (5 epochs — dataset is huge) → closed-loop
on v + closed-loop deterministic baseline (`lewm_pusht`). Configs verified to compose.

---

## What it means

- **The D-MPC pipeline is now domain-portable.** Everything that was TwoRoom-specific
  (parametrization, eval-harness column handling, plan config) is fixed and reused
  verbatim; only the dataset/env/encoder change. That the same code drops onto a 360×
  bigger, differently-shaped dataset with only config edits is the main infra result.
- **eps-loss / open-loop MSE remain deprecated as selection metrics** (2026-06-14
  lesson). E6 is judged on closed-loop success vs the deterministic baseline, full stop.
- **The headline question is still open.** If diffusion ≥ deterministic on PushT, the
  TwoRoom negative was a domain artifact and the D-MPC premise holds where dynamics are
  multimodal. If diffusion still loses even here, that's a stronger negative — the
  approach doesn't pay its cost on these tasks.
- **Process cost was real.** This setup ate a lot of wall-clock on non-science:
  HF-access spelunking, a `.h5.zst` decompression, a login-node usage-policy penalty
  (now: all heavy work via SLURM), and the encoder version-skew. Logged so the *next*
  domain (Reacher/Cube, also in `quentinll/lewm`) is a 1-hour port, not a day.

---

## What's next

1. **Converter — DONE** (job 3261427): `checkpoints/lewm_pusht/weights.pt` written,
   strict load exact.
2. **Chain LAUNCHED** (cache `3261439` → v/x0 sweep `3261442` → closed-loop v `3261443`
   + deterministic `3261444`); cache bumped to top of queue, currently pending on
   GPU-partition QOS behind the user's other jobs. Stages:
   - Cache ~2M frames through the encoder (~30–45 min, GPU).
   - v/x0 diffusion, 5 epochs (~15.6k batches/epoch; size-limited — extend if promising).
   - Closed-loop: 50 episodes, CEM N=64, vs the deterministic `lewm_pusht` predictor.
3. **Decision rule:**
   - diffusion (v or x0) **≥** deterministic success rate → D-MPC premise confirmed on
     multimodal dynamics; write it up, then tune (inference steps / eta / K, more epochs).
   - diffusion **<** deterministic → stronger negative; latent D-MPC w/ diffusion not
     worth it on these tasks. Sanity-check via a direct multimodality measurement of
     PushT next-latents before concluding.
4. **Watch for:** open-loop MSE again anti-ranking closed-loop (seen on TwoRoom); the
   5-epoch cap being too short on 2M frames (check the train-loss curve isn't still
   dropping steeply at the cutoff).

**Blocker risk:** GPU-partition QOS is shared with the user's `rmax`/`concurrent` jobs;
the chain may queue. `scontrol top <jobid>` reorders the user's own queue without
killing anything.

---

## Artifacts
- Encoder (converted): `checkpoints/lewm_pusht/weights.pt`
- Dataset: `datasets/pusht_expert_train.h5` (46 GB, ~2M frames)
- Latents (pending): `datasets/pusht_latent.lance`
- Scripts: `scripts/data/{fetch_lewm_dataset,inspect_dataset,convert_lewm_checkpoint}.py`,
  `scripts/slurm/{cache_latents_pusht,latent_dmpc_paramtype_pusht,plan_pusht_diffusion}.sbatch`
