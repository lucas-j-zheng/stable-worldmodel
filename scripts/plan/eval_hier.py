"""E18: closed-loop subgoal-proposer eval (H-JEPA rung 5, minimal instantiation).

Wraps the stock WorldModelPolicy: every `replan_every` env steps, replace the
env-provided goal IMAGE with a SUBGOAL image K steps ahead, produced by a
k-NN proposer over the training dataset conditioned on (state_t, goal_state)
-- the exact conditional validated by the E9/E13 benches. Two arms isolate
mean-vs-sample with the SAME bank and zero training:

  sample : subgoal = future frame of ONE RANDOM neighbor (generative sampler;
           the kNN head that won the bench).
  mean   : subgoal = future frame of the neighbor whose future STATE is
           nearest the neighbors' MEAN future state (the conservative-strong
           MSE analog: the conditional mean, snapped to the data manifold --
           a raw mean cannot be rendered as an image, and snapping only helps
           it).
  off    : no wrapper (flat baseline; equals eval_wm.py).

The level-1 planner, budget, and success criterion are untouched -- the ONLY
difference between arms is which goal image the planner sees between replans.
Run via SLURM (same config family as eval_wm.py).
"""

import os

os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig
from sklearn import preprocessing
from sklearn.neighbors import BallTree
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm

GOAL_CANDS = ('goal_state', 'variation_target_position', 'goal', 'goal_pos')


def img_transform(cfg, dtype=torch.float32):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(dtype, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


class SubgoalBank:
    """(state_t, goal_state) -> pointer to the same episode's frame at t+K.

    Startup loads only the small state/goal columns; pixel frames are loaded
    lazily per chosen subgoal (few hundred per eval)."""

    def __init__(self, dataset_name, subgoal_offset, seed=0):
        self.ds = swm.data.load_dataset(dataset_name)
        self.K = int(subgoal_offset)
        present = set(self.ds.column_names)
        gcol = next((c for c in GOAL_CANDS if c in present), None)
        conds, ptrs, futs = [], [], []
        for ep in range(len(self.ds.lengths)):
            T = int(self.ds.lengths[ep])
            if T <= self.K:
                continue
            d = self.ds.load_episode(ep)
            s = np.asarray(d['state'], np.float32).reshape(T, -1)
            g = (np.asarray(d[gcol], np.float32).reshape(T, -1)
                 if gcol else np.repeat(s[-1][None], T, 0))
            for t in range(T - self.K):
                conds.append(np.concatenate([s[t], g[t]]))
                ptrs.append((ep, t))
            futs.append(s[self.K:])
        cond = np.asarray(conds, np.float32)
        self.fut_state = np.concatenate(futs, 0)
        self.ptrs = np.asarray(ptrs, np.int64)
        self.scaler = preprocessing.StandardScaler().fit(cond)
        self.tree = BallTree(self.scaler.transform(cond))
        self.rng = np.random.default_rng(seed)
        # typical K-step travel distance -- sets the final-goal handoff radius
        d0 = cond[:, : self.fut_state.shape[1]]
        self.hop = float(np.median(
            np.linalg.norm(self.fut_state - d0, axis=1)))
        print(f'[bank] {cond.shape[0]} rows, K={self.K}, goal col={gcol}, '
              f'median {self.K}-step hop={self.hop:.2f}')

    def propose(self, state, goal_state, arm, k=32):
        q = self.scaler.transform(
            np.concatenate([state, goal_state]).reshape(1, -1))
        _, nbr = self.tree.query(q, k=k)
        nbr = nbr[0]
        if arm == 'sample':
            j = int(self.rng.choice(nbr))
        else:                                       # 'mean' (snap-to-manifold)
            mu = self.fut_state[nbr].mean(0)
            j = int(nbr[np.argmin(
                np.linalg.norm(self.fut_state[nbr] - mu, axis=1))])
        ep, t = self.ptrs[j]
        d = self.ds.load_episode(int(ep))
        px = np.asarray(d['pixels'])
        return px[int(t) + self.K]                   # (H, W, C) uint8


class LatentSubgoalPolicy(swm.policy.BasePolicy):
    """E19 latent-cost arms: inject `goal_emb` directly (LeWM.get_cost uses a
    provided goal_emb and skips goal-image encoding). This is the ONLY
    interface that can expose the RAW conditional mean closed-loop -- an
    off-manifold mean cannot be rendered as a goal image (R23/R24 lesson).

    arm='sampleL': z* = embedding of ONE random neighbor's future frame.
    arm='meanL'  : z* = MEAN of the k neighbors' future-frame embeddings --
                   the raw mean in the model's own embedding space.
    Handoff envs (goal within one hop) get z* = embedding of the true goal
    image, so all arms aim at the real goal at the end.
    """

    def __init__(self, inner, bank, model, transform, arm,
                 replan_every=10, k=8):
        super().__init__()
        self.type = 'world_model'
        self.inner, self.bank, self.arm = inner, bank, arm
        self.model, self.tf = model, transform
        self.replan_every, self.k = int(replan_every), int(k)
        self._step = 0
        self._z = None                                # (n_envs, D) torch

    def set_env(self, env):
        self.env = env
        self.inner.set_env(env)

    def set_seed(self, seed):
        if hasattr(self.inner, 'set_seed'):
            self.inner.set_seed(seed)

    @torch.no_grad()
    def _embed_frames(self, frames):
        """frames: list of (C,H,W)|(H,W,C) uint8 -> (n, D) cuda embeddings."""
        from torchvision import tv_tensors
        xs = []
        for f in frames:
            if f.shape[-1] == 3:                      # HWC -> CHW
                f = np.transpose(f, (2, 0, 1))
            xs.append(self.tf(tv_tensors.Image(torch.from_numpy(
                np.ascontiguousarray(f)))))
        px = torch.stack(xs).unsqueeze(1).to('cuda')  # (n, 1, C, H, W)
        out = self.model.encode({'pixels': px})
        return out['emb'][:, 0]                       # (n, D)

    def get_action(self, info_dict, **kw):
        goal = info_dict['goal']
        n = goal.shape[0]
        if self._step % self.replan_every == 0 or self._z is None:
            zs = []
            for i in range(n):
                s = np.asarray(info_dict['state'][i], np.float32).reshape(-1)
                g = np.asarray(
                    info_dict['goal_state'][i], np.float32).reshape(-1)
                if np.linalg.norm(g - s) <= 1.25 * self.bank.hop:
                    gf = np.asarray(goal[i])
                    gf = gf[0] if gf.ndim == 4 else gf
                    zs.append(self._embed_frames([gf])[0])
                    continue
                q = self.bank.scaler.transform(
                    np.concatenate([s, g]).reshape(1, -1))
                _, nbr = self.bank.tree.query(q, k=self.k)
                frames = []
                for j in nbr[0]:
                    ep, t = self.bank.ptrs[j]
                    d = self.bank.ds.load_episode(int(ep))
                    frames.append(
                        np.asarray(d['pixels'])[int(t) + self.bank.K])
                embs = self._embed_frames(frames)
                if self.arm == 'meanL':
                    zs.append(embs.mean(0))           # RAW off-manifold mean
                else:                                  # sampleL
                    zs.append(embs[int(self.bank.rng.integers(len(embs)))])
            self._z = torch.stack(zs)                 # (n, D)
        info_dict = dict(info_dict)
        # flat (n, D): the solver slices per env and inserts the sample dim
        # itself; criterion right-aligns via [..., -1:, :].expand_as(pred).
        info_dict['goal_emb'] = self._z
        self._step += 1
        return self.inner.get_action(info_dict, **kw)


class SubgoalPolicy(swm.policy.BasePolicy):
    """Rewrites info_dict['goal'] with proposer subgoal frames, then defers."""

    def __init__(self, inner, bank, arm, replan_every=25, k=32):
        super().__init__()
        self.type = 'world_model'
        self.inner, self.bank, self.arm = inner, bank, arm
        self.replan_every, self.k = int(replan_every), int(k)
        self._step = 0
        self._cached = None                          # (n_envs, ...) frames

    def set_env(self, env):
        self.env = env
        self.inner.set_env(env)

    def set_seed(self, seed):
        if hasattr(self.inner, 'set_seed'):
            self.inner.set_seed(seed)

    def get_action(self, info_dict, **kw):
        goal = info_dict['goal']
        if self._step % self.replan_every == 0 or self._cached is None:
            frames = []
            for i in range(goal.shape[0]):
                s = np.asarray(info_dict['state'][i], np.float32).reshape(-1)
                g = np.asarray(
                    info_dict['goal_state'][i], np.float32).reshape(-1)
                # FINAL-GOAL HANDOFF (R23 fix): once the true goal is within
                # ~one subgoal hop, aim directly at it -- flat behavior.
                if np.linalg.norm(g - s) <= 1.25 * self.bank.hop:
                    frames.append(None)
                else:
                    frames.append(self.bank.propose(s, g, self.arm, self.k))
            self._cached = frames
        new_goal = np.array(goal)                    # copy, keep dtype/shape

        def fit(f, shape):
            # dataset frames and env goal slots disagree on channel order
            # (either side may be CHW or HWC) -- try both transposes
            shape = tuple(shape)
            if f.shape == shape:
                return f
            if f.transpose(2, 0, 1).shape == shape:      # HWC -> CHW
                return f.transpose(2, 0, 1)
            if f.transpose(1, 2, 0).shape == shape:      # CHW -> HWC
                return f.transpose(1, 2, 0)
            raise ValueError(f'frame {f.shape} vs goal slot {shape}')

        for i, f in enumerate(self._cached):
            if f is None:                            # handoff: keep true goal
                continue
            flat = new_goal[i]
            # goal may carry a leading time dim (n, 1, C, H, W)
            if flat.ndim == 4:
                flat[0] = fit(f, flat[0].shape)
            else:
                flat[...] = fit(f, flat.shape)
            new_goal[i] = flat
        info_dict = dict(info_dict)
        info_dict['goal'] = new_goal
        self._step += 1
        return self.inner.get_action(info_dict, **kw)


@hydra.main(version_base=None, config_path='./config',
            config_name='tworoom_mm_diffusion')
def run(cfg: DictConfig):
    import scripts.plan.eval_wm as base  # reuse dataset/episode helpers

    arm = cfg.get('hier_arm', 'sample')              # sample | mean | off
    sub_k = int(cfg.get('hier_subgoal_offset', 8))
    replan = int(cfg.get('hier_replan_every', 25))
    bank_name = cfg.get('hier_bank', cfg.eval.dataset_name)

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {'pixels': img_transform(cfg, img_dtype),
                 'goal': img_transform(cfg, img_dtype)}

    dataset = base.get_dataset(cfg, cfg.eval.dataset_name)
    col_name = 'ep_idx' if 'ep_idx' in dataset.column_names else 'episode_idx'
    ep_indices, _ = np.unique(
        dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        pr = preprocessing.StandardScaler()
        cd = dataset.get_col_data(col)
        cd = cd[~np.isnan(cd).any(axis=1)]
        pr.fit(cd)
        process[col] = pr
        if col != 'action':
            process[f'goal_{col}'] = pr

    model = swm.wm.utils.load_pretrained(cfg.policy)
    model = model.to('cuda').eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    inner = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform)

    if arm == 'off':
        policy = inner
    elif arm in ('sampleL', 'meanL'):
        bank = SubgoalBank(bank_name, sub_k, seed=cfg.seed)
        policy = LatentSubgoalPolicy(
            inner, bank, model, transform['goal'], arm, replan, k=8)
    else:
        bank = SubgoalBank(bank_name, sub_k, seed=cfg.seed)
        policy = SubgoalPolicy(inner, bank, arm, replan, k=32)

    # episode/start sampling — identical to eval_wm.py
    episode_len = base.get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    msd = {e: max_start_idx[i] for i, e in enumerate(ep_indices)}
    per_row = np.array(
        [msd[e] for e in dataset.get_col_data(col_name)])
    valid = np.nonzero(dataset.get_col_data('step_idx') <= per_row)[0]
    g = np.random.default_rng(cfg.seed)
    pick = np.sort(valid[g.choice(len(valid) - 1, size=cfg.eval.num_eval,
                                  replace=False)])
    eval_eps = dataset.get_col_data(col_name)[pick]
    eval_start = dataset.get_col_data('step_idx')[pick]

    from omegaconf import OmegaConf

    world.set_policy(policy)
    results_path = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
        cfg.policy).parent
    results_path.mkdir(parents=True, exist_ok=True)
    eval_options = (OmegaConf.to_container(cfg.eval.get('options'),
                                           resolve=True)
                    if cfg.eval.get('options') is not None else None)
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_eps.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get('callables'),
                                         resolve=True),
        options=eval_options,
        video=results_path,
    )
    metrics.pop('proprio_trajectories', None)
    metrics.pop('goal_proprio', None)
    succ = np.asarray(metrics.get('episode_successes', []))
    print(f'[hier] arm={arm} offset={cfg.eval.goal_offset_steps} '
          f'success={int(succ.sum())}/{len(succ)}')
    out = results_path / cfg.output.filename
    out.write_text(f'arm={arm} offset={cfg.eval.goal_offset_steps} '
                   f'success={int(succ.sum())}/{len(succ)}\n{metrics}\n')
    print(f'[hier] wrote {out}')


if __name__ == '__main__':
    run()
