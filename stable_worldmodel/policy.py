from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

import numpy as np
import torch
from torchvision import tv_tensors

from stable_worldmodel.planning.solver import Solver
from stable_worldmodel.protocols import Actionable, Transformable


@dataclass(frozen=True)
class PlanConfig:
    """Configuration for the MPC planning loop.

    Attributes:
        horizon: Planning horizon in number of steps.
        receding_horizon: Number of steps to execute before re-planning.
        history_len: Number of past observations to consider.
        action_block: Number of times each action is repeated (frameskip).
        warm_start: Whether to use the previous plan to initialize the next one.
    """

    horizon: int
    receding_horizon: int
    history_len: int = 1
    action_block: int = 1
    warm_start: bool = True

    @property
    def plan_len(self) -> int:
        """Total plan length in environment steps."""
        return self.horizon * self.action_block


class BasePolicy:
    """Base class for agent policies.

    Attributes:
        env: The environment the policy is associated with.
        type: A string identifier for the policy type.
    """

    env: Any
    type: str

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the base policy.

        Args:
            **kwargs: Additional configuration parameters.
        """
        self.env = None
        self.type = 'base'
        for arg, value in kwargs.items():
            setattr(self, arg, value)

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Get action from the policy given the observation.

        Args:
            obs: The current observation from the environment.
            **kwargs: Additional parameters for action selection.

        Returns:
            Selected action as a numpy array.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def set_env(self, env: Any) -> None:
        """Associate this policy with an environment.

        Args:
            env: The environment to associate.
        """
        self.env = env

    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        """Pre-process and transform observations.

        Applies preprocessing (via `self.process`) and transformations (via `self.transform`)
        to observation data. Used by subclasses like FeedForwardPolicy and WorldModelPolicy.
        Returns a new dict; the input is not mutated.

        Args:
            info_dict: Raw observation dictionary from the environment.

        Returns:
            A dictionary of processed tensors.

        Raises:
            ValueError: If an expected numpy array is missing for processing.
        """
        out = {}
        for k, v in info_dict.items():
            is_numpy = isinstance(v, (np.ndarray | np.generic))

            if hasattr(self, 'process') and k in self.process:
                if not is_numpy:
                    raise ValueError(
                        f"Expected numpy array for key '{k}' in process, got {type(v)}"
                    )

                # flatten extra dimensions if needed
                shape = v.shape
                if len(shape) > 2:
                    v = v.reshape(-1, *shape[2:])

                # process and reshape back
                v = self.process[k].transform(v)
                v = v.reshape(shape)

            # collapse env and time dimensions for transform (e, t, ...) -> (e * t, ...)
            # then restore after transform
            if hasattr(self, 'transform') and k in self.transform:
                shape = None
                if is_numpy or torch.is_tensor(v):
                    if v.ndim > 2:
                        shape = v.shape
                        v = v.reshape(-1, *shape[2:])
                if k.startswith('pixels') or k.startswith('goal'):
                    # permute channel first for transform
                    if is_numpy:
                        v = np.transpose(v, (0, 3, 1, 2))
                    else:
                        v = v.permute(0, 3, 1, 2)
                v = torch.stack(
                    [self.transform[k](tv_tensors.Image(x)) for x in v]
                )
                is_numpy = isinstance(v, (np.ndarray | np.generic))

                if shape is not None:
                    v = v.reshape(*shape[:2], *v.shape[1:])

            if is_numpy and v.dtype.kind not in 'USO':
                v = torch.from_numpy(v)

            out[k] = v

        return out


class RandomPolicy(BasePolicy):
    """Policy that samples random actions from the action space."""

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        """Initialize the random policy.

        Args:
            seed: Random seed applied to the action space when the environment
                is attached. If None, the action space uses its own default RNG,
                making action sampling non-deterministic across runs.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'random'
        self.seed = seed

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Get a random action from the environment's action space.

        Args:
            obs: The current observation (ignored).
            **kwargs: Additional parameters (ignored).

        Returns:
            A randomly sampled action.
        """
        return self.env.action_space.sample()

    def set_env(self, env: Any) -> None:
        """Attach the environment and seed its action space.

        Args:
            env: The environment to attach. If the policy was constructed
                with a seed, the environment's action space is seeded so
                action sampling is reproducible.
        """
        super().set_env(env)
        if self.seed is not None:
            self._set_seed(self.seed)

    def _set_seed(self, seed: int) -> None:
        if self.env is not None:
            self.env.action_space.seed(seed)


class ExpertPolicy(BasePolicy):
    """Policy using expert demonstrations or heuristics."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the expert policy.

        Args:
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'expert'

    def get_action(
        self, obs: Any, goal_obs: Any, **kwargs: Any
    ) -> np.ndarray | None:
        """Get action from the expert policy.

        Args:
            obs: The current observation.
            goal_obs: The goal observation.
            **kwargs: Additional parameters.

        Returns:
            The expert action, or None if not available.
        """
        # Implement expert policy logic here
        pass


class FeedForwardPolicy(BasePolicy):
    """Feed-Forward Policy using a neural network model.

    Actions are computed via a single forward pass through the model.
    Useful for imitation learning policies like Goal-Conditioned Behavioral Cloning (GCBC).

    Attributes:
        model: Neural network model implementing the Actionable protocol.
        process: Dictionary of data preprocessors for specific keys.
        transform: Dictionary of tensor transformations (e.g., image transforms).
    """

    def __init__(
        self,
        model: Actionable,
        process: dict[str, Transformable] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]]
        | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the feed-forward policy.

        Args:
            model: Neural network model with a `get_action` method.
            process: Dictionary of data preprocessors for specific keys.
            transform: Dictionary of tensor transformations (e.g., image transforms).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'feed_forward'
        self.model = model.eval()
        self.process = process or {}
        self.transform = transform or {}

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        """Get action via a forward pass through the neural network model.

        Args:
            info_dict: Current state information containing at minimum a 'goal' key.
            **kwargs: Additional parameters (unused).

        Returns:
            The selected action as a numpy array.

        Raises:
            AssertionError: If environment not set or 'goal' not in info_dict.
        """
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'goal' in info_dict, "'goal' must be provided in info_dict"

        # Prepare the info dict (transforms and normalizes inputs)
        info_dict = self._prepare_info(info_dict)

        # Add goal_pixels key for GCBC model
        if 'goal' in info_dict:
            info_dict['goal_pixels'] = info_dict['goal']

        # Move all tensors to the model's device
        device = next(self.model.parameters()).device
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                info_dict[k] = v.to(device)

        # Get action from model
        with torch.no_grad():
            action = self.model.get_action(info_dict)

        # Convert to numpy
        if torch.is_tensor(action):
            action = action.cpu().detach().numpy()

        # post-process action
        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)

        return action


class WorldModelPolicy(BasePolicy):
    """Policy using a world model and planning solver for action selection."""

    def __init__(
        self,
        solver: Solver,
        config: PlanConfig,
        process: dict[str, Transformable] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]]
        | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the world model policy.

        Args:
            solver: The planning solver to use.
            config: MPC planning configuration.
            process: Dictionary of data preprocessors for specific keys.
            transform: Dictionary of tensor transformations (e.g., image transforms).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        self.type = 'world_model'
        self.cfg = config
        self.solver = solver
        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer: list[deque[torch.Tensor]] | None = None
        self._next_init: torch.Tensor | None = None
        self._info_history: list[dict[str, deque]] | None = None

    @property
    def flatten_receding_horizon(self) -> int:
        """Receding horizon in environment steps (with frameskip)."""
        return self.cfg.receding_horizon * self.cfg.action_block

    @property
    def info_history_window_len(self) -> int:
        """Raw env frames retained for policy history.

        Observation keys are strided by ``action_block`` (needs
        ``(history_len - 1) * action_block + 1`` frames); the ``action`` key is
        kept dense and packed into ``history_len`` blocks of ``action_block``
        raw actions (needs ``history_len * action_block`` frames). The latter is
        the larger requirement, so size the window to it.
        """
        return max(1, self.cfg.history_len * self.cfg.action_block)

    def set_env(self, env: Any) -> None:
        """Configure the policy and solver for the given environment.

        Args:
            env: The environment to associate with the policy.
        """
        self.env = env
        n_envs = getattr(env, 'num_envs', 1)
        self.solver.configure(
            action_space=env.action_space, n_envs=n_envs, config=self.cfg
        )
        self._action_buffer = [
            deque(maxlen=self.flatten_receding_horizon) for _ in range(n_envs)
        ]
        self._info_history = [
            defaultdict(lambda: deque(maxlen=self.info_history_window_len))
            for _ in range(n_envs)
        ]

        assert isinstance(self.solver, Solver), (
            'Solver must implement the Solver protocol'
        )

    @staticmethod
    def _is_history_key(key: str, value: Any) -> bool:
        """Whether ``value`` should be stacked over policy history."""
        if key.startswith('_') or key.startswith('goal'):
            return False
        if not (torch.is_tensor(value) or isinstance(value, np.ndarray)):
            return False
        return value.ndim >= 3 and value.shape[1] > 0

    @staticmethod
    def _copy_frame(value: torch.Tensor | np.ndarray):
        if torch.is_tensor(value):
            return value.detach().clone()
        return value.copy()

    @staticmethod
    def _slice_value(value: Any, indices: list[int] | torch.Tensor):
        if torch.is_tensor(value):
            return value[indices]
        if isinstance(value, np.ndarray):
            return value[indices]
        if isinstance(value, list):
            return [value[int(i)] for i in indices]
        return value

    def _append_info_history(
        self, info_dict: dict, dead: np.ndarray
    ) -> None:
        if self._info_history is None:
            return

        for key, value in info_dict.items():
            if not self._is_history_key(key, value):
                continue
            for env_i in range(len(self._info_history)):
                if dead[env_i]:
                    continue
                frame = value[env_i, -1]
                self._info_history[env_i][key].append(
                    self._copy_frame(frame)
                )

    def _stack_history_for_env(self, env_i: int, key: str, current):
        assert self._info_history is not None
        frames = list(self._info_history[env_i].get(key, ()))
        if not frames:
            frames = [self._copy_frame(current[env_i, -1])]

        if key == 'action':
            return self._stack_action_blocks(frames)

        stride = max(1, self.cfg.action_block)
        selected = []
        for i in range(self.cfg.history_len):
            offset = (self.cfg.history_len - 1 - i) * stride
            idx = len(frames) - 1 - offset
            selected.append(frames[max(0, idx)])

        if torch.is_tensor(selected[0]):
            return torch.stack(selected, dim=0)
        return np.stack(selected, axis=0)

    def _stack_action_blocks(self, frames: list):
        """Pack raw actions into ``(history_len, action_block * action_dim)``.

        Mirrors the training layout: ``data.buffer`` keeps the ``action`` column
        dense and reshapes it to ``(history_len, frameskip * action_dim)``, so
        each token conditions the denoiser on the full block of raw actions that
        drives one latent transition. Subsampling actions like an observation
        (the strided path above) would instead drop ``action_block - 1`` of
        every ``action_block`` raw actions, leaving the wrong last dim and
        falling back to zeros downstream. Left-pads with zeros at episode start
        when fewer than ``history_len * action_block`` raw actions exist.
        """
        block = max(1, self.cfg.action_block)
        need = self.cfg.history_len * block
        recent = frames[-need:]
        if len(recent) < need:
            zero = (
                torch.zeros_like(recent[0])
                if torch.is_tensor(recent[0])
                else np.zeros_like(recent[0])
            )
            recent = [zero] * (need - len(recent)) + recent

        if torch.is_tensor(recent[0]):
            stacked = torch.stack(recent, dim=0)
        else:
            stacked = np.stack(recent, axis=0)
        return stacked.reshape(self.cfg.history_len, -1)

    def _make_solver_info(
        self, info_dict: dict, env_indices: list[int]
    ) -> dict:
        solver_info = {}
        idx_tensor = torch.as_tensor(env_indices, dtype=torch.long)

        for key, value in info_dict.items():
            if self._is_history_key(key, value):
                stacked = [
                    self._stack_history_for_env(env_i, key, value)
                    for env_i in env_indices
                ]
                if torch.is_tensor(stacked[0]):
                    solver_info[key] = torch.stack(stacked, dim=0)
                else:
                    solver_info[key] = np.stack(stacked, axis=0)
                continue

            indices = idx_tensor if torch.is_tensor(value) else env_indices
            solver_info[key] = self._slice_value(value, indices)

        return solver_info

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        """Get action via planning with the world model.

        Args:
            info_dict: Current state information from the environment.
            **kwargs: Additional parameters for planning.

        Returns:
            The selected action(s) as a numpy array.
        """
        assert hasattr(self, 'env'), 'Environment not set for the policy'

        info_dict = self._prepare_info(info_dict)
        n_envs = self.env.num_envs

        needs_flush = info_dict.pop('_needs_flush', None)
        if needs_flush is not None:
            for i in range(n_envs):
                if needs_flush[i]:
                    self._action_buffer[i].clear()
                    if self._info_history is not None:
                        self._info_history[i].clear()
                    if self._next_init is not None:
                        self._next_init[i] = 0

        terminated = info_dict.get('terminated')
        dead = (
            np.asarray(terminated, dtype=bool)
            if terminated is not None
            else np.zeros(n_envs, dtype=bool)
        )
        self._append_info_history(info_dict, dead)

        replan_idx = [
            i
            for i in range(n_envs)
            if len(self._action_buffer[i]) == 0 and not dead[i]
        ]

        if replan_idx:
            idx_tensor = torch.as_tensor(replan_idx, dtype=torch.long)
            sliced = self._make_solver_info(info_dict, replan_idx)

            sliced_init = (
                self._next_init[idx_tensor]
                if self._next_init is not None
                else None
            )

            outputs = self.solver(sliced, init_action=sliced_init)

            actions = outputs['actions']
            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]

            if self.cfg.warm_start and rest.shape[1] > 0:
                if self._next_init is None:
                    self._next_init = torch.zeros(
                        n_envs, rest.shape[1], rest.shape[2], dtype=rest.dtype
                    )
                self._next_init[idx_tensor] = rest
            elif not self.cfg.warm_start:
                self._next_init = None

            plan = plan.reshape(
                len(replan_idx), self.flatten_receding_horizon, -1
            )

            for row, env_i in enumerate(replan_idx):
                self._action_buffer[env_i].extend(plan[row])

        single_shape = self.env.single_action_space.shape
        is_discrete = 'Discrete' in type(self.env.single_action_space).__name__

        action = torch.full(
            (n_envs, *single_shape),
            fill_value=0 if is_discrete else float('nan'),
            dtype=torch.long if is_discrete else torch.float32,
        )

        for i in range(n_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()

        action = action.reshape(*self.env.action_space.shape)
        action = action.numpy()

        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)

        return action


# Alias for backward compatibility and type hinting
Policy = BasePolicy
