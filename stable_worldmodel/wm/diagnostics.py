from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_1d_float(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    return arr


def _finite_xy(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = _as_1d_float(x, 'x')
    y = _as_1d_float(y, 'y')
    if x.shape != y.shape:
        raise ValueError(
            f'x and y must have the same shape, got {x.shape} and {y.shape}'
        )
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _safe_float(value) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def rankdata(values) -> np.ndarray:
    """Return average ranks, like scipy.stats.rankdata(method='average')."""

    values = _as_1d_float(values, 'values')
    if values.size == 0:
        return values

    sorter = np.argsort(values, kind='mergesort')
    sorted_values = values[sorter]
    group_start = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    starts = np.flatnonzero(group_start)
    ends = np.r_[starts[1:], len(values)]

    sorted_ranks = np.empty(len(values), dtype=np.float64)
    for start, end in zip(starts, ends):
        # Ranks are 1-indexed; ties receive the average rank.
        sorted_ranks[start:end] = 0.5 * (start + end - 1) + 1.0

    ranks = np.empty(len(values), dtype=np.float64)
    ranks[sorter] = sorted_ranks
    return ranks


def pearson_corr(x, y) -> float:
    x, y = _finite_xy(x, y)
    if x.size < 2:
        return float('nan')

    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float('nan')
    return float(np.sum(x * y) / denom)


def spearman_corr(x, y) -> float:
    x, y = _finite_xy(x, y)
    if x.size < 2:
        return float('nan')
    return pearson_corr(rankdata(x), rankdata(y))


def distance_regression_metrics(
    latent_distance,
    target_distance,
    *,
    num_bins: int = 5,
) -> dict:
    """Summarize whether latent L2 predicts a target distance.

    The linear regression treats latent L2 as the predictor and target distance
    as the response, matching the diagnostic question: can raw latent distance
    act as a proxy for temporal/geodesic distance?
    """

    x, y = _finite_xy(latent_distance, target_distance)
    out = {
        'n': int(x.size),
        'latent_distance_mean': _safe_float(np.mean(x)) if x.size else None,
        'latent_distance_std': _safe_float(np.std(x)) if x.size else None,
        'target_distance_mean': _safe_float(np.mean(y)) if y.size else None,
        'target_distance_std': _safe_float(np.std(y)) if y.size else None,
        'pearson': None,
        'spearman': None,
        'slope': None,
        'intercept': None,
        'r2': None,
        'mae': None,
        'bins': [],
    }
    if x.size < 2:
        return out

    out['pearson'] = _safe_float(pearson_corr(x, y))
    out['spearman'] = _safe_float(spearman_corr(x, y))

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_var = np.sum(x_centered * x_centered)
    if x_var > 0:
        slope = np.sum(x_centered * y_centered) / x_var
        intercept = y.mean() - slope * x.mean()
        pred = slope * x + intercept
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum(y_centered * y_centered)
        out['slope'] = _safe_float(slope)
        out['intercept'] = _safe_float(intercept)
        out['mae'] = _safe_float(np.mean(np.abs(y - pred)))
        if ss_tot > 0:
            out['r2'] = _safe_float(1.0 - ss_res / ss_tot)

    out['bins'] = binned_distance_summary(
        latent_distance=x,
        target_distance=y,
        num_bins=num_bins,
    )
    return out


def binned_distance_summary(
    latent_distance,
    target_distance,
    *,
    num_bins: int = 5,
) -> list[dict]:
    x, y = _finite_xy(latent_distance, target_distance)
    if x.size == 0 or num_bins <= 0:
        return []

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(y, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return []

    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        is_last = hi == edges[-1]
        mask = (y >= lo) & (y <= hi if is_last else y < hi)
        if not np.any(mask):
            continue
        bins.append(
            {
                'target_min': _safe_float(lo),
                'target_max': _safe_float(hi),
                'n': int(mask.sum()),
                'latent_distance_mean': _safe_float(np.mean(x[mask])),
                'latent_distance_std': _safe_float(np.std(x[mask])),
                'target_distance_mean': _safe_float(np.mean(y[mask])),
            }
        )
    return bins


def sample_temporal_pairs(
    num_steps: int,
    num_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_steps < 2 or num_pairs <= 0:
        return np.empty((0, 2), dtype=np.int64)

    first = rng.integers(0, num_steps, size=num_pairs)
    second = rng.integers(0, num_steps - 1, size=num_pairs)
    second = second + (second >= first)
    return np.stack([first, second], axis=1).astype(np.int64)


def sample_goal_segments(
    num_steps: int,
    num_segments: int,
    rng: np.random.Generator,
    *,
    min_distance: int = 2,
) -> np.ndarray:
    min_distance = max(int(min_distance), 1)
    if num_steps <= min_distance or num_segments <= 0:
        return np.empty((0, 2), dtype=np.int64)

    segments = np.empty((num_segments, 2), dtype=np.int64)
    for i in range(num_segments):
        goal = int(rng.integers(min_distance, num_steps))
        start = int(rng.integers(0, goal - min_distance + 1))
        segments[i] = (start, goal)
    return segments


def temporal_distance_metrics(
    latents,
    times,
    pair_indices,
    *,
    num_bins: int = 5,
) -> dict:
    latents = np.asarray(latents, dtype=np.float64)
    times = _as_1d_float(times, 'times')
    pair_indices = np.asarray(pair_indices, dtype=np.int64)

    if latents.ndim != 2:
        raise ValueError(
            f'latents must have shape (T, D), got {latents.shape}'
        )
    if latents.shape[0] != times.shape[0]:
        raise ValueError(
            'latents and times must have same T, got '
            f'{latents.shape[0]} and {times.shape[0]}'
        )
    if pair_indices.size == 0:
        pair_indices = np.empty((0, 2), dtype=np.int64)
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError('pair_indices must have shape (N, 2)')

    if pair_indices.shape[0] == 0:
        return distance_regression_metrics([], [], num_bins=num_bins)

    valid = (
        (pair_indices[:, 0] >= 0)
        & (pair_indices[:, 0] < latents.shape[0])
        & (pair_indices[:, 1] >= 0)
        & (pair_indices[:, 1] < latents.shape[0])
        & (pair_indices[:, 0] != pair_indices[:, 1])
    )
    pair_indices = pair_indices[valid]
    if pair_indices.shape[0] == 0:
        return distance_regression_metrics([], [], num_bins=num_bins)

    first = pair_indices[:, 0]
    second = pair_indices[:, 1]
    latent_l2 = np.linalg.norm(latents[first] - latents[second], axis=1)
    target_distance = np.abs(times[first] - times[second])
    return distance_regression_metrics(
        latent_l2, target_distance, num_bins=num_bins
    )


def goal_monotonicity_metrics(
    latents,
    times,
    segments: Sequence[tuple[int, int]] | np.ndarray,
    *,
    tolerance: float = 0.0,
    num_bins: int = 5,
) -> dict:
    latents = np.asarray(latents, dtype=np.float64)
    times = _as_1d_float(times, 'times')
    segments = np.asarray(segments, dtype=np.int64)

    if latents.ndim != 2:
        raise ValueError(
            f'latents must have shape (T, D), got {latents.shape}'
        )
    if latents.shape[0] != times.shape[0]:
        raise ValueError(
            'latents and times must have same T, got '
            f'{latents.shape[0]} and {times.shape[0]}'
        )
    if segments.size == 0:
        segments = np.empty((0, 2), dtype=np.int64)
    if segments.ndim != 2 or segments.shape[1] != 2:
        raise ValueError('segments must have shape (N, 2)')

    all_latent_distance = []
    all_steps_to_goal = []
    segment_spearman = []
    adjacent_total = 0
    adjacent_decreases = 0
    valid_segments = 0

    for start, goal in segments:
        if start < 0 or goal >= latents.shape[0] or goal <= start:
            continue

        idx = np.arange(start, goal + 1)
        distances = np.linalg.norm(latents[idx] - latents[goal], axis=1)
        steps_to_goal = times[goal] - times[idx]
        if np.any(steps_to_goal < 0):
            continue

        valid_segments += 1
        all_latent_distance.append(distances)
        all_steps_to_goal.append(steps_to_goal)

        if distances.size > 1:
            adjacent_total += distances.size - 1
            adjacent_decreases += int(
                np.sum(np.diff(distances) <= float(tolerance))
            )
            corr = spearman_corr(distances, steps_to_goal)
            if np.isfinite(corr):
                segment_spearman.append(corr)

    if all_latent_distance:
        latent_distance = np.concatenate(all_latent_distance)
        steps_to_goal = np.concatenate(all_steps_to_goal)
    else:
        latent_distance = np.array([], dtype=np.float64)
        steps_to_goal = np.array([], dtype=np.float64)

    metrics = distance_regression_metrics(
        latent_distance, steps_to_goal, num_bins=num_bins
    )
    metrics.update(
        {
            'segments': int(valid_segments),
            'adjacent_transitions': int(adjacent_total),
            'monotone_decrease_fraction': (
                _safe_float(adjacent_decreases / adjacent_total)
                if adjacent_total
                else None
            ),
            'segment_spearman_mean': (
                _safe_float(np.mean(segment_spearman))
                if segment_spearman
                else None
            ),
            'segment_spearman_std': (
                _safe_float(np.std(segment_spearman))
                if segment_spearman
                else None
            ),
        }
    )
    return metrics
