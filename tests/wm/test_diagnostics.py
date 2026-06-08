import math

import numpy as np

from stable_worldmodel.wm.diagnostics import (
    goal_monotonicity_metrics,
    rankdata,
    sample_goal_segments,
    sample_temporal_pairs,
    spearman_corr,
    temporal_distance_metrics,
)


def test_rankdata_uses_average_ranks_for_ties():
    ranks = rankdata([3.0, 1.0, 1.0, 2.0])
    np.testing.assert_allclose(ranks, [4.0, 1.5, 1.5, 3.0])


def test_spearman_corr_detects_monotone_relationship():
    assert spearman_corr([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman_corr([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_temporal_distance_metrics_linear_latent_is_perfect():
    times = np.arange(8, dtype=np.float64)
    latents = times[:, None]
    pairs = np.array(
        [(i, j) for i in range(len(times)) for j in range(i + 1, len(times))]
    )

    metrics = temporal_distance_metrics(latents, times, pairs)

    assert metrics['n'] == len(pairs)
    assert math.isclose(metrics['pearson'], 1.0)
    assert math.isclose(metrics['spearman'], 1.0)
    assert math.isclose(metrics['r2'], 1.0)


def test_goal_monotonicity_linear_latent_is_perfect():
    times = np.arange(8, dtype=np.float64)
    latents = times[:, None]
    segments = np.array([[0, 4], [2, 7]])

    metrics = goal_monotonicity_metrics(latents, times, segments)

    assert metrics['segments'] == 2
    assert math.isclose(metrics['pearson'], 1.0)
    assert math.isclose(metrics['spearman'], 1.0)
    assert math.isclose(metrics['monotone_decrease_fraction'], 1.0)
    assert math.isclose(metrics['segment_spearman_mean'], 1.0)


def test_goal_monotonicity_flags_nonmonotone_latent_path():
    times = np.arange(5, dtype=np.float64)
    latents = np.array([[4.0], [1.0], [3.0], [0.5], [0.0]])

    metrics = goal_monotonicity_metrics(latents, times, [[0, 4]])

    assert metrics['segments'] == 1
    assert metrics['monotone_decrease_fraction'] < 1.0


def test_sampling_helpers_return_valid_indices():
    rng = np.random.default_rng(0)
    pairs = sample_temporal_pairs(5, 100, rng)
    segments = sample_goal_segments(5, 100, rng, min_distance=2)

    assert pairs.shape == (100, 2)
    assert np.all(pairs[:, 0] != pairs[:, 1])
    assert np.all((pairs >= 0) & (pairs < 5))

    assert segments.shape == (100, 2)
    assert np.all(segments[:, 1] - segments[:, 0] >= 2)
    assert np.all((segments >= 0) & (segments < 5))
