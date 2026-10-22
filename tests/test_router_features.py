from __future__ import annotations

import unittest

import numpy as np
from sklearn.decomposition import PCA

from seminspect.router.features import ROUTER_FEATURE_NAMES, RouterObservation, raw_router_features, validate_feature_schema


def _observation(probability: float, offset: float = 0.0) -> RouterObservation:
    return RouterObservation(
        calibrated_probability=probability,
        detector_raw_score=0.2 + offset,
        patch_score_max=0.3,
        patch_score_top5_std=0.1,
        patch_score_q95=0.25,
        cls_normal_distance=0.2,
        cls_embedding=np.linspace(0.1 + offset, 1.0 + offset, 384, dtype=np.float32),
        intensity_mean=0.5,
        intensity_std=0.1,
        robust_dynamic_range=0.3,
        laplacian_variance=0.4,
        tenengrad=0.5,
        highpass_mad=0.01,
    )


class RouterFeatureTest(unittest.TestCase):
    def test_router_feature_schema_has_53_ordered_columns(self) -> None:
        self.assertEqual(len(ROUTER_FEATURE_NAMES), 53)
        validate_feature_schema()

    def test_router_feature_schema_contains_no_cost_or_budget_column(self) -> None:
        joined = " ".join(ROUTER_FEATURE_NAMES).lower()
        self.assertNotIn("cost", joined)
        self.assertNotIn("budget", joined)

    def test_q0_has_zero_history(self) -> None:
        pca = PCA(n_components=32, svd_solver="full").fit(np.random.default_rng(0).normal(size=(40, 384)))
        values = raw_router_features(_observation(0.2), quality_index=0, pca=pca)
        self.assertEqual(values.shape, (53,))
        self.assertEqual(values[-1], 0.0)
        self.assertTrue(np.allclose(values[-4:-1], 0.0))
