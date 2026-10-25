"""The shared credit utilities are usable without importing a training backend."""
from types import SimpleNamespace
import unittest

import torch

from ovc_release.credit.calibration import diagonal_metric
from ovc_release.credit.config import CreditOptions
from ovc_release.credit.policy import gspo_token_loss
from ovc_release.credit.visual import compute_visual_grounding_rho


class CreditTests(unittest.TestCase):
    def test_estimator_options_are_explicit_and_validated(self):
        self.assertEqual(CreditOptions.from_mapping().sigma_scope, 'response')
        for values in ({'sigma_scope': 'unknown'}, {'stats_scope': 'unknown'},
                       {'ridge_epsilon': 0}, {'ridge_epsilon': float('nan')},
                       {'distance_chunk_size': -1}, {'bounded_packing': 'yes'}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CreditOptions.from_mapping(values)

    def test_evidence_metric_matches_regularized_diagonal_formula(self):
        positive = torch.tensor([1., 3., 9.], requires_grad=True)
        negative = torch.tensor([2., 4., 5.], requires_grad=True)
        result = diagonal_metric(positive, negative, epsilon=1e-6,
                                 shrink=.1, min_var_scale=1e-3, exclude_visual=True)
        pooled = positive.detach() + negative.detach()
        expected = .9 * pooled + .1 * pooled[:-1].mean()
        expected = torch.clamp(expected + 1e-6, min=pooled[:-1].mean() * 1e-3 + 1e-6)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertFalse(result.requires_grad)

    def test_gspo_preserves_individual_token_credit(self):
        lp = torch.tensor([[-2., -2.]], dtype=torch.float64, requires_grad=True)
        config = SimpleNamespace(clip_ratio=.2, clip_ratio_low=.2,
                                 clip_ratio_high=.2, global_batch_info={})
        def aggregate(loss_mat, loss_mask, loss_agg_mode):
            self.assertEqual(loss_agg_mode, 'seq-mean-token-mean')
            return ((loss_mat * loss_mask).sum(-1) / loss_mask.sum(-1)).mean()
        loss, _ = gspo_token_loss(
            lp.detach().clone(), lp, torch.tensor([[1., 3.]], dtype=lp.dtype),
            torch.ones_like(lp), config=config, aggregate_loss=aggregate,
            masked_mean=lambda values, mask: (values * mask).sum() / mask.sum())
        loss.backward()
        torch.testing.assert_close(lp.grad, torch.tensor([[-.5, -1.5]], dtype=lp.dtype))

    def test_rve_stops_estimator_gradients(self):
        torch.manual_seed(42)
        full = torch.randn(2, 6, 4, requires_grad=True)
        images = torch.zeros(2, 6, dtype=torch.bool)
        images[:, 0] = True
        text = torch.zeros_like(images)
        text[:, 1:3] = True
        rho, _, _ = compute_visual_grounding_rho(
            full, full[:, -3:], images, torch.ones(2, 3, dtype=torch.bool),
            text_token_mask=text, signal_version='v2')
        self.assertFalse(rho.requires_grad)


if __name__ == '__main__':
    unittest.main()
