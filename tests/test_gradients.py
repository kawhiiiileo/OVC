"""Check token-local autograd through the released engine's actual policy losses."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def losses():
    path = ROOT / 'verl/trainer/ppo/core_algos.py'
    functions = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in (
                'agg_loss', 'compute_policy_loss_vanilla', 'compute_policy_loss_gspo'):
            node.decorator_list = []
            functions.append(node)

    class ActorConfig(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

    env = dict(torch=torch, ActorConfig=ActorConfig, AlgoConfig=type('AlgoConfig', (), {}),
               verl_F=SimpleNamespace(masked_sum=lambda x, m: (x * m).sum(),
                                      masked_mean=lambda x, m: (x * m).sum() / m.sum()))
    exec(compile('from __future__ import annotations\n' +
                 ast.unparse(ast.Module(body=functions, type_ignores=[])), str(path), 'exec'), env)
    return env, ActorConfig(clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.2,
                            clip_ratio_c=10., global_batch_info={})


class GradientTests(unittest.TestCase):
    def test_engine_gspo_keeps_individual_token_credit(self):
        env, config = losses()
        lp = torch.tensor([[-2., -2.]], dtype=torch.float64, requires_grad=True)
        loss, _ = env['compute_policy_loss_gspo'](
            old_log_prob=lp.detach().clone(), log_prob=lp,
            advantages=torch.tensor([[1., 3.]], dtype=lp.dtype),
            response_mask=torch.ones_like(lp), loss_agg_mode='seq-mean-token-mean', config=config)
        loss.backward()
        torch.testing.assert_close(lp.grad, torch.tensor([[-.5, -1.5]], dtype=lp.dtype))
