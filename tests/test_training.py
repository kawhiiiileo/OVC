"""Exercise the public launcher without starting CUDA, Ray or a training job."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
from ovc_release import train


class TrainingTests(unittest.TestCase):
    def fixture(self, root):
        model = root / 'model'
        model.mkdir()
        (model / 'config.json').write_text(json.dumps({
            'model_type': 'qwen3_5', 'image_token_id': 248056,
            'text_config': {'num_hidden_layers': 32, 'hidden_size': 4096}}))
        for name in ('train.parquet', 'val.parquet'):
            (root / name).touch()
        return ['ovc_release.train', '--model', str(model), '--train-file', str(root / 'train.parquet'),
                '--val-file', str(root / 'val.parquet'), '--output', str(root / 'output')]

    def test_dry_run_resolves_single_recipe_without_side_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.fixture(root)
            output = io.StringIO()
            with patch.object(sys, 'argv', args + ['--dry-run']), contextlib.redirect_stdout(output), \
                    patch.object(train, 'require_cluster') as cluster:
                train.main()
            cfg = json.loads(output.getvalue())
            self.assertEqual(cfg['trainer']['nnodes'], 16)
            self.assertEqual(cfg['trainer']['n_gpus_per_node'], 8)
            self.assertEqual(cfg['trainer']['resume_mode'], 'disable')
            self.assertEqual(cfg['actor_rollout_ref']['actor']['h_layer_idx'], 16)
            self.assertEqual(cfg['actor_rollout_ref']['actor']['impl'], 'Mahalanobis')
            self.assertEqual(cfg['data']['train_batch_size'], 256)
            OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
            self.assertNotIn('__MATH77K', output.getvalue())
            cluster.assert_not_called()
            self.assertFalse((root / 'output').exists())

    def test_launch_reaches_engine_with_capture_env_on_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.fixture(root)
            source = root / 'snapshot'
            source.mkdir()
            (source / 'source.py').write_text('# training snapshot\n')
            engine = ModuleType('verl.trainer.main_ppo')
            engine.run_ppo = Mock()
            old_cwd = Path.cwd()
            try:
                with patch.object(sys, 'argv', args + ['--launch', '--ray-address', 'head:6379']), \
                        patch.object(sys, 'path', list(sys.path)), patch.dict(os.environ), \
                        patch.dict(sys.modules, {'verl.trainer.main_ppo': engine}), \
                        patch.object(train, 'require_cluster', return_value={'total_gpus': 128}) as cluster, \
                        patch.object(train, 'materialize_runtime', return_value=source):
                    train.main()
                cluster.assert_called_once_with('head:6379', nnodes=16, gpus_per_node=8)
            finally:
                os.chdir(old_cwd)
            engine.run_ppo.assert_called_once()
            cfg = engine.run_ppo.call_args.args[0]
            env = cfg.ray_kwargs.ray_init.runtime_env.env_vars
            self.assertEqual(env.VERL_DELTA_H_LAYER_IDX, '16')
            self.assertEqual(cfg.actor_rollout_ref.actor.strategy, 'fsdp2')
            self.assertEqual(cfg.ray_kwargs.ray_init.address, 'head:6379')
            self.assertNotIn('num_gpus', cfg.ray_kwargs.ray_init)
            self.assertNotIn('num_cpus', cfg.ray_kwargs.ray_init)
            self.assertNotIn('_node_ip_address', cfg.ray_kwargs.ray_init)
            self.assertEqual(cfg.trainer.nnodes * cfg.trainer.n_gpus_per_node, 128)
            responses = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n
            self.assertEqual(responses // 128, 4)
            self.assertEqual(responses % 128, 0)
            self.assertTrue((root / 'output/config.json').is_file())
            self.assertTrue((root / 'output/runtime_sources.json').is_file())

    def test_missing_cluster_address_and_same_dataset_fail_before_connection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.fixture(root)
            with patch.object(sys, 'argv', args + ['--launch']), \
                    patch.object(train, 'require_cluster') as cluster, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train.main()
                cluster.assert_not_called()
            args[args.index('--val-file') + 1] = str(root / 'train.parquet')
            with patch.object(sys, 'argv', args + ['--launch', '--ray-address', 'head:6379']), \
                    patch.object(train, 'require_cluster') as cluster, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train.main()
                cluster.assert_not_called()


class ClusterTests(unittest.TestCase):
    def nodes(self):
        return [{'Alive': True, 'Resources': {'GPU': 8}} for _ in range(16)]

    def test_full_cluster_accepts_optional_cpu_head(self):
        nodes = self.nodes() + [{'Alive': True, 'Resources': {'CPU': 32}}]
        result = train.validate_cluster_resources(nodes, {'GPU': 128}, nnodes=16, gpus_per_node=8)
        self.assertEqual(result['total_gpus'], 128)

    def test_partial_or_wrong_topology_is_rejected(self):
        for nodes in [self.nodes()[:-1],
                      [{'Alive': True, 'Resources': {'GPU': 16}} for _ in range(8)],
                      self.nodes()[:-1] + [{'Alive': False, 'Resources': {'GPU': 8}}]]:
            with self.subTest(nodes=len(nodes)), self.assertRaisesRegex(RuntimeError, '16 GPU nodes'):
                train.validate_cluster_resources(nodes, {'GPU': 128}, nnodes=16, gpus_per_node=8)

    def test_busy_cluster_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, '128 available Ray GPUs'):
            train.validate_cluster_resources(self.nodes(), {'GPU': 127}, nnodes=16, gpus_per_node=8)

    def test_cluster_connection_is_closed_when_check_fails(self):
        ray = Mock()
        ray.is_initialized.return_value = False
        ray.nodes.return_value = self.nodes()[:-1]
        ray.available_resources.return_value = {'GPU': 120}
        with patch.dict(sys.modules, {'ray': ray}), self.assertRaises(RuntimeError):
            train.require_cluster('head:6379', nnodes=16, gpus_per_node=8)
        ray.init.assert_called_once_with(address='head:6379', logging_level='ERROR')
        ray.shutdown.assert_called_once()
