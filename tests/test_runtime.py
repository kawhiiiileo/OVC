"""Verify the actual training snapshot and its source/overwrite guards."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ovc_release.config import ROOT
from ovc_release.runtime import inventory, materialize_runtime


class RuntimeTests(unittest.TestCase):
    def test_training_snapshot_contains_engine_and_reward(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = materialize_runtime(Path(temp) / 'runtime')
            for name, expected in inventory().items():
                self.assertEqual(hashlib.sha256((runtime / name).read_bytes()).hexdigest(), expected)
            self.assertTrue((runtime / 'verl/trainer/main_ppo.py').is_file())
            self.assertTrue((runtime / 'verl/models/transformers/qwen3_5.py').is_file())
            self.assertTrue((runtime / 'recipe/math77k/reward.py').is_file())
            with self.assertRaises(FileExistsError):
                materialize_runtime(runtime)

    def test_changed_source_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'verl').mkdir()
            (root / 'verl/test.py').write_text('# edited\n')
            destination = root / 'runtime'
            with patch('ovc_release.runtime.ROOT', root), patch('ovc_release.runtime.inventory',
                    return_value={'verl/test.py': hashlib.sha256(b'# original\n').hexdigest()}):
                with self.assertRaisesRegex(ValueError, 'Runtime source changed'):
                    materialize_runtime(destination)
                self.assertFalse(destination.exists())
                materialize_runtime(destination, verify=False)
                self.assertEqual((destination / 'verl/test.py').read_text(), '# edited\n')

    def test_unsafe_paths_fail_even_with_source_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ('../outside.py', '/outside.py', 'evaluation/test.py'):
                with self.subTest(name=name), patch('ovc_release.runtime.inventory', return_value={name: ''}):
                    destination = Path(temp) / 'runtime'
                    with self.assertRaises(ValueError):
                        materialize_runtime(destination, verify=False)
                    self.assertFalse(destination.exists())

    def test_public_scope_has_one_configuration(self):
        self.assertEqual([p.name for p in (ROOT / 'configs').rglob('*.json')],
                         ['qwen3_5_9b_ovc.json'])
        self.assertFalse((ROOT / 'evaluation').exists())
        self.assertFalse((ROOT / 'backends').exists())
        self.assertEqual(json.loads((ROOT / 'provenance/source_manifest.json').read_text())['files'], inventory())
