"""Prevent train/validation overlap and preserve image types across write batches."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


class PreprocessTests(unittest.TestCase):
    def convert(self, root, val_size):
        source = root / 'raw.parquet'
        pq.write_table(pa.Table.from_pylist([
            {'problem': f'question {i}', 'answer': str(i), 'images': [] if i < 4 else ['image-data']}
            for i in range(8)]), source)
        destination = root / 'converted'
        command = [sys.executable, str(ROOT / 'recipe/math77k/preprocess.py'),
                   '--input', str(source), '--output-dir', str(destination),
                   '--val-size', str(val_size), '--batch-size', '2']
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        return destination, command, json.loads(result.stdout)

    def test_holdout_is_excluded_from_training(self):
        with tempfile.TemporaryDirectory() as temp:
            destination, command, summary = self.convert(Path(temp), 3)
            train = pq.read_table(destination / 'train.parquet').to_pylist()
            val = pq.read_table(destination / 'val.parquet').to_pylist()
            train_ids = {r['extra_info']['index'] for r in train}
            val_ids = {r['extra_info']['index'] for r in val}
            self.assertFalse(train_ids & val_ids)
            self.assertEqual(train_ids | val_ids, set(range(8)))
            self.assertEqual((summary['train_rows'], summary['val_rows']), (5, 3))
            self.assertEqual(train[-1]['images'], ['image-data'])
            previous = (destination / 'train.parquet').read_bytes()
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((destination / 'train.parquet').read_bytes(), previous)

    def test_external_validation_mode_does_not_copy_training_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            destination, _, summary = self.convert(Path(temp), 0)
            self.assertEqual(summary['train_rows'], 8)
            self.assertIsNone(summary['val_file'])
            self.assertFalse((destination / 'val.parquet').exists())
