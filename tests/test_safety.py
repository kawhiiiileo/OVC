import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_scan', ROOT / 'scripts/check_release.py')
scan_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan_module)


class SafetyTests(unittest.TestCase):
    def test_candidate_passes(self):
        report = scan_module.scan()
        self.assertEqual(report['findings'], [])

    def test_detects_without_printing_secret_value(self):
        with tempfile.TemporaryDirectory(prefix='ovc_scan_') as temp:
            root = Path(temp)
            secret = 'sk' + '-' + 'aB7z9Q'*6
            (root / 'bad.txt').write_text(secret)
            (root / 'provenance').mkdir()
            (root / 'provenance/source_manifest.json').write_text(json.dumps({'files': {}}))
            report = scan_module.scan(root)
            self.assertEqual(report['status'], 'FAIL')
            self.assertNotIn(secret, json.dumps(report))
            self.assertEqual(report['findings'][0]['reason'], 'credential_pattern')

    def test_detects_machine_paths_and_internal_wording(self):
        examples = [
            ('/mnt/' + 'cpfs/researcher/models/model', 'machine_specific_path'),
            ('/home/' + 'researcher/work/data.parquet', 'machine_specific_path'),
            ('Historical' + ' experiments', 'internal_release_wording'),
            ('not yet' + ' curated', 'internal_release_wording'),
            ('CPT' + '-fullSFT', 'internal_release_wording'),
        ]
        for value, reason in examples:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory(prefix='ovc_scan_') as temp:
                root = Path(temp)
                (root / 'provenance').mkdir()
                (root / 'provenance/source_manifest.json').write_text(json.dumps({'files': {}}))
                (root / 'example.txt').write_text(value)
                report = scan_module.scan(root)
                self.assertEqual(report['status'], 'FAIL')
                self.assertIn(reason, {item['reason'] for item in report['findings']})
                self.assertNotIn(value, json.dumps(report))

    def test_git_metadata_is_outside_working_tree_scan(self):
        with tempfile.TemporaryDirectory(prefix='ovc_scan_') as temp:
            root = Path(temp)
            (root / 'provenance').mkdir()
            (root / 'provenance/source_manifest.json').write_text(json.dumps({'files': {}}))
            (root / '.git').mkdir()
            (root / '.git/config').write_text('[core]\n\tbare = false\n')
            self.assertEqual(scan_module.scan(root)['findings'], [])


if __name__ == '__main__':
    unittest.main()
