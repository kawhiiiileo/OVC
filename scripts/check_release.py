"""Read-only source, syntax and sensitive-payload gate; never print matched secrets."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import warnings

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'.env', '.netrc', '.ssh', '.wandb', '__pycache__', 'node_modules'}
SUFFIXES = {'.pt', '.pth', '.safetensors', '.pkl', '.pickle', '.parquet', '.arrow', '.xlsx', '.wandb', '.pem', '.key'}
PATTERNS = {
    'credential_pattern': re.compile(rb'\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{24,}|AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    'private_key': re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----'),
    'private_workspace_path': re.compile(rb'/data/(?:MLLM-RL|experiments|ldn)(?:/|\b)|/home/' + rb'server/|/root/\.ssh/'),
    'machine_specific_path': re.compile(rb'/(?:mnt/(?:cpfs|data_cpfs)|home/[A-Za-z0-9_.-]+)/'),
    'internal_release_wording': re.compile(
        rb'(?i)\bhistorical\s*(?:experiments|7b)\b|old\s+dashboards|'
        rb'not\s+yet\s+curated|for\s+this\s+candidate|'
        rb'cpt[-_\s]*full[-_\s]*sft|DelTA\s*/\s*OVC'),
}


def scan(root=ROOT):
    findings, python_files, json_files, syntax_warnings = [], 0, 0, 0
    for path in sorted(root.rglob('*')):
        rel = str(path.relative_to(root))
        # Git metadata is not part of the checked working-tree payload.
        if '.git' in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            findings.append({'file': rel, 'reason': 'symlink_not_allowed'})
            continue
        if not path.is_file():
            continue
        if any(part in FORBIDDEN for part in path.relative_to(root).parts) or path.suffix in SUFFIXES:
            findings.append({'file': rel, 'reason': 'forbidden_payload'})
            continue
        content = path.read_bytes()
        for reason, pattern in PATTERNS.items():
            if pattern.search(content):
                findings.append({'file': rel, 'reason': reason})
        try:
            if path.suffix == '.py':
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', SyntaxWarning)
                    ast.parse(content, filename=rel)
                syntax_warnings += len(caught)
                python_files += 1
            elif path.suffix == '.json':
                json.loads(content)
                json_files += 1
        except (SyntaxError, ValueError, UnicodeError):
            findings.append({'file': rel, 'reason': 'syntax_or_json_error'})
    manifest = root / 'provenance/source_manifest.json'
    if not manifest.is_file():
        findings.append({'file': 'provenance/source_manifest.json', 'reason': 'missing_manifest'})
    else:
        for rel, expected in json.loads(manifest.read_text())['files'].items():
            path = root / rel
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                findings.append({'file': rel, 'reason': 'source_hash_mismatch'})
    return {'status': 'PASS' if not findings else 'FAIL', 'python_files': python_files,
            'json_files': json_files, 'upstream_syntax_warnings': syntax_warnings,
            'findings': findings, 'gpu_use': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    report = scan(args.root.resolve())
    print(json.dumps(report, indent=2))
    raise SystemExit(report['status'] != 'PASS')


if __name__ == '__main__':
    main()
