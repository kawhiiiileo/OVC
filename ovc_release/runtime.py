"""Snapshot the editable Qwen3.5 training source for a reproducible run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .config import ROOT


def inventory():
    return json.loads((ROOT / 'provenance/source_manifest.json').read_text())['files']


def _safe_file(root, name):
    relative = Path(name)
    if (not relative.parts or relative.is_absolute() or '..' in relative.parts
            or relative.parts[0] not in {'verl', 'recipe', 'ovc_release'}):
        raise ValueError(f'Invalid runtime source path: {name}')
    target = root / relative
    if any(part.is_symlink() for part in [target, *target.parents]):
        raise ValueError(f'Symlink in runtime source path: {name}')
    return target


def materialize_runtime(destination, *, verify=True):
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Runtime destination already exists: {destination}')
    files = inventory()
    sources = {}
    for name, expected in files.items():
        _safe_file(destination, name)
        source = _safe_file(ROOT, name)
        if not source.is_file():
            raise ValueError(f'Missing runtime source: {name}')
        if verify and hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Runtime source changed: {name}')
        sources[name] = source
    destination.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        target = _safe_file(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if verify:
        actual = {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                  for name in files}
        if actual != files:
            raise ValueError('Copied runtime differs from its source SHA manifest')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-source-edits', action='store_true',
                        help='Snapshot edited source while retaining path and overwrite checks')
    args = parser.parse_args()
    try:
        destination = materialize_runtime(args.output, verify=not args.allow_source_edits)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({'destination': str(destination), 'files': len(inventory()),
                      'source_verification': not args.allow_source_edits}))


if __name__ == '__main__':
    main()
