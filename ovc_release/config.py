"""Resolve the single Qwen3.5-9B OVC recipe without importing CUDA libraries."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs/qwen3_5_9b_ovc.json'


def build(*, model, train_file, val_file, output):
    output = Path(output).expanduser().resolve()
    replacements = {
        '__MODEL_PATH__': str(Path(model).expanduser().resolve()),
        '__TRAIN_FILE__': str(Path(train_file).expanduser().resolve()),
        '__VAL_FILE__': str(Path(val_file).expanduser().resolve()),
        '__OUTPUT_DIR__': str(output),
        '__CACHE_DIR__': str(output / 'cache/datasets'),
        '__MATH77K_REWARD_PATH__': str(output / 'runtime/recipe/math77k/reward.py'),
    }

    def resolve(value):
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, str):
            for placeholder, replacement in replacements.items():
                value = value.replace(placeholder, replacement)
            if re.fullmatch(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)[eE][-+]?\d+', value):
                return float(value)
        return value

    return resolve(json.loads(CONFIG_PATH.read_text()))
