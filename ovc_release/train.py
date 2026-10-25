"""Train Qwen3.5-9B with GSPO + OVC on 16 x 8 GPUs; dry-run by default."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from .config import build
from .runtime import materialize_runtime


def validate_cluster_resources(nodes, available, *, nnodes, gpus_per_node):
    """Require the dedicated topology rather than advertising nonexistent local GPUs."""
    workers = [node for node in nodes if node.get('Alive')
               and node.get('Resources', {}).get('GPU', 0) > 0]
    expected = nnodes * gpus_per_node
    if len(workers) != nnodes or any(node['Resources']['GPU'] != gpus_per_node for node in workers):
        raise RuntimeError(f'Require {nnodes} GPU nodes with {gpus_per_node} GPUs each ({expected} total)')
    free = available.get('GPU', 0)
    if free < expected:
        raise RuntimeError(f'Require {expected} available Ray GPUs; found {free}')
    return {'nnodes': nnodes, 'gpus_per_node': gpus_per_node,
            'total_gpus': expected, 'available_gpus': free}


def require_cluster(address, *, nnodes, gpus_per_node):
    """Inspect an existing cluster; disconnect without stopping any Ray node."""
    import ray
    if ray.is_initialized():
        raise RuntimeError('Run the training entry in a fresh Python process')
    ray.init(address=address, logging_level='ERROR')
    try:
        return validate_cluster_resources(ray.nodes(), ray.available_resources(),
                                          nnodes=nnodes, gpus_per_node=gpus_per_node)
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--val-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ray-address', help='Existing dedicated Ray head address, e.g. HEAD_IP:6379')
    parser.add_argument('--wandb-project', help='Optional; authenticate externally, never put keys in config')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--allow-source-edits', action='store_true',
                        help='Use modified source for development; record copied runtime hashes')
    args = parser.parse_args()
    if args.dry_run and args.launch:
        parser.error('Choose --dry-run or --launch, not both')
    cfg = build(model=args.model, train_file=args.train_file,
                val_file=args.val_file, output=args.output)
    if args.ray_address:
        cfg['ray_kwargs']['ray_init']['address'] = args.ray_address
    if args.wandb_project:
        cfg['trainer']['project_name'] = args.wandb_project
        cfg['trainer']['logger'] = ['console', 'file', 'wandb']
    if not args.launch:
        print(json.dumps(cfg, indent=2))
        return
    if cfg['trainer']['resume_mode'] != 'disable':
        parser.error('This fresh launcher must not silently replay a continuation as a fresh run')
    if not args.ray_address:
        parser.error('--launch requires --ray-address for the existing 128-GPU Ray cluster')
    model = Path(cfg['actor_rollout_ref']['model']['path'])
    train = Path(cfg['data']['train_files'])
    validation = Path(cfg['data']['val_files'])
    for path in [model / 'config.json', train, validation]:
        if not path.is_file():
            parser.error(f'Required input not found: {path}')
    if train.samefile(validation):
        parser.error('Training and validation must use separate datasets')
    model_config = json.loads((model / 'config.json').read_text())
    if (model_config.get('model_type') != 'qwen3_5'
            or model_config.get('text_config', {}).get('num_hidden_layers') != 32
            or model_config.get('text_config', {}).get('hidden_size') != 4096):
        parser.error('--model must point to the Qwen3.5-9B checkpoint')
    image_token_id = model_config.get('image_token_id')
    if image_token_id != cfg['actor_rollout_ref']['actor']['visual_grounding']['image_token_ids'][0]:
        parser.error('Model image token ID does not match the OVC configuration')
    out = Path(args.output).expanduser().resolve()
    if out.exists():
        parser.error('Output already exists; this entry never resumes or overwrites a run')
    cluster = require_cluster(args.ray_address, nnodes=cfg['trainer']['nnodes'],
                              gpus_per_node=cfg['trainer']['n_gpus_per_node'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'cluster.json').write_text(json.dumps(cluster, indent=2) + '\n')
    source = materialize_runtime(out / 'runtime', verify=not args.allow_source_edits)
    source_hashes = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source.rglob('*') if p.is_file()}
    (out / 'runtime_sources.json').write_text(json.dumps(
        {'model': 'Qwen3.5-9B', 'files': source_hashes}, indent=2) + '\n')
    os.environ.update(PYTHONPATH=str(source) + os.pathsep + os.environ.get('PYTHONPATH', ''),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False',
        NCCL_CUMEM_ENABLE='0', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        VERL_DELTA_H_LAYER_IDX=str(cfg['actor_rollout_ref']['actor']['h_layer_idx']),
        VERL_FILE_LOGGER_PATH=str(out / 'metrics/train.jsonl'))
    os.environ.pop('RAY_ADDRESS', None)
    worker_env = {key: os.environ[key] for key in ('PYTHONPATH', 'VERL_DELTA_H_LAYER_IDX',
        'PYTORCH_CUDA_ALLOC_CONF', 'NCCL_CUMEM_ENABLE', 'TOKENIZERS_PARALLELISM',
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VERL_FILE_LOGGER_PATH')}
    cfg['ray_kwargs']['ray_init'] = dict(address=args.ray_address,
        runtime_env={'env_vars': worker_env})
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    sys.path.insert(0, str(source))
    os.chdir(source)  # Resolve Hydra configuration relative to the runtime.
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo
    run_ppo(OmegaConf.create(cfg))
    (out / 'TRAIN_DRIVER_RETURNED.json').write_text(json.dumps({'status': 'RETURNED_SUCCESS'}) + '\n')


if __name__ == '__main__':
    main()
