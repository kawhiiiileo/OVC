#!/usr/bin/env python3
"""Convert raw Math77K parquet into disjoint verl training/validation files."""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def make_record(row, split, index, data_source):
    problem = row.get('problem') or ''
    answer = '' if row.get('answer') is None else str(row['answer'])
    return {
        'data_source': data_source,
        'prompt': [{'role': 'user', 'content': problem}],
        'images': row.get('images') or [],
        'ability': 'math',
        'reward_model': {'style': 'rule', 'ground_truth': answer},
        'extra_info': {'split': split, 'index': index, 'answer': answer, 'question': problem},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Parquet with problem/answer/images columns')
    parser.add_argument('--output-dir', required=True, help='New directory for converted files')
    parser.add_argument('--val-size', type=int, default=1024,
                        help='Hold out this many leading rows; use 0 with a separate validation dataset')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--max-rows', type=int, default=-1)
    parser.add_argument('--data-source', default='math')
    args = parser.parse_args()
    if args.batch_size <= 0 or args.val_size < 0 or args.max_rows == 0 or args.max_rows < -1:
        parser.error('batch-size must be positive, val-size nonnegative, and max-rows positive or -1')
    source = pq.ParquetFile(Path(args.input).expanduser())
    missing = {'problem', 'answer', 'images'} - set(source.schema_arrow.names)
    if missing:
        parser.error('Missing input columns: ' + ', '.join(sorted(missing)))
    count = source.metadata.num_rows
    if args.max_rows > 0:
        count = min(count, args.max_rows)
    if count <= args.val_size:
        parser.error('Input must contain training rows after the validation holdout')
    output = Path(args.output_dir).expanduser()
    if output.exists():
        parser.error('Output directory already exists; choose a new directory')
    # Fix the schema up front so an image-free batch cannot infer a null image type.
    schema = pa.schema([
        ('data_source', pa.string()),
        ('prompt', pa.list_(pa.struct([('role', pa.string()), ('content', pa.string())]))),
        ('images', source.schema_arrow.field('images').type),
        ('ability', pa.string()),
        ('reward_model', pa.struct([('style', pa.string()), ('ground_truth', pa.string())])),
        ('extra_info', pa.struct([('split', pa.string()), ('index', pa.int64()),
                                  ('answer', pa.string()), ('question', pa.string())])),
    ])
    output.mkdir(parents=True, exist_ok=False)
    train_count = val_count = total = 0
    train_writer = val_writer = None
    try:
        train_writer = pq.ParquetWriter(output / 'train.parquet', schema, compression='zstd')
        if args.val_size:
            val_writer = pq.ParquetWriter(output / 'val.parquet', schema, compression='zstd')
        for batch in source.iter_batches(batch_size=args.batch_size, columns=['problem', 'answer', 'images']):
            training, validation = [], []
            for row in batch.to_pylist():
                if total == count:
                    break
                if total < args.val_size:
                    validation.append(make_record(row, 'validation', total, args.data_source))
                    val_count += 1
                else:
                    training.append(make_record(row, 'train', total, args.data_source))
                    train_count += 1
                total += 1
            if training:
                train_writer.write_table(pa.Table.from_pylist(training, schema=schema))
            if validation:
                val_writer.write_table(pa.Table.from_pylist(validation, schema=schema))
            if total == count:
                break
    finally:
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()
    summary = {'train_file': str(output / 'train.parquet'), 'train_rows': train_count,
               'val_file': str(output / 'val.parquet') if args.val_size else None,
               'val_rows': val_count, 'total_rows': total}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
