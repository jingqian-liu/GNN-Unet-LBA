#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Run inference + evaluation on a list of checkpoints (no training).
Useful for checking whether test results are deterministic across runs.

Usage:
    python scripts/exps/test_ckpts.py \
        --ckpts path/to/epoch10_val0.62.pt path/to/epoch20_val0.63.pt \
        --test_set /path/to/test \
        --task PLA \
        --gpu 0 \
        --n_runs 2        # repeat each checkpoint this many times
"""
import os
import sys
import json
import argparse
from argparse import Namespace
from multiprocessing import Process

import numpy as np

PROJ_DIR = os.path.abspath(os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
))
sys.path.append(PROJ_DIR)

from inference import main as infer_proc
from evaluate import main as eval_proc
from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='Test checkpoints without training')
    parser.add_argument('--ckpts', type=str, nargs='+', required=True,
                        help='One or more checkpoint paths to evaluate')
    parser.add_argument('--test_set', type=str, required=True,
                        help='Path to the test set')
    parser.add_argument('--task', type=str, required=True,
                        choices=['PPA', 'PLA', 'LEP', 'PDBBind', 'NL'],
                        help='Task type')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Directory to save result files (default: same dir as first ckpt)')
    parser.add_argument('--fragment', type=str, default=None,
                        help='Fragmentation for small molecules')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=1,
                        help='How many times to run each checkpoint (>1 tests determinism)')
    return parser.parse_args()


def main(args):
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.ckpts[0]))
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}  # ckpt -> list of metric dicts across runs

    for ckpt in args.ckpts:
        ckpt_name = os.path.splitext(os.path.basename(ckpt))[0]
        print_log(f'Evaluating checkpoint: {ckpt}')
        run_metrics = []

        for run_i in range(args.n_runs):
            result_path = os.path.join(out_dir, f'{ckpt_name}_run{run_i}_results.jsonl')

            # inference
            namespace = Namespace(
                test_set=args.test_set,
                pdb_dir=None,
                task=args.task,
                fragment=args.fragment,
                ckpt=ckpt,
                save_path=result_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                gpu=args.gpu,
            )
            p = Process(target=infer_proc, args=(namespace,))
            p.start()
            p.join()
            p.close()
            print_log(f'  Run {run_i}: results saved to {result_path}')

            # evaluate
            namespace = Namespace(predictions=result_path, reference=None)
            metrics = eval_proc(namespace)
            run_metrics.append(metrics)
            print_log(f'  Run {run_i} metrics: { {k: round(float(v.statistic if hasattr(v, "statistic") else v), 4) for k, v in metrics.items()} }')

        all_results[ckpt_name] = run_metrics

    # summary
    print()
    print('=' * 60)
    print('Summary')
    print('=' * 60)
    for ckpt_name, run_metrics in all_results.items():
        print(f'\nCheckpoint: {ckpt_name}')
        metric_names = list(run_metrics[0].keys())
        for metric_name in metric_names:
            values = [m[metric_name] for m in run_metrics]
            if hasattr(values[0], 'statistic'):
                values = [v.statistic for v in values]
            values = [float(v) for v in values]
            if len(values) == 1:
                print(f'  {metric_name}: {round(values[0], 4)}')
            else:
                print(f'  {metric_name}: {[round(v, 4) for v in values]}  '
                      f'mean={round(np.mean(values), 4)}  std={round(np.std(values), 4)}')
    print()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    print(f'Project directory: {PROJ_DIR}')
    main(parse())
