import argparse
import copy
import csv
import datetime as dt
import json
import os
import traceback
import yaml

from analyze_gradient_study import analyze_runs
from config.config_utils import LoadConfig
from runnetworks import RunNetworks


METHODS = ('baseline', 'balance', 'pcgrad', 'balance_pcgrad')


def _write_manifest(path, manifest):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = path + '.tmp'
    with open(temporary_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Run baseline, gradient balancing, PCGrad, and their combination sequentially.'
    )
    parser.add_argument('--train-path', required=True, nargs='+')
    parser.add_argument('--validation-path', required=True, nargs='+')
    parser.add_argument('--initial-checkpoint', default='')
    parser.add_argument('--model', default='Release')
    parser.add_argument('--output-root', default='./data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--validation-batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--validation-workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.0002)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--diagnostic-interval', type=int, default=10)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--no-augmentation', action='store_true')
    parser.add_argument('--crop-size', type=int, default=0)
    parser.add_argument('--balance-min', type=float, default=0.1)
    parser.add_argument('--balance-max', type=float, default=10.0)
    parser.add_argument('--balance-ema-beta', type=float, default=0.9)
    return parser.parse_args()


def _experiment_config(base_config, args, method, suite_id):
    config = copy.deepcopy(base_config)
    config['run']['type'] = 'train'
    config['run']['seed'] = args.seed
    config['run']['model'] = args.model
    config['run']['data_path'] = os.path.abspath(args.output_root)
    config['run']['experiment_name'] = f'gradient_{suite_id}_{method}'
    config['run']['data_crop']['use'] = args.crop_size > 0
    if args.crop_size > 0:
        config['run']['data_crop']['size'] = args.crop_size

    train = config['train']
    train['dataset_path'] = [os.path.abspath(path) for path in args.train_path]
    train['epochs'] = args.epochs
    train['batch_size'] = args.batch_size
    train['numberworks'] = args.workers
    train['learning_rate'] = args.learning_rate
    train['model_save_every'] = args.save_every
    train['sample_save_every'] = max(1, train.get('sample_save_every', 1))
    train['aug'] = not args.no_augmentation
    train['strategy'] = 'none'
    train['gradient_method'] = method
    train['mult_stage_loss']['use'] = False
    train['pretrained'] = {
        'use': bool(args.initial_checkpoint),
        'model_path': os.path.abspath(args.initial_checkpoint) if args.initial_checkpoint else '',
    }
    train['gradient_diagnostics'] = {
        'enabled': True,
        'interval': args.diagnostic_interval,
        'filename': 'gradient_diagnostics.csv',
    }
    train['gradient_balance'] = {
        'min_weight': args.balance_min,
        'max_weight': args.balance_max,
        'ema_beta': args.balance_ema_beta,
        'eps': 1e-12,
    }

    config['validation']['dataset_path'] = [
        os.path.abspath(path) for path in args.validation_path
    ]
    config['validation']['batch_size'] = args.validation_batch_size
    config['validation']['numberworks'] = args.validation_workers
    return config


def main():
    args = _parse_args()
    if args.initial_checkpoint and not os.path.isfile(args.initial_checkpoint):
        raise FileNotFoundError(f'Initial checkpoint not found: {args.initial_checkpoint}')
    for path in [*args.train_path, *args.validation_path]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f'Dataset directory not found: {path}')

    suite_id = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    study_dir = os.path.abspath(os.path.join(args.output_root, 'gradient_studies', suite_id))
    manifest_path = os.path.join(study_dir, 'gradient_study_manifest.json')
    manifest = {
        'suite_id': suite_id,
        'created_at': dt.datetime.now().isoformat(timespec='seconds'),
        'arguments': vars(args),
        'runs': [],
    }
    _write_manifest(manifest_path, manifest)

    base_config = LoadConfig()
    for method in args.methods:
        record = {'method': method, 'status': 'starting'}
        manifest['runs'].append(record)
        _write_manifest(manifest_path, manifest)
        try:
            config = _experiment_config(base_config, args, method, suite_id)
            runner = RunNetworks(config)
            record['run_dir'] = os.path.abspath(runner.data_root_path)
            os.makedirs(record['run_dir'], exist_ok=True)
            with open(
                os.path.join(record['run_dir'], 'experiment_config.yaml'),
                'w',
                encoding='utf-8',
            ) as config_file:
                yaml.safe_dump(config, config_file, allow_unicode=True, sort_keys=False)
            record['status'] = 'running'
            _write_manifest(manifest_path, manifest)
            print(f'\n===== Gradient study: {method} =====')
            runner.work()
            record['status'] = 'completed'
            record['completed_at'] = dt.datetime.now().isoformat(timespec='seconds')
            _write_manifest(manifest_path, manifest)
        except Exception as error:
            record['status'] = 'failed'
            record['error'] = repr(error)
            record['traceback'] = traceback.format_exc()
            _write_manifest(manifest_path, manifest)
            analyze_runs(manifest['runs'], os.path.join(study_dir, 'analysis'))
            print(f'Experiment {method} failed. Progress is saved in {manifest_path}')
            raise

    comparison = analyze_runs(manifest['runs'], os.path.join(study_dir, 'analysis'))
    manifest['completed_at'] = dt.datetime.now().isoformat(timespec='seconds')
    manifest['comparison_rows'] = len(comparison)
    _write_manifest(manifest_path, manifest)
    print(f'\nAll experiments completed. Study manifest: {manifest_path}')
    print(f'Comparison: {os.path.join(study_dir, "analysis", "experiment_comparison.csv")}')


if __name__ == '__main__':
    main()
