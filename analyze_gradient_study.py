import argparse
import csv
import json
import math
import os
from collections import defaultdict
from statistics import mean, median


LAYERS = (
    'encoder_shallow',
    'encoder_middle',
    'encoder_deep',
    'shared_decoder',
    'shared_all',
)


def _read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as csv_file:
        return list(csv.DictReader(csv_file))


def _number(value, default=float('nan')):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_runs(run_records, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    comparison_rows = []
    gradient_rows = []
    epoch_rows = []

    for record in run_records:
        if record.get('status') != 'completed':
            continue
        method = record['method']
        run_dir = record['run_dir']
        logs_path = os.path.join(run_dir, 'logs.csv')
        diagnostics_path = os.path.join(run_dir, 'gradient_diagnostics.csv')
        best_path = os.path.join(run_dir, 'best_metrics.json')
        if not (os.path.isfile(logs_path) and os.path.isfile(best_path)):
            continue

        logs = _read_csv(logs_path)
        with open(best_path, encoding='utf-8') as best_file:
            best = json.load(best_file)
        best_epoch = int(best['epoch'])
        best_log = next((row for row in logs if int(row['epoch']) == best_epoch), {})
        comparison_rows.append({
            'method': method,
            'run_dir': os.path.abspath(run_dir),
            'best_epoch': best_epoch,
            'best_foreground_miou': best.get('ValidationForegroundMIoU', ''),
            'best_miou': best.get('ValidationMIoU', ''),
            'best_mean_dice': best.get('ValidationMeanDice', ''),
            'best_overlap_iou': best.get('ValidationOverlapIoU', ''),
            'best_validation_mse': best.get('ValidationMSE', ''),
            'best_validation_psnr': best.get('ValidationPSNR', ''),
            'segmentation_gradient_weight': best_log.get('SegmentationGradientWeight', ''),
            'training_gradient_conflict_rate': best_log.get('GradientConflictRate', ''),
            'best_checkpoint': os.path.abspath(os.path.join(run_dir, 'models', 'best.pth')),
        })

        for row in logs:
            epoch_rows.append({
                'method': method,
                'epoch': int(row['epoch']),
                'foreground_miou': row.get('ValidationForegroundMIoU', ''),
                'miou': row.get('ValidationMIoU', ''),
                'mean_dice': row.get('ValidationMeanDice', ''),
                'overlap_iou': row.get('ValidationOverlapIoU', ''),
                'segmentation_gradient_weight': row.get('SegmentationGradientWeight', ''),
                'training_gradient_conflict_rate': row.get('GradientConflictRate', ''),
            })

        if os.path.isfile(diagnostics_path):
            diagnostics = _read_csv(diagnostics_path)
            grouped = defaultdict(list)
            for row in diagnostics:
                grouped[row['layer']].append(row)
            for layer in LAYERS:
                rows = grouped.get(layer, [])
                if not rows:
                    continue
                cosines = [_number(row['cosine']) for row in rows]
                ratios = [_number(row['norm_ratio']) for row in rows]
                conflicts = [_number(row['is_conflict']) for row in rows]
                cosines = [value for value in cosines if math.isfinite(value)]
                ratios = [value for value in ratios if math.isfinite(value)]
                conflicts = [value for value in conflicts if math.isfinite(value)]
                gradient_rows.append({
                    'method': method,
                    'layer': layer,
                    'samples': len(rows),
                    'conflict_rate': mean(conflicts) if conflicts else '',
                    'mean_cosine': mean(cosines) if cosines else '',
                    'median_norm_ratio': median(ratios) if ratios else '',
                })

    comparison_rows.sort(key=lambda row: _number(row['best_foreground_miou'], -1.0), reverse=True)
    _write_csv(os.path.join(output_dir, 'experiment_comparison.csv'), comparison_rows)
    _write_csv(os.path.join(output_dir, 'gradient_summary.csv'), gradient_rows)
    _write_csv(os.path.join(output_dir, 'epoch_metrics.csv'), epoch_rows)
    _plot_results(epoch_rows, output_dir)
    return comparison_rows


def _plot_results(epoch_rows, output_dir):
    if not epoch_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed; CSV summaries were created without plots.')
        return

    grouped = defaultdict(list)
    for row in epoch_rows:
        grouped[row['method']].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row['epoch'])

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method, rows in grouped.items():
        epochs = [row['epoch'] for row in rows]
        foreground = [_number(row['foreground_miou']) for row in rows]
        overlap = [_number(row['overlap_iou']) for row in rows]
        axes[0].plot(epochs, foreground, label=method)
        axes[1].plot(epochs, overlap, label=method)
    axes[0].set_title('Validation foreground mIoU')
    axes[1].set_title('Validation overlap IoU')
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Score')
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, 'validation_comparison.png'), dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method, rows in grouped.items():
        epochs = [row['epoch'] for row in rows]
        weights = [_number(row['segmentation_gradient_weight']) for row in rows]
        conflicts = [_number(row['training_gradient_conflict_rate']) for row in rows]
        axes[0].plot(epochs, weights, label=method)
        if any(math.isfinite(value) for value in conflicts):
            axes[1].plot(epochs, conflicts, label=method)
    axes[0].set_title('Segmentation gradient weight')
    axes[1].set_title('Shared-gradient conflict rate')
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, 'gradient_training_comparison.png'), dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Summarize a four-method gradient study.')
    parser.add_argument('--manifest', required=True, help='Path to gradient_study_manifest.json')
    parser.add_argument('--output', help='Summary directory; defaults beside the manifest.')
    args = parser.parse_args()
    with open(args.manifest, encoding='utf-8') as manifest_file:
        manifest = json.load(manifest_file)
    output_dir = args.output or os.path.join(os.path.dirname(args.manifest), 'analysis')
    rows = analyze_runs(manifest['runs'], output_dir)
    print(f'Analyzed {len(rows)} completed experiments: {os.path.abspath(output_dir)}')


if __name__ == '__main__':
    main()
