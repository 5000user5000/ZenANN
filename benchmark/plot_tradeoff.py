#!/usr/bin/env python3
"""
Plot Recall-QPS trade-off curves from benchmark results
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_benchmark_result(json_file):
    """Load benchmark result from JSON file"""
    with open(json_file) as f:
        data = json.load(f)
    return data


def plot_recall_qps_tradeoff(results_files, output_file='recall_qps_tradeoff.png'):
    """
    Plot Recall vs QPS trade-off curves

    Args:
        results_files: List of JSON result files to compare
        output_file: Output image file path
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Recall-QPS Trade-off Curves', fontsize=16, fontweight='bold')

    k_values = [1, 10, 100]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_files)))

    for idx, k in enumerate(k_values):
        ax = axes[idx]

        for file_idx, result_file in enumerate(results_files):
            data = load_benchmark_result(result_file)
            metadata = data['metadata']
            results = data['results']

            # Extract data
            nprobes = [r['nprobe'] for r in results]
            qps_values = [r['qps_batch'] for r in results]

            # Handle missing recall@k values
            recall_key = f'recall@{k}'
            if recall_key not in results[0]:
                print(f"Warning: {recall_key} not found in {result_file}, skipping this file")
                continue

            recall_values = [r[recall_key] * 100 for r in results]

            # Create label
            dataset = metadata.get('dataset', 'unknown')
            nlist = metadata.get('nlist', '?')
            label = f"{dataset} (nlist={nlist})"

            # Plot
            ax.plot(recall_values, qps_values, 'o-',
                   color=colors[file_idx], label=label, linewidth=2, markersize=8)

            # Annotate nprobe values
            for i, nprobe in enumerate(nprobes):
                if i % 2 == 0:  # Only annotate every other point to avoid clutter
                    ax.annotate(f'np={nprobe}',
                              (recall_values[i], qps_values[i]),
                              textcoords="offset points",
                              xytext=(0, 10), ha='center', fontsize=8)

        ax.set_xlabel(f'Recall@{k} (%)', fontsize=12)
        ax.set_ylabel('QPS (queries/sec)', fontsize=12)
        ax.set_title(f'Recall@{k} vs QPS', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Add target line for Recall@10
        if k == 10:
            ax.axvline(x=95, color='red', linestyle='--', linewidth=2, label='Target (95%)')
            ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.show()


def plot_latency_distribution(results_files, output_file='latency_distribution.png'):
    """
    Plot latency percentiles across nprobe values
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Latency Analysis', fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_files)))

    for file_idx, result_file in enumerate(results_files):
        data = load_benchmark_result(result_file)
        metadata = data['metadata']
        results = data['results']

        nprobes = [r['nprobe'] for r in results]
        p50 = [r['latency_p50_ms'] for r in results]
        p95 = [r['latency_p95_ms'] for r in results]
        p99 = [r['latency_p99_ms'] for r in results]
        mean_latency = [r['latency_mean_ms'] for r in results]

        dataset = metadata.get('dataset', 'unknown')
        nlist = metadata.get('nlist', '?')
        label = f"{dataset} (nlist={nlist})"

        # Plot 1: Latency vs nprobe
        ax = axes[0]
        ax.plot(nprobes, mean_latency, 'o-', color=colors[file_idx],
               label=f'{label} (mean)', linewidth=2)
        ax.plot(nprobes, p95, 's--', color=colors[file_idx],
               label=f'{label} (p95)', linewidth=1.5, alpha=0.7)

        # Plot 2: p50 vs p95
        ax = axes[1]
        ax.plot(p50, p95, 'o-', color=colors[file_idx],
               label=label, linewidth=2, markersize=8)

        # Annotate nprobe values
        for i, nprobe in enumerate(nprobes):
            if i % 2 == 0:
                ax.annotate(f'np={nprobe}',
                          (p50[i], p95[i]),
                          textcoords="offset points",
                          xytext=(5, 5), ha='left', fontsize=8)

    # Configure plot 1
    axes[0].set_xlabel('nprobe', fontsize=12)
    axes[0].set_ylabel('Latency (ms)', fontsize=12)
    axes[0].set_title('Latency vs nprobe', fontsize=14, fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    # Configure plot 2
    axes[1].set_xlabel('p50 Latency (ms)', fontsize=12)
    axes[1].set_ylabel('p95 Latency (ms)', fontsize=12)
    axes[1].set_title('p50 vs p95 Latency', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.show()


def generate_report(results_files, output_file='benchmark_report.txt'):
    """Generate a text report comparing all results"""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BENCHMARK COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")

        for result_file in results_files:
            data = load_benchmark_result(result_file)
            metadata = data['metadata']
            results = data['results']

            dataset = metadata.get('dataset', 'unknown')
            nlist = metadata.get('nlist', '?')
            build_time = metadata.get('build_time_sec', '?')
            bytes_per_vec = metadata.get('bytes_per_vector', '?')

            f.write(f"\nDataset: {dataset}\n")
            f.write(f"  nlist: {nlist}\n")
            f.write(f"  Build time: {build_time:.2f} s\n")
            f.write(f"  Memory: {bytes_per_vec:.2f} bytes/vector\n")
            f.write(f"\n  {'nprobe':<8} {'QPS':<10} {'p50(ms)':<10} {'p95(ms)':<10} {'R@1':<8} {'R@10':<8} {'R@100':<8}\n")
            f.write("  " + "-"*70 + "\n")

            for r in results:
                f.write(f"  {r['nprobe']:<8} {r['qps_batch']:<10.1f} "
                       f"{r['latency_p50_ms']:<10.3f} {r['latency_p95_ms']:<10.3f} "
                       f"{r.get('recall@1', 0)*100:<8.2f} "
                       f"{r.get('recall@10', 0)*100:<8.2f} "
                       f"{r.get('recall@100', 0)*100:<8.2f}\n")

            # Find best config for Recall@10 >= 95%
            candidates = [r for r in results if r.get('recall@10', 0) >= 0.95]
            if candidates:
                best = max(candidates, key=lambda x: x['qps_batch'])
                f.write(f"\n  ✅ Best config (R@10≥95%): nprobe={best['nprobe']}, "
                       f"QPS={best['qps_batch']:.1f}, R@10={best['recall@10']*100:.2f}%\n")
            else:
                f.write(f"\n  ⚠️  No config achieves R@10≥95%\n")

            f.write("\n" + "-"*80 + "\n")

    print(f"Report saved to: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_tradeoff.py <result1.json> [result2.json ...]")
        print("\nExample:")
        print("  python3 plot_tradeoff.py benchmark_results/sift_nlist1024_*.json")
        sys.exit(1)

    result_files = sys.argv[1:]

    # Verify files exist
    for f in result_files:
        if not Path(f).exists():
            print(f"Error: File not found: {f}")
            sys.exit(1)

    print(f"Plotting {len(result_files)} result file(s)...")

    # Generate plots
    plot_recall_qps_tradeoff(result_files, 'recall_qps_tradeoff.png')
    plot_latency_distribution(result_files, 'latency_distribution.png')

    # Generate report
    generate_report(result_files, 'benchmark_report.txt')

    print("\nDone!")


if __name__ == '__main__':
    main()
