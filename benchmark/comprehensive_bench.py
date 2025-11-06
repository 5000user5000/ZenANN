#!/usr/bin/env python3
"""
Comprehensive Benchmark for ZenANN IVFFlatIndex

Measures all required metrics:
- Recall@k for k ∈ {1, 10, 100}
- QPS (Queries Per Second)
- Latency (p50, p95)
- Index build time
- Memory usage (bytes/vector)
- Recall-QPS trade-off curve across nprobe values

Supports SIFT1M and GIST1M datasets.
"""

import sys
import os
import time
import argparse
import numpy as np
import json
import csv
from pathlib import Path
from datetime import datetime

# Add build directory to path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', 'build')))


def load_fvecs(filename):
    """Load .fvecs format file"""
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError(f"Non-uniform vector sizes in {filename}")
    fv = fv[:, 1:]
    return fv.copy()


def load_ivecs(filename):
    """Load .ivecs format file"""
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0), dtype=np.int32)
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    fv = fv[:, 1:]
    return fv


def compute_recall_at_k(predicted, groundtruth, k):
    """Compute Recall@k"""
    n_queries = len(predicted)
    recalls = []

    for i in range(n_queries):
        # Get top-k from ground truth
        true_set = set(groundtruth[i, :k])
        # Get top-k from predictions (handle variable length)
        pred_k = min(k, len(predicted[i]))
        pred_set = set(predicted[i][:pred_k])
        # Calculate recall
        intersection = len(true_set & pred_set)
        recall = intersection / k
        recalls.append(recall)

    return np.mean(recalls)


def measure_latencies(index, queries, k, nprobe):
    """Measure per-query latencies for percentile calculation"""
    latencies = []

    for query in queries:
        t0 = time.perf_counter()
        result = index.search(query.tolist(), k, nprobe)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # Convert to milliseconds

    return np.array(latencies)


def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_single_config(index, queries, gt, nlist, nprobe, k_values, n_base_vectors):
    """
    Benchmark a single configuration (nlist, nprobe)
    Returns metrics for all k values
    """
    import zenann

    print(f"\n{'='*70}")
    print(f"Testing: nlist={nlist}, nprobe={nprobe}")
    print(f"{'='*70}")

    # 1. Measure batch QPS (use max k)
    max_k = max(k_values)
    print(f"Measuring batch QPS (k={max_k})...")

    t0 = time.perf_counter()
    all_results = index.search_batch(queries.tolist(), max_k, nprobe)
    t_batch = time.perf_counter() - t0
    qps_batch = len(queries) / t_batch

    # Collect results
    results_array = []
    for res in all_results:
        results_array.append(res.indices)

    # 2. Measure per-query latencies (sample for p50/p95)
    print(f"Measuring latencies (sampling 1000 queries)...")
    sample_size = min(1000, len(queries))
    sample_indices = np.random.choice(len(queries), sample_size, replace=False)
    sample_queries = queries[sample_indices]

    latencies = measure_latencies(index, sample_queries, max_k, nprobe)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    mean_latency = np.mean(latencies)

    print(f"  QPS (batch): {qps_batch:.2f}")
    print(f"  Latency - Mean: {mean_latency:.3f} ms")
    print(f"  Latency - p50: {p50_latency:.3f} ms")
    print(f"  Latency - p95: {p95_latency:.3f} ms")
    print(f"  Latency - p99: {p99_latency:.3f} ms")

    # 3. Compute Recall@k for multiple k values
    print(f"Computing Recall@k...")
    recalls = {}

    for k in k_values:
        # Pad results to have at least k elements
        padded_results = []
        for res in results_array:
            if len(res) < k:
                padded = list(res) + [-1] * (k - len(res))
            else:
                padded = res[:k]
            padded_results.append(padded)

        recall = compute_recall_at_k(padded_results, gt, k)
        recalls[f'recall@{k}'] = recall
        print(f"  Recall@{k}: {recall*100:.2f}%")

    # 4. Measure memory usage
    mem_after = get_memory_usage_mb()

    return {
        'nlist': nlist,
        'nprobe': nprobe,
        'qps_batch': qps_batch,
        'latency_mean_ms': mean_latency,
        'latency_p50_ms': p50_latency,
        'latency_p95_ms': p95_latency,
        'latency_p99_ms': p99_latency,
        **recalls,  # Unpack all recall values
        'memory_mb': mem_after,
    }


def main(args):
    import zenann

    print("="*70)
    print("ZenANN Comprehensive Benchmark")
    print("="*70)

    # 1. Load dataset
    print(f"\nLoading dataset...")
    base = load_fvecs(args.base)
    queries = load_fvecs(args.query)
    gt = load_ivecs(args.groundtruth)

    print(f"  Base vectors: {base.shape}")
    print(f"  Queries: {queries.shape}")
    print(f"  Ground truth: {gt.shape}")

    n_base = base.shape[0]
    dim = base.shape[1]

    # 2. Build or load index
    mem_before = get_memory_usage_mb()

    if args.index_file and os.path.exists(args.index_file):
        print(f"\nLoading index from {args.index_file}...")
        index = zenann.IVFFlatIndex.read_index(args.index_file)
        print("Index loaded.")
        build_time = 0  # Not measured when loading
    else:
        print(f"\nBuilding IVF index (nlist={args.nlist})...")
        index = zenann.IVFFlatIndex(dim=dim, nlist=args.nlist, nprobe=1)

        t0 = time.perf_counter()
        index.build(base)
        build_time = time.perf_counter() - t0

        print(f"  Build time: {build_time:.3f} s")

        if args.index_file:
            print(f"  Saving index to {args.index_file}...")
            index.write_index(args.index_file)

    mem_after = get_memory_usage_mb()
    index_memory_mb = mem_after - mem_before
    bytes_per_vector = (index_memory_mb * 1024 * 1024) / n_base

    print(f"\nIndex memory usage:")
    print(f"  Total: {index_memory_mb:.2f} MB")
    print(f"  Per vector: {bytes_per_vector:.2f} bytes/vector")
    print(f"  Ratio to raw data: {bytes_per_vector / (dim * 4):.2f}x")

    # 3. Parse nprobe values
    nprobe_list = [int(x.strip()) for x in args.nprobe_list.split(',')]
    k_values = [int(x.strip()) for x in args.k_list.split(',')]

    print(f"\nTesting configurations:")
    print(f"  nprobe values: {nprobe_list}")
    print(f"  k values: {k_values}")

    # 4. Run benchmarks for each nprobe
    results = []

    for nprobe in nprobe_list:
        result = benchmark_single_config(
            index, queries, gt, args.nlist, nprobe, k_values, n_base
        )
        result['build_time_sec'] = build_time
        result['bytes_per_vector'] = bytes_per_vector
        results.append(result)

    # 5. Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_name = Path(args.base).stem.split('_')[0]  # e.g., "sift" from "sift_base.fvecs"

    # Save as JSON
    json_file = output_dir / f'{dataset_name}_nlist{args.nlist}_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump({
            'metadata': {
                'dataset': dataset_name,
                'n_base': n_base,
                'n_queries': len(queries),
                'dimension': dim,
                'nlist': args.nlist,
                'nprobe_list': nprobe_list,
                'k_values': k_values,
                'build_time_sec': build_time,
                'bytes_per_vector': bytes_per_vector,
                'timestamp': timestamp,
            },
            'results': results
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {json_file}")

    # Save as CSV
    csv_file = output_dir / f'{dataset_name}_nlist{args.nlist}_{timestamp}.csv'
    if results:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to: {csv_file}")

    # 6. Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Recall-QPS Trade-off")
    print(f"{'='*70}")
    print(f"{'nprobe':<8} {'QPS':<10} {'p50(ms)':<10} {'p95(ms)':<10} ", end='')
    for k in k_values:
        print(f"{'R@'+str(k):<10} ", end='')
    print()
    print("-"*70)

    for r in results:
        print(f"{r['nprobe']:<8} {r['qps_batch']:<10.1f} {r['latency_p50_ms']:<10.3f} {r['latency_p95_ms']:<10.3f} ", end='')
        for k in k_values:
            recall = r[f'recall@{k}'] * 100
            print(f"{recall:<10.2f} ", end='')
        print()

    print("="*70)

    # Check if Recall@10 >= 0.95 is achieved
    max_recall_10 = max(r['recall@10'] for r in results)
    if max_recall_10 >= 0.95:
        print(f"\n✅ Target achieved: Recall@10 = {max_recall_10*100:.2f}% >= 95%")
        best_config = max(results, key=lambda x: x['recall@10'])
        print(f"   Best config: nprobe={best_config['nprobe']}, QPS={best_config['qps_batch']:.1f}")
    else:
        print(f"\n⚠️  Target not met: Best Recall@10 = {max_recall_10*100:.2f}% < 95%")
        print(f"   Consider increasing nprobe or nlist")

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comprehensive benchmark for ZenANN IVFFlatIndex',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # SIFT1M benchmark
  python3 comprehensive_bench.py \\
      --base data/sift/sift_base.fvecs \\
      --query data/sift/sift_query.fvecs \\
      --groundtruth data/sift/sift_groundtruth.ivecs \\
      --nlist 1024 \\
      --nprobe-list "1,2,4,8,16,32,64,128" \\
      --k-list "1,10,100"

  # GIST1M benchmark
  python3 comprehensive_bench.py \\
      --base data/gist/gist_base.fvecs \\
      --query data/gist/gist_query.fvecs \\
      --groundtruth data/gist/gist_groundtruth.ivecs \\
      --nlist 1024 \\
      --nprobe-list "1,4,16,64,256" \\
      --k-list "1,10,100"
        """
    )

    parser.add_argument('--base', required=True, help='Path to base.fvecs')
    parser.add_argument('--query', required=True, help='Path to query.fvecs')
    parser.add_argument('--groundtruth', required=True, help='Path to groundtruth.ivecs')
    parser.add_argument('--nlist', type=int, default=1024, help='Number of IVF clusters (default: 1024)')
    parser.add_argument('--nprobe-list', type=str, default='1,4,8,16,32,64',
                        help='Comma-separated nprobe values to test (default: 1,4,8,16,32,64)')
    parser.add_argument('--k-list', type=str, default='1,10,100',
                        help='Comma-separated k values for Recall@k (default: 1,10,100)')
    parser.add_argument('--index-file', default=None,
                        help='Path to save/load index (optional, speeds up repeated tests)')
    parser.add_argument('--output-dir', default='benchmark_results',
                        help='Output directory for results (default: benchmark_results)')

    args = parser.parse_args()

    # Check dependencies
    try:
        import psutil
    except ImportError:
        print("ERROR: psutil module required for memory measurement")
        print("Install with: pip install psutil")
        sys.exit(1)

    main(args)
