#!/usr/bin/env python3
"""
IVFFlat Search Profiling Tool
==============================

This tool analyzes the time breakdown of different stages in IVFFlat search.
It runs multiple search queries and collects profiling data to identify bottlenecks.

Usage:
------
1. Make sure the profiling version is built:
   $ make profiling

2. Set library path:
   $ export LD_LIBRARY_PATH=extern/faiss/build/install/lib:$LD_LIBRARY_PATH

3. Run with random test data (quick test):
   $ python3 benchmark/ivf_profiling.py --random

4. Run with custom parameters:
   $ python3 benchmark/ivf_profiling.py --random --random_base 50000 --nlist 256 --nprobe 16

5. Run with real dataset:
   $ python3 benchmark/ivf_profiling.py --base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs

Output:
-------
- Time breakdown for each search stage (centroid distance, list scanning, etc.)
- Average, min, max values for each stage
- Percentage of total time for each stage
- Identification of bottleneck for CUDA optimization

Notes:
------
- Index will be saved to ./index/ directory for reuse
- Built index is shared between runs with same parameters
"""

import sys
import os
import argparse
import numpy as np
import re
import subprocess
import tempfile

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', 'build')))


def load_fvecs(filename, max_vectors=None):
    """Load float vectors from .fvecs format"""
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if max_vectors:
        fv = fv[:max_vectors]
    return fv.copy()


def parse_profile_line(line):
    """Parse profiling output"""
    match = re.search(
        r'PROFILE: centroid_dist=([\d.]+)ms, centroid_select=([\d.]+)ms, '
        r'list_scan=([\d.]+)ms, final_sort=([\d.]+)ms, total=([\d.]+)ms',
        line
    )
    if match:
        return {
            'centroid_dist': float(match.group(1)),
            'centroid_select': float(match.group(2)),
            'list_scan': float(match.group(3)),
            'final_sort': float(match.group(4)),
            'total': float(match.group(5))
        }
    return None


def run_profiling_searches(index_file, query_data, nprobe, k, num_queries):
    """Run searches using subprocess to capture C++ stderr"""

    # Create temporary files for data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        script_path = f.name
        f.write(f"""
import sys
import os
import numpy as np

sys.path.append('{os.path.abspath('build')}')
from zenann import IVFFlatIndex

# Load pre-built index
index = IVFFlatIndex.read_index('{index_file}')

# Load queries
queries = np.load('{tempfile.gettempdir()}/prof_queries.npy')

# Run searches
for query in queries:
    result = index.search(query.tolist(), {k}, {nprobe})
""")

    # Save queries only (index is already saved)
    np.save(f'{tempfile.gettempdir()}/prof_queries.npy', query_data[:num_queries])

    # Run subprocess and capture stderr
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = 'extern/faiss/build/install/lib:' + env.get('LD_LIBRARY_PATH', '')

    result = subprocess.run(
        ['python3', script_path],
        capture_output=True,
        text=True,
        env=env
    )

    # Clean up
    os.unlink(script_path)

    # Parse profiling data from stderr
    profile_data = []
    for line in result.stderr.split('\n'):
        if 'PROFILE:' in line:
            data = parse_profile_line(line)
            if data:
                profile_data.append(data)

    return profile_data


def main(args):
    print("=" * 70)
    print("IVFFlat Search Profiling Tool")
    print("=" * 70)

    # Load or generate dataset
    if args.random:
        print("\n1. Generating random test data...")
        print(f"   Base: {args.random_base:,} vectors x {args.random_dim}D")
        print(f"   Queries: {args.num_queries:,} vectors")
        np.random.seed(42)
        base = np.random.randn(args.random_base, args.random_dim).astype(np.float32)
        queries = np.random.randn(args.num_queries, args.random_dim).astype(np.float32)
    else:
        print("\n1. Loading dataset...")
        base = load_fvecs(args.base, max_vectors=args.max_base)
        queries = load_fvecs(args.query, max_vectors=args.num_queries)
        print(f"   Base vectors: {base.shape[0]:,} x {base.shape[1]}D")
        print(f"   Query vectors: {queries.shape[0]:,} x {queries.shape[1]}D")

    # Build index once
    print(f"\n2. Building IVF index (nlist={args.nlist})...")
    from zenann import IVFFlatIndex

    # Create index directory if not exists
    os.makedirs('index', exist_ok=True)

    # Generate index filename based on parameters
    if args.random:
        index_file = f'index/prof_random_{args.random_base}_{args.random_dim}d_nlist{args.nlist}.bin'
    else:
        # Use base filename for real data
        base_name = os.path.splitext(os.path.basename(args.base))[0]
        index_file = f'index/prof_{base_name}_nlist{args.nlist}.bin'

    # Check if index already exists
    if os.path.exists(index_file):
        print(f"   Loading existing index from {index_file}")
        index = IVFFlatIndex.read_index(index_file)
    else:
        print(f"   Building new index...")
        index = IVFFlatIndex(dim=base.shape[1], nlist=args.nlist, nprobe=args.nprobe)
        index.build(base.tolist())
        index.write_index(index_file)
        print(f"   Index saved to {index_file}")

    print(f"\n3. Running profiling (nprobe={args.nprobe}, k={args.k})...")
    print(f"   Processing {args.num_queries} queries...")

    # Run profiling with pre-built index
    profile_data = run_profiling_searches(
        index_file, queries, args.nprobe, args.k, args.num_queries
    )

    if not profile_data:
        print("\n   ERROR: No profiling data collected!")
        print("   Make sure the library was built with 'make profiling'")
        return

    print(f"   Collected {len(profile_data)} profiling samples")

    # Analyze results
    print(f"\n4. Profiling Results (averaged over {len(profile_data)} queries)")
    print("=" * 70)

    # Calculate statistics
    keys = ['centroid_dist', 'centroid_select', 'list_scan', 'final_sort', 'total']
    stats = {}

    for key in keys:
        values = [d[key] for d in profile_data]
        stats[key] = {
            'mean': np.mean(values),
            'min': np.min(values),
            'max': np.max(values),
            'std': np.std(values)
        }

    # Print results
    print("\nTime Breakdown (milliseconds per query):")
    print("-" * 70)
    print(f"{'Stage':<25} {'Mean':>10} {'Min':>10} {'Max':>10} {'% Total':>10}")
    print("-" * 70)

    total_mean = stats['total']['mean']
    stage_names = {
        'centroid_dist': 'Centroid Distance',
        'centroid_select': 'Centroid Selection',
        'list_scan': 'List Scanning',
        'final_sort': 'Final Sorting'
    }

    for key in keys[:-1]:
        name = stage_names[key]
        mean = stats[key]['mean']
        min_val = stats[key]['min']
        max_val = stats[key]['max']
        pct = (mean / total_mean * 100) if total_mean > 0 else 0
        print(f"{name:<25} {mean:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {pct:>9.1f}%")

    print("-" * 70)
    print(f"{'TOTAL':<25} {total_mean:>10.4f} {stats['total']['min']:>10.4f} "
          f"{stats['total']['max']:>10.4f} {100.0:>9.1f}%")

    # Performance metrics
    qps = 1000.0 / total_mean
    print(f"\nPerformance:")
    print(f"  Average latency: {total_mean:.4f} ms/query")
    print(f"  Throughput: {qps:.2f} QPS (single query)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find the bottleneck
    bottleneck = max(keys[:-1], key=lambda k: stats[k]['mean'])
    bottleneck_pct = (stats[bottleneck]['mean'] / total_mean * 100)

    print(f"\nBottleneck: {stage_names[bottleneck]}")
    print(f"  - Time: {stats[bottleneck]['mean']:.4f} ms ({bottleneck_pct:.1f}% of total)")
    print(f"  - This is the PRIMARY target for GPU acceleration")

    # Recommendations
    print(f"\nCUDA Optimization Targets:")
    for key in keys[:-1]:
        pct = (stats[key]['mean'] / total_mean * 100)
        if pct > 10:  # Only show significant stages
            print(f"  ✓ {stage_names[key]}: {stats[key]['mean']:.4f} ms ({pct:.1f}%)")
            if key == 'centroid_dist':
                print(f"    → Parallelize distance computation across {args.nlist} centroids")
            elif key == 'list_scan':
                print(f"    → Parallelize distance computation across candidates in lists")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="IVFFlat Search Profiling Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with random data
  python3 benchmark/ivf_profiling_simple.py --random

  # Larger test
  python3 benchmark/ivf_profiling_simple.py --random --random_base 50000 --nlist 256 --nprobe 16

  # With real dataset
  python3 benchmark/ivf_profiling_simple.py --base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
        """
    )

    # Data source
    parser.add_argument("--random", action="store_true",
                        help="Use randomly generated data")
    parser.add_argument("--base", help="Path to base.fvecs")
    parser.add_argument("--query", help="Path to query.fvecs")

    # Random data parameters
    parser.add_argument("--random_base", type=int, default=10000,
                        help="Number of base vectors (default: 10000)")
    parser.add_argument("--random_dim", type=int, default=128,
                        help="Dimension (default: 128)")

    # Index parameters
    parser.add_argument("--nlist", type=int, default=100,
                        help="Number of clusters (default: 100)")
    parser.add_argument("--nprobe", type=int, default=10,
                        help="Clusters to probe (default: 10)")
    parser.add_argument("--k", type=int, default=10,
                        help="Nearest neighbors (default: 10)")

    # Profiling parameters
    parser.add_argument("--num_queries", type=int, default=100,
                        help="Number of queries (default: 100)")
    parser.add_argument("--max_base", type=int, default=None,
                        help="Max base vectors from file")

    args = parser.parse_args()

    if not args.random and (not args.base or not args.query):
        parser.error("--base and --query required unless --random specified")

    main(args)
