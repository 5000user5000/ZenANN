# ZenANN: Vector Similarity Search Library

## Basic Information

**ZenANN** is an approximate nearest neighbor (ANN) similarity search library for Python developers with **multiple optimization variants**. It provides several indexing methods including **IVF** (Inverted File Index), **HNSW** (Hierarchical Navigable Small World), and **KD-Tree** for exact search.

**Build Variants:**
- **naive**: Baseline version with no optimizations (single-threaded, scalar operations)
- **openmp**: Multi-threaded parallelization using OpenMP
- **simd**: SIMD vectorization using AVX2 intrinsics
- **full**: Complete optimization with OpenMP + SIMD (default)
- **cuda**: GPU acceleration with CUDA kernels
- **profiling**: Full version with detailed performance profiling enabled

All variants provide the same API and functional correctness, differing only in performance characteristics.

## Purpose

ZenANN serves as both a **production-ready library** and a **teaching tool** for understanding parallel optimization techniques in vector similarity search.

Similarity search is a fundamental problem in many domains, including information retrieval, natural language processing, and recommendation systems. The challenge is to efficiently find the nearest neighbors of a query vector in high-dimensional space.

**Approximate nearest neighbor (ANN)** search trades off a small loss in accuracy for significant speed improvements. This implementation provides:
- **Correctness**: All algorithms produce accurate results across all build variants
- **Performance**: Multiple optimization levels from baseline to fully optimized
- **Flexibility**: Choose the appropriate variant for your use case
- **Educational value**: Compare performance impact of different optimization techniques

**Implemented Optimizations:**
- Multi-threading with OpenMP (centroid search, list probing, batch queries)
- SIMD vectorization with AVX2 (L2 distance calculations)
- GPU acceleration with CUDA (IVF-Flat search kernels)
- Conditional compilation for easy performance comparison

**CUDA Implementation Highlights:**
- Full GPU-accelerated IVF-Flat search pipeline
- Hybrid kernel strategy: fast shared-memory kernel for small k, heap-based kernel for large k
- Automatic kernel selection based on shared memory requirements
- Supports k up to 100 with all nprobe values (1-64)
- Achieves 82K+ QPS on SIFT1M dataset (k=10, nprobe=1)

## Target Users

ZenANN is ideal for:
- **Students** learning about ANN algorithms and parallel programming optimization
- **Researchers** comparing different optimization techniques and their performance impact
- **Educators** teaching high-performance computing with real-world examples
- **Developers** needing a flexible ANN library with controllable optimization levels
- **Data Scientists** requiring vector similarity search in Python applications

## System Architecture

ZenANN is implemented in C++ with a Python API using pybind11.

### Index Hierarchy

An abstract base class provides a unified interface for different index types:

1. **IndexBase** - Abstract base class
   - Common API: `build()`, `search()`, `train()`
   - Manages vector storage through `VectorStore`

2. **KDTreeIndex** - Exact nearest neighbor search
   - Tree-based partitioning for exact search
   - Useful for small datasets or validation

3. **IVFFlatIndex** - Inverted file index
   - K-means clustering for coarse quantization
   - Optional OpenMP parallelization for centroid search and list probing
   - Optional SIMD optimization for distance calculations

4. **HNSWIndex** - Hierarchical navigable small world graph
   - Built on Faiss's HNSW implementation
   - Graph-based approximate search

### Implementation Notes

- **Conditional compilation** controls optimization features via `ENABLE_SIMD` and `ENABLE_OPENMP` flags
- **naive variant**: Scalar operations, single-threaded
- **openmp variant**: Multi-threaded with OpenMP pragmas
- **simd variant**: AVX2 vectorized L2 distance calculations
- **full variant**: Combines OpenMP + SIMD for maximum performance
- All variants use standard C++ STL containers for data structures

### Processing Flow

1. **Initialize** - Create an index instance (e.g., `IVFFlatIndex`, `HNSWIndex`)
2. **Build** - Add vector data using `build(data)`
   - For IVF: automatically trains K-means clustering
   - For HNSW: constructs the graph structure
3. **Search** - Query for nearest neighbors using `search(query, k)`
   - Returns `SearchResult` with indices and distances
4. **Batch Search** - Process multiple queries with `search_batch(queries, k)`
5. **Persistence** - Save/load index with `write_index()` / `read_index()`

## API Examples

### Basic Usage

```python
import zenann
import numpy as np

# Create random data (1000 vectors, 128 dimensions)
data = np.random.rand(1000, 128).astype('float32')
query = np.random.rand(128).astype('float32')

# Option 1: IVF Index
ivf_index = zenann.IVFFlatIndex(dim=128, nlist=10, nprobe=3)
ivf_index.build(data)
results = ivf_index.search(query, k=5)

# Option 2: HNSW Index
hnsw_index = zenann.HNSWIndex(dim=128, M=16, efConstruction=200)
hnsw_index.build(data)
hnsw_index.set_ef_search(50)
results = hnsw_index.search(query, k=5)

# Option 3: KD-Tree (exact search)
kd_index = zenann.KDTreeIndex(dim=128)
kd_index.build(data)
results = kd_index.search(query, k=5)

# Access results
print("Indices:", results.indices)
print("Distances:", results.distances)
```

### Batch Search

```python
# Search multiple queries at once
queries = np.random.rand(100, 128).astype('float32')
batch_results = ivf_index.search_batch(queries, k=5)

for i, result in enumerate(batch_results):
    print(f"Query {i}: {result.indices[:3]}")  # Top 3 results
```

### Save and Load Index

```python
# Save index to disk
ivf_index.write_index("my_index.bin")

# Load index from disk
loaded_index = zenann.IVFFlatIndex.read_index("my_index.bin")
```

### Using CUDA Acceleration

The CUDA variant provides the same Python API with GPU acceleration:

```python
import zenann
import numpy as np

# Build with CUDA variant first: make cuda
data = np.random.rand(1000000, 128).astype('float32')
queries = np.random.rand(10000, 128).astype('float32')

# Same API, GPU-accelerated backend
ivf_index = zenann.IVFFlatIndex(dim=128, nlist=1024, nprobe=16)
ivf_index.build(data)

# GPU-accelerated search
results = ivf_index.search_batch(queries, k=10)

# CUDA achieves 82K+ QPS on SIFT1M (k=10, nprobe=1)
```

**Note**: The CUDA variant automatically uses GPU for IVF-Flat search operations. No API changes required.

## Benchmarking

ZenANN provides comprehensive benchmarking tools to evaluate performance across different optimization variants.

### Quick Start

```bash
# Set library path
export LD_LIBRARY_PATH=extern/faiss/build/install/lib:$LD_LIBRARY_PATH

# Run comprehensive benchmark on SIFT1M
python3 benchmark/comprehensive_bench.py \
    --base data/sift/sift_base.fvecs \
    --query data/sift/sift_query.fvecs \
    --groundtruth data/sift/sift_groundtruth.ivecs \
    --nlist 1024 \
    --nprobe-list "1,2,4,8,16,32,64" \
    --k-list "1,10,100" \
    --index-file sift_index.bin \
    --output-dir benchmark_results

# Generate Recall-QPS trade-off plots
python3 benchmark/plot_tradeoff.py benchmark_results/*.json
```

### Benchmark Metrics

The benchmark suite measures:
- **QPS (Queries Per Second)**: Throughput for batch queries
- **Latency**: Mean, p50, p95, p99 response times
- **Recall@k**: Accuracy for k=1, 10, 100
- **Index Build Time**: Time to construct the index
- **Memory Usage**: Bytes per vector

### Comparing Variants

```bash
# Test OpenMP variant
make openmp
python3 benchmark/comprehensive_bench.py ... --output-dir results_openmp

# Test CUDA variant
make cuda
python3 benchmark/comprehensive_bench.py ... --output-dir results_cuda

# Compare results
python3 benchmark/plot_tradeoff.py results_*/*.json
```

See [benchmark/BENCHMARK_GUIDE.md](benchmark/BENCHMARK_GUIDE.md) for detailed instructions.

## Build and Test

### Requirements

**Base Requirements:**
- C++17 compiler (g++, clang++)
- Python >= 3.10
- CMake >= 3.17 (for Faiss)
- Ninja build system
- OpenBLAS

**Additional Requirements for CUDA variant:**
- CUDA Toolkit >= 10.0
- NVIDIA GPU with compute capability >= 6.0 (Pascal or newer)

### Build Instructions

```bash
# 1. Clone repository with submodules
git clone --recursive https://github.com/5000user5000/ZenANN.git
cd ZenANN

# 2. Build Faiss dependency
cd extern/faiss
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX="${PWD}/build/install" \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_GPU=OFF
cmake --build build
cmake --install build
cd ../..

# 3. Build ZenANN (choose a variant)
make              # Build full version (default, OpenMP + SIMD)
make full         # Same as above
make naive        # Build naive version (no optimizations)
make openmp       # Build OpenMP-only version
make simd         # Build SIMD-only version
make cuda         # Build CUDA version (GPU acceleration)
make profiling    # Build profiling version (Full + timing)

# 4. Run tests
LD_LIBRARY_PATH=extern/faiss/build/install/lib pytest tests/
```

### Build Variants

Choose the appropriate variant for your needs:

| Target | Optimizations | Use Case |
|--------|--------------|----------|
| `make naive` | None | Baseline reference, debugging |
| `make openmp` | Multi-threading only | Study OpenMP impact |
| `make simd` | SIMD (AVX2) only | Study vectorization impact |
| `make full` | OpenMP + SIMD | Production use (default) |
| `make cuda` | GPU kernels | GPU acceleration, highest QPS |
| `make profiling` | Full + timing | Performance analysis |

**CUDA Build Notes:**
- Ensure `nvcc` is in your PATH and CUDA Toolkit is properly installed
- Adjust `CUDA_ARCH` in Makefile to match your GPU (sm_60=Pascal, sm_75=Turing, sm_86=Ampere)
- The CUDA variant uses pure GPU acceleration (no OpenMP/SIMD)
- Hybrid kernel strategy automatically handles k values up to 100

### Running Tests

All unit tests validate **functional correctness** only (not performance):

```bash
pytest tests/ -v

# Test specific index
pytest tests/test_ivfflat.py -v
pytest tests/test_hnsw.py -v
pytest tests/test_kdtree.py -v
```

## Performance Characteristics

All variants provide **correct results** with different performance profiles:

| Variant | Performance | Key Features |
|---------|-------------|--------------|
| **naive** | Baseline (1x) | Single-threaded, scalar operations |
| **openmp** | ~10x faster | Multi-threaded parallelization |
| **simd** | ~3x faster | AVX2 vectorized distance calculations |
| **full** | ~15-20x faster | Combined OpenMP + SIMD optimizations , highest QPS in k = 100 |
| **cuda** | ~20-25x faster | GPU parallelization, highest QPS in k=1,10|

**Performance factors:**
- Actual speedup depends on dataset size, dimensionality, and hardware
- OpenMP scales with CPU core count (tested on 8-core systems)
- SIMD provides consistent 3x speedup for L2 distance calculations
- Combining optimizations often yields multiplicative benefits
- CUDA achieves 82K+ QPS on SIFT1M (k=10, nprobe=1) with NVIDIA GPUs

**Optimization breakdown:**
- **Distance calculations**: SIMD provides ~3x speedup (processes 8 floats per instruction with AVX2)
- **Centroid search**: OpenMP parallelizes across centroids; CUDA uses GPU threads
- **List probing**: OpenMP parallelizes across probe lists; CUDA uses 2D grid mapping
- **Batch queries**: OpenMP parallelizes across multiple queries; CUDA processes batch on GPU
- **Top-K selection**: CUDA uses hybrid strategy (shared memory vs heap-based) for optimal performance

## Project Structure

```
ZenANN/
├── include/zenann/       # C++ headers
│   ├── IndexBase.h
│   ├── IVFFlatIndex.h
│   ├── HNSWIndex.h
│   ├── KDTreeIndex.h
│   ├── VectorStore.h
│   ├── SimdUtils.h      # L2 distance with optional SIMD (conditional compilation)
│   └── CudaUtils.h      # CUDA kernel declarations
├── src/                  # C++ implementation (with conditional OpenMP pragmas)
│   ├── IndexBase.cpp
│   ├── IVFFlatIndex.cpp
│   ├── KDTreeIndex.cpp
│   ├── HNSWIndex.cpp
│   └── CudaUtils.cu     # CUDA kernel implementations
├── python/               # Python bindings (pybind11)
├── tests/                # Unit tests (pytest)
├── benchmark/            # Performance benchmarks
│   ├── comprehensive_bench.py  # Complete benchmark suite
│   ├── ivf-bench.py            # IVF-specific benchmarks
│   ├── hnsw-bench.py           # HNSW-specific benchmarks
│   ├── plot_tradeoff.py        # Visualization tools
│   └── BENCHMARK_GUIDE.md      # Benchmarking documentation
├── doc/                  # Technical documentation
│   ├── cuda.md          # CUDA implementation guide
│   └── cuda-fix.md      # CUDA k=100 fix documentation
├── extern/faiss/         # Faiss submodule
└── Makefile              # Build configuration with multiple targets
```

### Core Documentation
- **tests/** - Usage examples in test files
- **Makefile** - Run `make help` for build variant information

## Engineering Infrastructure

- **Build**: GNU Make
- **Testing**: pytest
- **CI/CD**: GitHub Actions (tests full variant)
- **Version Control**: Git

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
