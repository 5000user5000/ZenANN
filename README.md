# ZenANN: Vector Similarity Search Library

## Basic Information

**ZenANN** is an approximate nearest neighbor (ANN) similarity search library for Python developers with **multiple optimization variants**. It provides several indexing methods including **IVF** (Inverted File Index), **HNSW** (Hierarchical Navigable Small World), and **KD-Tree** for exact search.

**Build Variants:**
- **naive**: Baseline version with no optimizations (single-threaded, scalar operations)
- **openmp**: Multi-threaded parallelization using OpenMP
- **simd**: SIMD vectorization using AVX2 intrinsics
- **full**: Complete optimization with OpenMP + SIMD (default)

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
- Conditional compilation for easy performance comparison

**Future Optimization Directions:**
- Cache-aware data layouts
- GPU acceleration (CUDA)

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

## Build and Test

### Requirements

- C++17 compiler (g++, clang++)
- Python >= 3.10
- CMake >= 3.17 (for Faiss)
- Ninja build system
- OpenBLAS

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
| **simd** | ~3 faster | AVX2 vectorized distance calculations |
| **full** | ~15-20x faster | Combined OpenMP + SIMD optimizations |

**Performance factors:**
- Actual speedup depends on dataset size, dimensionality, and hardware
- OpenMP scales with CPU core count (tested on 8-core systems)
- SIMD provides consistent 3x speedup for L2 distance calculations
- Combining optimizations often yields multiplicative benefits

**Optimization breakdown:**
- **Distance calculations**: SIMD provides ~3x speedup (processes 8 floats per instruction with AVX2)
- **Centroid search**: OpenMP parallelizes across centroids
- **List probing**: OpenMP parallelizes across probe lists with dynamic scheduling
- **Batch queries**: OpenMP parallelizes across multiple queries

## Project Structure

```
ZenANN/
├── include/zenann/       # C++ headers
│   ├── IndexBase.h
│   ├── IVFFlatIndex.h
│   ├── HNSWIndex.h
│   ├── KDTreeIndex.h
│   ├── VectorStore.h
│   └── SimdUtils.h      # L2 distance with optional SIMD (conditional compilation)
├── src/                  # C++ implementation (with conditional OpenMP pragmas)
├── python/               # Python bindings (pybind11)
├── tests/                # Unit tests (pytest)
├── benchmark/            # Performance benchmarks
├── extern/faiss/         # Faiss submodule
└── Makefile              # Build configuration with multiple targets
```

## Documentation

- **uml.md** - Architecture diagrams (Mermaid)
- **tests/** - Usage examples in test files
- **Makefile** - Run `make help` for build variant information

## Engineering Infrastructure

- **Build**: GNU Make, CMake
- **Testing**: pytest
- **CI/CD**: GitHub Actions (tests full variant)
- **Version Control**: Git

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
