# ZenANN: Vector Similarity Search Library (Naive Baseline Implementation)

## Basic Information

**ZenANN** is a straightforward implementation of approximate nearest neighbor (ANN) similarity search library for Python developers. This is a **naive baseline version** that provides multiple indexing methods, such as **IVF** (Inverted File Index), **HNSW** (Hierarchical Navigable Small World), and **KD-Tree** for exact search.

**Key Characteristics of This Version:**
- **No parallelization**: Single-threaded execution only (no OpenMP)
- **No SIMD**: Scalar computation for distance calculations
- **Baseline implementation**: Serves as a performance reference for optimization studies
- **Functional correctness**: All algorithms work correctly, just not optimized for speed

## Purpose

This naive implementation serves as a **baseline reference** for understanding and optimizing vector similarity search algorithms.

Similarity search is a fundamental problem in many domains, including information retrieval, natural language processing, and recommendation systems. The challenge is to efficiently find the nearest neighbors of a query vector in high-dimensional space.

**Approximate nearest neighbor (ANN)** search trades off a small loss in accuracy for significant speed improvements. This implementation focuses on:
- **Correctness**: All algorithms produce accurate results
- **Simplicity**: Clean, understandable code without optimization complexity
- **Baseline**: Performance reference for measuring optimization improvements

**Potential Optimization Directions** (not implemented in this version):
- Multi-threading (OpenMP, pthread)
- SIMD vectorization (AVX2, AVX-512)
- Cache-aware data layouts
- GPU acceleration

## Target Users

This baseline implementation is ideal for:
- **Students** learning about ANN algorithms and parallel programming optimization
- **Researchers** needing a clean reference implementation for comparison
- **Educators** teaching high-performance computing and algorithm optimization
- **Developers** who want to understand ANN algorithms before applying optimizations

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

3. **IVFFlatIndex** - Inverted file index (naive implementation)
   - K-means clustering for coarse quantization
   - Sequential search within clusters
   - **No OpenMP parallelization** in this version

4. **HNSWIndex** - Hierarchical navigable small world graph
   - Built on Faiss's HNSW implementation
   - Graph-based approximate search

### Implementation Notes

- All distance calculations use **scalar operations** (no SIMD)
- All loops are **sequential** (no multi-threading)
- Data structures use standard C++ STL containers

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

# 3. Build ZenANN
make

# 4. Run tests
LD_LIBRARY_PATH=extern/faiss/build/install/lib pytest tests/
```

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

This naive implementation provides:
- ✅ **Correct results** - All algorithms work properly
- ⚠️ **Slower performance** - 10-50x slower than optimized versions
- 📊 **Baseline metrics** - Reference for measuring optimization gains

Expected performance (compared to parallelized version):
- IVF search: ~10x slower (no OpenMP)
- Distance calculation: ~4-8x slower (no SIMD)
- Batch queries: ~N x slower (N = CPU cores, no parallelization)

## Project Structure

```
ZenANN/
├── include/zenann/       # C++ headers
│   ├── IndexBase.h
│   ├── IVFFlatIndex.h
│   ├── HNSWIndex.h
│   ├── KDTreeIndex.h
│   ├── VectorStore.h
│   └── SimdUtils.h      # Naive L2 distance (no SIMD)
├── src/                  # C++ implementation
├── python/               # Python bindings (pybind11)
├── tests/                # Unit tests (pytest)
├── benchmark/            # Performance benchmarks
├── extern/faiss/         # Faiss submodule
├── claude.md             # Restoration guide to parallelized version
└── Makefile              # Build configuration
```

## Documentation

- **claude.md** - Complete record of parallelization removal and restoration instructions
- **uml.md** - Architecture diagrams (Mermaid)
- **tests/** - Usage examples in test files

## Engineering Infrastructure

- **Build**: GNU Make, CMake
- **Testing**: pytest
- **CI/CD**: GitHub Actions
- **Version Control**: Git

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
