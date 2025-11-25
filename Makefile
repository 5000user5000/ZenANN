CXX := g++
NVCC := nvcc
BASE_CXXFLAGS := -std=c++17 -O3 -fPIC

# CUDA configuration
CUDA_PATH ?= /usr/local/cuda
CUDA_ARCH := -arch=sm_60  # Adjust for your GPU (sm_60=Pascal, sm_75=Turing, sm_86=Ampere)
NVCC_FLAGS := -O3 --compiler-options '-fPIC' $(CUDA_ARCH)

# Python / pybind11 include flags
PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
PYTHON_INCLUDE    := $(shell python3-config --includes)
PYTHON_LIB        := $(shell python3-config --ldflags)

# Faiss submodule install path
FAISS_ROOT := extern/faiss/build/install

# Project includes
PROJECT_INCLUDE := -I./include -I./include/zenann

# Aggregate includes
ALL_INCLUDES := $(PYBIND11_INCLUDES) $(PYTHON_INCLUDE) $(PROJECT_INCLUDE) -I$(FAISS_ROOT)/include

# Python linking flags only (faiss linked below)
ALL_LIBS := $(PYTHON_LIB)

# Source files
SOURCES := \
    src/IndexBase.cpp \
    src/IVFFlatIndex.cpp \
    src/KDTreeIndex.cpp \
    src/HNSWIndex.cpp \
    python/zenann_pybind.cpp

# Extension suffix (.so or .cpython-XYm-x86_64-linux-gnu.so, etc.)
EXT_SUFFIX := $(shell python3-config --extension-suffix)

# Platform‐specific linker flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # on macOS use dynamic_lookup
    BASE_LDFLAGS := -undefined dynamic_lookup
else
    # on Linux embed rpath to pick up our extern/faiss libfaiss.so
    BASE_LDFLAGS := -Wl,-rpath,$$ORIGIN/../extern/faiss/build/install/lib
endif

# ============================================================================
# Version-specific build configurations
# ============================================================================

# Output target (all versions output to the same file)
TARGET := build/zenann$(EXT_SUFFIX)

# NAIVE: No parallelization (baseline)
NAIVE_CXXFLAGS := $(BASE_CXXFLAGS)
NAIVE_LDFLAGS := $(BASE_LDFLAGS)

# OPENMP: Only OpenMP parallelization
OPENMP_CXXFLAGS := $(BASE_CXXFLAGS) -fopenmp -DENABLE_OPENMP
OPENMP_LDFLAGS := $(BASE_LDFLAGS) -fopenmp

# SIMD: Only SIMD vectorization (AVX2)
SIMD_CXXFLAGS := $(BASE_CXXFLAGS) -march=native -DENABLE_SIMD
SIMD_LDFLAGS := $(BASE_LDFLAGS)

# FULL: OpenMP + SIMD (fully optimized)
FULL_CXXFLAGS := $(BASE_CXXFLAGS) -march=native -fopenmp -DENABLE_OPENMP -DENABLE_SIMD
FULL_LDFLAGS := $(BASE_LDFLAGS) -fopenmp

# CUDA: Pure CUDA acceleration (no OpenMP/SIMD to avoid conflicts)
CUDA_CXXFLAGS := $(BASE_CXXFLAGS) -DENABLE_CUDA
CUDA_LDFLAGS := $(BASE_LDFLAGS) -L$(CUDA_PATH)/lib64 -lcudart
CUDA_INCLUDES := $(ALL_INCLUDES) -I$(CUDA_PATH)/include

# PROFILING: Full version with profiling enabled
PROFILING_CXXFLAGS := $(BASE_CXXFLAGS) -march=native -fopenmp -DENABLE_OPENMP -DENABLE_SIMD -DENABLE_PROFILING
PROFILING_LDFLAGS := $(BASE_LDFLAGS) -fopenmp

# ============================================================================
# Targets
# ============================================================================

.PHONY: all clean prepare naive openmp simd full cuda profiling help

# Default target: build full version
all: full

prepare:
	@mkdir -p build

# Build naive version (no parallelization)
naive: prepare
	$(CXX) $(NAIVE_CXXFLAGS) $(ALL_INCLUDES) -shared -o $(TARGET) \
	    $(SOURCES) \
	    -L$(FAISS_ROOT)/lib -lfaiss \
	    $(ALL_LIBS) \
	    $(NAIVE_LDFLAGS)
	@echo "✓ Built NAIVE version: $(TARGET)"

# Build OpenMP-only version
openmp: prepare
	$(CXX) $(OPENMP_CXXFLAGS) $(ALL_INCLUDES) -shared -o $(TARGET) \
	    $(SOURCES) \
	    -L$(FAISS_ROOT)/lib -lfaiss \
	    $(ALL_LIBS) \
	    $(OPENMP_LDFLAGS)
	@echo "✓ Built OPENMP version: $(TARGET)"

# Build SIMD-only version
simd: prepare
	$(CXX) $(SIMD_CXXFLAGS) $(ALL_INCLUDES) -shared -o $(TARGET) \
	    $(SOURCES) \
	    -L$(FAISS_ROOT)/lib -lfaiss \
	    $(ALL_LIBS) \
	    $(SIMD_LDFLAGS)
	@echo "✓ Built SIMD version: $(TARGET)"

# Build full version (OpenMP + SIMD)
full: prepare
	$(CXX) $(FULL_CXXFLAGS) $(ALL_INCLUDES) -shared -o $(TARGET) \
	    $(SOURCES) \
	    -L$(FAISS_ROOT)/lib -lfaiss \
	    $(ALL_LIBS) \
	    $(FULL_LDFLAGS)
	@echo "✓ Built FULL version: $(TARGET)"

# Build CUDA version (Pure CUDA, no OpenMP/SIMD)
cuda: prepare
	@echo "Building CUDA kernel..."
	@$(NVCC) $(NVCC_FLAGS) -c src/CudaUtils.cu -o build/CudaUtils.o \
	    $(PROJECT_INCLUDE) -I$(CUDA_PATH)/include
	@echo "Building C++ sources with CUDA support..."
	@$(CXX) $(CUDA_CXXFLAGS) $(CUDA_INCLUDES) -c src/IndexBase.cpp -o build/IndexBase.o
	@$(CXX) $(CUDA_CXXFLAGS) $(CUDA_INCLUDES) -c src/IVFFlatIndex.cpp -o build/IVFFlatIndex.o
	@$(CXX) $(CUDA_CXXFLAGS) $(CUDA_INCLUDES) -c src/KDTreeIndex.cpp -o build/KDTreeIndex.o
	@$(CXX) $(CUDA_CXXFLAGS) $(CUDA_INCLUDES) -c src/HNSWIndex.cpp -o build/HNSWIndex.o
	@$(CXX) $(CUDA_CXXFLAGS) $(CUDA_INCLUDES) -c python/zenann_pybind.cpp -o build/zenann_pybind.o
	@echo "Linking with CUDA runtime..."
	@$(CXX) -shared -o $(TARGET) \
	    build/IndexBase.o build/IVFFlatIndex.o build/KDTreeIndex.o build/HNSWIndex.o \
	    build/zenann_pybind.o build/CudaUtils.o \
	    -L$(FAISS_ROOT)/lib -lfaiss \
	    $(ALL_LIBS) \
	    $(CUDA_LDFLAGS)
	@echo "✓ Built CUDA version: $(TARGET)"
	@echo "Note: This version uses pure CUDA (no OpenMP/SIMD)"

# Build profiling version (Full with profiling enabled)
profiling: prepare
	$(CXX) $(PROFILING_CXXFLAGS) $(ALL_INCLUDES) -shared -o $(TARGET) \
	    $(SOURCES) \
	    -L$(FAISS_ROOT)/lib -lfaiss \
	    $(ALL_LIBS) \
	    $(PROFILING_LDFLAGS)
	@echo "✓ Built PROFILING version: $(TARGET)"

# Clean all builds
clean:
	rm -rf build

# Help message
help:
	@echo "ZenANN Build System - Multiple Optimization Versions"
	@echo ""
	@echo "Available targets:"
	@echo "  make naive   - Build naive version (no parallelization)"
	@echo "  make openmp  - Build OpenMP-only version"
	@echo "  make simd    - Build SIMD-only version (AVX2)"
	@echo "  make full       - Build fully optimized version (OpenMP + SIMD)"
	@echo "  make profiling  - Build profiling version (Full + detailed timing)"
	@echo "  make cuda       - Build CUDA version (Pure GPU acceleration)"
	@echo "  make all        - Build full version (default)"
	@echo "  make clean   - Remove all built files"
	@echo ""
	@echo "Note: All versions output to build/zenann.so"
	@echo "Each build will overwrite the previous one."
	@echo ""
	@echo "Usage:"
	@echo "  import build.zenann as zenann    # Always works regardless of version"
