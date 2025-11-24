#include "CudaUtils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <algorithm>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace zenann {
namespace cuda {

// ============================================================================
// Constants and Helper Structures
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int MAX_K = 128;  // Maximum k we support efficiently

/**
 * Pair structure for (distance, id) with device-side operators
 */
struct DistIdPair {
    float dist;
    int id;

    __device__ DistIdPair() : dist(INFINITY), id(-1) {}
    __device__ DistIdPair(float d, int i) : dist(d), id(i) {}

    __device__ bool operator<(const DistIdPair& other) const {
        return dist < other.dist || (dist == other.dist && id < other.id);
    }
};

// ============================================================================
// Kernel A: Compute queries × centroids distances
// ============================================================================

/**
 * Kernel A: Batch compute L2 distances from queries to centroids
 *
 * Grid: (num_queries, 1, 1)
 * Block: (256, 1, 1)
 *
 * Design (Section 3.1):
 * - One block per query
 * - Query loaded into shared memory (all threads share)
 * - Each thread computes distances to multiple centroids
 * - Uses L2 expansion: ||q||² + ||c||² - 2·q·c
 *
 * Memory access pattern: COALESCED
 * - Threads stride through centroids with step = blockDim.x
 */
__global__ void kernel_a_compute_coarse_distances(
    const float* __restrict__ queries,      // [Q x dim]
    const float* __restrict__ centroids,    // [nlist x dim]
    float* __restrict__ coarse_dists,       // [Q x nlist] OUTPUT
    int num_queries,
    int num_centroids,
    int dim
) {
    int q = blockIdx.x;
    if (q >= num_queries) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Shared memory for query vector
    extern __shared__ float shared_query[];

    // Step 1: Cooperatively load query into shared memory
    const float* query_ptr = queries + q * dim;
    for (int d = tid; d < dim; d += block_size) {
        shared_query[d] = query_ptr[d];
    }
    __syncthreads();

    // Step 2: Each thread computes distances to multiple centroids
    float* out_ptr = coarse_dists + q * num_centroids;

    for (int c = tid; c < num_centroids; c += block_size) {
        const float* centroid_ptr = centroids + c * dim;

        // Compute L2 distance: ||q - c||² = Σ(q[i] - c[i])²
        float sum = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < dim; ++d) {
            float diff = shared_query[d] - centroid_ptr[d];
            sum += diff * diff;
        }

        out_ptr[c] = sum;
    }
}

// ============================================================================
// Kernel B: Select top-nprobe centroids for each query
// ============================================================================

/**
 * Kernel B: Select top-nprobe nearest centroids using parallel selection
 *
 * Grid: (num_queries, 1, 1)
 * Block: (256, 1, 1)
 *
 * Design (Section 3.2):
 * - One block per query
 * - Load coarse distances to shared memory
 * - Parallel iterative min-finding (nprobe rounds)
 * - Each round: all threads cooperate to find minimum, then mark as selected
 *
 * Complexity: O(nprobe × (nlist/threads + log(threads)))
 */
__global__ void kernel_b_select_top_nprobe(
    const float* __restrict__ coarse_dists,  // [Q x nlist]
    int* __restrict__ selected_lists,        // [Q x nprobe] OUTPUT
    int num_queries,
    int num_centroids,
    int nprobe
) {
    int q = blockIdx.x;
    if (q >= num_queries) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Shared memory layout:
    // [distances: nlist floats] [reduction_vals: 256 floats] [reduction_ids: 256 ints]
    extern __shared__ char smem[];
    float* shared_dists = (float*)smem;
    float* reduction_vals = shared_dists + num_centroids;
    int* reduction_ids = (int*)(reduction_vals + block_size);

    // Load distances to shared memory
    const float* in_ptr = coarse_dists + q * num_centroids;
    for (int c = tid; c < num_centroids; c += block_size) {
        shared_dists[c] = in_ptr[c];
    }
    __syncthreads();

    int* out_ptr = selected_lists + q * nprobe;

    // Perform nprobe rounds of parallel min-finding
    for (int round = 0; round < nprobe; ++round) {
        // Phase 1: Each thread finds local minimum
        float local_min = INFINITY;
        int local_min_id = -1;

        for (int c = tid; c < num_centroids; c += block_size) {
            if (shared_dists[c] < local_min) {
                local_min = shared_dists[c];
                local_min_id = c;
            }
        }

        reduction_vals[tid] = local_min;
        reduction_ids[tid] = local_min_id;
        __syncthreads();

        // Phase 2: Parallel reduction to find global minimum
        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (reduction_vals[tid + stride] < reduction_vals[tid]) {
                    reduction_vals[tid] = reduction_vals[tid + stride];
                    reduction_ids[tid] = reduction_ids[tid + stride];
                }
            }
            __syncthreads();
        }

        // Phase 3: Thread 0 records result and marks as selected
        if (tid == 0) {
            int selected_id = reduction_ids[0];
            if (selected_id >= 0) {
                out_ptr[round] = selected_id;
                shared_dists[selected_id] = INFINITY;  // Mark as used
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Kernel C: Scan inverted lists with thread-local top-k
// ============================================================================

/**
 * Device function: Insert into thread-local top-k heap
 *
 * Maintains a small sorted array (size k) in registers
 * Uses insertion sort for small k (very efficient for k <= 32)
 */
__device__ inline void insert_to_local_topk(
    DistIdPair* local_topk,
    int& local_size,
    int max_k,
    float dist,
    int id
) {
    // If not full, just append and sort
    if (local_size < max_k) {
        local_topk[local_size] = DistIdPair(dist, id);
        local_size++;

        // Bubble up (insertion sort style)
        for (int i = local_size - 1; i > 0 && local_topk[i] < local_topk[i-1]; --i) {
            DistIdPair tmp = local_topk[i];
            local_topk[i] = local_topk[i-1];
            local_topk[i-1] = tmp;
        }
    }
    // If full and new element is better than worst
    else if (dist < local_topk[max_k - 1].dist) {
        // Insert in sorted position
        int insert_pos = max_k - 1;
        while (insert_pos > 0 && dist < local_topk[insert_pos - 1].dist) {
            local_topk[insert_pos] = local_topk[insert_pos - 1];
            insert_pos--;
        }
        local_topk[insert_pos] = DistIdPair(dist, id);
    }
}

/**
 * Device function: Merge thread-local top-k into block-level top-k
 *
 * All threads write their results to shared memory, then thread 0
 * performs final selection to get block's top-k
 */
__device__ void merge_block_topk(
    DistIdPair* local_topk,
    int local_size,
    DistIdPair* shared_candidates,
    DistIdPair* block_topk,
    int k,
    int tid,
    int block_size
) {
    // Write thread-local results to shared memory
    // Each thread gets 'k' slots (not MAX_K) to save shared memory
    int base = tid * k;
    int write_count = min(local_size, k);

    for (int i = 0; i < write_count; ++i) {
        shared_candidates[base + i] = local_topk[i];
    }
    for (int i = write_count; i < k; ++i) {
        shared_candidates[base + i] = DistIdPair();  // INFINITY
    }
    __syncthreads();

    // Thread 0 performs merge
    if (tid == 0) {
        int total_candidates = block_size * k;

        // Simple k-pass selection (good for small k)
        for (int ki = 0; ki < k; ++ki) {
            DistIdPair best = DistIdPair();
            int best_idx = -1;

            for (int i = 0; i < total_candidates; ++i) {
                if (shared_candidates[i] < best) {
                    best = shared_candidates[i];
                    best_idx = i;
                }
            }

            if (best_idx >= 0) {
                block_topk[ki] = best;
                shared_candidates[best_idx] = DistIdPair();  // Mark as used
            } else {
                block_topk[ki] = DistIdPair();
            }
        }
    }
    __syncthreads();
}

/**
 * Kernel C: Scan inverted lists and compute partial top-k
 *
 * Grid: (num_queries, nprobe, 1) - 2D GRID!
 * Block: (128, 1, 1)
 *
 * Design (Section 3.3):
 * - Each block handles one (query, list) pair
 * - Query loaded into shared memory
 * - Each thread maintains thread-local top-k in registers
 * - Threads stride through list vectors
 * - Block-level merge at end
 * - Output: partial top-k for this (query, list)
 *
 * This is the CORE of the search!
 */
__global__ void kernel_c_scan_lists(
    const float* __restrict__ queries,        // [Q x dim]
    const int* __restrict__ selected_lists,   // [Q x nprobe]
    const float* __restrict__ vectors,        // [N_total x dim]
    const int* __restrict__ list_offsets,     // [nlist + 1]
    const size_t* __restrict__ ids,           // [N_total] - IMPORTANT: size_t, not int!
    DistIdPair* __restrict__ partial_topk,    // [Q x nprobe x k] OUTPUT
    int num_queries,
    int nprobe,
    int k,
    int dim
) {
    int q = blockIdx.x;
    int probe = blockIdx.y;

    if (q >= num_queries || probe >= nprobe) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Get actual list ID for this (query, probe) pair
    int list_id = selected_lists[q * nprobe + probe];
    int list_start = list_offsets[list_id];
    int list_end = list_offsets[list_id + 1];
    int list_size = list_end - list_start;

    // Shared memory layout:
    // [query: dim floats] [candidates: block_size * MAX_K pairs]
    extern __shared__ char smem[];
    float* shared_query = (float*)smem;
    DistIdPair* shared_candidates = (DistIdPair*)(shared_query + dim);

    // Load query into shared memory
    const float* query_ptr = queries + q * dim;
    for (int d = tid; d < dim; d += block_size) {
        shared_query[d] = query_ptr[d];
    }
    __syncthreads();

    // Thread-local top-k (in registers!)
    DistIdPair local_topk[MAX_K];
    int local_size = 0;
    int max_local_k = min(k, MAX_K);

    // Scan list vectors with stride
    for (int idx = list_start + tid; idx < list_end; idx += block_size) {
        // Get the actual vector ID from the inverted list (size_t!)
        size_t vec_id = ids[idx];

        // Access the vector using its ID
        const float* vec_ptr = vectors + vec_id * dim;

        // Compute L2 distance
        float sum = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < dim; ++d) {
            float diff = shared_query[d] - vec_ptr[d];
            sum += diff * diff;
        }

        // Insert into thread-local top-k
        insert_to_local_topk(local_topk, local_size, max_local_k, sum, vec_id);
    }
    __syncthreads();

    // Merge all thread-local top-k into block-level top-k
    DistIdPair* block_result = (DistIdPair*)smem;  // Reuse shared memory
    merge_block_topk(local_topk, local_size, shared_candidates, block_result, k, tid, block_size);

    // Write block result to global memory
    if (tid == 0) {
        DistIdPair* out_ptr = partial_topk + (q * nprobe + probe) * k;
        for (int i = 0; i < k; ++i) {
            out_ptr[i] = block_result[i];
        }
    }
}

// ============================================================================
// Kernel D: Merge partial top-k results
// ============================================================================

/**
 * Kernel D: Merge nprobe partial top-k into final top-k
 *
 * Grid: (num_queries, 1, 1)
 * Block: (256, 1, 1)
 *
 * Design (Section 3.4):
 * - One block per query
 * - Read nprobe × k candidates from global memory
 * - Perform final top-k selection
 * - Write to output
 */
__global__ void kernel_d_merge_final_topk(
    const DistIdPair* __restrict__ partial_topk,  // [Q x nprobe x k]
    float* __restrict__ out_distances,            // [Q x k] OUTPUT
    int* __restrict__ out_indices,                // [Q x k] OUTPUT
    int num_queries,
    int nprobe,
    int k
) {
    int q = blockIdx.x;
    if (q >= num_queries) return;

    int tid = threadIdx.x;

    int total_candidates = nprobe * k;

    // Shared memory for candidates
    extern __shared__ DistIdPair smem_candidates[];

    // Load all partial results to shared memory
    const DistIdPair* in_ptr = partial_topk + q * nprobe * k;
    for (int i = tid; i < total_candidates; i += blockDim.x) {
        smem_candidates[i] = in_ptr[i];
    }
    __syncthreads();

    // Thread 0 performs final selection
    if (tid == 0) {
        // Simple k-pass selection
        for (int ki = 0; ki < k; ++ki) {
            DistIdPair best = DistIdPair();
            int best_idx = -1;

            for (int i = 0; i < total_candidates; ++i) {
                if (smem_candidates[i] < best) {
                    best = smem_candidates[i];
                    best_idx = i;
                }
            }

            if (best_idx >= 0 && best.id >= 0) {
                out_distances[q * k + ki] = best.dist;
                out_indices[q * k + ki] = best.id;
                smem_candidates[best_idx] = DistIdPair();  // Mark as used
            } else {
                out_distances[q * k + ki] = INFINITY;
                out_indices[q * k + ki] = -1;
            }
        }
    }
}

// ============================================================================
// Memory Managers (keep existing implementations)
// ============================================================================

GpuCentroidsManager::GpuCentroidsManager()
    : d_centroids_(nullptr), num_centroids_(0), dim_(0) {}

GpuCentroidsManager::~GpuCentroidsManager() {
    free();
}

void GpuCentroidsManager::upload_centroids(
    const float* centroids_flat,
    size_t num_centroids,
    size_t dim
) {
    free();
    num_centroids_ = num_centroids;
    dim_ = dim;

    size_t size_bytes = num_centroids * dim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_centroids_, size_bytes));
    CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_flat, size_bytes, cudaMemcpyHostToDevice));

    std::cout << "[CUDA] Uploaded " << num_centroids << " centroids ("
              << (size_bytes / 1024.0 / 1024.0) << " MB) to GPU" << std::endl;
}

void GpuCentroidsManager::free() {
    if (d_centroids_) {
        CUDA_CHECK(cudaFree(d_centroids_));
        d_centroids_ = nullptr;
    }
}

GpuDataManager::GpuDataManager()
    : d_data_(nullptr), num_vectors_(0), dim_(0) {}

GpuDataManager::~GpuDataManager() {
    free();
}

void GpuDataManager::upload_data(
    const float* data_flat,
    size_t num_vectors,
    size_t dim
) {
    free();
    num_vectors_ = num_vectors;
    dim_ = dim;

    size_t size_bytes = num_vectors * dim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_data_, size_bytes));
    CUDA_CHECK(cudaMemcpy(d_data_, data_flat, size_bytes, cudaMemcpyHostToDevice));

    std::cout << "[CUDA] Uploaded " << num_vectors << " base vectors ("
              << (size_bytes / 1024.0 / 1024.0) << " MB) to GPU" << std::endl;
}

void GpuDataManager::free() {
    if (d_data_) {
        CUDA_CHECK(cudaFree(d_data_));
        d_data_ = nullptr;
    }
}

GpuInvertedListsManager::GpuInvertedListsManager()
    : d_list_data_(nullptr), d_list_offsets_(nullptr), d_list_sizes_(nullptr),
      nlist_(0), total_size_(0) {}

GpuInvertedListsManager::~GpuInvertedListsManager() {
    free();
}

void GpuInvertedListsManager::upload_lists(
    const std::vector<std::vector<size_t>>& lists,
    size_t nlist
) {
    free();
    nlist_ = nlist;

    // Build offsets array (prefix sum)
    std::vector<int> offsets(nlist + 1);
    offsets[0] = 0;
    for (size_t i = 0; i < nlist; ++i) {
        offsets[i + 1] = offsets[i] + lists[i].size();
    }
    total_size_ = offsets[nlist];

    // Flatten list data and IDs
    std::vector<size_t> flattened_ids(total_size_);
    size_t pos = 0;
    for (size_t i = 0; i < nlist; ++i) {
        for (size_t id : lists[i]) {
            flattened_ids[pos++] = id;
        }
    }

    // Upload to GPU
    CUDA_CHECK(cudaMalloc(&d_list_data_, total_size_ * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_list_offsets_, (nlist + 1) * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_list_data_, flattened_ids.data(),
                          total_size_ * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_list_offsets_, offsets.data(),
                          (nlist + 1) * sizeof(int), cudaMemcpyHostToDevice));

    std::cout << "[CUDA] Uploaded inverted lists: " << nlist << " lists, "
              << total_size_ << " total entries" << std::endl;
}

void GpuInvertedListsManager::free() {
    if (d_list_data_) {
        CUDA_CHECK(cudaFree(d_list_data_));
        d_list_data_ = nullptr;
    }
    if (d_list_offsets_) {
        CUDA_CHECK(cudaFree(d_list_offsets_));
        d_list_offsets_ = nullptr;
    }
    if (d_list_sizes_) {
        CUDA_CHECK(cudaFree(d_list_sizes_));
        d_list_sizes_ = nullptr;
    }
}

// ============================================================================
// Complete GPU Search Pipeline (4-Kernel Architecture from cuda.md)
// ============================================================================

/**
 * Complete 4-kernel GPU search pipeline following cuda.md design
 *
 * Pipeline:
 *   Kernel A: queries × centroids distances
 *   Kernel B: select top-nprobe per query
 *   Kernel C: scan lists with 2D grid (Q, nprobe)
 *   Kernel D: merge partial top-k
 */
void batch_search_gpu_pipeline_v2(
    const float* queries,
    const float* d_centroids,
    const float* d_base_data,
    const size_t* d_list_data,
    const int* d_list_offsets,
    float* result_distances,
    size_t* result_indices,
    size_t num_queries,
    size_t num_centroids,
    size_t nprobe,
    size_t k,
    size_t dim
) {
    // Allocate device memory
    float *d_queries, *d_coarse_dists;
    int *d_selected_lists, *d_out_indices;
    float *d_out_distances;
    DistIdPair *d_partial_topk;

    size_t queries_size = num_queries * dim * sizeof(float);
    size_t coarse_dists_size = num_queries * num_centroids * sizeof(float);
    size_t selected_lists_size = num_queries * nprobe * sizeof(int);
    size_t partial_topk_size = num_queries * nprobe * k * sizeof(DistIdPair);
    size_t output_size = num_queries * k * sizeof(float);
    size_t output_indices_size = num_queries * k * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_queries, queries_size));
    CUDA_CHECK(cudaMalloc(&d_coarse_dists, coarse_dists_size));
    CUDA_CHECK(cudaMalloc(&d_selected_lists, selected_lists_size));
    CUDA_CHECK(cudaMalloc(&d_partial_topk, partial_topk_size));
    CUDA_CHECK(cudaMalloc(&d_out_distances, output_size));
    CUDA_CHECK(cudaMalloc(&d_out_indices, output_indices_size));

    // Copy queries to GPU
    CUDA_CHECK(cudaMemcpy(d_queries, queries, queries_size, cudaMemcpyHostToDevice));

    // ========================================================================
    // Kernel A: Compute queries × centroids distances
    // ========================================================================
    {
        int grid_size = num_queries;
        int block_size = 256;
        size_t smem_size = dim * sizeof(float);

        kernel_a_compute_coarse_distances<<<grid_size, block_size, smem_size>>>(
            d_queries, d_centroids, d_coarse_dists,
            num_queries, num_centroids, dim
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // ========================================================================
    // Kernel B: Select top-nprobe centroids
    // ========================================================================
    {
        int grid_size = num_queries;
        int block_size = 256;
        size_t smem_size = num_centroids * sizeof(float) +
                          block_size * (sizeof(float) + sizeof(int));

        kernel_b_select_top_nprobe<<<grid_size, block_size, smem_size>>>(
            d_coarse_dists, d_selected_lists,
            num_queries, num_centroids, nprobe
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // ========================================================================
    // Kernel C: Scan inverted lists (2D grid!)
    // ========================================================================
    {
        dim3 grid_size(num_queries, nprobe);  // 2D GRID!
        int block_size = 128;

        // Shared memory: only for query vector + small merge buffer
        // We use thread-local top-k in registers (MAX_K per thread)
        // For merge, we only need k * block_size space, not MAX_K * block_size
        size_t smem_size = dim * sizeof(float) +        // query vector
                          k * block_size * sizeof(DistIdPair);  // merge buffer

        // Check shared memory limit (typically 48KB)
        const size_t MAX_SMEM = 48 * 1024;
        if (smem_size > MAX_SMEM) {
            // Reduce block_size to fit in shared memory
            int max_block_size = (MAX_SMEM - dim * sizeof(float)) / (k * sizeof(DistIdPair));
            block_size = std::max(32, std::min(block_size, max_block_size));
            smem_size = dim * sizeof(float) + k * block_size * sizeof(DistIdPair);
        }

        kernel_c_scan_lists<<<grid_size, block_size, smem_size>>>(
            d_queries, d_selected_lists, d_base_data, d_list_offsets,
            d_list_data,  // size_t* - no cast needed!
            d_partial_topk,
            num_queries, nprobe, k, dim
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // ========================================================================
    // Kernel D: Merge partial top-k
    // ========================================================================
    {
        int grid_size = num_queries;
        int block_size = 256;
        size_t smem_size = nprobe * k * sizeof(DistIdPair);

        // Check shared memory limit
        const size_t MAX_SMEM = 48 * 1024;
        if (smem_size > MAX_SMEM) {
            // Reduce block_size if needed (though it doesn't affect smem here)
            block_size = 128;
        }

        kernel_d_merge_final_topk<<<grid_size, block_size, smem_size>>>(
            d_partial_topk, d_out_distances, d_out_indices,
            num_queries, nprobe, k
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Synchronize and copy results
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(result_distances, d_out_distances, output_size, cudaMemcpyDeviceToHost));

    // Convert int* to size_t*
    int* temp_indices = new int[num_queries * k];
    CUDA_CHECK(cudaMemcpy(temp_indices, d_out_indices, output_indices_size, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < num_queries * k; ++i) {
        result_indices[i] = (temp_indices[i] >= 0) ? temp_indices[i] : 0;
    }
    delete[] temp_indices;

    // Cleanup
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_coarse_dists));
    CUDA_CHECK(cudaFree(d_selected_lists));
    CUDA_CHECK(cudaFree(d_partial_topk));
    CUDA_CHECK(cudaFree(d_out_distances));
    CUDA_CHECK(cudaFree(d_out_indices));
}

// ============================================================================
// Utility Functions
// ============================================================================

bool is_cuda_available() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

void initialize_cuda() {
    if (!is_cuda_available()) {
        std::cerr << "Warning: No CUDA device available" << std::endl;
        return;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "[CUDA] Device: " << prop.name << std::endl;
    std::cout << "[CUDA] Compute: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "[CUDA] Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));  // Warm up
}

// ============================================================================
// Legacy API Compatibility Layer
// ============================================================================

/**
 * Legacy API: batch_compute_centroid_distances
 * Redirects to Kernel A from the new pipeline
 */
void batch_compute_centroid_distances(
    const float* queries,
    const float* d_centroids,
    float* distances,
    size_t num_queries,
    size_t num_centroids,
    size_t dim
) {
    // Allocate device memory
    float *d_queries, *d_distances;

    size_t queries_size = num_queries * dim * sizeof(float);
    size_t distances_size = num_queries * num_centroids * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_queries, queries_size));
    CUDA_CHECK(cudaMalloc(&d_distances, distances_size));

    // Copy queries to device
    CUDA_CHECK(cudaMemcpy(d_queries, queries, queries_size, cudaMemcpyHostToDevice));

    // Launch Kernel A
    int threads_per_block = 256;
    int num_blocks = num_queries;

    kernel_a_compute_coarse_distances<<<num_blocks, threads_per_block>>>(
        d_queries,
        d_centroids,
        d_distances,
        num_queries,
        num_centroids,
        dim
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(distances, d_distances, distances_size, cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_distances));
}

/**
 * Legacy API: batch_search_gpu_pipeline (old version)
 * Redirects to batch_search_gpu_pipeline_v2
 */
void batch_search_gpu_pipeline(
    const float* queries,
    const float* d_centroids,
    const float* d_base_data,
    const size_t* d_list_data,
    const int* d_list_offsets,
    const int* d_list_sizes,  // Note: v2 doesn't use this parameter
    float* result_distances,
    size_t* result_indices,
    size_t num_queries,
    size_t num_centroids,
    size_t nprobe,
    size_t k,
    size_t dim
) {
    // Simply redirect to the new v2 implementation
    batch_search_gpu_pipeline_v2(
        queries,
        d_centroids,
        d_base_data,
        d_list_data,
        d_list_offsets,
        result_distances,
        result_indices,
        num_queries,
        num_centroids,
        nprobe,
        k,
        dim
    );
}

/**
 * Legacy API: batch_scan_selected_lists (stub implementation)
 * This was for CPU coarse + GPU fine, not used in optimized path
 */
void batch_scan_selected_lists(
    const float* queries,
    const size_t* selected_lists,
    const float* d_base_data,
    const size_t* d_list_data,
    const int* d_list_offsets,
    const int* d_list_sizes,
    float* result_distances,
    size_t* result_indices,
    size_t num_queries,
    size_t nprobe,
    size_t k,
    size_t dim
) {
    // Stub: not implemented as it's not used in the optimized pipeline
    std::cerr << "Warning: batch_scan_selected_lists is deprecated. Use batch_search_gpu_pipeline_v2." << std::endl;
}

/**
 * Legacy API: batch_search_with_lists (stub implementation)
 * This was an old experimental API, not used
 */
void batch_search_with_lists(
    const float* queries,
    const float* d_centroids,
    const float* d_base_data,
    const size_t* d_list_data,
    const int* d_list_offsets,
    const int* d_list_sizes,
    float* result_distances,
    size_t* result_indices,
    size_t num_queries,
    size_t num_centroids,
    size_t nprobe,
    size_t k,
    size_t dim
) {
    // Stub: not implemented as it's not used
    std::cerr << "Warning: batch_search_with_lists is deprecated. Use batch_search_gpu_pipeline_v2." << std::endl;
}

} // namespace cuda
} // namespace zenann
