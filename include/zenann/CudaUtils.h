#pragma once

#include <cstddef>
#include <vector>

namespace zenann {
namespace cuda {

/**
 * GPU Memory Manager for persistent centroid data
 */
class GpuCentroidsManager {
public:
    GpuCentroidsManager();
    ~GpuCentroidsManager();

    /**
     * Upload centroids to GPU memory (call once after training)
     * @param centroids_flat Flattened centroid vectors [num_centroids * dim]
     * @param num_centroids  Number of centroids
     * @param dim           Vector dimension
     */
    void upload_centroids(const float* centroids_flat, size_t num_centroids, size_t dim);

    /**
     * Get device pointer to centroids
     */
    const float* get_device_ptr() const { return d_centroids_; }

    /**
     * Check if centroids are uploaded
     */
    bool is_ready() const { return d_centroids_ != nullptr; }

    /**
     * Free GPU memory
     */
    void free();

private:
    float* d_centroids_;
    size_t num_centroids_;
    size_t dim_;
};

/**
 * GPU Memory Manager for base vectors (all data points)
 */
class GpuDataManager {
public:
    GpuDataManager();
    ~GpuDataManager();
    void upload_data(const float* data_flat, size_t num_vectors, size_t dim);
    const float* get_device_ptr() const { return d_data_; }
    bool is_ready() const { return d_data_ != nullptr; }
    void free();
private:
    float* d_data_;
    size_t num_vectors_;
    size_t dim_;
};

/**
 * GPU Memory Manager for inverted lists structure
 */
class GpuInvertedListsManager {
public:
    GpuInvertedListsManager();
    ~GpuInvertedListsManager();
    void upload_lists(const std::vector<std::vector<size_t>>& lists, size_t nlist);
    const size_t* get_list_data_ptr() const { return d_list_data_; }
    const int* get_list_offsets_ptr() const { return d_list_offsets_; }
    const int* get_list_sizes_ptr() const { return d_list_sizes_; }
    bool is_ready() const { return d_list_data_ != nullptr; }
    void free();
private:
    size_t* d_list_data_;      // Flattened list IDs
    int* d_list_offsets_;      // Starting offset for each list
    int* d_list_sizes_;        // Size of each list
    size_t nlist_;
    size_t total_size_;
};

/**
 * Batch compute L2 distances from multiple queries to centroids on GPU
 *
 * This is the main optimized function that processes multiple queries in parallel.
 *
 * @param queries          Query vectors (host, [num_queries x dim])
 * @param d_centroids      Centroid vectors (device, [num_centroids x dim])
 * @param distances        Output distances (host, [num_queries x num_centroids])
 * @param num_queries      Number of query vectors
 * @param num_centroids    Number of centroid vectors
 * @param dim              Vector dimension
 */
void batch_compute_centroid_distances(
    const float* queries,
    const float* d_centroids,
    float* distances,
    size_t num_queries,
    size_t num_centroids,
    size_t dim
);

/**
 * Complete GPU pipeline for IVF search (OPTIMIZED for 100K+ QPS)
 *
 * All computation on GPU: centroid distances → select lists → scan lists
 * Minimal CPU-GPU data transfer for maximum throughput.
 *
 * Pipeline:
 *   1. GPU: Compute query-centroid distances
 *   2. GPU: Select top-nprobe nearest centroids
 *   3. GPU: Scan selected inverted lists
 *
 * @param queries          Query vectors (host, [num_queries x dim])
 * @param d_centroids      Centroid vectors (device, [num_centroids x dim])
 * @param d_base_data      All base vectors (device, [num_base x dim])
 * @param d_list_data      Flattened list IDs (device)
 * @param d_list_offsets   List starting offsets (device)
 * @param d_list_sizes     List sizes (device)
 * @param result_distances Output distances (host, [num_queries x k])
 * @param result_indices   Output indices (host, [num_queries x k])
 * @param num_queries      Number of queries
 * @param num_centroids    Number of centroids
 * @param nprobe           Number of lists to probe per query
 * @param k                Number of nearest neighbors
 * @param dim              Vector dimension
 */
void batch_search_gpu_pipeline(
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
);

/**
 * Complete GPU pipeline V2 for IVF search (CLEAN IMPLEMENTATION)
 *
 * 4-Kernel Architecture following cuda.md specification:
 *   Kernel A: Compute queries × centroids distances
 *   Kernel B: Select top-nprobe nearest centroids per query
 *   Kernel C: Scan inverted lists with 2D grid (query, probe)
 *   Kernel D: Merge partial top-k results
 *
 * @param queries          Query vectors (host, [num_queries x dim])
 * @param d_centroids      Centroid vectors (device, [num_centroids x dim])
 * @param d_base_data      All base vectors (device, [num_base x dim])
 * @param d_list_data      Flattened list IDs (device)
 * @param d_list_offsets   List starting offsets (device)
 * @param result_distances Output distances (host, [num_queries x k])
 * @param result_indices   Output indices (host, [num_queries x k])
 * @param num_queries      Number of queries
 * @param num_centroids    Number of centroids
 * @param nprobe           Number of lists to probe per query
 * @param k                Number of nearest neighbors
 * @param dim              Vector dimension
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
);

/**
 * Scan selected lists on GPU (optimized version without shared memory issues)
 *
 * This version assumes CPU has already computed centroid distances and selected
 * which lists to probe. GPU only does the heavy lifting: scanning list contents.
 *
 * @param queries          Query vectors (host, [num_queries x dim])
 * @param selected_lists   List IDs to scan for each query (host, [num_queries x nprobe])
 * @param d_base_data      All base vectors (device, [num_base x dim])
 * @param d_list_data      Flattened list IDs (device)
 * @param d_list_offsets   List starting offsets (device)
 * @param d_list_sizes     List sizes (device)
 * @param result_distances Output distances (host, [num_queries x k])
 * @param result_indices   Output indices (host, [num_queries x k])
 * @param num_queries      Number of queries
 * @param nprobe           Number of lists to probe per query
 * @param k                Number of nearest neighbors
 * @param dim              Vector dimension
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
);

/**
 * Complete GPU search with list scanning (coarse + fine-grained search)
 *
 * NOTE: This function has shared memory limitations and may fail for large nlist.
 * Consider using batch_scan_selected_lists instead.
 *
 * @param queries          Query vectors (host, [num_queries x dim])
 * @param d_centroids      Centroid vectors (device, [num_centroids x dim])
 * @param d_base_data      All base vectors (device, [num_base x dim])
 * @param d_list_data      Flattened list IDs (device)
 * @param d_list_offsets   List starting offsets (device)
 * @param d_list_sizes     List sizes (device)
 * @param result_distances Output distances (host, [num_queries x k])
 * @param result_indices   Output indices (host, [num_queries x k])
 * @param num_queries      Number of queries
 * @param num_centroids    Number of centroids
 * @param nprobe           Number of lists to probe
 * @param k                Number of nearest neighbors
 * @param dim              Vector dimension
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
);

/**
 * Check if CUDA is available
 */
bool is_cuda_available();

/**
 * Initialize CUDA (optional)
 */
void initialize_cuda();

} // namespace cuda
} // namespace zenann
