#include "IVFFlatIndex.h"
#include "SimdUtils.h"
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif
#ifdef ENABLE_CUDA
#include "CudaUtils.h"
#endif
#include <limits>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>

#ifdef ENABLE_PROFILING
#define PROFILE_START(name) auto name##_start = std::chrono::high_resolution_clock::now();
#define PROFILE_END(name, var) var = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - name##_start).count();
#else
#define PROFILE_START(name)
#define PROFILE_END(name, var)
#endif

namespace zenann {

IVFFlatIndex::IVFFlatIndex(size_t dim, size_t nlist, size_t nprobe)
    : IndexBase(dim), nlist_(nlist), nprobe_(nprobe) {}

IVFFlatIndex::~IVFFlatIndex() = default;

void IVFFlatIndex::train() {
    const auto& data = datastore_->getAll();
    if (data.empty()) return;

    // Calculate centroids with K-means algorithm
    kmeans(data);

    // Construct inverted list structure
    lists_.assign(nlist_, idList());
    for (size_t id = 0; id < data.size(); ++id) {
        const auto& v = data[id];
        float best_dist = std::numeric_limits<float>::max();
        size_t best_c = 0;
        for (size_t c = 0; c < nlist_; ++c) {
            float d = 0.0f;
            for (size_t k = 0; k < dimension_; ++k) {
                float diff = v[k] - centroids_[c][k];
                d += diff * diff;
            }
            if (d < best_dist) {
                best_dist = d;
                best_c = c;
            }
        }
        lists_[best_c].push_back(id);
    }

#ifdef ENABLE_CUDA
    // Upload centroids to GPU
    centroids_flat_.resize(nlist_ * dimension_);
    for (size_t c = 0; c < nlist_; ++c) {
        for (size_t d = 0; d < dimension_; ++d) {
            centroids_flat_[c * dimension_ + d] = centroids_[c][d];
        }
    }
    gpu_centroids_manager_.upload_centroids(centroids_flat_.data(), nlist_, dimension_);

    // Upload base data to GPU
    std::vector<float> data_flat(data.size() * dimension_);
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t d = 0; d < dimension_; ++d) {
            data_flat[i * dimension_ + d] = data[i][d];
        }
    }
    gpu_data_manager_.upload_data(data_flat.data(), data.size(), dimension_);

    // Upload inverted lists to GPU
    gpu_lists_manager_.upload_lists(lists_, nlist_);
#endif
}

SearchResult IVFFlatIndex::search(const Vector& query, size_t k) const {
    using Pair = std::pair<float, size_t>;
    std::vector<Pair> cdist(nlist_);
    std::vector<Pair> heap;

    // Calculate distance from query to all centroids
    PROFILE_START(centroid_distance)

#ifdef ENABLE_CUDA
    // GPU version: Use batch API even for single query
    std::vector<float> distances(nlist_);

    if (gpu_centroids_manager_.is_ready()) {
        cuda::batch_compute_centroid_distances(
            query.data(),
            gpu_centroids_manager_.get_device_ptr(),
            distances.data(),
            1,  // num_queries = 1
            nlist_,
            dimension_
        );

        // Store results with indices
        for (size_t c = 0; c < nlist_; ++c) {
            cdist[c] = {distances[c], c};
        }
    } else {
        // Fallback to CPU if GPU not initialized
        for (size_t c = 0; c < nlist_; ++c) {
            float d = l2_distance(query.data(), centroids_[c].data(), dimension_);
            cdist[c] = {d, c};
        }
    }
#else
    // CPU version: Compute distances with optional OpenMP
    #ifdef ENABLE_OPENMP
        #pragma omp parallel for schedule(static)
    #endif
    for (size_t c = 0; c < nlist_; ++c) {
        float d = l2_distance(query.data(), centroids_[c].data(), dimension_);
        cdist[c] = {d, c};
    }
#endif

    double time_centroid_distance = 0;
    PROFILE_END(centroid_distance, time_centroid_distance)

    // Select top-nprobe nearest centroids
    PROFILE_START(centroid_selection)
    std::partial_sort(cdist.begin(), cdist.begin() + nprobe_, cdist.end(),
        [](auto& a, auto& b) {
            return a.first < b.first;
        }
    );
    double time_centroid_selection = 0;
    PROFILE_END(centroid_selection, time_centroid_selection)

    // Probe nprobe nearest lists
    heap.reserve(k);
    const auto& data = datastore_->getAll();

    PROFILE_START(list_scanning)

#ifdef ENABLE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t pi = 0; pi < nprobe_; ++pi) {
        size_t c = cdist[pi].second;

#ifdef ENABLE_OPENMP
        // Thread-local heap for this cluster
        std::vector<Pair> local;
        local.reserve(k);
#endif

        // Search within this cluster's inverted list
        for (size_t id : lists_[c]) {
            float dist = l2_distance(query.data(), data[id].data(), dimension_);

#ifdef ENABLE_OPENMP
            if (local.size() < k) {
                local.emplace_back(dist, id);
                if (local.size() == k) {
                    std::make_heap(local.begin(), local.end());
                }
            } else if (dist < local.front().first) {
                std::pop_heap(local.begin(), local.end());
                local.back() = {dist, id};
                std::push_heap(local.begin(), local.end());
            }
#else
            if (heap.size() < k) {
                heap.emplace_back(dist, id);
                if (heap.size() == k) {
                    std::make_heap(heap.begin(), heap.end());
                }
            } else if (dist < heap.front().first) {
                std::pop_heap(heap.begin(), heap.end());
                heap.back() = {dist, id};
                std::push_heap(heap.begin(), heap.end());
            }
#endif
        }

#ifdef ENABLE_OPENMP
        // Merge local results into global heap (thread-safe)
        #pragma omp critical
        {
            for (auto& p : local) {
                if (heap.size() < k) {
                    heap.push_back(p);
                    if (heap.size() == k) {
                        std::make_heap(heap.begin(), heap.end());
                    }
                } else if (p.first < heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = p;
                    std::push_heap(heap.begin(), heap.end());
                }
            }
        }
#endif
    }
    double time_list_scanning = 0;
    PROFILE_END(list_scanning, time_list_scanning)

    PROFILE_START(final_sorting)
    std::sort(heap.begin(), heap.end(),
              [](const Pair& a, const Pair& b) { return a.first < b.first; });
    double time_final_sorting = 0;
    PROFILE_END(final_sorting, time_final_sorting)

#ifdef ENABLE_PROFILING
    // Output profiling data to stderr
    std::cerr << "PROFILE: centroid_dist=" << time_centroid_distance
              << "ms, centroid_select=" << time_centroid_selection
              << "ms, list_scan=" << time_list_scanning
              << "ms, final_sort=" << time_final_sorting
              << "ms, total=" << (time_centroid_distance + time_centroid_selection + time_list_scanning + time_final_sorting)
              << "ms" << std::endl;
#endif

    SearchResult res;
    res.distances.resize(heap.size());
    res.indices.resize(heap.size());
    for (size_t i = 0; i < heap.size(); ++i) {
        res.distances[i] = heap[i].first;
        res.indices[i]   = heap[i].second;
    }
    return res;
}

SearchResult IVFFlatIndex::search(const Vector& query, size_t k, size_t nprobe) const {
    size_t old = nprobe_;
    const_cast<IVFFlatIndex*>(this)->nprobe_ = nprobe;
    SearchResult res = search(query, k);
    const_cast<IVFFlatIndex*>(this)->nprobe_ = old;
    return res;
}

std::vector<SearchResult> IVFFlatIndex::search_batch(const Dataset& queries, size_t k) const {
    const size_t nq = queries.size();
    std::vector<SearchResult> results(nq);

#ifdef ENABLE_CUDA
    if (gpu_centroids_manager_.is_ready() &&
        gpu_data_manager_.is_ready() &&
        gpu_lists_manager_.is_ready() &&
        nq > 0) {
        // OPTIMIZED: Complete GPU pipeline for 100K+ QPS
        // All computation on GPU: centroid distances → select lists → scan lists

        // Flatten queries
        std::vector<float> queries_flat(nq * dimension_);
        for (size_t i = 0; i < nq; ++i) {
            for (size_t d = 0; d < dimension_; ++d) {
                queries_flat[i * dimension_ + d] = queries[i][d];
            }
        }

        // Allocate output buffers
        std::vector<float> result_distances(nq * k);
        std::vector<size_t> result_indices(nq * k);

        // Call complete GPU pipeline V2 (4-kernel architecture from cuda.md)
        cuda::batch_search_gpu_pipeline_v2(
            queries_flat.data(),
            gpu_centroids_manager_.get_device_ptr(),
            gpu_data_manager_.get_device_ptr(),
            gpu_lists_manager_.get_list_data_ptr(),
            gpu_lists_manager_.get_list_offsets_ptr(),
            result_distances.data(),
            result_indices.data(),
            nq,
            nlist_,
            nprobe_,
            k,
            dimension_
        );

        // Convert results to SearchResult format
        for (size_t q = 0; q < nq; ++q) {
            results[q].distances.resize(k);
            results[q].indices.resize(k);

            size_t valid_count = 0;
            for (size_t i = 0; i < k; ++i) {
                float dist = result_distances[q * k + i];
                if (dist < INFINITY) {
                    results[q].distances[valid_count] = dist;
                    results[q].indices[valid_count] = result_indices[q * k + i];
                    valid_count++;
                }
            }

            // Resize to actual valid count
            results[q].distances.resize(valid_count);
            results[q].indices.resize(valid_count);
        }
    } else {
        // Fallback: use individual search calls
        #ifdef ENABLE_OPENMP
            #pragma omp parallel for schedule(dynamic)
        #endif
        for (size_t i = 0; i < nq; ++i) {
            results[i] = search(queries[i], k);
        }
    }
#else
    // CPU version: use individual search calls with optional parallelization
    #ifdef ENABLE_OPENMP
        #pragma omp parallel for schedule(dynamic)
    #endif
    for (size_t i = 0; i < nq; ++i) {
        results[i] = search(queries[i], k);
    }
#endif

    return results;
}

std::vector<SearchResult> IVFFlatIndex::search_batch(const Dataset& queries, size_t k, size_t nprobe) const {
    size_t old = nprobe_;
    const_cast<IVFFlatIndex*>(this)->nprobe_ = nprobe;
    auto res = search_batch(queries, k);
    const_cast<IVFFlatIndex*>(this)->nprobe_ = old;
    return res;
}

void IVFFlatIndex::kmeans(const Dataset& data, size_t iterations) {
    size_t n = data.size();
    std::mt19937 rng(123);
    std::uniform_int_distribution<size_t> dist(0, n-1);

    // Initialized with random centroids
    centroids_.clear();
    centroids_.reserve(nlist_);
    for (size_t i = 0; i < nlist_; ++i) {
        centroids_.push_back(data[dist(rng)]);
    }

    std::vector<size_t> assignment(n);
    for (size_t it = 0; it < iterations; ++it) {

        // Assignment step (E)
        for (size_t i = 0; i < n; ++i) {
            const auto& v = data[i];
            float best = std::numeric_limits<float>::max();
            size_t best_c = 0;
            for (size_t c = 0; c < nlist_; ++c) {
                float d = 0.0f;
                for (size_t k = 0; k < dimension_; ++k) {
                    float diff = v[k] - centroids_[c][k];
                    d += diff * diff;
                }
                if (d < best) {
                    best = d;
                    best_c = c;
                }
            }
            assignment[i] = best_c;
        }

        // Update Step (M)
        std::vector<Vector> sums(nlist_, Vector(dimension_, 0.0f));
        std::vector<size_t> counts(nlist_, 0);
        for (size_t i = 0; i < n; ++i) {
            size_t c = assignment[i];
            for (size_t k = 0; k < dimension_; ++k) {
                sums[c][k] += data[i][k];
            }
            counts[c]++;
        }
        for (size_t c = 0; c < nlist_; ++c) {
            if (counts[c] > 0) {
                for (size_t k = 0; k < dimension_; ++k) {
                    sums[c][k] /= counts[c];
                }
                centroids_[c].swap(sums[c]);
            }
        }
    }
}

void IVFFlatIndex::write_index(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    // Basic params
    out.write(reinterpret_cast<const char*>(&dimension_), sizeof(dimension_));
    out.write(reinterpret_cast<const char*>(&nlist_), sizeof(nlist_));
    out.write(reinterpret_cast<const char*>(&nprobe_), sizeof(nprobe_));

    // Write raw data
    const auto& data = datastore_->getAll();
    size_t N = data.size();
    out.write(reinterpret_cast<const char*>(&N), sizeof(N));
    for (const auto& v : data) {
        out.write(reinterpret_cast<const char*>(v.data()), sizeof(float)*v.size());
    }

    // Write centroids
    size_t C = centroids_.size();
    out.write(reinterpret_cast<const char*>(&C), sizeof(C));
    for (const auto& cvec : centroids_) {
        out.write(reinterpret_cast<const char*>(cvec.data()), sizeof(float)*cvec.size());
    }

    // Write inverted lists
    size_t L = lists_.size();
    out.write(reinterpret_cast<const char*>(&L), sizeof(L));
    for (const auto& lst : lists_) {
        size_t sz = lst.size();
        out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        out.write(reinterpret_cast<const char*>(lst.data()), sizeof(size_t)*sz);
    }
}

std::shared_ptr<IVFFlatIndex> IVFFlatIndex::read_index(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    // Read basic params
    size_t dim, nlist, nprobe;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&nlist), sizeof(nlist));
    in.read(reinterpret_cast<char*>(&nprobe), sizeof(nprobe));

    auto idx = std::make_shared<IVFFlatIndex>(dim, nlist, nprobe);

    // Read raw data
    size_t N;
    in.read(reinterpret_cast<char*>(&N), sizeof(N));
    Dataset data(N, Vector(dim));
    for (auto& v : data) {
        in.read(reinterpret_cast<char*>(v.data()), sizeof(float)*dim);
    }
    idx->datastore_->add(data);

    // Read centroids
    size_t C;
    in.read(reinterpret_cast<char*>(&C), sizeof(C));
    idx->centroids_.resize(C, Vector(dim));
    for (auto& cvec : idx->centroids_) {
        in.read(reinterpret_cast<char*>(cvec.data()), sizeof(float)*dim);
    }

    // Read inverted lists
    size_t L;
    in.read(reinterpret_cast<char*>(&L), sizeof(L));
    idx->lists_.resize(L);
    for (auto& lst : idx->lists_) {
        size_t sz;
        in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        lst.resize(sz);
        in.read(reinterpret_cast<char*>(lst.data()), sizeof(size_t)*sz);
    }

#ifdef ENABLE_CUDA
    // Upload data to GPU after loading from disk
    const auto& loaded_data = idx->datastore_->getAll();

    // Upload centroids
    idx->centroids_flat_.resize(idx->nlist_ * idx->dimension_);
    for (size_t c = 0; c < idx->nlist_; ++c) {
        for (size_t d = 0; d < idx->dimension_; ++d) {
            idx->centroids_flat_[c * idx->dimension_ + d] = idx->centroids_[c][d];
        }
    }
    idx->gpu_centroids_manager_.upload_centroids(idx->centroids_flat_.data(), idx->nlist_, idx->dimension_);

    // Upload base data
    std::vector<float> data_flat(loaded_data.size() * idx->dimension_);
    for (size_t i = 0; i < loaded_data.size(); ++i) {
        for (size_t d = 0; d < idx->dimension_; ++d) {
            data_flat[i * idx->dimension_ + d] = loaded_data[i][d];
        }
    }
    idx->gpu_data_manager_.upload_data(data_flat.data(), loaded_data.size(), idx->dimension_);

    // Upload inverted lists
    idx->gpu_lists_manager_.upload_lists(idx->lists_, idx->nlist_);
#endif

    return idx;
}
}
