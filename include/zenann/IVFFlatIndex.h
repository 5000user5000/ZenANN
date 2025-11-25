#pragma once

#include "IndexBase.h"
#include <vector>
#ifdef ENABLE_CUDA
#include "CudaUtils.h"
#endif

namespace zenann {

class IVFFlatIndex : public IndexBase {
public:
    IVFFlatIndex(size_t dim, size_t nlist, size_t nprobe = 1);
    ~IVFFlatIndex() override;
    void train() override;
    SearchResult search(const Vector& query, size_t k) const override;
    SearchResult search(const Vector& query, size_t k, size_t nprobe) const;
    std::vector<SearchResult> search_batch(const Dataset& queries, size_t k) const;
    std::vector<SearchResult> search_batch(const Dataset& queries, size_t k, size_t nprobe) const;
    void write_index(const std::string& filename) const;
    static std::shared_ptr<IVFFlatIndex> read_index(const std::string& filename);
private:
    size_t                     nlist_;
    size_t                     nprobe_;
    Dataset                    centroids_;
    std::vector<idList>        lists_;
#ifdef ENABLE_CUDA
    mutable std::vector<float>           centroids_flat_;         // Flattened centroids
    mutable cuda::GpuCentroidsManager    gpu_centroids_manager_;  // GPU centroid manager
    mutable cuda::GpuDataManager         gpu_data_manager_;       // GPU data manager
    mutable cuda::GpuInvertedListsManager gpu_lists_manager_;     // GPU inverted lists manager
#endif
    void kmeans(const Dataset& data, size_t iterations = 10);
};

} 
