# IVF-Flat GPU Search - Shared Memory 優化方案

## 問題描述

當 `k=100` 時，Kernel C 和 Kernel D 的 shared memory 使用量超過硬體限制（48KB）：

### Kernel C 的問題

```
shared memory 需求 = dim * sizeof(float) + k * block_size * sizeof(DistIdPair)
                   = 128 * 4 + 100 * 128 * 8
                   = 512 + 102,400
                   = ~100 KB ❌ 超過 48KB 限制
```

**原因**：
- Query vector: `dim * sizeof(float)` (通常 512 bytes，還好)
- Merge buffer: `k * block_size * sizeof(DistIdPair)` (k=100, block_size=128 時需要 100KB)

### Kernel D 的問題

```
shared memory 需求 = nprobe * k * sizeof(DistIdPair)
                   = 64 * 100 * 8
                   = 51,200 bytes = 50 KB ❌ 超過 48KB 限制
```

**原因**：
- 需要將所有 partial candidates 載入 shared memory 做 merge

---

## 解決方案概覽

我們提供三種方案，從簡單到複雜：

| 方案 | 複雜度 | 效能影響 | 適用情境 |
|------|--------|---------|---------|
| 方案 1: 兩階段 Kernel C | 中 | 小 | k > 50 |
| 方案 2: Global Memory Merge | 低 | 中等 | k > 50 |
| 方案 3: 動態策略選擇 | 高 | 最小 | 生產環境 |

---

## 方案 1: 兩階段 Kernel C (推薦)

### 核心思想

將原本的 Kernel C 拆成兩個階段：

1. **Kernel C-Scan**: 每個 thread 掃描 list，維護 thread-local top-k，直接寫到 global memory
2. **Kernel C-Merge**: 合併同一個 (query, probe) 的所有 thread results

### 優點

- 完全避免 shared memory 限制
- 保持原有的並行度
- 只增加一次 global memory 寫入/讀取

### 缺點

- 需要額外的 global memory 空間：`Q * nprobe * block_size * k * sizeof(DistIdPair)`
- 多一次 kernel launch (overhead 很小)

---

## 實作細節

### 1. 新增輔助資料結構

在 `batch_search_gpu_pipeline_v2` 中增加中間 buffer：

```cpp
// 新增: thread-level top-k 儲存空間
DistIdPair *d_thread_topk;
int threads_per_block = 128;
size_t thread_topk_size = num_queries * nprobe * threads_per_block * k * sizeof(DistIdPair);

CUDA_CHECK(cudaMalloc(&d_thread_topk, thread_topk_size));
```

---

### 2. 修改 Kernel C - 只做 Scan，不做 Merge

**函數簽名**：

```cuda
__global__ void kernel_c_scan_lists_v2(
    const float* __restrict__ queries,        // [Q x dim]
    const int* __restrict__ selected_lists,   // [Q x nprobe]
    const float* __restrict__ vectors,        // [N_total x dim]
    const int* __restrict__ list_offsets,     // [nlist + 1]
    const size_t* __restrict__ ids,           // [N_total]
    DistIdPair* __restrict__ thread_topk,     // [Q x nprobe x block_size x k] OUTPUT
    int num_queries,
    int nprobe,
    int k,
    int dim
)
```

**主要改動**：

```cuda
{
    // ... 前面的邏輯相同 ...

    // Thread-local top-k (in registers)
    DistIdPair local_topk[MAX_K];
    int local_size = 0;
    int max_local_k = min(k, MAX_K);

    // Scan list (與原版相同)
    for (int idx = list_start + tid; idx < list_end; idx += block_size) {
        size_t vec_id = ids[idx];
        const float* vec_ptr = vectors + vec_id * dim;

        float sum = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < dim; ++d) {
            float diff = shared_query[d] - vec_ptr[d];
            sum += diff * diff;
        }

        insert_to_local_topk(local_topk, local_size, max_local_k, sum, vec_id);
    }

    // ========== 修改點: 直接寫到 global memory ==========

    // 計算這個 thread 在 global buffer 中的位置
    int thread_global_idx = (q * nprobe + probe) * blockDim.x + tid;
    DistIdPair* my_output = thread_topk + thread_global_idx * k;

    // 寫入 thread-local top-k
    for (int i = 0; i < local_size; ++i) {
        my_output[i] = local_topk[i];
    }

    // 填充剩餘位置為 INFINITY
    for (int i = local_size; i < k; ++i) {
        my_output[i] = DistIdPair();  // (INFINITY, -1)
    }

    // 不需要 __syncthreads() 和 merge_block_topk()
}
```

**Shared Memory 需求**：

```
只需要 query vector: dim * sizeof(float) (通常 < 1KB) ✅
```

---

### 3. 新增 Kernel C-Merge - 合併 Thread Results

**函數簽名**：

```cuda
__global__ void kernel_c_merge_thread_topk(
    const DistIdPair* __restrict__ thread_topk,  // [Q x nprobe x block_size x k]
    DistIdPair* __restrict__ partial_topk,       // [Q x nprobe x k] OUTPUT
    int num_queries,
    int nprobe,
    int k,
    int threads_per_block
)
```

**實作 (簡單版本)**：

```cuda
{
    int q = blockIdx.x;
    int probe = blockIdx.y;

    if (q >= num_queries || probe >= nprobe) return;

    int tid = threadIdx.x;

    // 總共有 threads_per_block * k 個候選
    int total_candidates = threads_per_block * k;
    const DistIdPair* input_base = thread_topk + (q * nprobe + probe) * total_candidates;

    // 使用 thread 0 做簡單的 k-pass selection
    // (可以優化成多個 threads 協作，但 k=100 時單 thread 也夠快)
    if (tid == 0) {
        DistIdPair best_k[MAX_K];
        bool used[MAX_CANDIDATES];  // 或用其他方式標記

        // 初始化
        for (int i = 0; i < total_candidates; ++i) {
            used[i] = false;
        }

        // k 輪選擇
        for (int round = 0; round < k; ++round) {
            DistIdPair best = DistIdPair();
            int best_idx = -1;

            // 線性掃描找最小值
            for (int i = 0; i < total_candidates; ++i) {
                if (!used[i] && input_base[i] < best) {
                    best = input_base[i];
                    best_idx = i;
                }
            }

            if (best_idx >= 0 && best.id >= 0) {
                best_k[round] = best;
                used[best_idx] = true;
            } else {
                best_k[round] = DistIdPair();
            }
        }

        // 寫出結果
        DistIdPair* output = partial_topk + (q * nprobe + probe) * k;
        for (int i = 0; i < k; ++i) {
            output[i] = best_k[i];
        }
    }
}
```

**Grid/Block 配置**：

```cpp
dim3 grid_size(num_queries, nprobe);
int block_size = 32;  // 只需要少量 threads (甚至只用 1 個)

kernel_c_merge_thread_topk<<<grid_size, block_size>>>(
    d_thread_topk, d_partial_topk,
    num_queries, nprobe, k, threads_per_block_in_scan
);
```

**複雜度分析**：

- 每個 block 處理 `threads_per_block * k` 個候選 (例如 128 * 100 = 12,800)
- k 輪選擇，每輪 O(threads_per_block * k)
- 總複雜度：O(k² * threads_per_block) ≈ O(1.28M) operations
- 對於 GPU 來說這不算多，且是 embarrassingly parallel

---

### 4. 修改 Kernel D - 使用 Global Memory 或 Optimized Selection

#### 選項 A: 簡單版本 - 單 Thread Selection

```cuda
__global__ void kernel_d_merge_final_topk_v2(
    const DistIdPair* __restrict__ partial_topk,
    float* __restrict__ out_distances,
    int* __restrict__ out_indices,
    int num_queries,
    int nprobe,
    int k
) {
    int q = blockIdx.x;
    if (q >= num_queries) return;

    int tid = threadIdx.x;

    if (tid == 0) {
        int total_candidates = nprobe * k;
        const DistIdPair* input = partial_topk + q * total_candidates;

        // 簡單的 k-pass selection (不使用 shared memory)
        bool used[MAX_CANDIDATES];
        for (int i = 0; i < total_candidates; ++i) {
            used[i] = false;
        }

        for (int round = 0; round < k; ++round) {
            DistIdPair best = DistIdPair();
            int best_idx = -1;

            for (int i = 0; i < total_candidates; ++i) {
                if (!used[i] && input[i] < best) {
                    best = input[i];
                    best_idx = i;
                }
            }

            if (best_idx >= 0 && best.id >= 0) {
                out_distances[q * k + round] = best.dist;
                out_indices[q * k + round] = best.id;
                used[best_idx] = true;
            } else {
                out_distances[q * k + round] = INFINITY;
                out_indices[q * k + round] = -1;
            }
        }
    }
}
```

**問題**：`used` 陣列太大 (nprobe * k 可能到 6400)

#### 選項 B: 優化版本 - Heap-based Selection

```cuda
__global__ void kernel_d_merge_final_topk_heap(
    const DistIdPair* __restrict__ partial_topk,
    float* __restrict__ out_distances,
    int* __restrict__ out_indices,
    int num_queries,
    int nprobe,
    int k
) {
    int q = blockIdx.x;
    if (q >= num_queries) return;

    int tid = threadIdx.x;

    if (tid == 0) {
        int total_candidates = nprobe * k;
        const DistIdPair* input = partial_topk + q * total_candidates;

        // 使用 min-heap (size = k) 做 streaming selection
        DistIdPair heap[MAX_K];
        int heap_size = 0;

        // 遍歷所有 candidates
        for (int i = 0; i < total_candidates; ++i) {
            DistIdPair cand = input[i];
            if (cand.id < 0) continue;  // 跳過無效項

            if (heap_size < k) {
                // Heap 還沒滿，直接插入
                heap[heap_size++] = cand;
                if (heap_size == k) {
                    // 建立 min-heap
                    for (int j = k/2 - 1; j >= 0; --j) {
                        heapify_down(heap, k, j);
                    }
                }
            } else if (cand < heap[0]) {
                // 新元素比 heap 頂小，替換並重新 heapify
                heap[0] = cand;
                heapify_down(heap, k, 0);
            }
        }

        // 排序 heap 並輸出
        for (int i = heap_size - 1; i > 0; --i) {
            swap(heap[0], heap[i]);
            heapify_down(heap, i, 0);
        }

        for (int i = 0; i < heap_size; ++i) {
            out_distances[q * k + i] = heap[i].dist;
            out_indices[q * k + i] = heap[i].id;
        }
        for (int i = heap_size; i < k; ++i) {
            out_distances[q * k + i] = INFINITY;
            out_indices[q * k + i] = -1;
        }
    }
}

// 輔助函數
__device__ void heapify_down(DistIdPair* heap, int size, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < size && heap[left].dist < heap[largest].dist) {
        largest = left;
    }
    if (right < size && heap[right].dist < heap[largest].dist) {
        largest = right;
    }

    if (largest != i) {
        DistIdPair tmp = heap[i];
        heap[i] = heap[largest];
        heap[largest] = tmp;
        heapify_down(heap, size, largest);
    }
}

__device__ void swap(DistIdPair& a, DistIdPair& b) {
    DistIdPair tmp = a;
    a = b;
    b = tmp;
}
```

**複雜度**：O(total_candidates * log k) = O(nprobe * k * log k)

---

## 方案 2: Global Memory Merge (最簡單)

如果不想改太多，可以只修改 Kernel D：

### 實作步驟

1. 保持 Kernel C 不變
2. 修改 Kernel D 使用選項 B (Heap-based)
3. 調整 Kernel C 的 block_size 使其 shared memory 不超過限制

### 動態調整 Block Size

```cpp
// 在 batch_search_gpu_pipeline_v2 中
{
    dim3 grid_size(num_queries, nprobe);

    // 計算最大可用的 block_size
    const size_t MAX_SMEM = 48 * 1024;
    size_t query_smem = dim * sizeof(float);
    size_t available = MAX_SMEM - query_smem;

    // merge buffer 需要 k * block_size * sizeof(DistIdPair)
    int max_block_size = available / (k * sizeof(DistIdPair));
    max_block_size = (max_block_size / 32) * 32;  // Round down to multiple of 32

    int block_size;
    if (max_block_size >= 32) {
        block_size = min(128, max_block_size);
    } else {
        // k 太大，無法在 shared memory 做 merge
        // 降級為只掃描，不做 block-level merge
        block_size = 128;
        // ... 需要修改 kernel 或使用方案 1
    }

    size_t smem_size = query_smem + k * block_size * sizeof(DistIdPair);

    kernel_c_scan_lists<<<grid_size, block_size, smem_size>>>(
        d_queries, d_selected_lists, d_base_data, d_list_offsets,
        d_list_data, d_partial_topk,
        num_queries, nprobe, k, dim
    );
}
```

---

## 方案 3: 動態策略選擇 (生產環境推薦)

根據 k 值自動選擇最佳策略：

```cpp
void batch_search_gpu_pipeline_adaptive(
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
    // 決定使用哪種策略
    const size_t MAX_SMEM = 48 * 1024;
    size_t kernel_c_smem = dim * sizeof(float) + k * 128 * sizeof(DistIdPair);
    size_t kernel_d_smem = nprobe * k * sizeof(DistIdPair);

    bool use_two_stage = (kernel_c_smem > MAX_SMEM) || (kernel_d_smem > MAX_SMEM);

    if (use_two_stage) {
        // 使用方案 1: 兩階段 merge
        batch_search_gpu_pipeline_two_stage(
            queries, d_centroids, d_base_data, d_list_data, d_list_offsets,
            result_distances, result_indices,
            num_queries, num_centroids, nprobe, k, dim
        );
    } else {
        // 使用原版 (一階段 shared memory merge)
        batch_search_gpu_pipeline_v2(
            queries, d_centroids, d_base_data, d_list_data, d_list_offsets,
            result_distances, result_indices,
            num_queries, num_centroids, nprobe, k, dim
        );
    }
}
```

---

## 效能預期

### 方案 1 (兩階段)

| 階段 | 額外開銷 |
|------|---------|
| Kernel C-Scan | 無 (與原版相同) |
| Global memory write | ~1-2ms (對於大 batch) |
| Kernel C-Merge launch | ~0.01ms |
| Kernel C-Merge compute | ~0.5-1ms |
| **總額外開銷** | **~2-4ms** |

對於 k=100 的情況，這個開銷是可接受的。

### 方案 2 (Global Memory)

| 階段 | 額外開銷 |
|------|---------|
| Kernel C (reduced block_size) | +10-20% (因為並行度下降) |
| Kernel D (heap-based) | -10% (反而更快，因為避免 shared memory 爭用) |
| **總額外開銷** | **~5-10%** |

---

## 實作優先順序

### Phase 1: 快速修復 (1-2 小時)
- ✅ 實作 Kernel D heap-based 版本
- ✅ 添加動態 block_size 調整
- ✅ 測試 k=100 是否能跑

### Phase 2: 完整方案 (3-4 小時)
- ✅ 實作兩階段 Kernel C
- ✅ 實作 Kernel C-Merge
- ✅ 整合到 batch_search_gpu_pipeline_v2
- ✅ 完整測試與 benchmark

### Phase 3: 優化 (選做)
- 🔄 Kernel C-Merge 使用多 threads 協作
- 🔄 使用 warp-level primitives 優化
- 🔄 實作 adaptive strategy selection

---

## 參考資料

- **CUDA Shared Memory Limit**: 48KB (Compute Capability 3.x-8.x)
- **Max Shared Memory (dynamic config)**: 96KB (Compute Capability 7.0+, 需要 `cudaFuncSetAttribute`)
- **Alternative**: 使用 L1 cache (自動管理)

---

## 下一步

建議從 **Phase 1** 開始實作，這樣可以快速驗證解決方案是否可行。
