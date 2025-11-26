# CUDA k=100 問題修復說明

## 問題描述

### 原始問題
原始 CUDA 實現在 `k=100` 時會崩潰，原因是 **Kernel D 的 shared memory 超過硬件限制**：

```
Shared memory 需求 = nprobe × k × sizeof(DistIdPair)
                   = nprobe × k × 8 bytes

當 k=100, nprobe=64 時：
  = 64 × 100 × 8 = 51,200 bytes > 48KB (GPU 限制)
  → 崩潰！
```

### 性能要求
- **k=10**: nprobe=1 需要達到 **81K+ QPS**
- **k=100**: 所有 nprobe 值（1-64）都要能運行，不能崩潰

## 修復過程

### 嘗試 1: Pure Heap-based Kernel ❌
**想法**: 使用 heap-based selection 替換 shared memory，避免內存限制。

**實現**:
- 單個 kernel 使用 max-heap streaming selection
- O(nprobe × k × log k) 複雜度
- 只使用寄存器，無 shared memory 需求

**結果**:
- ✅ k=100 所有 nprobe 都能運行
- ❌ nprobe=1 性能從 81K 降到 **53K QPS (-35%)**

**原因**: heap 初始化和維護開銷對小 nprobe 不划算。

---

### 嘗試 2: Hybrid Kernel with Branch ❌
**想法**: 在單個 kernel 內用 if-else 選擇算法。

**實現**:
```cpp
__global__ void kernel_d(..., bool use_shared_mem) {
    if (use_shared_mem) {
        // Shared memory path
    } else {
        // Heap path
    }
}
```

**結果**:
- ✅ k=100 所有 nprobe 都能運行
- ❌ nprobe=1 性能只有 **50K QPS (-38%)**

**原因**:
1. 分支判斷雖然在 kernel 內，但可能影響編譯器優化
2. 額外的參數傳遞影響寄存器分配

---

### 嘗試 3: 兩個獨立 Kernel + MAX_K=100 ❌
**想法**: 用兩個完全獨立的 kernel，避免分支開銷。

**實現**:
- `kernel_d_merge_final_topk_fast`: Shared memory 版本
- `kernel_d_merge_final_topk_heap`: Heap 版本
- Host 側根據需求選擇調用哪個
- 同時降低 MAX_K 從 128 到 100 以減少寄存器壓力

**結果**:
- ✅ k=100 所有 nprobe 都能運行
- ❌ nprobe=1 性能只有 **51K QPS (-37%)**

**原因**: **MAX_K=100 影響了 Kernel C！**
```cpp
// Kernel C 中也使用 MAX_K
DistIdPair local_topk[MAX_K];
```
降低 MAX_K 改變了 Kernel C 的寄存器分配模式，影響了整體性能。

---

### 最終解決方案: 兩個獨立 Kernel + MAX_K=128 ✅

**核心想法**:
1. 保持原始 MAX_K=128 不變（不影響 Kernel C）
2. 使用兩個完全獨立的 kernel（零分支開銷）
3. Host 側根據 shared memory 需求自動選擇

**實現細節**:

#### 1. Fast Kernel (原始高性能版本)
```cpp
__global__ void kernel_d_merge_final_topk_fast(
    const DistIdPair* __restrict__ partial_topk,
    float* __restrict__ out_distances,
    int* __restrict__ out_indices,
    int num_queries, int nprobe, int k
) {
    // 多線程並行載入到 shared memory
    extern __shared__ DistIdPair smem_candidates[];
    for (int i = tid; i < total_candidates; i += blockDim.x) {
        smem_candidates[i] = in_ptr[i];
    }
    __syncthreads();

    // Thread 0 執行 k-pass 選擇
    if (tid == 0) {
        for (int ki = 0; ki < k; ++ki) {
            // 找到最小值並標記為已使用
            DistIdPair best = 找最小值();
            標記為已使用();
        }
    }
}
```

**特點**:
- Grid: (num_queries, 1, 1)
- Block: (256, 1, 1) - 多線程並行載入
- Shared memory: nprobe × k × 8 bytes
- 複雜度: O(k² × nprobe)，但常數極小
- **性能**: nprobe=1 達到 **82,952 QPS**

#### 2. Heap Kernel (省內存版本)
```cpp
__global__ void kernel_d_merge_final_topk_heap(
    const DistIdPair* __restrict__ partial_topk,
    float* __restrict__ out_distances,
    int* __restrict__ out_indices,
    int num_queries, int nprobe, int k
) {
    if (tid == 0) {
        DistIdPair heap[MAX_K];
        int heap_size = 0;

        // Streaming insertion with max-heap
        for (int i = 0; i < total_candidates; ++i) {
            if (cand < heap[0]) {
                heap[0] = cand;
                heapify_down_max(heap, k, 0);
            }
        }

        // Sort heap for ascending order
        heap_sort(heap, heap_size);
    }
}
```

**特點**:
- Grid: (num_queries, 1, 1)
- Block: (32, 1, 1) - 單線程即可
- Shared memory: 0 bytes（全部用寄存器）
- 複雜度: O(nprobe × k × log k)
- **性能**: ~60% of fast kernel，但無內存限制

#### 3. 自動選擇邏輯
```cpp
const size_t MAX_SMEM = 48 * 1024;
size_t required_smem = nprobe * k * sizeof(DistIdPair);

if (required_smem <= MAX_SMEM) {
    // 使用 fast kernel (高性能)
    kernel_d_merge_final_topk_fast<<<num_queries, 256, required_smem>>>(
        d_partial_topk, d_out_distances, d_out_indices,
        num_queries, nprobe, k
    );
} else {
    // 使用 heap kernel (省內存)
    kernel_d_merge_final_topk_heap<<<num_queries, 32, 0>>>(
        d_partial_topk, d_out_distances, d_out_indices,
        num_queries, nprobe, k
    );
}
```

**切換閾值分析 (k=100)**:
```
nprobe=1:   100 × 8 = 800 B      → fast kernel
nprobe=2:   200 × 8 = 1.6 KB     → fast kernel
nprobe=4:   400 × 8 = 3.2 KB     → fast kernel
nprobe=8:   800 × 8 = 6.4 KB     → fast kernel
nprobe=16:  1600 × 8 = 12.8 KB   → fast kernel
nprobe=32:  3200 × 8 = 25.6 KB   → fast kernel
nprobe=64:  6400 × 8 = 51.2 KB   → heap kernel (> 48KB)
```

## 性能驗證

### k=10 性能測試
```
======================================================================
nprobe   QPS        p50(ms)    p95(ms)    R@10
----------------------------------------------------------------------
1        82952.2    0.013      0.013      38.01      ✅ 超越 81K 目標
2        66272.8    0.016      0.016      54.14      ✅
4        46521.1    0.022      0.022      70.63      ✅
8        29284.9    0.034      0.034      84.17      ✅
16       16807.6    0.060      0.060      93.36      ✅
32       9122.0     0.110      0.110      97.95      ✅
64       4809.8     0.208      0.208      99.56      ✅
======================================================================
```

### k=100 穩定性測試
```
======================================================================
nprobe   QPS        p50(ms)    p95(ms)    R@100
----------------------------------------------------------------------
1        65868.5    0.018      0.018      28.46      ✅ fast kernel
2        49883.7    0.024      0.024      43.05      ✅ fast kernel
4        32562.9    0.036      0.036      59.42      ✅ fast kernel
8        18399.0    0.060      0.060      74.76      ✅ fast kernel
16       9939.6     0.107      0.107      86.90      ✅ fast kernel
32       4559.1     0.225      0.225      94.37      ✅ fast kernel
64       3030.5     0.336      0.336      97.66      ✅ heap kernel (無崩潰！)
======================================================================
```

## 關鍵要點總結

### ✅ 成功達成
1. **性能恢復**: k=10, nprobe=1 達到 **82,952 QPS**（超越 81K 目標 +1.7%）
2. **穩定性**: k=100 所有 nprobe（1-64）都能運行，無崩潰
3. **零性能損失**: 小 nprobe 使用原始 fast kernel，保持原有性能
4. **自動適配**: Host 側根據內存需求自動選擇最優 kernel

### 🔑 關鍵教訓
1. **MAX_K 不能隨意修改**: 影響多個 kernel 的寄存器分配
2. **避免 kernel 內分支**: 使用獨立 kernel 而非 if-else
3. **測試環境需一致**: 相同的 k 值和測試腳本
4. **分而治之**: 針對不同場景使用專門優化的 kernel

### 📊 性能對比
| 版本 | nprobe=1 QPS | k=100 支援 | 說明 |
|------|--------------|------------|------|
| 原始版本 | 81,548 | ❌ 崩潰 | Shared memory 超限 |
| Pure Heap | 53,015 | ✅ | 性能損失 -35% |
| Hybrid (branch) | 50,715 | ✅ | 性能損失 -38% |
| 兩個 Kernel (MAX_K=100) | 51,228 | ✅ | MAX_K 影響 Kernel C |
| **最終方案** | **82,952** | ✅ | **完美解決** 🎉 |

## 代碼修改位置

### 修改的文件
1. `src/CudaUtils.cu`
   - 添加 `kernel_d_merge_final_topk_fast()` (line 496-544)
   - 添加 `kernel_d_merge_final_topk_heap()` (line 563-621)
   - 修改 Kernel D 調用邏輯 (line 874-902)
   - 添加迭代式 `heapify_down_max()` (line 446-468)

### 保持不變
- `MAX_K = 128` (line 26) - **關鍵！**
- `THREAD_LOCAL_K = 10` (line 31)
- Kernel A, B, C 的實現
- 所有其他參數和配置

## 未來優化方向

當前方案已經完美解決 k=100 問題並保持性能。如需進一步提升，可考慮：

1. **cuBLAS GEMM for Kernel A** (預期 5-10x 加速)
   - 使用矩陣乘法計算 query-centroid 距離
   - 適合大 batch size (256+)

2. **Persistent Memory Allocation**
   - 避免重複 cudaMalloc/cudaFree 開銷
   - 適合連續多次查詢場景

3. **CUDA Streams Pipelining**
   - 並行執行多個 batch
   - 隱藏 memory transfer 延遲

## 測試命令

### 性能測試 (k=10)
```bash
export LD_LIBRARY_PATH=extern/faiss/build/install/lib:$LD_LIBRARY_PATH
python3 benchmark/comprehensive_bench.py \
    --base data/sift/sift_base.fvecs \
    --query data/sift/sift_query.fvecs \
    --groundtruth data/sift/sift_groundtruth.ivecs \
    --nlist 1024 \
    --nprobe-list "1,2,4,8,16,32,64" \
    --k-list "10" \
    --index-file sift_index.bin \
    --output-dir benchmark_results
```

### 穩定性測試 (k=100)
```bash
export LD_LIBRARY_PATH=extern/faiss/build/install/lib:$LD_LIBRARY_PATH
python3 benchmark/comprehensive_bench.py \
    --base data/sift/sift_base.fvecs \
    --query data/sift/sift_query.fvecs \
    --groundtruth data/sift/sift_groundtruth.ivecs \
    --nlist 1024 \
    --nprobe-list "1,2,4,8,16,32,64" \
    --k-list "100" \
    --index-file sift_index.bin \
    --output-dir benchmark_results
```

## 結論

通過使用**兩個獨立的專門優化 kernel** + **保持原始 MAX_K=128**，成功實現：
- ✅ 性能無損失（82K+ QPS）
- ✅ k=100 全面支援（無崩潰）
- ✅ 自動適配（透明切換）

這是一個兼顧**性能、穩定性和可維護性**的優雅解決方案。🎉
