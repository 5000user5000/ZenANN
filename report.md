# 平程期末專案報告
## Parallel Acceleration of the Inverted File Index for Approximate Nearest Neighbor Search

**組員:** 周哲瑋、邱德晏、陳冠霖

---

## 1. Introduction / Motivation (引言與動機)

### 1.1 專案背景

在現代人工智慧應用中，從大型語言模型的檢索增強生成 (RAG) 到推薦系統、以圖搜圖，**向量相似度搜尋 (Vector Similarity Search)** 已成為不可或缺的核心技術。然而，隨著數據量呈現爆炸性增長，傳統的暴力搜尋 (Brute-force k-NN) 因其 O(N) 的線性複雜度，在處理百萬甚至十億級別的數據時：

* **計算成本過高**：需要與所有向量計算距離
* **延遲無法接受**：無法滿足線上服務的低延遲需求 (< 100ms)
* **資源浪費嚴重**：大部分計算都在不相關的向量上

### 1.2 為何選擇 IVF-Flat

我們採用 **倒排檔案索引 (Inverted File Index, IVF-Flat)** 演算法來解決這個問題：

* **原理**：透過 K-means 將高維向量空間劃分為多個 Voronoi Cells (聚類)
* **效果**：將搜尋範圍從全域縮小至少數幾個最相關的聚類 (nprobe)
* **優勢**：大幅降低計算複雜度，同時保持良好的召回率 (Recall)

### 1.3 為何需要平行化

儘管 IVF 演算法本身具備效率優勢，但在大規模數據下仍存在性能瓶頸：

* **距離計算密集**：需要計算大量向量間的 L2 距離
* **批量查詢需求**：實際應用常需處理大量並發查詢
* **硬體未充分利用**：單執行緒無法發揮現代多核 CPU 和 GPU 的潛力

**本專案目標**：透過現代平行運算技術（OpenMP、SIMD、CUDA），在不犧牲搜尋準確度的前提下，極大化查詢吞吐量 (QPS)。

---

## 2. Problem Statement (問題陳述)

### 2.1 IVF-Flat 演算法的兩階段流程

IVF-Flat 的查詢可分為兩個主要階段：

**階段 1：粗粒度量化 (Coarse Quantization)**
* 將查詢向量與所有聚類中心 (Centroids) 計算距離
* 找出最近的 nprobe 個聚類
* **瓶頸**：需要計算 nlist 次距離（通常 nlist = 100~1024）

**階段 2：細粒度搜尋 (Fine Search)**
* 在選定的 nprobe 個聚類的倒排列表中掃描所有候選向量
* 計算每個候選向量與查詢向量的距離
* 維護 Top-K 結果
* **瓶頸**：
  - 大量距離計算（可能有數萬至數十萬候選）
  - 倒排列表長度不均導致負載不平衡
  - Top-K 維護需要頻繁比較與更新

### 2.2 平行化挑戰

**CPU 端挑戰**：
1. **資料層級**：如何高效向量化 L2 距離計算
2. **任務層級**：如何平行化質心搜尋、列表掃描、批量查詢
3. **同步問題**：多執行緒寫入 Top-K heap 的 race condition

**GPU 端挑戰**：
1. **記憶體限制**：Shared Memory 容量限制 (48KB)
2. **負載平衡**：倒排列表長度差異巨大
3. **不規則存取**：倒排列表的隨機訪問模式
4. **大 k 值問題**：k=100 時 Shared Memory 需求爆炸

---

## 3. Proposed Solution: ZenANN (提出的解決方案)

我們開發了 **ZenANN**，一個高效能向量搜尋函式庫，提供多種最佳化版本：

### 3.1 系統架構

* **語言**：C++17 核心邏輯 + Python 介面 (pybind11)
* **索引類型**：IVF-Flat, HNSW, KD-Tree
* **編譯變體**：
  - **naive**：單執行緒基準版本（無優化）
  - **openmp**：CPU 多執行緒版本
  - **simd**：CPU SIMD 向量化版本（AVX2）
  - **full**：CPU 完整優化（OpenMP + SIMD）
  - **cuda**：GPU 加速版本
  - **profiling**：效能分析版本

### 3.2 平行化策略概覽

| 層級 | 技術 | 目標 | 預期加速 |
|------|------|------|----------|
| **資料平行** | SIMD (AVX2) | L2 距離計算 | 4-8x |
| **任務平行** | OpenMP | 質心搜尋、列表掃描、批量查詢 | Nx（N=核心數）|
| **大規模平行** | CUDA | 所有計算卸載至 GPU | 20-80x |

---

## 4. Changes Since Proposal (與提案的差異)

### 4.1 原始計畫
我們的提案規劃了完整的 CPU (OpenMP + SIMD) 和 GPU (CUDA) 平行化實作，並成功完成了所有實作。

### 4.2 關鍵轉折
在開發和測試 GPU 實作時，我們發現了一個更深層且關鍵的挑戰：

**問題**：當 k=100 時，現有平行化策略會因 GPU Shared Memory 容量限制而崩潰
* **原因**：Shared Memory 需求 = nprobe × k × 8 bytes
* **實例**：nprobe=64, k=100 → 51.2 KB > 48 KB (硬體上限)
* **影響**：這不是性能問題，而是功能性失敗

### 4.3 專案重心調整
認識到解決這個硬體層級瓶頸比單純展示加速比更有意義，我們調整了專案重點：

**從**：廣泛的實作調查（展示各種平行化技術）
**到**：深入解決真實世界的平行運算限制（記憶體瓶頸）

**貢獻**：
* 設計並實作創新的 Hybrid CUDA Kernel 架構
* 解決大 k 值的 Shared Memory 瓶頸
* 在保持高性能的同時支援 k=1~100 全範圍

---

## 5. How Parallelization is Done (平行化實作細節)

我們的平行化策略是一個多層次的系統性方法，從 CPU 的指令級優化，到多核心協同，再到 GPU 的大規模平行化與硬體感知優化。

### **層次一：CPU 多核心平行化 (OpenMP + SIMD)**

我們將 CPU 優化視為一個金字塔，從最底層的指令級平行化到最高層的任務級平行化，旨在榨乾處理器的每一分效能。

#### **5.1 資料級平行化：SIMD (Single Instruction, Multiple Data)**

*   **目標：** 加速最核心、最頻繁的運算——**L2 距離計算**。
*   **位置：** `include/zenann/SimdUtils.h`
*   **設計決策 (`Why`):** 距離計算是典型的**資料平行**任務，向量的每個維度計算都相互獨立，非常適合 SIMD。這是整個系統的基礎瓶頸，優化它的投資回報率最高。
*   **實作細節 (`How`):**
    *   **優化前 (Naive):**
        ```cpp
        float d = 0.f;
        for (size_t i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            d += diff * diff;
        }
        ```
    *   **優化後 (AVX2 Intrinsics):** 我們使用 `<immintrin.h>` 的 AVX2 指令集，通過 `__m256` 暫存器一次處理 8 個 32-bit 浮點數。使用 `_mm256_fmadd_ps` (Fused Multiply-Add) 指令進一步提升效率。
        ```cpp
        __m256 acc = _mm256_setzero_ps();
        for (size_t i = 0; i < dim; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }
        // ... 水平加總 acc 並處理剩餘元素 ...
        ```
    *   **預期加速：** 僅此項即可對距離計算帶來 **4-8 倍**的理論加速。

#### **5.2 任務級平行化：OpenMP**

我們使用 OpenMP 對演算法中可並行的迴圈進行多執行緒處理。

*   **A. 批量查詢 (`search_batch`) & 質心距離計算 - 無數據依賴的優化**
    *   **設計決策:** 這是最理想的平行化場景，每個查詢或每個質心距離的計算都完全獨立 ("Embarrassingly Parallel")。
    *   **實作 (`search_batch`):** 使用 `#pragma omp parallel for schedule(dynamic)`。選擇 `dynamic` 是因為不同查詢的複雜度可能略有不同，有助於負載平衡。
    *   **實作 (`centroid distance`):** 使用 `#pragma omp parallel for schedule(static)`。因每個質心的計算量完全相同，`static` 排程可以減少執行緒調度的開銷。

*   **B. 列表掃描 (List Probing in `search`) - 處理同步與負載平衡**
    *   **挑戰:** 這是個複雜的點，因為 (1) 不同倒排列表長度差異巨大，導致**負載不平衡**；(2) 所有執行緒需要更新**同一個全局 Top-k 結果堆 (Heap)**，會產生**競爭條件 (Race Condition)**。
    *   **設計決策:** 我們的策略是讓每個執行緒維護自己的**本地堆 (Thread-local Heap)**，待所有執行緒完成後再進行合併。
    *   **實作:**
        ```cpp
        #pragma omp parallel
        {
            std::vector<Pair> local_heap; // 每個執行緒自己的 heap
            #pragma omp for schedule(dynamic) nowait
            for (size_t pi = 0; pi < nprobe_; ++pi) {
                // ... 搜尋 list 並更新 local_heap ...
            }

            #pragma omp critical
            {
                // 將 local_heap 合併到全局 heap
            }
        }
        ```
    *   選擇 `schedule(dynamic)` 是解決負載不平衡問題的關鍵。

*   **C. 索引訓練 (`kmeans`) - 處理 Race Condition**
    *   **E-Step (分配):** 將每個數據點分配給最近的質心，此過程點與點之間獨立，使用簡單的 `#pragma omp parallel for`。
    *   **M-Step (更新):** 更新質心位置時，多個執行緒可能同時寫入同一個質心的累加值。我們的實作使用 OpenMP 的 `reduction` 子句來安全地進行並行累加，避免了手動同步的複雜性。

### **層次二：GPU 大規模平行化與進階優化**

在最大化 CPU 效能後，我們轉向 GPU 以尋求數量級的效能提升。我們的基礎 GPU 版本並非簡單移植，而是從一開始就建立在多個 CUDA 最佳實踐之上，以確保高效率。

#### **5.3 基礎 GPU 平行化策略 (Foundational GPU Parallelization)**

我們初始的 GPU 實作遵循了以下幾個核心優化原則：

1.  **最小化 CPU-GPU 傳輸開銷 (Zero Transfer Overhead)**
    *   **設計決策:** PCIe 帶寬遠低於 GPU 記憶體帶寬，是常見瓶頸。
    *   **實作:** 我們採用**索引常駐 (Index Residency)** 策略。巨大的索引資料（數GB的向量）在初始化時一次性從 CPU 複製到 GPU Global Memory。在查詢階段，只有數 KB 的查詢向量和結果需要通過 PCIe 傳輸。此外，通過**批量處理 (Batch Processing)**，單次傳輸的固定開銷被攤銷到大量查詢上，極大地提升了吞吐量。

2.  **記憶體合併存取 (Memory Coalescing)**
    *   **設計決策:** GPU 記憶體以 32 位元組或 128 位元組的塊進行存取。當一個 Warp (32 個執行緒) 的請求能被合併為單一記憶體事務時，效率最高。
    *   **實作:** 我們確保了所有向量資料在 Global Memory 中採用**Row-major**佈局。在 Kernel 中，一個 Warp 裡的所有連續執行緒 (`threadIdx.x`, `threadIdx.x+1`, ...) 會被分配去處理連續的記憶體區段（例如一個向量的不同維度），從而觸發合併存取，最大化記憶體帶寬利用率。

3.  **共享記憶體重用 (Shared Memory Reuse)**
    *   **設計決策:** Shared Memory 是 SM 內部的低延遲讀寫區，相當於用戶可程式化的快取。
    *   **實作:** 在計算質心距離等 Kernel 中，會被重複讀取的**查詢向量 (Query Vector)** 被首先載入到 Shared Memory。之後 Block 內的所有執行緒都從 Shared Memory 讀取查詢向量，避免了數百次對高延遲 Global Memory 的重複訪問。

4.  **暫存器級 Top-k 維護 (Register-level Top-k)**
    *   **設計決策:** 暫存器是 GPU 中最快的記憶體。在列表掃描時，如果每個執行緒都頻繁更新位於 Global 或 Shared Memory 的 Top-k 列表，會造成嚴重的延遲和競爭。
    *   **實作:** 每個執行緒在自己的私有暫存器中維護一個極小的、固定大小的陣列來存放局部的 Top-k 候選。由於 `k` 值通常不大，這個小陣列的操作（如插入排序）對暫存器來說幾乎沒有開銷，極大降低了記憶體流量。

有了這些基礎優化，我們的 GPU 版本在標準測試下表現出色。然而，當我們將 `k` 值推向極限進行壓力測試時，一個更根本的硬體限制浮現了出來。

#### **5.4 瓶頸分析：共享記憶體為何會爆炸？**

*   **背景:** 共享記憶體是 GPU SM 內部的高速暫存空間，對 CUDA 核心可見，是執行緒塊 (Thread Block) 內合作的關鍵。但它非常有限（我們平台上為 48 KB）。
*   **Kernel C (列表掃描):**
    *   **傳統設計:** 一個 Block 內的所有 Thread 掃描列表，將各自的 Top-k 候選放入共享記憶體，在 Block 結束前進行一次合併。
    *   **問題:** 所需空間 `k * block_size * 8` bytes，當 `k=100`, `block_size=128` 時，需要約 **100 KB**，遠超上限。
*   **Kernel D (最終合併):**
    *   **傳統設計:** 每個查詢的 `nprobe` 個候選列表（共 `nprobe * k` 個結果）被讀入共享記憶體進行最終排序。
    *   **問題:** 所需空間 `nprobe * k * 8` bytes，當 `nprobe=64`, `k=100` 時，需要 **50 KB**，同樣超出上限。

#### **5.5 CUDA 4-Kernel Pipeline 架構**

在深入瓶頸解決方案前，先說明我們的完整 GPU 搜尋流程：

**Pipeline 概覽**：
```
Query Batch → [Kernel A] → [Kernel B] → [Kernel C] → [Kernel D] → Results
```

**Kernel A: Coarse Distance Calculation（質心距離計算）**
* **目標**：計算每個查詢向量與所有 nlist 個質心的距離
* **Grid/Block 映射**：
  - Grid: `(num_queries, 1, 1)`
  - Block: `(256, 1, 1)`
* **優化技術**：
  - Query vector 載入 Shared Memory（重複使用）
  - Coalesced memory access for centroids
  - 使用 `__ldg()` 唯讀快取

**Kernel B: Probe Selection（探測列表選擇）**
* **目標**：為每個查詢選出最近的 nprobe 個質心
* **方法**：Parallel reduction / partial sort in shared memory
* **輸出**：`d_nprobe_ids[num_queries * nprobe]`

**Kernel C: List Scanning（列表掃描）**
* **目標**：在選中的倒排列表中掃描候選向量，計算距離
* **Grid/Block 映射**：
  - Grid: `(num_queries, nprobe, 1)` - 2D grid
  - 每個 block 處理一個 `(query, probe_list)` 對
* **Thread-local Top-k**：每個執行緒在 registers 維護局部 Top-k
* **輸出**：`d_partial_topk[num_queries * nprobe * k]`

**Kernel D: Final Merge（最終合併）**
* **目標**：合併每個查詢的 nprobe × k 個候選，產生最終 Top-k
* **這是瓶頸所在** → 見下一節的解決方案

---

#### **5.6 Hybrid Kernel 解決方案：兩個獨立的 Kernel D**

經過多次嘗試（詳見 `doc/cuda-fix.md`），我們最終採用 **Hybrid Kernel 架構**：

**方案演進歷程**：
1. ❌ **Pure Heap-based Kernel**：k=100 能運行，但 k=10 性能降 35%
2. ❌ **Single Kernel with Branch**：分支判斷影響性能，降 38%
3. ❌ **兩個 Kernel + MAX_K=100**：影響了 Kernel C，降 37%
4. ✅ **兩個獨立 Kernel + MAX_K=128**：完美解決！

**最終設計：兩個專用 Kernel**

**Kernel D-Fast（高性能版，使用 Shared Memory）**
```cpp
__global__ void kernel_d_merge_final_topk_fast(
    const DistIdPair* partial_topk,
    float* out_distances, int* out_indices,
    int num_queries, int nprobe, int k
) {
    extern __shared__ DistIdPair smem_candidates[];

    // 多執行緒並行載入到 shared memory
    for (int i = tid; i < nprobe * k; i += blockDim.x) {
        smem_candidates[i] = partial_topk[query_offset + i];
    }
    __syncthreads();

    // Thread 0 執行 k-pass selection
    if (tid == 0) {
        for (int ki = 0; ki < k; ++ki) {
            // 找最小值並標記
        }
    }
}
```
* **適用**：Shared Memory 需求 ≤ 48KB
* **配置**：Grid(num_queries, 1, 1), Block(256, 1, 1)
* **性能**：k=10, nprobe=1 達到 **82,952 QPS**

**Kernel D-Heap（省記憶體版，使用 Registers）**
```cpp
__global__ void kernel_d_merge_final_topk_heap(
    const DistIdPair* partial_topk,
    float* out_distances, int* out_indices,
    int num_queries, int nprobe, int k
) {
    if (tid == 0) {
        DistIdPair heap[MAX_K];  // 在暫存器中
        int heap_size = 0;

        // Streaming insertion with max-heap
        for (int i = 0; i < nprobe * k; ++i) {
            DistIdPair cand = partial_topk[...];
            if (heap_size < k || cand.dist < heap[0].dist) {
                heap[0] = cand;
                heapify_down_max(heap, k, 0);
            }
        }

        // Heap sort 產生遞增序列
        heap_sort(heap, heap_size);
    }
}
```
* **適用**：Shared Memory 需求 > 48KB
* **配置**：Grid(num_queries, 1, 1), Block(32, 1, 1)
* **性能**：約 60% of fast kernel，但無記憶體限制

**自動選擇邏輯**（在 Host 端）：
```cpp
size_t required_smem = nprobe * k * sizeof(DistIdPair);
const size_t MAX_SMEM = 48 * 1024;

if (required_smem <= MAX_SMEM) {
    // 使用 fast kernel
    kernel_d_merge_final_topk_fast<<<num_queries, 256, required_smem>>>(...);
} else {
    // 使用 heap kernel
    kernel_d_merge_final_topk_heap<<<num_queries, 32, 0>>>(...);
}
```

**切換閾值（k=100）**：
- nprobe ≤ 60：fast kernel (shared memory)
- nprobe > 60：heap kernel (registers)

**為何這個方案成功**：
1. **零分支開銷**：兩個完全獨立的 kernel，編譯器可完全優化
2. **保持 MAX_K=128**：不影響 Kernel C 的暫存器分配
3. **自動適配**：根據硬體限制透明切換

---

## 6. Challenges Encountered (遇到的挑戰)

1.  **Race Conditions (CPU):** 在 `kmeans` 的 M-step 和列表探測的合併階段，多執行緒並行寫入共享數據結構。必須使用 `reduction` 或 `critical` section 等同步機制來保證結果正確性，但這也帶來了額外開銷。
2.  **Load Imbalancing (CPU):** 在列表探測階段，各 `nprobe` 列表的長度極不均勻。若使用 `schedule(static)`，會導致部分執行緒很快完成而其他仍在忙碌。改用 `schedule(dynamic)` 解決了此問題，但增加了調度開銷。
3.  **記憶體階層的權衡 (GPU):** 我們的 GPU 解決方案是一個典型的**空間換時間（或可行性）**的權衡。我們放棄了使用高速但有限的共享記憶體，轉而使用容量巨大但慢得多的全域記憶體。這雖然帶來了額外的延遲，但卻讓演算法從「不可用」變為「可用」。
4.  **CUDA Kernel 複雜度 (GPU):** 在 Kernel 中手動實作 Heap 演算法，比標準的排序複雜得多，需要精細管理暫存器和裝置函數，增加了開發與除錯的難度。

---

## 7. Evaluation (效能評估)

> **📊 數據來源說明**：
> 完整的 benchmark 數據、詳細圖表、以及實驗結果分析請參考以下 GitHub Discussions：
> - [CPU 平行化分析與數據](https://github.com/5000user5000/ZenANN/discussions/4)
> - [CUDA 實作與性能數據](https://github.com/5000user5000/ZenANN/discussions/15)
> - [CUDA k=100 修復過程與數據](doc/cuda-fix.md)
>
> 本章節提供代表性數據和分析框架，供製作投影片參考。

### 7.1 實驗平台 (Platform)

**硬體配置**：
* **GPU**：NVIDIA RTX A6000
  - Compute Capability: 8.6 (Ampere)
  - CUDA Cores: 10,752
  - Shared Memory per SM: 48 KB
  - Global Memory: 48 GB GDDR6
* **CPU**：Intel Core i7-12700
  - 核心數：12 cores (8P+4E)
  - 測試時使用：8 threads (OpenMP)
  - 支援 AVX2 指令集
* **CUDA Toolkit**：13.0
* **編譯器**：g++ 11.4, nvcc 13.0

**資料集**：
* **SIFT1M**（主要測試）
  - 向量數量：1,000,000
  - 維度：128
  - 查詢數量：10,000
  - Ground truth：k=100
* **GIST1M**（額外驗證）
  - 向量數量：1,000,000
  - 維度：960
  - 查詢數量：1,000

**索引參數**：
* **nlist**：1024（質心數量）
* **nprobe**：1, 2, 4, 8, 16, 32, 64（探測列表數）
* **k**：1, 10, 100（返回鄰居數）

---

### 7.2 評估指標 (Metrics)

1. **QPS (Queries Per Second)**：吞吐量，越高越好
2. **Recall@k**：召回率，與 ground truth 的重疊比例
3. **Latency**：延遲（p50, p95, p99），越低越好
4. **Speedup**：相對於 naive 版本的加速比

---

### 7.3 實驗結果

#### **7.3.1 CUDA 版本性能測試（SIFT1M, k=10）**

**投影片重點**：這是我們的最佳性能，展示 GPU 加速的威力。

| nprobe | QPS | Latency (ms) | Recall@10 | 使用的 Kernel |
|--------|-----|--------------|-----------|---------------|
| 1 | **82,952** | 0.013 | 38.01% | Fast |
| 2 | 66,273 | 0.016 | 54.14% | Fast |
| 4 | 46,521 | 0.022 | 70.63% | Fast |
| 8 | 29,285 | 0.034 | 84.17% | Fast |
| 16 | 16,808 | 0.060 | 93.36% | Fast |
| 32 | 9,122 | 0.110 | 97.95% | Fast |
| 64 | 4,810 | 0.208 | 99.56% | Fast |

**關鍵數據**：
- **最高 QPS**：82,952（nprobe=1）
- **最高 Recall**：99.56%（nprobe=64）
- **實用平衡點**：nprobe=8，QPS=29,285，Recall=84.17%

---

#### **7.3.2 CUDA k=100 穩定性測試（驗證瓶頸修復）**

**投影片重點**：證明我們解決了 k=100 的崩潰問題，且保持性能。

| nprobe | QPS | Latency (ms) | Recall@100 | 使用的 Kernel |
|--------|-----|--------------|------------|---------------|
| 1 | 65,869 | 0.018 | 28.46% | Fast ✅ |
| 2 | 49,884 | 0.024 | 43.05% | Fast ✅ |
| 4 | 32,563 | 0.036 | 59.42% | Fast ✅ |
| 8 | 18,399 | 0.060 | 74.76% | Fast ✅ |
| 16 | 9,940 | 0.107 | 86.90% | Fast ✅ |
| 32 | 4,559 | 0.225 | 94.37% | Fast ✅ |
| 64 | **3,031** | 0.336 | 97.66% | **Heap ✅（無崩潰！）** |

**關鍵成就**：
- ✅ **所有 nprobe 值都能運行**（原本 nprobe≥64 會崩潰）
- ✅ **nprobe=64 自動切換到 Heap Kernel**
- ✅ **性能損失可接受**（Heap kernel 約為 Fast kernel 的 60-70%）

---

#### **7.3.3 CPU 版本性能比較（SIFT1M, k=10, nprobe=16）**

**投影片重點**：展示不同優化技術的累積效果。

| 版本 | QPS | Speedup | 關鍵技術 |
|------|-----|---------|----------|
| **naive** | 245 | 1.0x（基準） | 無優化 |
| **simd** | 1,127 | 4.6x | AVX2 向量化 |
| **openmp** | 1,842 | 7.5x | 多執行緒（8 threads）|
| **full** | **3,654** | **14.9x** | OpenMP + SIMD |
| **cuda** | 16,808 | **68.6x** | GPU 加速 |

**關鍵洞察**：
- SIMD 單獨貢獻：4.6x
- OpenMP 單獨貢獻：7.5x
- **組合效果**：14.9x（接近相乘）
- **GPU 質的飛躍**：68.6x

---

#### **7.3.4 Recall-QPS Trade-off（做投影片的同學需要畫圖）**

**圖表說明**：
- **X 軸**：QPS（越右越好）
- **Y 軸**：Recall@10（越高越好）
- **曲線**：不同 nprobe 的點連線

**數據點**（CUDA, k=10）：
```
nprobe=1:  QPS=82,952, Recall=38.01%
nprobe=2:  QPS=66,273, Recall=54.14%
nprobe=4:  QPS=46,521, Recall=70.63%
nprobe=8:  QPS=29,285, Recall=84.17%  ← 推薦平衡點
nprobe=16: QPS=16,808, Recall=93.36%
nprobe=32: QPS=9,122,  Recall=97.95%
nprobe=64: QPS=4,810,  Recall=99.56%
```

**投影片文字**：
「隨著 nprobe 增加，Recall 提升但 QPS 下降。nprobe=8 是速度與準確度的最佳平衡點。」

---

#### **7.3.5 Hybrid Kernel 性能對比**

**投影片重點**：證明 Hybrid 設計的優越性。

| 方案 | k=10, nprobe=1 QPS | k=100 支援 | 說明 |
|------|-------------------|-----------|------|
| 原始版本 | 81,548 | ❌ 崩潰 | Shared memory 超限 |
| Pure Heap | 53,015 (-35%) | ✅ | 性能大幅下降 |
| Hybrid (branch) | 50,715 (-38%) | ✅ | 分支開銷大 |
| 兩個 Kernel (MAX_K=100) | 51,228 (-37%) | ✅ | 影響 Kernel C |
| **最終方案（Hybrid）** | **82,952 (+1.7%)** | ✅ | **完美解決** 🎉 |

---

### 7.4 與基準的比較

**與 FAISS 的定位**：
* FAISS 是成熟的生產級函式庫，經過多年優化
* ZenANN 是學術探索專案，專注於：
  1. **教學價值**：清晰展示不同優化層級的效果
  2. **問題深度**：深入解決 Shared Memory 瓶頸
  3. **模組化設計**：可獨立測試各優化技術

**性能對比**（僅供參考）：
* FAISS IVF-Flat (GPU)：~100K QPS（k=10, nprobe=1）
* ZenANN (CUDA)：~83K QPS
* **差距合理**：FAISS 使用了更多底層優化（如 cuBLAS GEMM）

---

## 8. Related Work (相關研究)

Approximate Nearest Neighbor (ANN) 搜尋是一個成熟的研究領域，主要有三大範式：

### 8.1 三大 ANN 範式

1. **Hashing-based（基於哈希）**
   - 代表：LSH (Locality-Sensitive Hashing)
   - 特點：查詢速度快，但召回率較低
   - 適用：超高維空間（>1000 維）

2. **Graph-based（基於圖）**
   - 代表：HNSW (Hierarchical Navigable Small World)
   - 特點：高召回率，記憶體效率高
   - 缺點：不規則訪問模式，GPU 加速困難

3. **Clustering/Quantization-based（基於聚類/量化）**
   - 代表：IVF, PQ, IVFPQ
   - 特點：**記憶體效率高，適合 GPU 加速**
   - 本專案選擇：IVF-Flat

### 8.2 與 FAISS 的關係

**FAISS（Facebook AI Similarity Search）**是目前最成熟的 IVF 實作：
* 提供高度優化的 CPU 和 GPU 索引
* 使用 cuBLAS GEMM 優化質心距離計算
* 支援多種量化方法（PQ, SQ, OPQ）

**ZenANN 的定位**：
* **不是競爭關係**：我們不打算取代 FAISS
* **學術探索**：深入研究 GPU 加速的基礎挑戰
* **獨立驗證**：獨立解決 Shared Memory 瓶頸問題
* **教育價值**：清晰展示優化過程和權衡決策

**共同挑戰**：
* 我們解決的 Shared Memory 瓶頸，FAISS 也必須處理
* 我們的 Hybrid Kernel 設計，是對這類硬體感知優化問題的一種解法

---

## 9. Conclusion (總結)

*   **成果總結:**
    1.  我們成功診斷出一個基於 GPU 的 IVF-Flat 實作中由硬體限制引發的 Shared Memory 瓶頸。
    2.  通過**重新設計 CUDA Kernel**，採用**兩階段合併**與**基於 Heap 的流式選擇**等平行化策略，我們成功解除了演算法對 `k` 值的限制。
    3.  效能評估證實，我們的方案在犧牲極小的效能開銷下，大幅提升了演算法的**可用性與擴展性**，使其能勝任更多元的真實世界應用場景。

*   **專案心得 (Takeaways):**
    *   深刻體會到在平行程式設計中，**演算法必須與硬體架構深度配合**。單純的邏輯正確是不夠的，必須考慮如 Shared Memory、快取、記憶體頻寬等硬體限制。
    *   **沒有萬能的優化**。我們用 Global Memory 頻寬和額外的 Kernel 啟動開銷，換取了 Shared Memory 的使用彈性，這是一個基於明確目標的工程權衡 (trade-off)。

*   **組員貢獻 (Contributions):**
    *   Group 4
    *   周哲瑋: Modular OpenMP & SIMD Integration，Benchmark script，evaluation SIFT1M data for CPU，CUDA optimization, Documentation
    *   邱德晏: Project base，OpenMP & SIMD Implementation，CUDA implementation，evaluation SIFT1M data for GPU 
    *   陳冠霖: Presentation，Result Visualization，CUDA optimization，Documentation Refinement，evaluation GIST1M data
