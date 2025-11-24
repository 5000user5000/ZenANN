# IVF-Flat GPU Search 實作說明

## 0. 問題範圍與前提

### 0.1 你要做的是什麼？

我們要把一個已經訓練好的 **IVF Index（coarse quantizer + inverted lists，沒有 PQ 壓縮）**，
在 **GPU 上實作 search 階段**，支援：

* 輸入：一批查詢向量（queries），維度為 `d`
* 動作：

  1. 對所有 coarse centroids 算距離 → 選前 `nprobe` 個 list
  2. 在被選中的 lists 裡面，掃描所有向量，計算距離
  3. 為每個 query 找出 top-`k` 近鄰
* 輸出：每個 query 的 top-k IDs 和 distance

### 0.2 重要假設

* Feature 維度：`d`（通常 64, 128, 256…）
* Coarse centroids 個數：`nlist`
* 總向量數：`N_total`
* GPU 端 index 為常駐（常態情況下一次 load 上去，重複用很多次查詢）

距離型式可以是：

* L2：`||q - x||^2`
* 或 inner product：`- q·x`（當成「距離」，越小越好）

以下說明以 L2 為例，inner product 只是少一個 norm term。

---

## 1. Index 在 GPU 上的資料佈局

實作效率的 80% 來自這一段。

### 1.1 Coarse centroids

建議 layout（row-major）：

* `coarseCentroids`：大小 `nlist * d`，排列為 `[c0[0..d-1], c1[0..d-1], ..., c_{nlist-1}[0..d-1]]`
* `coarseNorms`（可選）：大小 `nlist`，裡面存 `||c_i||^2`，查詢時少算一部分 FLOPs

> 為什麼這樣排：
> 一個 centroid 是一坨連續資料，方便一個 warp 拿來做內積或 L2。

### 1.2 Inverted lists：向量與 ID

IVF-Flat 的 inverted list 通常就兩件事：

1. **所有向量**本體
2. 每個 list 的 offset / size

建議在 GPU 上：

* `vectors`：大小 `N_total * d`

  * 所有 list 的向量集中在一個大陣列裡
  * 某個向量 `x_idx` 在 array 中的起始位置是 `vectors[idx * d]`
* `ids`：大小 `N_total`

  * `ids[idx]` 是該向量對應的全域 vector id
* `listOffsets`：大小 `nlist + 1`

  * 第 `i` 個 list 的 index range 是 `[listOffsets[i], listOffsets[i+1])`，
  * list size = `listOffsets[i+1] - listOffsets[i]`

> 直覺：
> 每個 list 就是大陣列上一段連續區間。要掃描一個 list，就是掃這段的所有 row（每 row 一個向量）。

---

## 2. Host 端整體 Search 流程

你可以想像一個高階 API：

> 給定一批 query，回傳各自的 top-k。

### 2.1 輸入 / 輸出定義

* 輸入：

  * `queries`（host）：大小 `Q * d`，`Q` 是 query 個數
  * `nprobe`：每個 query 要探訪幾個 list
  * `k`：要找幾個近鄰
* 輸出：

  * `outIds`（host）：`Q * k`
  * `outDists`（host）：`Q * k`

### 2.2 Pipeline 分步

對於一次 query batch：

1. **H2D**：把 queries 複製到 GPU（`d_queries`）
2. **Kernel A**：算每個 query 對所有 centroids 的距離 → `d_coarseDists`（`Q * nlist`）
3. **Kernel B**：從 `nlist` 個 centroids 裡選出該 query 的 top-`nprobe` → `d_nprobeIdx`（`Q * nprobe`）
4. **Kernel C**：掃描 inverted lists，對所有候選向量算距離，產生各 `(query, list)` 的 local top-k → `d_partialTopk`
5. **Kernel D**：對每個 query 的所有 partial top-k 做 merge → `d_outIds`, `d_outDists`
6. **D2H**：結果 copy 回 host

---

## 3. Kernel 設計邏輯（用文字講清楚）

### 3.1 Kernel A：計算 queries × centroids 距離

**目的**：對每個 query，算它到所有 `nlist` centroids 的距離。

#### 3.1.1 設計理念

* Grid：

  * `blockIdx.x` = query index `q`
* Block：

  * `threadIdx.x` 負責不同 centroids（或不同維度的 slice）
* 典型做法：

  * 把 query `q` 的 `d` 維資料載入 shared memory（所有 threads 共用）
  * 每個 thread 在外圈迴圈中負責多個 centroids：

    * 對 centroid `c` 做 dot product / L2 累加
    * 利用 `||q||^2 + ||c||^2 − 2 q⋅c` 的公式

#### 3.1.2 核心運算風格

以 L2 為例：

* 先在 host 或另外一個小 kernel 算出：

  * `queryNorms[q] = ||q||^2`
  * `coarseNorms[c] = ||c||^2`（建 index 時就算好）

在 kernel 裡每一筆距離的計算變成：

* 先算 `dot(q, c)`
* 再做 `dist = queryNorms[q] + coarseNorms[c] − 2 * dot`

> 這樣可以把核心變成「大量 dot product」，對 GPU 很友善，未來可以轉 GEMM / tensor core。

---

### 3.2 Kernel B：每個 query 選出 top-nprobe 的 list

**目的**：從 `nlist` 個距離裡挑出最小的 `nprobe` 個。

#### 3.2.1 設計理念

* Grid：

  * 一個 block 處理一個 query
* Block 內：

  * 先把該 query 的 `nlist` 距離搬到 shared memory
  * 再在 shared memory 上做「部分排序」或「選擇演算法」（例如 quickselect + 小排序）
* 結果：

  * 輸出 `nprobe` 個 centroid index（`list id`）

#### 3.2.2 實作風格

* 為了簡單，可以一開始用：

  * 「在 shared memory BFS 型 partial sort」：

    * 先 copy 到 shared
    * 然後用一個簡單的 `nprobe`-pass，找到 `nprobe` 個最小值
    * 這樣是 O(nlist * nprobe)，但 nlist / nprobe 多數時候不會爆到不可用
* 未來可以換成：

  * 在 shared memory 上的 bitonic sort（對 nlist 不太大的情境）
  * 或使用 thrust / cub 的 device-level selection（但這會牽涉到 extra temp buffer 管理）

---

### 3.3 Kernel C：掃描 inverted lists（list scan）

**這是整個 search 的主體，實作要特別小心。**

#### 3.3.1 Grid mapping：用 (query, list) 二維 grid

設計：

* `blockIdx.x = query id (q)`
* `blockIdx.y = probe index (0..nprobe-1)`
  → 利用 `d_nprobeIdx[q * nprobe + probe]` 找到實際 `list id`

這樣的好處：

* 對每個 `(q, list)` 啟動一個 block，
  list 的不同長度會自然分散在不同 block 上，有利 load balance
* 也方便後面做「per-(q,list) local top-k」

#### 3.3.2 Block 裡要做的事情

對於 block 對應到 `(q, list)`：

1. 查出該 list 的範圍：

   * `start = listOffsets[list]`
   * `end   = listOffsets[list+1]`
2. 把 query `q` 的向量載入 shared memory（`d` 維）
3. 每個 thread 以 stride 方式掃這段 `[start, end)` 向量：

   * `for idx in [start + threadIdx.x, end, step blockDim.x]`：

     * 取出 `vectors[idx * d .. idx * d + d-1]`
     * 和 query 做 L2 distance
     * 更新 thread-local top-k
4. Block 結束前，把所有 thread 的 local top-k 收集到 shared memory，
   做一次 block-level top-k 合併，產生**該 `(q, list)` 的 k 個最佳候選**
5. 將這 k 個候選寫到 global memory 的 `partial` buffer：

   * 例如 `partialBase = (q * nprobe + probe) * k`

#### 3.3.3 thread-local top-k 怎麼做？

典型作法：

* 假設 `k` 不大（如 10, 20），
  每個 thread 用「**小陣列 + insertion sort**」維護 thread-local top-k：

  * 陣列大小可以 >= k（例如 `LOCAL_K=32`），
    這樣 thread 內候選稍微冗餘一點，讓 block 合併時選擇空間更大
* 每算出一個 distance 就試著插入：

  * 若陣列還沒滿 → 直接插入（保持排序）
  * 若陣列已滿，而且距離比最後一名還小 → 拿掉最後一名，插入適當位置

這段完全在 register 裡跑，效能很好。

#### 3.3.4 Block-level top-k 合併

做法（概念）：

1. 每個 thread 把自己的 thread-local top-k 搬到 shared memory 的一段區間：

   * 如果 thread 數是 `T`，每個 thread 有 `LOCAL_K` 候選 →
     總共 `T * LOCAL_K` 個候選
2. Block 內再做一次選擇：

   * 在 shared memory 上跑「small-N top-k」：

     * 可以是 bitonic sort（針對幾千以內元素）
     * 或者簡單的「重複在所有元素中找最小值，跑 k 次」（O(N * k)，N 是 `T * LOCAL_K`）

最後得到 block-level 的前 `k` 個候選，寫回 global memory。

---

### 3.4 Kernel D：對整個 query 的 partial top-k 做 merge

每個 query 對應 `nprobe` 個 list，每個 list 的 block 都給了你 `k` 個 candidate：

* 總共 `nprobe * k` 個候選
* 要在這裡整合成該 query 的 global top-k

設計：

* `blockIdx.x = query id (q)`
* Block 內：

  * 把該 query 對應的 `nprobe * k` 候選從 global memory 讀進 shared memory
  * 再在 shared memory 上跑一次 top-k
  * 最後把 `k` 個最佳寫到 `d_outIds[q * k .. q*k + k-1]` 和 `d_outDists[同樣範圍]`

實務上 `nprobe * k` 通常不會太大（例如 64 * 10 = 640），
所以這一階段的 cost 很小。

---

## 4. 查詢批次、大量併發與 stream 管理

### 4.1 Query batching（避免每次只丟一個 query）

GPU 的利用率很大一部分來自於：

* 一次 batch 送多個 query，例如 32～1024 個
* 所有 kernel 都以「`Q` 維」展開 grid（`blockIdx.x` 掃 query）

你可以設計一個「**固定 batch size 的 search**」：

1. 接收上層傳來的 `numQueries`，拆成多個 batch：

   * 每 batch 至多 `MAX_Q`（例如 256）
2. 每個 batch 跑一遍 A/B/C/D Kernels + H2D/D2H copy
3. 將這一 batch 的結果寫入最終 output buffer 的對應範圍

### 4.2 使用 CUDA streams 做 pipeline

如果要更進階，可以：

* 假設你有很多 batch
* 將不同 batch 的 kernel / H2D copy 交錯在不同 stream 上：

  * stream 0：batch 0 的 kernel
  * stream 1：batch 1 做 H2D copy
  * stream 2：batch −1 做 D2H copy
* 讓 GPU 計算與 PCIe 資料傳輸互相 overlap

---

## 5. 效能優化的重點 checklist

下面是你實作完一版 naive 之後，可以逐一檢查、優化的點：

### 5.1 記憶體讀寫是否 coalesced？

* `coarseCentroids`、`vectors` 都是 row-major，
  讓「不同 thread 處理不同 centroid / 向量，但在內層維度上是連續讀」
* 避免一個 thread 讀一條向量時寫成「每次 stride 很大」
  → 正確作法是固定一個 thread block 處理一個 query，
  block 中 threads 在維度上做「向量內的分工」，或在向量 index 上 stride

### 5.2 重複使用資料（shared memory）

* Query 本身常常會被同一個 block 重複用很多次：

  * 在 Kernel A（query × centroid）中：
    一個 block 處理一個 query → query 可以先載入 shared
  * 在 Kernel C（掃某 query 的某個 list）中：
    同樣是 query 向量，先載入 shared，再重複讀它跟多個向量做距離

### 5.3 減少運算：使用 L2 展開式

* 一次算好 `||q||^2` 和 `||x||^2`
* 計算距離只剩下「dot product + 加減」：

  * 對 GPU 來說，是最典型、最好優化的運算 pattern
* 如果你採 inner product search，很多情況下你可以直接用 `-q⋅x` 當距離，不需要 norms

### 5.4 利用半精度 / tensor cores（進階）

* 如果對 recall 要求允許，你可以：

  * 把向量與 centroids 存為 FP16
  * 在 dot product 時使用 FP16 input、FP32 accumulate
  * 或把 coarse matching 寫成 GEMM（查詢 batch × centroids），用 cublas + tensor cores
* 這一層先不做也可以，等 baseline version 跑起來再考慮

### 5.5 Load balancing：長短 list

* 現實中 inverted lists 的長度可能很不均：

  * 有的 list 幾百點、有的 list 幾十萬
* 如果你讓一個 block 吃完整個長 list，會導致嚴重 load 不均
* 解法：

  * 把 list 切成 chunk：

    * 例如每 `CHUNK_SIZE` 個向量為一段
    * 每個 `(q, list_chunk)` 用一個 block 處理
  * partial top-k 數量會變多，merge 階段稍微變複雜，但反而更均衡

---

## 6. 整體實作順序建議

如果你要從零開始寫，我會建議這樣排：

1. **先做最簡版，確保 correctness：**

   * 單個 query，用 GPU 算：

     1. coarse distance
     2. 手動選 nprobe（甚至先在 CPU 選）
     3. 掃 list，用 naive 寫法（每個 thread 做一條向量的距離，global top-k）
   * 先確保數值結果跟 CPU 版一致

2. **改成 batch query 與四個 kernel 分階段：**

   * Kernel A~D 拆清楚
   * grid/block mapping 確定下來

3. **引入 thread-local top-k + block-level merge：**

   * 避免大量 global atomic 或 global heap

4. **優化記憶體使用與 shared memory：**

   * query 放 shared
   * 擴充分工方式，讓 memory access 更 coalesced

5. **處理長尾 list（可選）：**

   * 若發現有某些 list 特別長導致整體 latency 被拖慢，再做 chunk 化 / persistent kernel

做到這裡，你就有一個「可工作的 IVF-Flat GPU Search 核心」，之後要再往 FP16 / GEMM / multi-GPU 進化都會比較自然。