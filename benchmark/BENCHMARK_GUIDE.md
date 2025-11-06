# ZenANN 綜合評估指南

本指南說明如何使用 `comprehensive_bench.py` 完成專案要求的所有評估指標。

---

## 📋 評估指標覆蓋

### ✅ 所有指標均已支援

| 評估項目 | 支援狀態 | 工具 |
|----------|----------|------|
| **資料集** |
| SIFT1M (128D) | ✅ | comprehensive_bench.py |
| GIST1M (960D) | ✅ | comprehensive_bench.py |
| **準確率** |
| Recall@1 | ✅ | comprehensive_bench.py |
| Recall@10 | ✅ | comprehensive_bench.py |
| Recall@100 | ✅ | comprehensive_bench.py |
| **性能** |
| QPS | ✅ | comprehensive_bench.py |
| p50 latency | ✅ | comprehensive_bench.py |
| p95 latency | ✅ | comprehensive_bench.py |
| **索引成本** |
| Index build time | ✅ | comprehensive_bench.py |
| bytes/vector | ✅ | comprehensive_bench.py |
| **視覺化** |
| Recall-QPS curve | ✅ | plot_tradeoff.py |

---

## 🚀 快速開始

### 步驟 1: 準備數據集

```bash
# 創建數據目錄
mkdir -p data

# 下載 SIFT1M
cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzvf sift.tar.gz

# 下載 GIST1M
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xzvf gist.tar.gz

cd ..
```

### 步驟 2: 安裝依賴

```bash
pip install psutil matplotlib numpy
```

### 步驟 3: 運行 Benchmark

```bash
# 設定環境變數
export LD_LIBRARY_PATH=extern/faiss/build/install/lib:$LD_LIBRARY_PATH

# SIFT1M 測試
python3 benchmark/comprehensive_bench.py \
    --base data/sift/sift_base.fvecs \
    --query data/sift/sift_query.fvecs \
    --groundtruth data/sift/sift_groundtruth.ivecs \
    --nlist 1024 \
    --nprobe-list "1,2,4,8,16,32,64,128,256" \
    --k-list "1,10,100" \
    --index-file sift_index.bin \
    --output-dir benchmark_results

# GIST1M 測試（可選,要注意會花相當多時間）
python3 benchmark/comprehensive_bench.py \
    --base data/gist/gist_base.fvecs \
    --query data/gist/gist_query.fvecs \
    --groundtruth data/gist/gist_groundtruth.ivecs \
    --nlist 1024 \
    --nprobe-list "1,4,16,64,256,512" \
    --k-list "1,10,100" \
    --index-file gist_index.bin \
    --output-dir benchmark_results
```

### 步驟 4: 生成報告和圖表

```bash
# 生成 Recall-QPS 曲線
python3 benchmark/plot_tradeoff.py benchmark_results/*.json
```

輸出文件：
- `recall_qps_tradeoff.png` - Recall vs QPS 曲線（3 個子圖，對應 k=1,10,100）
- `latency_distribution.png` - 延遲分析圖
- `benchmark_report.txt` - 文字報告

---

## 📊 輸出指標說明

### Console 輸出範例

```
======================================================================
Testing: nlist=1024, nprobe=16
======================================================================
Measuring batch QPS (k=100)...
  QPS (batch): 2450.32
  Latency - Mean: 0.408 ms
  Latency - p50: 0.385 ms
  Latency - p95: 0.612 ms
  Latency - p99: 0.758 ms
Computing Recall@k...
  Recall@1: 84.52%
  Recall@10: 95.28%
  Recall@100: 99.15%

======================================================================
SUMMARY: Recall-QPS Trade-off
======================================================================
nprobe   QPS        p50(ms)    p95(ms)    R@1        R@10       R@100
----------------------------------------------------------------------
1        12450.3    0.080      0.125      32.15      42.58      58.23
2        8920.5     0.112      0.185      52.34      65.87      78.45
4        5630.2     0.178      0.295      68.92      82.15      89.67
8        3580.1     0.279      0.448      79.45      91.23      95.82
16       2450.3     0.408      0.612      84.52      95.28      98.15
32       1680.5     0.595      0.891      87.89      97.45      99.32

✅ Target achieved: Recall@10 = 95.28% >= 95%
   Best config: nprobe=16, QPS=2450.3
```

### JSON 輸出

```json
{
  "metadata": {
    "dataset": "sift",
    "n_base": 1000000,
    "n_queries": 10000,
    "dimension": 128,
    "nlist": 1024,
    "nprobe_list": [1, 2, 4, 8, 16, 32],
    "k_values": [1, 10, 100],
    "build_time_sec": 45.234,
    "bytes_per_vector": 8.5,
    "timestamp": "20251106_150000"
  },
  "results": [
    {
      "nlist": 1024,
      "nprobe": 16,
      "qps_batch": 2450.32,
      "latency_mean_ms": 0.408,
      "latency_p50_ms": 0.385,
      "latency_p95_ms": 0.612,
      "latency_p99_ms": 0.758,
      "recall@1": 0.8452,
      "recall@10": 0.9528,
      "recall@100": 0.9915,
      "memory_mb": 128.5,
      "build_time_sec": 45.234,
      "bytes_per_vector": 8.5
    }
  ]
}
```

---

## 整體範例

### 以 openMP 為例

```bash
export LD_LIBRARY_PATH=extern/faiss/build/install/lib:$LD_LIBRARY_PATH

# 1. 測試 OpenMP
## "Testing OpenMP version..."
git checkout feature/openMP
make clean && make

python3 benchmark/comprehensive_bench.py \
    --base data/sift/sift_base.fvecs \
    --query data/sift/sift_query.fvecs \
    --groundtruth data/sift/sift_groundtruth.ivecs \
    --nlist 1024 \
    --nprobe-list "1,4,8,16,32,64" \
    --k-list "1,10,100" \
    --index-file sift_openmp.bin \
    --output-dir results_openmp

# 2. 生成對比圖表 (但注意要指定正確的 json，或是把之前的 json 清理)
python3 benchmark/plot_tradeoff.py \
    results_baseline/sift*.json \
    results_openmp/sift*.json

## "Done! Check recall_qps_tradeoff.png and benchmark_report.txt"
```

---

## 📈 預期結果

### Recall@10 ≥ 0.95 達成條件

根據文獻，對於 SIFT1M：

| nlist | nprobe | 預期 Recall@10 | 預期 QPS (baseline) |
|-------|--------|----------------|---------------------|
| 1024  | 16     | ~95%           | ~2000               |
| 1024  | 32     | ~97%           | ~1200               |
| 2048  | 32     | ~96%           | ~1500               |

### OpenMP 加速比預期

| 指標 | Baseline | OpenMP (8核心) | 加速比 |
|------|----------|----------------|--------|
| QPS  | 2000     | 8000-12000     | 4-6x   |
| p95 latency | 0.5 ms | 0.15 ms    | 3-4x   |
| Build time | 45 s  | 45 s          | 1x (未優化) |

---

## 🔍 故障排除

### 問題 1: Recall 太低

**症狀**：即使 nprobe=256 也達不到 95%

**解決**：
```bash
# 增加 nlist
python3 comprehensive_bench.py ... --nlist 2048
```

### 問題 2: QPS 沒有提升

**症狀**：OpenMP 版本 QPS 與 baseline 相同

**檢查**：
```bash
# 確認 OpenMP 編譯標誌
cat Makefile | grep fopenmp

# 確認運行時線程數
export OMP_NUM_THREADS=8
```

### 問題 3: 記憶體不足

**症狀**：`MemoryError` 或程序被殺

**解決**：使用索引文件避免重複建構
```bash
# 先建構並保存索引
python3 comprehensive_bench.py ... --index-file sift.bin

# 後續測試重用索引（跳過 build）
python3 comprehensive_bench.py ... --index-file sift.bin
```

---

## 💡 進階使用

### 自定義 nprobe 掃描範圍

```bash
# 細粒度掃描（找到精確的 Recall@10=95% 點）
--nprobe-list "10,12,14,16,18,20,22,24"

# 粗粒度掃描（快速探索）
--nprobe-list "1,8,64,512"
```

### 測試不同 nlist 配置

```bash
# 對比不同 nlist
for nlist in 512 1024 2048; do
    python3 comprehensive_bench.py \
        ... \
        --nlist $nlist \
        --output-dir results_nlist${nlist}
done

# 統一繪圖對比
python3 plot_tradeoff.py results_nlist*/*.json
```

---

## 📚 相關文檔

- `comprehensive_bench.py --help` - 完整參數說明
- `plot_tradeoff.py --help` - 繪圖工具說明
- `ivf-bench.py` - 簡化版測試（向下兼容）

---

## ✅ 檢查清單

完成評估前確認：

- [ ] SIFT1M 數據集已下載
- [ ] GIST1M 數據集已下載（可選）
- [ ] 已安裝 psutil, matplotlib
- [ ] 生成了 Recall-QPS 曲線圖
- [ ] 確認 Recall@10 ≥ 95% 在合理的 QPS 下達成
- [ ] 記錄了 OpenMP 加速比
- [ ] 保存了所有 JSON 結果文件
