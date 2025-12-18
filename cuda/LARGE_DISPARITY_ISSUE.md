# Large Disparity Range Issue Report

## 問題概述

在實現 CUDA 加速的 Semi-Global Matching 算法時，發現當視差範圍（disparity range）較大時（如 Motorcycle 資料集的 270），GPU 版本與 CPU 版本的結果出現顯著差異（42.63% 錯誤率），而小視差範圍（如 cone 的 64）則完全正確（0% 錯誤率）。

## 測試結果對比

| Dataset | Resolution | Disparity | CPU Time (ms) | GPU Time (ms) | Speedup | Error Rate | Status |
|---------|------------|-----------|---------------|---------------|---------|------------|--------|
| **cone** | 450×375 | 64 | 97 | 1 | 97× | **0.00%** | ✅ PASS |
| **Reindeer** | 671×555 | 128 | 416 | 3 | 139× | **0.00%** | ✅ PASS |
| **Cloth3** | 626×555 | 64 | 220 | 2 | 110× | **0.00%** | ✅ PASS |
| **Wood2** | 640×553 | 64 | 227 | 2 | 114× | **0.00%** | ✅ PASS |
| **Motorcycle** | 2964×2000 | **270** | 14,460 | 87 | 166× | **42.63%** | ❌ FAIL |

## 關鍵觀察

### 1. 視差範圍門檻
- **Disparity ≤ 128**: 完全正確（0% 錯誤）
- **Disparity = 270**: 嚴重錯誤（42.63% 錯誤）
- 問題不在於圖片大小，而在於 disparity 範圍本身

### 2. 錯誤模式
```
Mismatch at (3,0,d=0): CPU=10, GPU=120, diff=110
Mismatch at (3,0,d=1): CPU=10, GPU=130, diff=120
Mismatch at (3,0,d=2): CPU=30, GPU=140, diff=110
Mismatch at (3,0,d=3): CPU=50, GPU=150, diff=100
```

- 錯誤從第一個像素 (3,0) 就開始出現
- GPU 的值系統性地偏大（約 +110~120）
- 錯誤分佈均勻（42.63%），不是局部問題

### 3. CUDA 限制
- CUDA block 的單一維度最大值：**1024 threads**
- Motorcycle disparity = 270 < 1024，理論上不需要特殊處理
- 但實際配置：`block(270, 1, 1)` 仍然失敗

## 技術分析

### 原始實現（小 disparity 正常工作）
```cuda
// Grid: (height, 1, 1)
// Block: (disparity, 1, 1)  // 每個 thread 處理一個 disparity
__global__ void aggregate_horizontal_path_kernel(...) {
    extern __shared__ uint16_t prev_L[];
    const int d = threadIdx.x;  // 一對一映射
    
    // 訪問相鄰 disparity
    prev_L[d-1];  // ✅ 由 thread d-1 負責
    prev_L[d+1];  // ✅ 由 thread d+1 負責
}
```

### 嘗試的解決方案

#### 方案 1: Strided Access（失敗）
```cuda
// 當 disparity > 1024 時，每個 thread 處理多個 disparities
for (int d = threadIdx.x; d < max_disparity; d += num_threads) {
    // ❌ Race condition:
    // Thread 0 讀取 prev_L[1]（作為 d=0 的 term3）
    // Thread 1 同時寫入 prev_L[1]（處理 d=1）
    prev_L[d] = new_L;
}
```

**問題**: 在 strided loop 內部，讀取 `prev_L[d±1]` 時可能與其他線程的寫入衝突。

#### 方案 2: Double Buffering（仍然失敗）
```cuda
extern __shared__ uint16_t smem[];
uint16_t* prev_L = smem;
uint16_t* next_L = smem + max_disparity;

// Read from prev_L, write to next_L
for (int d = threadIdx.x; d < max_disparity; d += num_threads) {
    term2 = prev_L[d-1];  // 讀取
    next_L[d] = new_L;    // 寫入不同 buffer
}

// Swap buffers
uint16_t* tmp = prev_L; prev_L = next_L; next_L = tmp;
```

**結果**: Motorcycle (disparity=270) 仍然 42.63% 錯誤率，但 cone (64) 仍然 0% 錯誤。

### 奇怪的現象

1. **Disparity=270 不需要 strided access**
   - 270 < 1024，每個 disparity 都有對應的 thread
   - 理論上應該和 disparity=64 的邏輯完全相同
   - 但結果完全不同

2. **Shared Memory 配置**
   - 打印顯示：`Shared Memory: 1080 bytes per block`
   - 計算：270 × 2 (uint16_t) × 2 (double buffer) = 1080 ✅ 正確
   - RTX 4090 的 shared memory per block: 48KB >> 1080 bytes

3. **系統性偏差**
   - GPU 結果系統性地比 CPU 大 ~110
   - 這暗示可能是參數問題（如 P1, P2）或初始化問題
   - 但為什麼只在大 disparity 時出現？

## 可能的根本原因

### 假設 1: Vertical Aggregation 的 Transpose
- Vertical aggregation 使用 transpose 策略
- 當 disparity 變大時，transpose 矩陣也變大
- 可能的 memory layout 問題？

### 假設 2: Shared Memory Bank Conflicts
- 當 disparity > 某個閾值時，bank conflicts 急劇增加
- 導致非預期的數據讀寫順序
- 但這不應該影響正確性，只影響性能

### 假設 3: 編譯器優化問題
- 對於大 disparity，編譯器可能做了不同的優化
- Template instantiation 在 disparity > threshold 時行為改變？

### 假設 4: 初始化錯誤
- 錯誤從 (3,0) 開始，非常早期
- 可能是第一個 pixel 的初始化就有問題
- 但為什麼小 disparity 沒問題？

## 下一步調試方向

### 優先級 1: 隔離測試
1. ✅ 測試只有 horizontal aggregation（不含 vertical）
2. ⬜ 測試只有 vertical aggregation（不含 horizontal）
3. ⬜ 測試 disparity = 128 vs 129 vs 256 vs 257，找出確切的閾值

### 優先級 2: 深度檢查
1. ⬜ 打印 GPU 和 CPU 的初始化階段（第一個 pixel 的 prev_L）
2. ⬜ 打印前幾個 pixel 的 aggregation 結果
3. ⬜ 檢查 min_prev 的值是否正確

### 優先級 3: 算法驗證
1. ⬜ 回退到單一 buffer（不用 double buffering），測試 disparity=270
2. ⬜ 使用 global memory 而非 shared memory，測試正確性
3. ⬜ 簡化 kernel：移除 template，硬編碼方向

## 臨時解決方案

### 選項 A: 限制 Disparity 範圍
```cpp
if (max_disparity > 128) {
    fprintf(stderr, "Warning: Disparity > 128 not fully validated. Use CPU fallback.\n");
    // Fall back to CPU implementation
}
```

### 選項 B: 分段處理
```cpp
// 將大 disparity 拆分成多個小塊
const int MAX_D_PER_BLOCK = 128;
for (int d_start = 0; d_start < max_disparity; d_start += MAX_D_PER_BLOCK) {
    int d_end = min(d_start + MAX_D_PER_BLOCK, max_disparity);
    // Process [d_start, d_end) range
}
```

### 選項 C: 接受誤差
```cpp
// 如果性能仍然優異（166× speedup），且視覺效果可接受
// 可以標註為 "beta feature" 並記錄已知問題
```

## 效能影響

即使存在 42% 的數值誤差，Motorcycle 測試仍然達到：
- **Aggregation 加速**: 176× (13400ms → 76ms)
- **WTA 加速**: 96× (1060ms → 11ms)  
- **總體加速**: 166× (14460ms → 87ms)

視差圖已成功生成（14MB PNG），視覺質量待評估。

## 結論

這是一個與 disparity 範圍相關的邊界條件 bug，而非圖片大小問題。問題出現在 disparity > 128 的情況，雖然實現了雙緩衝來避免 race conditions，但仍有未知因素導致 42.63% 的數值誤差。需要進一步隔離測試來定位根本原因。

---

## ✅ 問題已解決！

### Root Cause (根本原因)
當 `max_disparity` 不是 32 的倍數時（如 270），最後一個 warp 只有部分線程活躍（270 % 32 = 14 個活躍線程）。但 `block_min_reduce()` 中使用 `__shfl_down_sync(0xffffffff, ...)` 假設所有 32 個 lane 都活躍，導致從不存在的 lane 讀取到未定義值！

**為什麼小 disparity 沒問題？**
- disparity=64: 64 = 32×2 ✅ (2 個完整 warp)
- disparity=128: 128 = 32×4 ✅ (4 個完整 warp)
- disparity=270: 270 = 32×8 + 14 ❌ (8.4375 個 warp，最後 warp 不完整)

### Solution: Host-side Padding Strategy

**實施方案**:
1. **Host Wrapper**: Round up block size to next multiple of 32
   ```cpp
   int threads_per_block = ((max_disparity + 31) / 32) * 32;  // 270 → 288
   ```

2. **Kernel**: Padding threads 設為 `UINT16_MAX`
   ```cuda
   if (d < max_disparity) {
       // Normal computation
   } else {
       prev_L[d] = UINT16_MAX;  // Won't affect min reduction
   }
   ```

3. **Key Insight**: ALL threads (including padding) must participate in `block_min_reduce()`, but padding threads don't write to global memory.

### Final Results

| Dataset | Disparity | Threads | Error Rate | Status |
|---------|-----------|---------|------------|--------|
| cone | 64 | 64 (2 warps) | 0.00% | ✅ PASS |
| Reindeer | 128 | 128 (4 warps) | 0.00% | ✅ PASS |
| **Motorcycle** | **270** | **288 (9 warps)** | **0.00%** | ✅ **PASS** |

**Motorcycle 性能**:
- Aggregation: 178× speedup (13.4s → 75ms)
- WTA: 96× speedup (1.06s → 11ms)
- **Total: 168× speedup** (14.4s → 86ms) ⚡

---

**Date**: 2025-12-18  
**GPU**: NVIDIA RTX 4090 (Compute Capability 8.9)  
**CUDA Version**: 12.x  
**Status**: ✅ RESOLVED - Host-side Padding Strategy Implemented
