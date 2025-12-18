/* -*-c++-*- SemiGlobalMatching CUDA - Copyright (C) 2025.
* Phase 2: Cost Aggregation (Horizontal + Vertical)
* Target: NVIDIA RTX 4090 (Compute Capability 8.9)
*/

#ifndef SGM_CUDA_PHASE2_CUH
#define SGM_CUDA_PHASE2_CUH

#include "sgm_cuda_common.cuh"
#include <stdint.h>

// SGM penalty parameters
#define SGM_P1 10
#define SGM_P2 120

/**
 * @brief Horizontal Cost Aggregation Kernel
 * 
 * Implements SGM aggregation along horizontal scanlines (left-to-right or right-to-left).
 * Uses the classic SGM formula with path-wise cost accumulation.
 * 
 * Grid: (Height, 1, 1) - Each block processes one full row
 * Block: (Max_Disparity, 1, 1) - Each thread handles one disparity value
 * 
 * Algorithm per row:
 * 1. Initialize prev_L from first pixel's costs
 * 2. For each pixel x in the sweep direction:
 *    - Load raw cost C(p,d)
 *    - Compute 4 terms: L(d), L(d-1)+P1, L(d+1)+P1, min_k(L(k))+P2
 *    - Apply normalization: new_L = C + min(terms) - min_prev
 *    - Accumulate to output buffer
 *    - Update prev_L and compute new min_prev
 * 
 * @tparam Direction +1 for left-to-right, -1 for right-to-left
 * @param cost_volume Input raw matching costs (from Phase 1)
 * @param total_cost_volume Output accumulated costs (must be initialized to 0)
 * @param width Image width
 * @param height Image height
 * @param max_disparity Maximum disparity value
 */
template<int Direction>
__global__ void aggregate_horizontal_path_kernel(
    const unsigned char* __restrict__ cost_volume,
    uint16_t* __restrict__ total_cost_volume,
    int width,
    int height,
    int max_disparity
);

/**
 * @brief Host wrapper for Phase 2 horizontal aggregation
 * 
 * Launches horizontal aggregation kernels for both directions:
 * 1. Left-to-right pass
 * 2. Right-to-left pass
 * 
 * Results are accumulated in total_cost_volume.
 * 
 * @param d_cost_volume Device pointer to raw matching costs (from Phase 1)
 * @param d_total_cost_volume Device pointer to accumulated costs (must be pre-initialized to 0)
 * @param width Image width
 * @param height Image height
 * @param max_disparity Maximum disparity value
 */
void launch_phase2_horizontal(
    const unsigned char* d_cost_volume,
    uint16_t* d_total_cost_volume,
    int width,
    int height,
    int max_disparity
);

/**
 * @brief Host wrapper for Phase 2 vertical aggregation via transpose strategy
 * 
 * Uses the "Transpose Strategy" to maintain memory coalescing:
 * 1. Transpose cost volume [H][W][D] -> [W][H][D]
 * 2. Reuse horizontal aggregation kernel on transposed volume (U->D and D->U)
 * 3. Transpose result back and accumulate to total_cost_volume
 * 
 * This strategy avoids slow vertical memory access patterns by converting
 * vertical aggregation into horizontal aggregation on transposed data.
 * 
 * @param d_cost_volume Device pointer to raw matching costs (from Phase 1)
 * @param d_total_cost_volume Device pointer to accumulated costs (accumulates on top of existing)
 * @param width Image width
 * @param height Image height
 * @param max_disparity Maximum disparity value
 */
void launch_vertical_aggregation(
    const unsigned char* d_cost_volume,
    uint16_t* d_total_cost_volume,
    int width,
    int height,
    int max_disparity
);

#endif // SGM_CUDA_PHASE2_CUH
