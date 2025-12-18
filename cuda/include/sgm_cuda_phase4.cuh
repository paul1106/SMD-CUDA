/* -*-c++-*- SemiGlobalMatching CUDA - Copyright (C) 2025.
* Phase 4: Winner-Takes-All (WTA) Disparity Computation
* Target: NVIDIA RTX 4090 (Compute Capability 8.9)
*/

#ifndef SGM_CUDA_PHASE4_CUH
#define SGM_CUDA_PHASE4_CUH

#include "sgm_cuda_common.cuh"
#include <stdint.h>

/**
 * @brief Winner-Takes-All kernel (integer disparity)
 * 
 * For each pixel (x,y), finds the disparity d that minimizes the aggregated cost.
 * Output is scaled to 0-255 for visualization.
 * 
 * Grid: (width/32, height/32, 1)
 * Block: (32, 32, 1) - Each thread processes one pixel
 * 
 * @param total_cost_volume Input aggregated costs from Phase 2+3 [H][W][D]
 * @param disparity_map Output disparity map [H][W], scaled to 0-255
 * @param width Image width
 * @param height Image height
 * @param max_disparity Maximum disparity value
 */
__global__ void winner_takes_all_kernel(
    const uint16_t* __restrict__ total_cost_volume,
    unsigned char* __restrict__ disparity_map,
    int width,
    int height,
    int max_disparity
);

/**
 * @brief Winner-Takes-All kernel with sub-pixel refinement
 * 
 * Uses quadratic interpolation for sub-pixel accuracy.
 * Formula: d_refined = d + (C(d-1) - C(d+1)) / (2 * (C(d-1) - 2*C(d) + C(d+1)))
 * 
 * Grid: (width/32, height/32, 1)
 * Block: (32, 32, 1)
 * 
 * @param total_cost_volume Input aggregated costs [H][W][D]
 * @param disparity_map Output float disparity map [H][W]
 * @param width Image width
 * @param height Image height
 * @param max_disparity Maximum disparity value
 */
__global__ void winner_takes_all_subpixel_kernel(
    const uint16_t* __restrict__ total_cost_volume,
    float* __restrict__ disparity_map,
    int width,
    int height,
    int max_disparity
);

/**
 * @brief Host wrapper for WTA (integer disparity)
 * 
 * Launches WTA kernel to generate disparity map from aggregated costs.
 * Output is uint8 scaled to 0-255 for direct visualization.
 * 
 * @param d_total_cost_volume Device pointer to aggregated costs
 * @param d_disparity_map Device pointer to output disparity map (uint8)
 * @param width Image width
 * @param height Image height
 * @param max_disparity Maximum disparity value
 */
void launch_wta(
    const uint16_t* d_total_cost_volume,
    unsigned char* d_disparity_map,
    int width,
    int height,
    int max_disparity
);

/**
 * @brief Host wrapper for WTA with sub-pixel refinement
 * 
 * Launches WTA kernel with quadratic interpolation for sub-pixel accuracy.
 * Output is float32 for higher precision.
 * 
 * @param d_total_cost_volume Device pointer to aggregated costs
 * @param d_disparity_map Device pointer to output disparity map (float)
 * @param width Image width
 * @param height Image height
 * @param max_disparity Maximum disparity value
 */
void launch_wta_subpixel(
    const uint16_t* d_total_cost_volume,
    float* d_disparity_map,
    int width,
    int height,
    int max_disparity
);

#endif // SGM_CUDA_PHASE4_CUH
