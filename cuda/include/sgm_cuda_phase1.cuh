/* -*-c++-*- SemiGlobalMatching CUDA - Copyright (C) 2025.
* Phase 1: Census Transform & Matching Cost Computation
* Target: NVIDIA RTX 4090 (Compute Capability 8.9)
*/

#ifndef SGM_CUDA_PHASE1_CUH
#define SGM_CUDA_PHASE1_CUH

#include "sgm_cuda_common.cuh"
#include <stdint.h>

/**
 * @brief Census Transform Kernel (9x7 window: 9 rows x 7 columns)
 * 
 * Uses tile-based shared memory optimization to reduce global memory reads.
 * Each thread computes one Census value by comparing 63 neighbors with center pixel.
 * Result is bit-packed into uint64_t.
 * 
 * @param image Input grayscale image (flattened 1D array)
 * @param census Output Census buffer (uint64_t per pixel)
 * @param width Image width
 * @param height Image height
 */
__global__ void census_transform_kernel(
    const unsigned char* __restrict__ image,
    uint64_t* __restrict__ census,
    int width,
    int height
);

/**
 * @brief Matching Cost Kernel (Hamming Distance)
 * 
 * Computes matching cost between left and right Census transforms.
 * For each pixel (x,y) and disparity d, calculates Hamming distance
 * between censusL[x,y] and censusR[x-d,y] using __popcll() intrinsic.
 * 
 * Cost volume layout: pixel-major [y][x][d]
 * Index = y * (width * max_disparity) + x * max_disparity + d
 * 
 * @param censusL Left Census buffer
 * @param censusR Right Census buffer
 * @param cost_volume Output cost volume (unsigned char)
 * @param width Image width
 * @param height Image height
 * @param max_disparity Maximum disparity value
 */
__global__ void matching_cost_kernel(
    const uint64_t* __restrict__ censusL,
    const uint64_t* __restrict__ censusR,
    unsigned char* __restrict__ cost_volume,
    int width,
    int height,
    int max_disparity
);

/**
 * @brief Host wrapper for Phase 1 kernels
 * 
 * Launches Census Transform and Matching Cost kernels with proper
 * grid/block configuration for RTX 4090.
 * 
 * @param d_imL Device pointer to left image
 * @param d_imR Device pointer to right image
 * @param d_censusL Device pointer to left Census buffer (allocated by caller)
 * @param d_censusR Device pointer to right Census buffer (allocated by caller)
 * @param d_cost_volume Device pointer to cost volume (allocated by caller)
 * @param width Image width
 * @param height Image height
 * @param max_disparity Maximum disparity value
 */
void launch_phase1(
    const unsigned char* d_imL,
    const unsigned char* d_imR,
    uint64_t* d_censusL,
    uint64_t* d_censusR,
    unsigned char* d_cost_volume,
    int width,
    int height,
    int max_disparity
);

#endif // SGM_CUDA_PHASE1_CUH
