/* -*-c++-*- SemiGlobalMatching CUDA - Copyright (C) 2025.
* Implementation: Matching Cost Kernel (Hamming Distance)
* Target: NVIDIA RTX 4090 (Compute Capability 8.9)
*/

#include "../include/sgm_cuda_phase1.cuh"

/**
 * Matching Cost Kernel - Computes Hamming distance between Census transforms
 * 
 * Strategy:
 * 1. Each thread handles one pixel (x, y)
 * 2. Iterates through all disparities [0, max_disparity)
 * 3. Calculates Hamming distance using __popcll() hardware intrinsic (single cycle)
 * 4. Stores result in pixel-major cost volume layout
 * 
 * Memory Layout:
 * - cost_volume[y][x][d] -> index = y * (width * max_disparity) + x * max_disparity + d
 * - This ensures all disparities for a pixel are contiguous (cache-friendly for aggregation)
 * 
 * Optimization Notes:
 * - Simple version: relies on L1 cache (128 KB on RTX 4090) for censusR reuse
 * - Could add shared memory caching for censusR in future optimization
 */
__global__ void matching_cost_kernel(
    const uint64_t* __restrict__ censusL,
    const uint64_t* __restrict__ censusR,
    unsigned char* __restrict__ cost_volume,
    int width,
    int height,
    int max_disparity
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Fetch left Census value once
    const uint64_t census_left = censusL[y * width + x];
    
    // Base offset for this pixel in cost volume
    const int cost_base = y * (width * max_disparity) + x * max_disparity;
    
    // Iterate through all disparities
    #pragma unroll 4
    for (int d = 0; d < max_disparity; d++) {
        unsigned char cost;
        
        // Check if corresponding right pixel is within bounds
        const int x_right = x - d;
        
        if (x_right >= 0) {
            // Fetch right Census value
            const uint64_t census_right = censusR[y * width + x_right];
            
            // Compute Hamming distance using hardware intrinsic
            // __popcll() counts set bits in 64-bit integer (single cycle on modern GPUs)
            const uint64_t xor_result = census_left ^ census_right;
            cost = static_cast<unsigned char>(__popcll(xor_result));
        } else {
            // Out of bounds: assign maximum penalty
            cost = INVALID_COST;
        }
        
        // Store in pixel-major layout
        cost_volume[cost_base + d] = cost;
    }
}
