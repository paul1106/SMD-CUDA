/* -*-c++-*- SemiGlobalMatching CUDA - Copyright (C) 2025.
* Implementation: Winner-Takes-All (WTA) for Disparity Map Generation
* Target: NVIDIA RTX 4090 (Compute Capability 8.9)
*/

#include "../include/sgm_cuda_phase4.cuh"

/**
 * Winner-Takes-All Kernel
 * 
 * For each pixel, find the disparity with minimum aggregated cost.
 * 
 * Grid: (width/32, height/32, 1) - 2D grid covering all pixels
 * Block: (32, 32, 1) - Each thread processes one pixel
 * 
 * Output: Disparity map scaled to 0-255 for visualization
 */
__global__ void winner_takes_all_kernel(
    const uint16_t* __restrict__ total_cost_volume,  // [H][W][D]
    unsigned char* __restrict__ disparity_map,        // [H][W]
    int width,
    int height,
    int max_disparity
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Find disparity with minimum cost
    uint16_t min_cost = UINT16_MAX;
    int best_disparity = 0;
    
    const int base_idx = y * (width * max_disparity) + x * max_disparity;
    
    for (int d = 0; d < max_disparity; d++) {
        uint16_t cost = total_cost_volume[base_idx + d];
        if (cost < min_cost) {
            min_cost = cost;
            best_disparity = d;
        }
    }
    
    // Scale disparity to 0-255 for visualization
    unsigned char scaled_disp = (unsigned char)((best_disparity * 255) / max_disparity);
    
    disparity_map[y * width + x] = scaled_disp;
}

/**
 * Winner-Takes-All Kernel with Sub-pixel Refinement
 * 
 * Uses quadratic interpolation for sub-pixel accuracy:
 * d_refined = d + (C(d-1) - C(d+1)) / (2 * (C(d-1) - 2*C(d) + C(d+1)))
 * 
 * Grid: (width/32, height/32, 1)
 * Block: (32, 32, 1)
 * 
 * Output: Float disparity map with sub-pixel precision
 */
__global__ void winner_takes_all_subpixel_kernel(
    const uint16_t* __restrict__ total_cost_volume,  // [H][W][D]
    float* __restrict__ disparity_map,                // [H][W]
    int width,
    int height,
    int max_disparity
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Find disparity with minimum cost
    uint16_t min_cost = UINT16_MAX;
    int best_disparity = 0;
    
    const int base_idx = y * (width * max_disparity) + x * max_disparity;
    
    for (int d = 0; d < max_disparity; d++) {
        uint16_t cost = total_cost_volume[base_idx + d];
        if (cost < min_cost) {
            min_cost = cost;
            best_disparity = d;
        }
    }
    
    // Sub-pixel refinement using quadratic interpolation
    float refined_disparity = best_disparity;
    
    if (best_disparity > 0 && best_disparity < max_disparity - 1) {
        float c_prev = total_cost_volume[base_idx + best_disparity - 1];
        float c_curr = total_cost_volume[base_idx + best_disparity];
        float c_next = total_cost_volume[base_idx + best_disparity + 1];
        
        // Quadratic interpolation formula
        float numerator = c_prev - c_next;
        float denominator = 2.0f * (c_prev - 2.0f * c_curr + c_next);
        
        if (fabsf(denominator) > 1e-5f) {
            float delta = numerator / denominator;
            // Clamp delta to reasonable range [-1, 1]
            delta = fmaxf(-1.0f, fminf(1.0f, delta));
            refined_disparity = best_disparity + delta;
        }
    }
    
    disparity_map[y * width + x] = refined_disparity;
}

/**
 * Host Wrapper: Launch WTA to generate disparity map
 */
void launch_wta(
    const uint16_t* d_total_cost_volume,
    unsigned char* d_disparity_map,
    int width,
    int height,
    int max_disparity
) {
    dim3 block(32, 32, 1);
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y, 1);
    
    winner_takes_all_kernel<<<grid, block>>>(
        d_total_cost_volume, d_disparity_map,
        width, height, max_disparity);
    
    cudaDeviceSynchronize();
}

/**
 * Host Wrapper: Launch WTA with sub-pixel refinement
 */
void launch_wta_subpixel(
    const uint16_t* d_total_cost_volume,
    float* d_disparity_map,
    int width,
    int height,
    int max_disparity
) {
    dim3 block(32, 32, 1);
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y, 1);
    
    winner_takes_all_subpixel_kernel<<<grid, block>>>(
        d_total_cost_volume, d_disparity_map,
        width, height, max_disparity);
    
    cudaDeviceSynchronize();
}
