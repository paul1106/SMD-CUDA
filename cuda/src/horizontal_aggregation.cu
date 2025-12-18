/* -*-c++-*- SemiGlobalMatching CUDA - Copyright (C) 2025.
* Implementation: Horizontal Cost Aggregation with Warp Shuffle Reduction
* Target: NVIDIA RTX 4090 (Compute Capability 8.9)
*/

#include "../include/sgm_cuda_phase2.cuh"

/**
 * Block-wide Min Reduction using Warp Shuffle
 * 
 * Strategy:
 * 1. Each thread holds one value (from prev_L[threadIdx.x])
 * 2. Use warp shuffle to find min within each warp (32 threads)
 * 3. First thread of each warp writes result to shared memory
 * 4. Warp 0 reduces the per-warp minimums
 * 5. Thread 0 broadcasts final result to all threads
 * 
 * Assumption: blockDim.x <= 1024 (valid for max_disparity <= 1024)
 */
__device__ __forceinline__ uint16_t block_min_reduce(uint16_t thread_val, int block_size) {
    __shared__ uint16_t warp_mins[32]; // Max 32 warps per block (1024 threads / 32)
    
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = (block_size + 31) / 32;
    
    // Step 1: Reduce within warp using shuffle
    uint16_t warp_min = thread_val;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        uint16_t other = __shfl_down_sync(0xffffffff, warp_min, offset);
        warp_min = min(warp_min, other);
    }
    
    // Step 2: First thread of each warp writes to shared memory
    if (lane_id == 0) {
        warp_mins[warp_id] = warp_min;
    }
    
    __syncthreads();
    
    // Step 3: Warp 0 reduces the per-warp minimums
    uint16_t block_min;
    if (warp_id == 0) {
        // Load from shared memory (or use a large value if out of range)
        warp_min = (lane_id < num_warps) ? warp_mins[lane_id] : UINT16_MAX;
        
        // Reduce within warp 0
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            uint16_t other = __shfl_down_sync(0xffffffff, warp_min, offset);
            warp_min = min(warp_min, other);
        }
        
        block_min = warp_min;
        
        // Thread 0 writes back to shared memory for broadcast
        if (lane_id == 0) {
            warp_mins[0] = block_min;
        }
    }
    
    __syncthreads();
    
    // Step 4: All threads read the final result
    return warp_mins[0];
}

/**
 * Horizontal Cost Aggregation Kernel (Template for Direction)
 * 
 * Direction: +1 (left-to-right), -1 (right-to-left)
 * 
 * Grid: (height, 1, 1)
 * Block: (max_disparity, 1, 1)
 */
template<int Direction>
__global__ void aggregate_horizontal_path_kernel(
    const unsigned char* __restrict__ cost_volume,
    uint16_t* __restrict__ total_cost_volume,
    int width,
    int height,
    int max_disparity
) {
    // Shared memory for previous pixel's aggregated costs
    extern __shared__ uint16_t prev_L[];
    
    const int y = blockIdx.x; // Each block processes one row
    const int d = threadIdx.x; // Each thread processes one disparity
    
    if (y >= height || d >= max_disparity) return;
    
    // Determine sweep direction
    const int x_start = (Direction > 0) ? 0 : (width - 1);
    const int x_end = (Direction > 0) ? width : -1;
    
    // ===== Initialization: Load first pixel's costs =====
    {
        const int idx = y * (width * max_disparity) + x_start * max_disparity + d;
        prev_L[d] = cost_volume[idx];
    }
    __syncthreads();
    
    // Find initial min_prev
    uint16_t min_prev = block_min_reduce(prev_L[d], max_disparity);
    
    // Accumulate first pixel (no aggregation, just raw cost)
    // Note: No atomicAdd needed - each kernel runs separately
    {
        const int idx = y * (width * max_disparity) + x_start * max_disparity + d;
        total_cost_volume[idx] += prev_L[d];
    }
    __syncthreads();
    
    // ===== Main Sweep: Iterate through row =====
    for (int x = x_start + Direction; x != x_end; x += Direction) {
        // Load raw cost for current pixel
        const int cost_idx = y * (width * max_disparity) + x * max_disparity + d;
        const uint16_t raw_cost = cost_volume[cost_idx];
        
        // Compute 4 terms for SGM formula
        const uint16_t term1 = prev_L[d]; // Same disparity
        
        // d-1 with boundary check
        // Note: Use 255+P1 for out-of-bounds to match CPU padding logic
        const uint16_t term2 = (d > 0) ? (prev_L[d - 1] + SGM_P1) : (255 + SGM_P1);
        
        // d+1 with boundary check  
        const uint16_t term3 = (d < max_disparity - 1) ? (prev_L[d + 1] + SGM_P1) : (255 + SGM_P1);
        
        // Any disparity with large penalty
        const uint16_t term4 = min_prev + SGM_P2;
        
        // Find minimum of 4 terms
        uint16_t selected_min = min(term1, min(term2, min(term3, term4)));
        
        // Apply SGM formula with normalization
        uint16_t new_L = raw_cost + selected_min - min_prev;
        
        // Clamp to prevent overflow (optional safety)
        new_L = min(new_L, (uint16_t)UINT16_MAX);
        
        // Accumulate to output buffer
        // Note: No atomicAdd needed - each direction runs separately
        total_cost_volume[cost_idx] += new_L;
        
        // Update prev_L for next iteration
        prev_L[d] = new_L;
        
        __syncthreads();
        
        // Compute new min_prev for next iteration
        min_prev = block_min_reduce(new_L, max_disparity);
        
        __syncthreads();
    }
}

// Explicit template instantiations
template __global__ void aggregate_horizontal_path_kernel<1>(
    const unsigned char*, uint16_t*, int, int, int);
    
template __global__ void aggregate_horizontal_path_kernel<-1>(
    const unsigned char*, uint16_t*, int, int, int);
