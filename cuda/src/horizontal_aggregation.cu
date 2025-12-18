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
 * Block: (min(max_disparity, 1024), 1, 1)
 * 
 * Note: For max_disparity > 1024, each thread handles multiple disparities using strided access
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
    // Note: blockDim.x may be > max_disparity (padded to multiple of 32)
    extern __shared__ uint16_t smem[];
    const int padded_size = blockDim.x;  // Actual allocated size (rounded up)
    uint16_t* prev_L = smem;              // Buffer 0: read from here
    uint16_t* next_L = smem + padded_size; // Buffer 1: write to here
    
    const int y = blockIdx.x; // Each block processes one row
    const int num_threads = blockDim.x;
    const int d = threadIdx.x; // May be >= max_disparity for padding threads
    
    if (y >= height) return;
    
    // Determine sweep direction
    const int x_start = (Direction > 0) ? 0 : (width - 1);
    const int x_end = (Direction > 0) ? width : -1;
    
    // Thread-local minimum for reduction
    uint16_t thread_min;
    
    // ===== Initialization: Load first pixel's costs =====
    // Padding threads (d >= max_disparity) set to UINT16_MAX so they don't affect min reduction
    if (d < max_disparity) {
        const int idx = y * (width * max_disparity) + x_start * max_disparity + d;
        prev_L[d] = cost_volume[idx];
    } else {
        // Padding thread: Set to infinity so it's ignored in min reduction
        prev_L[d] = UINT16_MAX;
    }
    __syncthreads();
    
    // Find initial min_prev
    // ALL threads (including padding) MUST participate in block_min_reduce
    // Padding threads already have UINT16_MAX, so they won't affect the result
    thread_min = prev_L[d];
    uint16_t min_prev = block_min_reduce(thread_min, num_threads);
    
    // Accumulate first pixel (only valid disparities write to global memory)
    if (d < max_disparity) {
        const int idx = y * (width * max_disparity) + x_start * max_disparity + d;
        total_cost_volume[idx] += prev_L[d];
    }
    __syncthreads();
    
    // ===== Main Sweep: Iterate through row =====
    // Double buffering: Read from prev_L, write to next_L, then swap
    
    for (int x = x_start + Direction; x != x_end; x += Direction) {
        // Compute new costs for valid disparities only
        if (d < max_disparity) {
            const int cost_idx = y * (width * max_disparity) + x * max_disparity + d;
            const uint16_t raw_cost = cost_volume[cost_idx];
            
            // Read from prev_L (safe: no writes to prev_L in this phase)
            const uint16_t term1 = prev_L[d];
            const uint16_t term2 = (d > 0) ? (prev_L[d - 1] + SGM_P1) : (255 + SGM_P1);
            const uint16_t term3 = (d < max_disparity - 1) ? (prev_L[d + 1] + SGM_P1) : (255 + SGM_P1);
            const uint16_t term4 = min_prev + SGM_P2;
            
            uint16_t selected_min = min(term1, min(term2, min(term3, term4)));
            uint16_t new_L = raw_cost + selected_min - min_prev;
            new_L = min(new_L, (uint16_t)UINT16_MAX);
            
            // Write to global memory
            total_cost_volume[cost_idx] += new_L;
            
            // Write to next_L buffer
            next_L[d] = new_L;
        } else {
            // Padding thread: maintain UINT16_MAX for next reduction
            next_L[d] = UINT16_MAX;
        }
        
        __syncthreads();
        
        // Swap buffers: next_L becomes prev_L for next iteration
        uint16_t* tmp = prev_L;
        prev_L = next_L;
        next_L = tmp;
        
        // Compute new min_prev for next iteration
        // ALL threads (including padding) MUST participate in reduction
        thread_min = prev_L[d];  // Padding threads have UINT16_MAX
        min_prev = block_min_reduce(thread_min, num_threads);
        
        __syncthreads();
    }
}

// Explicit template instantiations
template __global__ void aggregate_horizontal_path_kernel<1>(
    const unsigned char*, uint16_t*, int, int, int);
    
template __global__ void aggregate_horizontal_path_kernel<-1>(
    const unsigned char*, uint16_t*, int, int, int);
