/* -*-c++-*- SemiGlobalMatching CUDA - Copyright (C) 2025.
* Implementation: Host Wrapper for Phase 2 Horizontal Aggregation
* Target: NVIDIA RTX 4090 (Compute Capability 8.9)
*/

#include "../include/sgm_cuda_phase2.cuh"

/**
 * Host Wrapper for Horizontal Cost Aggregation
 * 
 * Launches two aggregation passes:
 * 1. Left-to-right (Direction = +1)
 * 2. Right-to-left (Direction = -1)
 * 
 * Both passes accumulate their results into total_cost_volume.
 */
void launch_phase2_horizontal(
    const unsigned char* d_cost_volume,
    uint16_t* d_total_cost_volume,
    int width,
    int height,
    int max_disparity
) {
    // CRITICAL FIX: Round up to multiple of 32 to ensure all warps are full
    // This prevents undefined behavior in __shfl_down_sync when using 0xffffffff mask
    // Example: disparity=270 -> threads=288 (9 full warps, not 8.4375 warps)
    
    int threads_per_block = ((max_disparity + 31) / 32) * 32;
    threads_per_block = min(threads_per_block, 1024); // Still respect CUDA limit
    
    dim3 grid(height, 1, 1);
    dim3 block(threads_per_block, 1, 1);
    
    // Shared memory size: Use padded thread count for allocation
    // Double buffering: prev_L + next_L
    size_t shared_mem_size = 2 * threads_per_block * sizeof(uint16_t);
    
    printf("Launching Phase 2 Horizontal Aggregation...\n");
    printf("  Grid: (%d, 1, 1), Block: (%d, 1, 1)\n", grid.x, block.x);
    printf("  Max Disparity: %d, Threads: %d\n", max_disparity, threads_per_block);
    printf("  Shared Memory: %zu bytes per block\n", shared_mem_size);
    
    // ===== Pass 1: Left-to-Right =====
    printf("  Direction: Left-to-Right\n");
    aggregate_horizontal_path_kernel<1><<<grid, block, shared_mem_size>>>(
        d_cost_volume, d_total_cost_volume, width, height, max_disparity
    );
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // ===== Pass 2: Right-to-Left =====
    printf("  Direction: Right-to-Left\n");
    aggregate_horizontal_path_kernel<-1><<<grid, block, shared_mem_size>>>(
        d_cost_volume, d_total_cost_volume, width, height, max_disparity
    );
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("Phase 2 Horizontal Aggregation completed.\n");
}
