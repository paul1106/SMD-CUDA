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
    // Grid: One block per row
    // Block: One thread per disparity
    dim3 grid(height, 1, 1);
    dim3 block(max_disparity, 1, 1);
    
    // Shared memory size: prev_L array
    size_t shared_mem_size = max_disparity * sizeof(uint16_t);
    
    printf("Launching Phase 2 Horizontal Aggregation...\n");
    printf("  Grid: (%d, 1, 1), Block: (%d, 1, 1)\n", grid.x, block.x);
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
