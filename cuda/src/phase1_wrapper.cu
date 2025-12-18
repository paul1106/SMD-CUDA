/* -*-c++-*- SemiGlobalMatching CUDA - Copyright (C) 2025.
* Implementation: Host Wrapper for Phase 1 Kernels
* Target: NVIDIA RTX 4090 (Compute Capability 8.9)
*/

#include "../include/sgm_cuda_phase1.cuh"

/**
 * Host Wrapper for Phase 1: Census Transform + Matching Cost
 * 
 * Launches both kernels with optimized grid/block configuration for RTX 4090
 * Uses synchronous launches for debugging (can be made async later)
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
) {
    // Configure grid and block dimensions
    // Using 32x32 blocks for optimal occupancy on RTX 4090
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(
        (width + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
        (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT
    );
    
    // ===== Step 1: Census Transform for Left Image =====
    printf("Launching Census Transform (Left)...\n");
    printf("  Grid: (%d, %d), Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);
    
    census_transform_kernel<<<grid, block>>>(
        d_imL, d_censusL, width, height
    );
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // ===== Step 2: Census Transform for Right Image =====
    printf("Launching Census Transform (Right)...\n");
    
    census_transform_kernel<<<grid, block>>>(
        d_imR, d_censusR, width, height
    );
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // ===== Step 3: Matching Cost Computation =====
    printf("Launching Matching Cost Kernel...\n");
    printf("  Max Disparity: %d\n", max_disparity);
    printf("  Cost Volume Size: %d x %d x %d = %zu bytes\n",
           width, height, max_disparity,
           (size_t)width * height * max_disparity);
    
    matching_cost_kernel<<<grid, block>>>(
        d_censusL, d_censusR, d_cost_volume,
        width, height, max_disparity
    );
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("Phase 1 completed successfully.\n");
}
