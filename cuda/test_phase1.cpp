/* -*-c++-*- SemiGlobalMatching CUDA - Copyright (C) 2025.
* Test Program: Phase 1 Validation (Census + Matching Cost)
* Compares CUDA implementation against CPU reference
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cstring>

// CPU reference implementation
#include "../SemiGlobalMatching/sgm_util.h"
#include "../SemiGlobalMatching/sgm_types.h"

// CUDA implementation
#include "include/sgm_cuda_phase1.cuh"
#include "include/sgm_cuda_validation.h"

using namespace std::chrono;

void print_usage(const char* prog_name) {
    printf("Usage: %s <left_image> <right_image> <max_disparity>\n", prog_name);
    printf("Example: %s Data/cone/im2.png Data/cone/im6.png 64\n", prog_name);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return -1;
    }

    // ===== Parse Arguments =====
    std::string path_left = argv[1];
    std::string path_right = argv[2];
    int max_disparity = atoi(argv[3]);

    printf("==============================================\n");
    printf("SGM CUDA Phase 1 Validation Test\n");
    printf("==============================================\n");
    printf("Left Image:  %s\n", path_left.c_str());
    printf("Right Image: %s\n", path_right.c_str());
    printf("Max Disparity: %d\n", max_disparity);
    printf("==============================================\n\n");

    // ===== Load Images =====
    cv::Mat img_left = cv::imread(path_left, cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread(path_right, cv::IMREAD_GRAYSCALE);

    if (img_left.empty() || img_right.empty()) {
        fprintf(stderr, "Error: Cannot load images\n");
        return -1;
    }

    if (img_left.size() != img_right.size()) {
        fprintf(stderr, "Error: Image dimensions mismatch\n");
        return -1;
    }

    const int width = img_left.cols;
    const int height = img_left.rows;
    const size_t img_size = width * height;

    printf("Image Size: %d x %d (%zu pixels)\n\n", width, height, img_size);

    // ===== Prepare CPU Buffers =====
    auto* h_imL = new uint8[img_size];
    auto* h_imR = new uint8[img_size];
    auto* h_censusL_cpu = new uint64[img_size]();
    auto* h_censusR_cpu = new uint64[img_size]();
    auto* h_censusL_gpu = new uint64[img_size]();
    auto* h_censusR_gpu = new uint64[img_size]();

    // Copy image data
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_imL[y * width + x] = img_left.at<uint8>(y, x);
            h_imR[y * width + x] = img_right.at<uint8>(y, x);
        }
    }

    // ===== CPU Reference: Census Transform =====
    printf("Running CPU Census Transform...\n");
    auto cpu_start = steady_clock::now();
    
    sgm_util::census_transform_9x7(h_imL, h_censusL_cpu, width, height);
    sgm_util::census_transform_9x7(h_imR, h_censusR_cpu, width, height);
    
    auto cpu_end = steady_clock::now();
    double cpu_census_time = duration_cast<milliseconds>(cpu_end - cpu_start).count();
    printf("  CPU Census Time: %.2f ms\n\n", cpu_census_time);

    // ===== GPU: Allocate Device Memory =====
    printf("Allocating GPU memory...\n");
    unsigned char *d_imL, *d_imR;
    uint64_t *d_censusL, *d_censusR;
    unsigned char *d_cost_volume;

    size_t census_size = img_size * sizeof(uint64_t);
    size_t cost_volume_size = img_size * max_disparity * sizeof(unsigned char);

    CHECK_CUDA(cudaMalloc(&d_imL, img_size));
    CHECK_CUDA(cudaMalloc(&d_imR, img_size));
    CHECK_CUDA(cudaMalloc(&d_censusL, census_size));
    CHECK_CUDA(cudaMalloc(&d_censusR, census_size));
    CHECK_CUDA(cudaMalloc(&d_cost_volume, cost_volume_size));

    printf("  Images: %zu bytes x 2\n", img_size);
    printf("  Census: %zu bytes x 2\n", census_size);
    printf("  Cost Volume: %zu bytes (%.2f MB)\n\n",
           cost_volume_size, cost_volume_size / 1024.0 / 1024.0);

    // ===== GPU: Copy Input Data =====
    CHECK_CUDA(cudaMemcpy(d_imL, h_imL, img_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_imR, h_imR, img_size, cudaMemcpyHostToDevice));

    // ===== GPU: Launch Phase 1 =====
    printf("Launching GPU Phase 1...\n");
    
    // Warm-up run (not timed)
    launch_phase1(d_imL, d_imR, d_censusL, d_censusR, d_cost_volume,
                  width, height, max_disparity);
    
    // Timed run
    CHECK_CUDA(cudaDeviceSynchronize());
    auto gpu_start = steady_clock::now();
    
    launch_phase1(d_imL, d_imR, d_censusL, d_censusR, d_cost_volume,
                  width, height, max_disparity);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    auto gpu_end = steady_clock::now();
    double gpu_time = duration_cast<milliseconds>(gpu_end - gpu_start).count();
    
    printf("  GPU Phase 1 Time: %.2f ms\n", gpu_time);
    printf("  Speedup vs CPU Census: %.2fx\n\n", cpu_census_time / gpu_time);

    // ===== Copy Results Back =====
    printf("Copying results back to host...\n");
    CHECK_CUDA(cudaMemcpy(h_censusL_gpu, d_censusL, census_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_censusR_gpu, d_censusR, census_size, cudaMemcpyDeviceToHost));

    // ===== Validation: Census Transform =====
    printf("\n========== VALIDATION: Census Transform ==========\n");
    bool census_left_valid = validate_census(h_censusL_cpu, h_censusL_gpu, width, height, false);
    bool census_right_valid = validate_census(h_censusR_cpu, h_censusR_gpu, width, height, false);

    // ===== CPU Reference: Matching Cost =====
    printf("Running CPU Matching Cost (for comparison)...\n");
    auto* h_cost_volume_cpu = new uint8[img_size * max_disparity];
    
    cpu_start = steady_clock::now();
    // Compute matching cost using CPU Hamming distance
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint64 censusL = h_censusL_cpu[y * width + x];
            for (int d = 0; d < max_disparity; d++) {
                uint8 cost;
                if (x - d >= 0) {
                    uint64 censusR = h_censusR_cpu[y * width + (x - d)];
                    cost = sgm_util::Hamming64(censusL, censusR);
                } else {
                    cost = 255;
                }
                int idx = y * (width * max_disparity) + x * max_disparity + d;
                h_cost_volume_cpu[idx] = cost;
            }
        }
    }
    cpu_end = steady_clock::now();
    double cpu_cost_time = duration_cast<milliseconds>(cpu_end - cpu_start).count();
    printf("  CPU Matching Cost Time: %.2f ms\n\n", cpu_cost_time);

    // Copy GPU cost volume
    auto* h_cost_volume_gpu = new uint8[img_size * max_disparity];
    CHECK_CUDA(cudaMemcpy(h_cost_volume_gpu, d_cost_volume, cost_volume_size, cudaMemcpyDeviceToHost));

    // ===== Validation: Cost Volume =====
    printf("========== VALIDATION: Cost Volume ==========\n");
    bool cost_valid = validate_cost_volume_sampled(
        h_cost_volume_cpu, h_cost_volume_gpu,
        width, height, max_disparity, 10000, 0
    );

    // ===== Summary =====
    printf("\n==============================================\n");
    printf("VALIDATION SUMMARY\n");
    printf("==============================================\n");
    printf("Census Transform (Left):  %s\n", census_left_valid ? "PASSED" : "FAILED");
    printf("Census Transform (Right): %s\n", census_right_valid ? "PASSED" : "FAILED");
    printf("Matching Cost:            %s\n", cost_valid ? "PASSED" : "FAILED");
    printf("\n");
    printf("Performance:\n");
    printf("  CPU Census:        %.2f ms\n", cpu_census_time);
    printf("  CPU Matching Cost: %.2f ms\n", cpu_cost_time);
    printf("  CPU Total:         %.2f ms\n", cpu_census_time + cpu_cost_time);
    printf("  GPU Phase 1:       %.2f ms\n", gpu_time);
    printf("  Speedup:           %.2fx\n", (cpu_census_time + cpu_cost_time) / gpu_time);
    printf("==============================================\n");

    // ===== Cleanup =====
    delete[] h_imL;
    delete[] h_imR;
    delete[] h_censusL_cpu;
    delete[] h_censusR_cpu;
    delete[] h_censusL_gpu;
    delete[] h_censusR_gpu;
    delete[] h_cost_volume_cpu;
    delete[] h_cost_volume_gpu;

    cudaFree(d_imL);
    cudaFree(d_imR);
    cudaFree(d_censusL);
    cudaFree(d_censusR);
    cudaFree(d_cost_volume);

    return (census_left_valid && census_right_valid && cost_valid) ? 0 : 1;
}
