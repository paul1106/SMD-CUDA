// Quick debug: Test only left-to-right direction
#include <opencv2/opencv.hpp>
#include <iostream>
#include "../SemiGlobalMatching/sgm_types.h"
#include "include/sgm_cuda_phase1.cuh"
#include "include/sgm_cuda_phase2.cuh"

int main() {
    // Load cone dataset
    cv::Mat img_l = cv::imread("../Data/cone/im2.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_r = cv::imread("../Data/cone/im6.png", cv::IMREAD_GRAYSCALE);
    
    int W = img_l.cols, H = img_l.rows, D = 64;
    
    // Phase 1: Generate cost volume
    auto *h_imL = new uint8[W*H];
    auto *h_imR = new uint8[W*H];
    for (int i = 0; i < W*H; i++) {
        h_imL[i] = img_l.data[i];
        h_imR[i] = img_r.data[i];
    }
    
    unsigned char *d_imL, *d_imR, *d_cost;
    uint64_t *d_censusL, *d_censusR;
    uint16_t *d_aggr;
    
    cudaMalloc(&d_imL, W*H);
    cudaMalloc(&d_imR, W*H);
    cudaMalloc(&d_censusL, W*H*sizeof(uint64_t));
    cudaMalloc(&d_censusR, W*H*sizeof(uint64_t));
    cudaMalloc(&d_cost, W*H*D);
    cudaMalloc(&d_aggr, W*H*D*sizeof(uint16_t));
    
    cudaMemcpy(d_imL, h_imL, W*H, cudaMemcpyHostToDevice);
    cudaMemcpy(d_imR, h_imR, W*H, cudaMemcpyHostToDevice);
    cudaMemset(d_aggr, 0, W*H*D*sizeof(uint16_t));
    
    launch_phase1(d_imL, d_imR, d_censusL, d_censusR, d_cost, W, H, D);
    
    // Test both directions separately
    dim3 grid(H, 1, 1);
    dim3 block(D, 1, 1);
    size_t smem = D * sizeof(uint16_t);
    
    uint16_t *d_aggr_lr, *d_aggr_rl;
    cudaMalloc(&d_aggr_lr, W*H*D*sizeof(uint16_t));
    cudaMalloc(&d_aggr_rl, W*H*D*sizeof(uint16_t));
    cudaMemset(d_aggr_lr, 0, W*H*D*sizeof(uint16_t));
    cudaMemset(d_aggr_rl, 0, W*H*D*sizeof(uint16_t));
    
    // L->R
    aggregate_horizontal_path_kernel<1><<<grid, block, smem>>>(
        d_cost, d_aggr_lr, W, H, D);
    // R->L  
    aggregate_horizontal_path_kernel<-1><<<grid, block, smem>>>(
        d_cost, d_aggr_rl, W, H, D);
    cudaDeviceSynchronize();
    
    // Copy back first row
    auto *h_aggr_lr = new uint16_t[W*D];
    auto *h_aggr_rl = new uint16_t[W*D];
    cudaMemcpy(h_aggr_lr, d_aggr_lr, W*D*sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_aggr_rl, d_aggr_rl, W*D*sizeof(uint16_t), cudaMemcpyDeviceToHost);
    
    auto *h_cost = new uint8[W*D];
    cudaMemcpy(h_cost, d_cost, W*D, cudaMemcpyDeviceToHost);
    
    printf("First row, pixel x=0:\n");
    printf("d\tRaw\tL->R\tR->L\n");
    for (int d = 0; d < 10; d++) {
        int idx = 0 * D + d;  // x=0
        printf("%d\t%d\t%d\t%d\n", d, h_cost[idx], 
               h_aggr_lr[idx], h_aggr_rl[idx]);
    }
    
    printf("\nFirst row, pixel x=1:\n");
    printf("d\tRaw\tL->R\tR->L\n");
    for (int d = 0; d < 10; d++) {
        int idx = 1 * D + d;  // x=1
        printf("%d\t%d\t%d\t%d\n", d, h_cost[idx], 
               h_aggr_lr[idx], h_aggr_rl[idx]);
    }
    
    return 0;
}
