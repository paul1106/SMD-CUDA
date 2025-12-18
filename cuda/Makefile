# Makefile for SemiGlobalMatching CUDA Phase 1 & 2
# Target: NVIDIA RTX 4090 (Compute Capability 8.9)

# Compiler settings
NVCC = nvcc
CXX = g++

# CUDA architecture (RTX 4090 is sm_89)
CUDA_ARCH = -arch=sm_89

# Include paths
CUDA_INC = -I./include
OPENCV_INC = $(shell pkg-config --cflags opencv4)
CPU_INC = -I../SemiGlobalMatching

# Library paths
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# Compiler flags
NVCC_FLAGS = $(CUDA_ARCH) -std=c++11 -O3 -lineinfo --use_fast_math
NVCC_FLAGS += -Xcompiler -fopenmp -Xcompiler -Wall
CXX_FLAGS = -std=c++11 -O3 -Wall -fopenmp

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = ../bin

# Source files - Phase 1
CUDA_SOURCES_P1 = $(SRC_DIR)/census_transform.cu \
                  $(SRC_DIR)/matching_cost.cu \
                  $(SRC_DIR)/phase1_wrapper.cu

# Source files - Phase 2
CUDA_SOURCES_P2 = $(SRC_DIR)/horizontal_aggregation.cu \
                  $(SRC_DIR)/vertical_aggregation.cu \
                  $(SRC_DIR)/phase2_wrapper.cu

# Source files - Phase 4
CUDA_SOURCES_P4 = $(SRC_DIR)/wta_disparity.cu

# All CUDA sources
CUDA_SOURCES = $(CUDA_SOURCES_P1) $(CUDA_SOURCES_P2) $(CUDA_SOURCES_P4)

CPU_SOURCES = ../SemiGlobalMatching/sgm_util.cpp \
              ../SemiGlobalMatching/stdafx.cpp

# Object files
CUDA_OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CUDA_SOURCES))
CPU_OBJECTS = $(BUILD_DIR)/sgm_util.o $(BUILD_DIR)/stdafx.o

# Test executables
TEST1_OBJECT = $(BUILD_DIR)/test_phase1.o
TEST2_OBJECT = $(BUILD_DIR)/test_phase2.o
TEST2_FULL_OBJECT = $(BUILD_DIR)/test_phase2_full.o
TARGET1 = $(BIN_DIR)/test_phase1_cuda
TARGET2 = $(BIN_DIR)/test_phase2_cuda
TARGET2_FULL = $(BIN_DIR)/test_phase2_full_cuda

# Default target
all: $(TARGET1) $(TARGET2) $(TARGET2_FULL)

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC) -c $< -o $@

# Compile CPU reference implementation
$(BUILD_DIR)/sgm_util.o: ../SemiGlobalMatching/sgm_util.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) $(CPU_INC) -c $< -o $@

$(BUILD_DIR)/stdafx.o: ../SemiGlobalMatching/stdafx.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) $(CPU_INC) -c $< -o $@

# Compile test programs
$(BUILD_DIR)/test_phase1.o: test_phase1.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC) $(CPU_INC) $(OPENCV_INC) -c $< -o $@

$(BUILD_DIR)/test_phase2.o: test_phase2.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC) $(CPU_INC) $(OPENCV_INC) -c $< -o $@

$(BUILD_DIR)/test_phase2_full.o: test_phase2_full.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC) $(CPU_INC) $(OPENCV_INC) -c $< -o $@

# Link test executables
$(TARGET1): $(CUDA_OBJECTS) $(CPU_OBJECTS) $(TEST1_OBJECT) | $(BIN_DIR)
	$(NVCC) $(CUDA_ARCH) -o $@ $^ $(OPENCV_LIBS) -Xcompiler -fopenmp
	@echo "Build complete: $(TARGET1)"

$(TARGET2): $(CUDA_OBJECTS) $(CPU_OBJECTS) $(TEST2_OBJECT) | $(BIN_DIR)
	$(NVCC) $(CUDA_ARCH) -o $@ $^ $(OPENCV_LIBS) -Xcompiler -fopenmp
	@echo "Build complete: $(TARGET2)"

$(TARGET2_FULL): $(CUDA_OBJECTS) $(CPU_OBJECTS) $(TEST2_FULL_OBJECT) | $(BIN_DIR)
	$(NVCC) $(CUDA_ARCH) -o $@ $^ $(OPENCV_LIBS) -Xcompiler -fopenmp
	@echo "Build complete: $(TARGET2_FULL)"

# Run Phase 1 tests
test_phase1_cone: $(TARGET1)
	@echo "========================================="
	@echo "Testing Phase 1 on cone dataset"
	@echo "========================================="
	$(TARGET1) ../Data/cone/im2.png ../Data/cone/im6.png 64

test_phase1_reindeer: $(TARGET1)
	@echo "========================================="
	@echo "Testing Phase 1 on Reindeer dataset"
	@echo "========================================="
	$(TARGET1) ../Data/Reindeer/view1.png ../Data/Reindeer/view5.png 128

# Run Phase 2 tests
test_phase2_cone: $(TARGET2)
	@echo "========================================="
	@echo "Testing Phase 2 on cone dataset"
	@echo "========================================="
	$(TARGET2) ../Data/cone/im2.png ../Data/cone/im6.png 64

test_phase2_reindeer: $(TARGET2)
	@echo "========================================="
	@echo "Testing Phase 2 on Reindeer dataset"
	@echo "========================================="
	$(TARGET2) ../Data/Reindeer/view1.png ../Data/Reindeer/view5.png 128

# Run all tests
test_all: test_phase1_cone test_phase1_reindeer test_phase2_cone test_phase2_reindeer

# Quick test aliases
test_cone: test_phase2_cone
test_reindeer: test_phase2_reindeer

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(TARGET1) $(TARGET2)

# Show GPU info
gpu_info:
	nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free --format=csv

.PHONY: all clean test_cone test_reindeer test_all gpu_info
