# Makefile for SemiGlobalMatching CPU baseline

CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -fopenmp
INCLUDES = -I/usr/include/opencv4
LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

SRC_DIR = SemiGlobalMatching
BUILD_DIR = build
BIN_DIR = bin

# Source files
SOURCES = $(SRC_DIR)/main.cpp \
          $(SRC_DIR)/SemiGlobalMatching.cpp \
          $(SRC_DIR)/sgm_util.cpp \
          $(SRC_DIR)/stdafx.cpp

# Object files
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Target executable
TARGET = $(BIN_DIR)/sgm_cpu

.PHONY: all clean run baseline

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LIBS) -o $(TARGET)
	@echo "Build complete: $(TARGET)"

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Clean complete"

# Run baseline test on cone dataset
baseline: $(TARGET)
	@echo "========================================="
	@echo "Running SGM CPU Baseline Test"
	@echo "Dataset: cone"
	@echo "========================================="
	$(TARGET) Data/cone/im2.png Data/cone/im6.png 0 64

# Run on different datasets
test-cone: $(TARGET)
	@echo "Testing on cone dataset..."
	$(TARGET) Data/cone/im2.png Data/cone/im6.png 0 64

test-reindeer: $(TARGET)
	@echo "Testing on Reindeer dataset..."
	$(TARGET) Data/Reindeer/view1.png Data/Reindeer/view5.png 0 128

test-cloth: $(TARGET)
	@echo "Testing on Cloth3 dataset..."
	$(TARGET) Data/Cloth3/view1.png Data/Cloth3/view5.png 0 64

test-all: test-cone test-reindeer test-cloth
	@echo "All tests complete!"
