# ─────────────────────────────────────────────────────────────────────────────
# Makefile — Parallel SHA-256 GPU Benchmark
# Author : Shruthi Chinnasamy (G25AIT1165) — IIT Jodhpur
# ─────────────────────────────────────────────────────────────────────────────

NVCC       := nvcc
TARGET     := sha256_bench

# ── Detect compute capability automatically (requires nvidia-smi) ────────────
# Falls back to sm_75 (Turing) if detection fails.
DETECTED   := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(DETECTED),)
    ARCH   := sm_75
else
    ARCH   := sm_$(DETECTED)
endif

# ── Compiler flags ────────────────────────────────────────────────────────────
NVCCFLAGS  := -O3                   \
              -arch=$(ARCH)         \
              --use_fast_math       \
              -std=c++17            \
              -Xcompiler -O3        \
              -lineinfo

INCLUDES   := -I.

SRCS       := main.cu

.PHONY: all clean run run_e0 run_e1 run_e2 run_e3 run_e4 run_e5

all: $(TARGET)
	@echo "Build complete — arch=$(ARCH)"
	@echo "Run with: ./$(TARGET) [0-5|all]"

$(TARGET): $(SRCS) sha256.cuh kernels.cuh sha256_cpu.h utils.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

# ── Convenience targets ───────────────────────────────────────────────────────
run:    $(TARGET); ./$(TARGET) all
run_e0: $(TARGET); ./$(TARGET) 0
run_e1: $(TARGET); ./$(TARGET) 1
run_e2: $(TARGET); ./$(TARGET) 2
run_e3: $(TARGET); ./$(TARGET) 3
run_e4: $(TARGET); ./$(TARGET) 4
run_e5: $(TARGET); ./$(TARGET) 5

clean:
	rm -f $(TARGET) *.o

# ── Colab/cloud build helper ─────────────────────────────────────────────────
# If nvcc is not in PATH, try the default CUDA toolkit location.
# Usage: make NVCC=/usr/local/cuda/bin/nvcc
