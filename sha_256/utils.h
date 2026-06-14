#pragma once
/*
 * utils.h
 * Dataset generation, hex printing, and GPU-vs-CPU verification helpers.
 *
 * Author : Shruthi Chinnasamy (G25AIT1165)
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "sha256_cpu.h"

// ─────────────────────────────────────────────────────────────────────────────
// Dataset: variable-length messages packed into a contiguous byte buffer.
// ─────────────────────────────────────────────────────────────────────────────
struct Dataset {
    uint8_t*  data;      // packed messages (host, may be pinned)
    uint32_t* offsets;   // offsets[i] = start of message i in data[]
    uint32_t* lengths;   // lengths[i] = byte length of message i
    uint32_t  N;         // number of messages
    size_t    total_bytes; // total bytes in data[]
    bool      is_pinned; // if true, data/offsets/lengths are cudaMallocHost'd
};

// Generate N messages with lengths drawn uniformly in [min_len, max_len].
// Deterministic: seeded with `seed`.
// If `pinned` is true, use cudaMallocHost for zero-copy async transfers.
static Dataset generate_dataset(uint32_t N, uint32_t min_len, uint32_t max_len,
                                uint32_t seed = 42, bool pinned = false)
{
    Dataset ds;
    ds.N         = N;
    ds.is_pinned = pinned;

    // Allocate offset and length arrays (host, not pinned — small)
    if (pinned) {
        cudaMallocHost(&ds.offsets, N * sizeof(uint32_t));
        cudaMallocHost(&ds.lengths, N * sizeof(uint32_t));
    } else {
        ds.offsets = new uint32_t[N];
        ds.lengths = new uint32_t[N];
    }

    // First pass: compute lengths and total size
    srand(seed);
    uint32_t range = max_len - min_len + 1;
    size_t total = 0;
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t len = min_len + (uint32_t)(rand() % range);
        ds.lengths[i] = len;
        ds.offsets[i] = (uint32_t)total;
        total += len;
    }
    ds.total_bytes = total;

    // Allocate data buffer
    if (pinned) {
        cudaMallocHost(&ds.data, total);
    } else {
        ds.data = new uint8_t[total];
    }

    // Second pass: fill with deterministic pattern
    srand(seed + 1);
    for (size_t b = 0; b < total; ++b)
        ds.data[b] = (uint8_t)(rand() & 0xFF);

    return ds;
}

static void free_dataset(Dataset& ds) {
    if (ds.is_pinned) {
        cudaFreeHost(ds.data);
        cudaFreeHost(ds.offsets);
        cudaFreeHost(ds.lengths);
    } else {
        delete[] ds.data;
        delete[] ds.offsets;
        delete[] ds.lengths;
    }
    ds.data = nullptr; ds.offsets = nullptr; ds.lengths = nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Hex printing utility
// ─────────────────────────────────────────────────────────────────────────────
static void print_hex(const char* label, const uint8_t* d, int n) {
    printf("%s: ", label);
    for (int i = 0; i < n; ++i) printf("%02x", d[i]);
    printf("\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Verify GPU output against CPU reference on a random sample.
// sample_frac: fraction of N to check (e.g., 0.01 = 1%)
// Returns number of mismatches (0 = all correct).
// ─────────────────────────────────────────────────────────────────────────────
static int verify_sample(const Dataset&   ds,
                          const uint8_t*   gpu_digests,  // host pointer, N*32 bytes
                          float            sample_frac = 0.01f,
                          uint32_t         seed = 99)
{
    uint32_t n_check = (uint32_t)fmax(1.0f, ds.N * sample_frac);
    n_check = (n_check > ds.N) ? ds.N : n_check;

    srand(seed);
    int mismatches = 0;

    for (uint32_t s = 0; s < n_check; ++s) {
        uint32_t idx = (uint32_t)(rand() % ds.N);
        uint8_t cpu_digest[32];
        sha256_cpu(ds.data + ds.offsets[idx], ds.lengths[idx], cpu_digest);

        const uint8_t* gpu_d = gpu_digests + (size_t)idx * 32;
        if (memcmp(cpu_digest, gpu_d, 32) != 0) {
            ++mismatches;
            if (mismatches <= 3) {   // print first few failures
                printf("[VERIFY FAIL] msg %u (len=%u)\n", idx, ds.lengths[idx]);
                print_hex("  CPU", cpu_digest, 32);
                print_hex("  GPU", gpu_d, 32);
            }
        }
    }

    if (mismatches == 0)
        printf("[VERIFY] OK — %u/%u samples match CPU reference.\n", n_check, ds.N);
    else
        printf("[VERIFY] FAIL — %d/%u mismatches detected!\n", mismatches, n_check);

    return mismatches;
}

// ─────────────────────────────────────────────────────────────────────────────
// Timing helper: CUDA event pair
// ─────────────────────────────────────────────────────────────────────────────
struct Timer {
    cudaEvent_t start, stop;
    Timer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin(cudaStream_t s = 0) { cudaEventRecord(start, s); }
    void end(cudaStream_t s = 0)   { cudaEventRecord(stop,  s); }
    float ms() {
        cudaEventSynchronize(stop);
        float t = 0; cudaEventElapsedTime(&t, start, stop); return t;
    }
};
