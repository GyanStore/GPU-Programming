/*
 * main.cu
 * ─────────────────────────────────────────────────────────────────────────────
 * Parallel SHA-256 on GPU — Complete Benchmark Suite
 *
 * Implements:
 *   B0 — Synchronous baseline (one kernel launch, single H2D + D2H)
 *   B2 — Stream-pipelined variant (S=3 streams, pinned host memory)
 *   CPU single-thread reference
 *
 * Experiments:
 *   E0  FIPS self-test (CPU reference)
 *   E1  Default: N=1M, len 16-200 B  →  B0 vs B2 vs CPU
 *   E2  Block-size sweep (128/256/512) for B0
 *   E3  Message-length crossover (fixed N=1M, len = 16/64/256/1K/4K)
 *   E4  Scalability (N = 10^4 .. 10^7, fixed avg len=128)
 *   E5  Operating map: grid (N x avg_len) → GPU speedup heatmap data
 *
 * Build (example):
 *   nvcc -O3 -arch=sm_86 -o sha256_bench main.cu
 *   ./sha256_bench
 *
 * Author  : Shruthi Chinnasamy (G25AIT1165)
 * Course  : GPU Programming — IIT Jodhpur
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

#include "sha256.cuh"
#include "kernels.cuh"
#include "sha256_cpu.h"
#include "utils.h"

// ─── Tuning constants ────────────────────────────────────────────────────────
static const int BLOCK_SIZE    = 256;   // threads per block (default)
static const int NUM_STREAMS   = 3;     // streams for B2 pipeline
static const int VERIFY_SAMPLE = 1000; // messages to verify per run

// ─── Grid size helper ────────────────────────────────────────────────────────
static inline int grid(uint32_t N, int block) {
    return (int)((N + block - 1) / block);
}

// ─────────────────────────────────────────────────────────────────────────────
// run_b0: Synchronous baseline
//   Returns struct with timing breakdown in milliseconds.
// ─────────────────────────────────────────────────────────────────────────────
struct TimingResult {
    float h2d_ms, kernel_ms, d2h_ms, total_ms, cpu_ms;
    double gpu_mhash, cpu_mhash;
    float speedup_kernel, speedup_e2e;
};

static TimingResult run_b0(const Dataset& ds, int block_size = BLOCK_SIZE,
                            bool verify = true)
{
    TimingResult res = {};
    size_t digest_bytes = (size_t)ds.N * SHA256_DIGEST_BYTES;

    // ── Device allocations ───────────────────────────────────────────────────
    uint8_t*  d_data;
    uint32_t* d_off;
    uint32_t* d_len;
    uint8_t*  d_digest;

    CUDA_CHECK(cudaMalloc(&d_data,   ds.total_bytes));
    CUDA_CHECK(cudaMalloc(&d_off,    ds.N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_len,    ds.N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_digest, digest_bytes));

    // ── Host output buffer ───────────────────────────────────────────────────
    uint8_t* h_digest = new uint8_t[digest_bytes];

    // ── Timers ───────────────────────────────────────────────────────────────
    Timer t_h2d, t_kern, t_d2h;

    // ── H2D transfer ─────────────────────────────────────────────────────────
    t_h2d.begin();
    CUDA_CHECK(cudaMemcpy(d_data, ds.data,    ds.total_bytes,          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_off,  ds.offsets, ds.N*sizeof(uint32_t),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_len,  ds.lengths, ds.N*sizeof(uint32_t),   cudaMemcpyHostToDevice));
    t_h2d.end();
    CUDA_CHECK(cudaDeviceSynchronize());
    res.h2d_ms = t_h2d.ms();

    // ── Kernel ───────────────────────────────────────────────────────────────
    t_kern.begin();
    sha256_kernel_b0<<<grid(ds.N, block_size), block_size>>>(
        d_data, d_off, d_len, d_digest, ds.N);
    t_kern.end();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    res.kernel_ms = t_kern.ms();

    // ── D2H transfer ─────────────────────────────────────────────────────────
    t_d2h.begin();
    CUDA_CHECK(cudaMemcpy(h_digest, d_digest, digest_bytes, cudaMemcpyDeviceToHost));
    t_d2h.end();
    CUDA_CHECK(cudaDeviceSynchronize());
    res.d2h_ms = t_d2h.ms();

    res.total_ms = res.h2d_ms + res.kernel_ms + res.d2h_ms;

    // ── CPU reference (single thread) ────────────────────────────────────────
    auto cpu_start = std::chrono::high_resolution_clock::now();
    uint8_t cpu_single[32];
    for (uint32_t i = 0; i < ds.N; ++i)
        sha256_cpu(ds.data + ds.offsets[i], ds.lengths[i], cpu_single);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    res.cpu_ms = (float)std::chrono::duration<double, std::milli>(
                     cpu_end - cpu_start).count();

    // ── Throughput metrics ───────────────────────────────────────────────────
    res.gpu_mhash        = ds.N / (res.kernel_ms * 1e3);   // Mhash/s (kernel only)
    res.cpu_mhash        = ds.N / (res.cpu_ms    * 1e3);
    res.speedup_kernel   = (float)(res.cpu_ms    / res.kernel_ms);
    res.speedup_e2e      = (float)(res.cpu_ms    / res.total_ms);

    // ── Verification ─────────────────────────────────────────────────────────
    if (verify) {
        Dataset tmp = ds;  // verify_sample takes const ref
        verify_sample(tmp, h_digest, (float)VERIFY_SAMPLE / ds.N);
    }

    // ── Cleanup ──────────────────────────────────────────────────────────────
    delete[] h_digest;
    cudaFree(d_data); cudaFree(d_off); cudaFree(d_len); cudaFree(d_digest);
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// run_b2: Multi-stream pinned-memory pipeline
//   Partitions dataset into NUM_STREAMS * ceil(N / NUM_STREAMS) batches.
//   H2D_b+1 overlaps with Kernel_b, D2H_b-1 overlaps with both.
// ─────────────────────────────────────────────────────────────────────────────
static TimingResult run_b2(const Dataset& ds, int block_size = BLOCK_SIZE,
                             bool verify = true)
{
    TimingResult res = {};
    const int S = NUM_STREAMS;
    size_t digest_bytes_total = (size_t)ds.N * SHA256_DIGEST_BYTES;

    // ── Per-stream batch sizing ───────────────────────────────────────────────
    uint32_t batch_n  = (ds.N + S - 1) / S;          // messages per batch
    int      n_batches = (int)((ds.N + batch_n - 1) / batch_n);

    // ── Pinned host buffers (one per stream) ─────────────────────────────────
    // We need data, offsets, lengths, and digest buffers per stream.
    // Sizes: data — batch_n * max_msg_len worst case.  We use total/S + margin.
    size_t batch_data_bytes   = (ds.total_bytes / S) + 4096 * 64; // headroom
    size_t batch_off_bytes    = batch_n * sizeof(uint32_t);
    size_t batch_len_bytes    = batch_n * sizeof(uint32_t);
    size_t batch_digest_bytes = (size_t)batch_n * SHA256_DIGEST_BYTES;

    uint8_t*  h_data[S];    // pinned host data buffers
    uint32_t* h_off[S];     // pinned host offset buffers
    uint32_t* h_len[S];     // pinned host length buffers
    uint8_t*  h_digest[S];  // pinned host digest output buffers

    uint8_t*  d_data[S];
    uint32_t* d_off[S];
    uint32_t* d_len[S];
    uint8_t*  d_digest[S];

    cudaStream_t streams[S];

    for (int s = 0; s < S; ++s) {
        CUDA_CHECK(cudaMallocHost(&h_data[s],   batch_data_bytes));
        CUDA_CHECK(cudaMallocHost(&h_off[s],    batch_off_bytes));
        CUDA_CHECK(cudaMallocHost(&h_len[s],    batch_len_bytes));
        CUDA_CHECK(cudaMallocHost(&h_digest[s], batch_digest_bytes));

        CUDA_CHECK(cudaMalloc(&d_data[s],   batch_data_bytes));
        CUDA_CHECK(cudaMalloc(&d_off[s],    batch_off_bytes));
        CUDA_CHECK(cudaMalloc(&d_len[s],    batch_len_bytes));
        CUDA_CHECK(cudaMalloc(&d_digest[s], batch_digest_bytes));

        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    // Host output — collect results from all streams in order
    uint8_t* h_out = new uint8_t[digest_bytes_total];

    // ── Pipeline dispatch ─────────────────────────────────────────────────────
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start, 0));

    for (int b = 0; b < n_batches; ++b) {
        int s = b % S;
        uint32_t msg_start = (uint32_t)b * batch_n;
        uint32_t msg_end   = msg_start + batch_n;
        if (msg_end > ds.N) msg_end = ds.N;
        uint32_t bN = msg_end - msg_start;

        // Build local offset/length arrays with byte offsets relative to this batch's data
        size_t data_bytes = 0;
        for (uint32_t i = 0; i < bN; ++i) {
            uint32_t gi = msg_start + i;          // global index
            h_off[s][i] = (uint32_t)data_bytes;
            h_len[s][i] = ds.lengths[gi];
            memcpy(h_data[s] + data_bytes, ds.data + ds.offsets[gi], ds.lengths[gi]);
            data_bytes += ds.lengths[gi];
        }

        // Async H2D
        CUDA_CHECK(cudaMemcpyAsync(d_data[s],   h_data[s],   data_bytes,
                                   cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_off[s],    h_off[s],    bN*sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_len[s],    h_len[s],    bN*sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, streams[s]));

        // Kernel on stream s
        sha256_kernel_b2<<<grid(bN, block_size), block_size, 0, streams[s]>>>(
            d_data[s], d_off[s], d_len[s], d_digest[s], bN);

        // Async D2H
        CUDA_CHECK(cudaMemcpyAsync(h_digest[s], d_digest[s],
                                   (size_t)bN * SHA256_DIGEST_BYTES,
                                   cudaMemcpyDeviceToHost, streams[s]));
    }

    // Sync all streams
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&res.total_ms, ev_start, ev_stop));

    // Collect digests from pinned buffers into h_out (in order)
    for (int b = 0; b < n_batches; ++b) {
        int s = b % S;
        uint32_t msg_start = (uint32_t)b * batch_n;
        uint32_t msg_end   = msg_start + batch_n;
        if (msg_end > ds.N) msg_end = ds.N;
        uint32_t bN = msg_end - msg_start;
        memcpy(h_out + (size_t)msg_start * SHA256_DIGEST_BYTES,
               h_digest[s],
               (size_t)bN * SHA256_DIGEST_BYTES);
    }

    // ── CPU reference ────────────────────────────────────────────────────────
    auto cpu_start = std::chrono::high_resolution_clock::now();
    uint8_t cpu_sink[32];
    for (uint32_t i = 0; i < ds.N; ++i)
        sha256_cpu(ds.data + ds.offsets[i], ds.lengths[i], cpu_sink);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    res.cpu_ms = (float)std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // approximate breakdown (kernel dominates once overlap occurs)
    res.kernel_ms      = res.total_ms;   // total pipeline time
    res.h2d_ms         = 0; res.d2h_ms = 0;
    res.gpu_mhash      = ds.N / (res.total_ms * 1e3);
    res.cpu_mhash      = ds.N / (res.cpu_ms   * 1e3);
    res.speedup_e2e    = res.cpu_ms / res.total_ms;
    res.speedup_kernel = res.speedup_e2e;

    // ── Verification ─────────────────────────────────────────────────────────
    if (verify)
        verify_sample(ds, h_out, (float)VERIFY_SAMPLE / ds.N);

    // ── Cleanup ──────────────────────────────────────────────────────────────
    delete[] h_out;
    for (int s = 0; s < S; ++s) {
        cudaFreeHost(h_data[s]); cudaFreeHost(h_off[s]);
        cudaFreeHost(h_len[s]);  cudaFreeHost(h_digest[s]);
        cudaFree(d_data[s]);     cudaFree(d_off[s]);
        cudaFree(d_len[s]);      cudaFree(d_digest[s]);
        cudaStreamDestroy(streams[s]);
    }
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// print_result: pretty-print timing breakdown
// ─────────────────────────────────────────────────────────────────────────────
static void print_result(const char* label, uint32_t N, const TimingResult& r) {
    printf("\n┌─────────────────────────────────────────────────────────┐\n");
    printf("│  %-54s│\n", label);
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│  Messages         : %10u                           │\n", N);
    printf("│  H2D transfer     : %10.3f ms                         │\n", r.h2d_ms);
    printf("│  Kernel           : %10.3f ms                         │\n", r.kernel_ms);
    printf("│  D2H transfer     : %10.3f ms                         │\n", r.d2h_ms);
    printf("│  GPU end-to-end   : %10.3f ms                         │\n", r.total_ms);
    printf("│  CPU single-core  : %10.3f ms                         │\n", r.cpu_ms);
    printf("│  GPU Mhash/s      : %10.2f                            │\n", r.gpu_mhash);
    printf("│  CPU Mhash/s      : %10.2f                            │\n", r.cpu_mhash);
    printf("│  Speedup (kernel) :     %6.1f×                         │\n", r.speedup_kernel);
    printf("│  Speedup (e2e)    :     %6.1f×                         │\n", r.speedup_e2e);
    printf("└─────────────────────────────────────────────────────────┘\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// E0 — FIPS 180-4 self-test
// ─────────────────────────────────────────────────────────────────────────────
static void experiment_E0() {
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  E0: FIPS 180-4 Self-Test (CPU reference)\n");
    printf("══════════════════════════════════════════════════════\n");
    bool ok = fips_selftest();
    if (!ok) {
        fprintf(stderr, "[FATAL] FIPS test failed — aborting.\n");
        exit(EXIT_FAILURE);
    }
    printf("[E0] All FIPS vectors verified ✓\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// E1 — Default benchmark: N=1M, len 16-200, B0 vs B2 vs CPU
// ─────────────────────────────────────────────────────────────────────────────
static void experiment_E1() {
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  E1: Default — 1M messages, len 16-200 B\n");
    printf("══════════════════════════════════════════════════════\n");

    const uint32_t N = 1000000;
    Dataset ds = generate_dataset(N, 16, 200, 42, false);

    TimingResult b0 = run_b0(ds, BLOCK_SIZE, true);
    print_result("B0 — Synchronous baseline", N, b0);

    // For B2 we need a re-created dataset (run_b2 doesn't modify ds)
    TimingResult b2 = run_b2(ds, BLOCK_SIZE, true);
    // B2 reports total pipeline time; compute pseudo-speedup vs B0
    printf("\n  B2 pipeline time      : %.3f ms\n",  b2.total_ms);
    printf("  B2 end-to-end speedup vs B0: %.2f×\n", b0.total_ms / b2.total_ms);
    printf("  B2 end-to-end speedup vs CPU: %.2f×\n", b2.speedup_e2e);

    free_dataset(ds);
}

// ─────────────────────────────────────────────────────────────────────────────
// E2 — Block-size sweep on B0
// ─────────────────────────────────────────────────────────────────────────────
static void experiment_E2() {
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  E2: Block-size sweep (B0, N=1M, len 16-200 B)\n");
    printf("══════════════════════════════════════════════════════\n");
    printf("  %-8s  %-12s  %-12s  %-10s\n",
           "BlkSize", "Kernel(ms)", "GPU Mh/s", "Speedup");

    const uint32_t N = 1000000;
    Dataset ds = generate_dataset(N, 16, 200, 42, false);
    int blksizes[] = {64, 128, 256, 512, 1024};
    for (int bs : blksizes) {
        TimingResult r = run_b0(ds, bs, false);
        printf("  %-8d  %-12.3f  %-12.2f  %-6.1f×\n",
               bs, r.kernel_ms, r.gpu_mhash, r.speedup_kernel);
    }
    free_dataset(ds);
}

// ─────────────────────────────────────────────────────────────────────────────
// E3 — Crossover: fixed N=1M, sweep average message length
// ─────────────────────────────────────────────────────────────────────────────
static void experiment_E3() {
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  E3: Length crossover (B0, N=1M)\n");
    printf("══════════════════════════════════════════════════════\n");
    printf("  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
           "Len(B)", "H2D(ms)", "Ker(ms)", "D2H(ms)", "E2E(ms)", "Speedup");

    const uint32_t N = 1000000;
    int lengths[] = {16, 64, 256, 1024, 4096};
    for (int l : lengths) {
        Dataset ds = generate_dataset(N, l, l, 7, false);
        TimingResult r = run_b0(ds, BLOCK_SIZE, false);
        printf("  %-8d  %-10.3f  %-10.3f  %-10.3f  %-10.3f  %-6.1f×\n",
               l, r.h2d_ms, r.kernel_ms, r.d2h_ms, r.total_ms, r.speedup_e2e);
        free_dataset(ds);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E4 — Scalability: sweep N, fixed len=128
// ─────────────────────────────────────────────────────────────────────────────
static void experiment_E4() {
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  E4: Scalability — B0 vs B2, len=128 B\n");
    printf("══════════════════════════════════════════════════════\n");
    printf("  %-10s  %-12s  %-12s  %-12s  %-10s\n",
           "N", "B0 E2E(ms)", "B2 E2E(ms)", "B0 Mh/s", "B2/B0");

    uint32_t Ns[] = {10000, 50000, 100000, 500000, 1000000, 5000000};
    for (uint32_t N : Ns) {
        Dataset ds = generate_dataset(N, 128, 128, 13, false);
        TimingResult b0 = run_b0(ds, BLOCK_SIZE, false);
        TimingResult b2 = run_b2(ds, BLOCK_SIZE, false);
        printf("  %-10u  %-12.3f  %-12.3f  %-12.2f  %-6.2f×\n",
               N, b0.total_ms, b2.total_ms, b0.gpu_mhash,
               b0.total_ms / b2.total_ms);
        free_dataset(ds);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E5 — Operating map: grid over (N, avg_len), output GPU speedup vs CPU
// ─────────────────────────────────────────────────────────────────────────────
static void experiment_E5() {
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  E5: GPU Operating Map (end-to-end speedup vs CPU)\n");
    printf("══════════════════════════════════════════════════════\n");

    uint32_t Ns[]   = {10000, 100000, 500000, 1000000, 5000000};
    int      lens[] = {16, 64, 256, 1024, 4096};
    int      nN     = 5, nL = 5;

    // CSV-style header for easy plotting
    printf("\n  GPU_SPEEDUP_CSV (copy into Python for heatmap)\n");
    printf("  N");
    for (int l = 0; l < nL; ++l) printf(",len=%d", lens[l]);
    printf("\n");

    for (int i = 0; i < nN; ++i) {
        printf("  %u", Ns[i]);
        for (int j = 0; j < nL; ++j) {
            Dataset ds = generate_dataset(Ns[i], lens[j], lens[j], 77, false);
            TimingResult r = run_b0(ds, BLOCK_SIZE, false);
            printf(",%.2f", r.speedup_e2e);
            free_dataset(ds);
        }
        printf("\n");
    }

    printf("\n  (Values < 1.0 = CPU is faster. Values > 1.0 = GPU wins.)\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Device info banner
// ─────────────────────────────────────────────────────────────────────────────
static void print_device_info() {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  GPU Device Info                                     ║\n");
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Name         : %-36s║\n", prop.name);
    printf("║  Compute cap  : sm_%d%d                                ║\n",
           prop.major, prop.minor);
    printf("║  SMs          : %-4d                                  ║\n",
           prop.multiProcessorCount);
    printf("║  Global mem   : %-5zu MB                              ║\n",
           prop.totalGlobalMem / (1024*1024));
    printf("║  Mem BW       : ~%-4.0f GB/s                           ║\n",
           2.0 * prop.memoryClockRate * 1e3 * prop.memoryBusWidth / 8.0 / 1e9);
    printf("║  Warp size    : %-4d                                  ║\n",
           prop.warpSize);
    printf("╚══════════════════════════════════════════════════════╝\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   Parallel SHA-256 on GPU — Benchmark Suite         ║\n");
    printf("║   Swaraj Mahindrakar — G25AIT1179 — IIT Jodhpur     ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    print_device_info();

    // Parse optional experiment selector: ./sha256_bench [0-5|all]
    bool run_all = true;
    int  sel     = -1;
    if (argc >= 2) {
        if (strcmp(argv[1], "all") != 0) {
            sel     = atoi(argv[1]);
            run_all = false;
        }
    }

    if (run_all || sel == 0) experiment_E0();
    if (run_all || sel == 1) experiment_E1();
    if (run_all || sel == 2) experiment_E2();
    if (run_all || sel == 3) experiment_E3();
    if (run_all || sel == 4) experiment_E4();
    if (run_all || sel == 5) experiment_E5();

    printf("\n[Done] All selected experiments complete.\n\n");
    return 0;
}
