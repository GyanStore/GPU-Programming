/*
 * kernels.cuh
 * Device-side SHA-256 compression function and kernel variants:
 *   - sha256_kernel_b0 : baseline, one thread per message
 *   - sha256_kernel_b2 : same kernel, used with stream pipeline on host side
 *
 * The compression function is self-contained per thread — no shared memory,
 * no inter-thread communication, no synchronisation barriers.
 *
 * Author : Shruthi Chinnasamy (G25AIT1165)
 */

#pragma once
#include "sha256.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// compress_block: absorb one 64-byte block into state[0..7]
// All state words and working variables live in registers.
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__
void compress_block(uint32_t state[8], const uint8_t block[64])
{
    // Build message schedule W[0..63]
    uint32_t W[64];
    #pragma unroll 16
    for (int i = 0; i < 16; ++i)
        W[i] = load_be32(block + 4 * i);

    #pragma unroll 48
    for (int i = 16; i < 64; ++i)
        W[i] = sig1(W[i-2]) + W[i-7] + sig0(W[i-15]) + W[i-16];

    // Working variables
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    // 64 rounds — fully unrolled
    #pragma unroll 64
    for (int i = 0; i < 64; ++i) {
        uint32_t T1 = h + SIG1(e) + CH(e,f,g) + d_K[i] + W[i];
        uint32_t T2 = SIG0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// ─────────────────────────────────────────────────────────────────────────────
// sha256_one_message: per-thread SHA-256 for a single variable-length message.
// Handles any number of full 64-byte blocks plus the final padded block(s).
// ─────────────────────────────────────────────────────────────────────────────
__device__
void sha256_one_message(const uint8_t* __restrict__ msg,
                         uint32_t                    len,
                         uint8_t*  __restrict__      digest)
{
    // Initialise state from constant memory
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) state[i] = d_H0[i];

    // ── Process all complete 64-byte blocks ──────────────────────────────
    uint32_t remaining = len;
    const uint8_t* ptr = msg;

    while (remaining >= 64) {
        compress_block(state, ptr);
        ptr       += 64;
        remaining -= 64;
    }

    // ── Build final padded block(s) in a thread-local buffer ─────────────
    // After padding: total length is a multiple of 64.
    // Rule: append 0x80, zero-fill to byte 56 of the block,
    //       then write 64-bit big-endian bit-length.
    // If remaining tail >= 56 we need TWO padding blocks.
    uint8_t buf[128];  // worst case: two 64-byte blocks
    for (int i = 0; i < 128; ++i) buf[i] = 0x00;  // zero-fill

    // Copy tail into buf
    for (uint32_t i = 0; i < remaining; ++i) buf[i] = ptr[i];

    // Append 0x80
    buf[remaining] = 0x80;

    // Write 64-bit big-endian bit-count into correct position
    uint64_t bitlen = (uint64_t)len * 8;

    int blocks_needed = (remaining < 56) ? 1 : 2;
    int bitlen_offset = blocks_needed * 64 - 8;  // last 8 bytes of last block

    buf[bitlen_offset + 0] = (uint8_t)(bitlen >> 56);
    buf[bitlen_offset + 1] = (uint8_t)(bitlen >> 48);
    buf[bitlen_offset + 2] = (uint8_t)(bitlen >> 40);
    buf[bitlen_offset + 3] = (uint8_t)(bitlen >> 32);
    buf[bitlen_offset + 4] = (uint8_t)(bitlen >> 24);
    buf[bitlen_offset + 5] = (uint8_t)(bitlen >> 16);
    buf[bitlen_offset + 6] = (uint8_t)(bitlen >>  8);
    buf[bitlen_offset + 7] = (uint8_t)(bitlen);

    // Compress final block(s)
    compress_block(state, buf);
    if (blocks_needed == 2)
        compress_block(state, buf + 64);

    // ── Write digest in big-endian order ─────────────────────────────────
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        store_be32(digest + 4 * i, state[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// B0 — Baseline kernel: one block of 256 threads, one thread per message.
// Grid covers all N messages.  Surplus threads exit immediately.
// ─────────────────────────────────────────────────────────────────────────────
__global__
void sha256_kernel_b0(const uint8_t* __restrict__ data,     // packed messages
                      const uint32_t* __restrict__ offsets,  // byte offset of each message
                      const uint32_t* __restrict__ lengths,  // byte length of each message
                            uint8_t* __restrict__ digests,   // output: N * 32 bytes
                            uint32_t             N)          // total message count
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    sha256_one_message(data    + offsets[idx],
                       lengths[idx],
                       digests + (uint64_t)idx * SHA256_DIGEST_BYTES);
}

// ─────────────────────────────────────────────────────────────────────────────
// B2 — Batch kernel: same logic as B0 but for a sub-batch of messages.
// The host stream-pipeline calls this once per batch per stream.
// batch_start: first global message index in this batch
// batch_n    : number of messages in this batch
// ─────────────────────────────────────────────────────────────────────────────
__global__
void sha256_kernel_b2(const uint8_t*  __restrict__ data,
                      const uint32_t* __restrict__ offsets,
                      const uint32_t* __restrict__ lengths,
                            uint8_t*  __restrict__ digests,
                            uint32_t              batch_n)
{
    uint32_t local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= batch_n) return;

    sha256_one_message(data    + offsets[local_idx],
                       lengths[local_idx],
                       digests + (uint64_t)local_idx * SHA256_DIGEST_BYTES);
}
