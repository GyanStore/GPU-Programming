#pragma once
/*
 * sha256.cuh
 * Shared device constants, macros, and type definitions for the
 * parallel SHA-256 GPU engine.
 *
 * Author : Shruthi Chinnasamy (G25AIT1165)
 * Course : GPU Programming — IIT Jodhpur
 */

#include <cstdint>
#include <cstdio>

// ─── SHA-256 initial hash values (first 32 bits of fractional parts of sqrt of first 8 primes) ───
__constant__ uint32_t d_H0[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

// ─── SHA-256 round constants (first 32 bits of fractional parts of cbrt of first 64 primes) ───
__constant__ uint32_t d_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbefu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

// ─── Bit manipulation macros ───
#define ROTR32(x, n)  (((x) >> (n)) | ((x) << (32u - (n))))
#define SHR(x, n)     ((x) >> (n))

#define CH(e, f, g)   (((e) & (f)) ^ (~(e) & (g)))
#define MAJ(a, b, c)  (((a) & (b)) ^ ((a) & (c)) ^ ((b) & (c)))

#define SIG0(a)  (ROTR32(a,  2) ^ ROTR32(a, 13) ^ ROTR32(a, 22))
#define SIG1(e)  (ROTR32(e,  6) ^ ROTR32(e, 11) ^ ROTR32(e, 25))
#define sig0(x)  (ROTR32(x,  7) ^ ROTR32(x, 18) ^ SHR(x,   3))
#define sig1(x)  (ROTR32(x, 17) ^ ROTR32(x, 19) ^ SHR(x,  10))

// Big-endian byte load (GPU-safe)
__device__ __forceinline__ uint32_t load_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
         | ((uint32_t)p[2] <<  8) |  (uint32_t)p[3];
}

// Big-endian byte store
__device__ __forceinline__ void store_be32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >>  8);
    p[3] = (uint8_t)(v);
}

// CUDA error check macro
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Digest size
#define SHA256_DIGEST_BYTES  32
