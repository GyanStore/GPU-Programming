#pragma once
/*
 * sha256_cpu.h
 * Pure-C++ reference SHA-256 — used for FIPS test-vector validation
 * and runtime correctness sampling against the GPU output.
 *
 * Author : Shruthi Chinnasamy (G25AIT1165)
 */

#include <cstdint>
#include <cstring>

// ─── Constants ───────────────────────────────────────────────────────────────
static const uint32_t SHA256_H0[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};
static const uint32_t SHA256_K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbefu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,
    0x923f82a4u,0xab1c5ed5u,0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
    0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,0xe49b69c1u,0xefbe4786u,
    0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,
    0x06ca6351u,0x14292967u,0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
    0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,0xa2bfe8a1u,0xa81a664bu,
    0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,
    0x5b9cca4fu,0x682e6ff3u,0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
    0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

#define CPU_ROTR(x,n)  (((x)>>(n))|((x)<<(32-(n))))
#define CPU_SHR(x,n)   ((x)>>(n))
#define CPU_CH(e,f,g)  (((e)&(f))^(~(e)&(g)))
#define CPU_MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))
#define CPU_S0(a)  (CPU_ROTR(a,2)^CPU_ROTR(a,13)^CPU_ROTR(a,22))
#define CPU_S1(e)  (CPU_ROTR(e,6)^CPU_ROTR(e,11)^CPU_ROTR(e,25))
#define CPU_s0(x)  (CPU_ROTR(x,7)^CPU_ROTR(x,18)^CPU_SHR(x,3))
#define CPU_s1(x)  (CPU_ROTR(x,17)^CPU_ROTR(x,19)^CPU_SHR(x,10))

static inline uint32_t be32(const uint8_t* p) {
    return ((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|(uint32_t)p[3];
}

static void cpu_compress(uint32_t st[8], const uint8_t blk[64]) {
    uint32_t W[64];
    for (int i = 0; i < 16; ++i) W[i] = be32(blk + 4*i);
    for (int i = 16; i < 64; ++i)
        W[i] = CPU_s1(W[i-2]) + W[i-7] + CPU_s0(W[i-15]) + W[i-16];

    uint32_t a=st[0],b=st[1],c=st[2],d=st[3];
    uint32_t e=st[4],f=st[5],g=st[6],h=st[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t T1 = h + CPU_S1(e) + CPU_CH(e,f,g) + SHA256_K[i] + W[i];
        uint32_t T2 = CPU_S0(a) + CPU_MAJ(a,b,c);
        h=g; g=f; f=e; e=d+T1;
        d=c; c=b; b=a; a=T1+T2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d;
    st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

// ─── Public API ──────────────────────────────────────────────────────────────

// Compute SHA-256 of `len` bytes at `msg`, write 32-byte digest to `out`.
static void sha256_cpu(const uint8_t* msg, size_t len, uint8_t out[32]) {
    uint32_t state[8];
    for (int i = 0; i < 8; ++i) state[i] = SHA256_H0[i];

    size_t remaining = len;
    const uint8_t* ptr = msg;

    // Full blocks
    while (remaining >= 64) {
        cpu_compress(state, ptr);
        ptr += 64; remaining -= 64;
    }

    // Padding
    uint8_t buf[128];
    memset(buf, 0, 128);
    memcpy(buf, ptr, remaining);
    buf[remaining] = 0x80;

    int blocks = (remaining < 56) ? 1 : 2;
    int off    = blocks * 64 - 8;
    uint64_t bitlen = (uint64_t)len * 8;
    for (int i = 0; i < 8; ++i)
        buf[off + i] = (uint8_t)(bitlen >> (56 - 8*i));

    cpu_compress(state, buf);
    if (blocks == 2) cpu_compress(state, buf + 64);

    for (int i = 0; i < 8; ++i) {
        out[4*i+0] = (uint8_t)(state[i] >> 24);
        out[4*i+1] = (uint8_t)(state[i] >> 16);
        out[4*i+2] = (uint8_t)(state[i] >>  8);
        out[4*i+3] = (uint8_t)(state[i]);
    }
}

// FIPS 180-4 test vectors — call once at startup.
// Returns true if all pass.
static bool fips_selftest() {
    struct TestCase { const char* input; uint8_t expected[32]; };
    TestCase cases[3] = {
        // SHA-256("") 
        {"", {0xe3,0xb0,0xc4,0x42,0x98,0xfc,0x1c,0x14,0x9a,0xfb,0xf4,0xc8,
              0x99,0x6f,0xb9,0x24,0x27,0xae,0x41,0xe4,0x64,0x9b,0x93,0x4c,
              0xa4,0x95,0x99,0x1b,0x78,0x52,0xb8,0x55}},
        // SHA-256("abc")  — FIPS 180-4 §B.1
        {"abc", {0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,
                 0x5d,0xae,0x2e,0xc7,0x3b,0x00,0x36,0x1b,0xbe,0xf0,0x46,0x91,
                 0x48,0xa2,0x4c,0x2b,0x0d,0x53,0xa9,0xef}},
        // SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         {0x24,0x8d,0x6a,0x61,0xd2,0x06,0x38,0xb8,0xe5,0xc0,0x26,0x93,
          0x0c,0x3e,0x60,0x39,0xa3,0x3c,0xe4,0x59,0x64,0xff,0x21,0x67,
          0xf6,0xec,0xed,0xd4,0x19,0xdb,0x06,0xc1}}
    };
    bool all_pass = true;
    for (int t = 0; t < 3; ++t) {
        uint8_t got[32];
        sha256_cpu((const uint8_t*)cases[t].input,
                   strlen(cases[t].input), got);
        bool pass = (memcmp(got, cases[t].expected, 32) == 0);
        printf("[FIPS] Test %d (\"%s...\"): %s\n",
               t, cases[t].input[0] ? cases[t].input : "<empty>",
               pass ? "PASS" : "FAIL");
        if (!pass) all_pass = false;
    }
    return all_pass;
}
