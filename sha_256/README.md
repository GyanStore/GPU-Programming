# Parallel SHA-256 on GPU — Code Package
**Shruthi | G25AIT1165 | IIT Jodhpur**

---

## File Structure

```
sha256_gpu/
├── sha256.cuh        # Device constants, macros, CUDA_CHECK
├── kernels.cuh       # compress_block(), sha256_kernel_b0, sha256_kernel_b2
├── sha256_cpu.h      # CPU reference + FIPS 180-4 self-test
├── utils.h           # Dataset generation, verification, Timer
├── main.cu           # All experiments (E0–E5), main()
├── Makefile          # Auto-detects GPU arch
└── plot_results.py   # Python plots (operating map, crossover, scalability)
```

---

## Build

### Local machine / lab workstation
```bash
cd sha256_gpu
make            # auto-detects your GPU's compute capability
```

### Google Colab
```python
# Cell 1 — upload all files, then:
!nvcc -O3 -arch=sm_75 -std=c++17 --use_fast_math \
      main.cu -o sha256_bench
```
(Replace sm_75 with your Colab GPU's compute capability — check with `!nvidia-smi`.)

### Manual nvcc
```bash
nvcc -O3 -arch=sm_86 -std=c++17 --use_fast_math main.cu -o sha256_bench
```

---

## Run

```bash
# All experiments (E0–E5):
./sha256_bench

# Single experiment:
./sha256_bench 0    # E0: FIPS self-test
./sha256_bench 1    # E1: Default 1M messages, B0 vs B2 vs CPU
./sha256_bench 2    # E2: Block-size sweep
./sha256_bench 3    # E3: Message-length crossover
./sha256_bench 4    # E4: Scalability B0 vs B2
./sha256_bench 5    # E5: Operating map (CSV output)
```

### Capture E5 for plotting
```bash
./sha256_bench 5 | tee e5_output.txt
python3 plot_results.py --e5 e5_output.txt
# Generates: operating_map.pdf, crossover.pdf, scalability.pdf
```

---

## What each experiment measures

| Exp | What | Key output |
|-----|------|------------|
| E0  | FIPS 180-4 correctness | PASS/FAIL per vector |
| E1  | B0 vs B2 vs CPU, N=1M, len 16-200 B | Timing table + speedup |
| E2  | Block size (64/128/256/512/1024), B0 | Best block size for your GPU |
| E3  | len = 16/64/256/1K/4K, B0, N=1M | H2D vs kernel crossover point |
| E4  | N = 10K..5M, B0 vs B2, len=128 | Scalability + pipeline benefit |
| E5  | 5×5 grid (N × len), B0 | GPU speedup heatmap data |

---

## Inserting results into the paper

1. **Table I** — fill GPU/CPU hardware from your machine.
2. **Table II** — copy numbers from E1 B0 output.
3. **Table III** — copy B0 sync / B2 pipeline total times from E1.
4. **Figure 1** — `crossover.pdf` from E3 data.
5. **Figure 2** — `operating_map.pdf` from E5 data.

---

## Notes

- The CPU reference is single-threaded and compiled `-O3`; it is a fair
  baseline for the paper's "single CPU core" comparison.
- For very small N (< 10,000), GPU launch overhead dominates — this is
  expected and reported honestly in the operating map.
- If you see FIPS FAIL, check your GPU's compute capability flag; using
  the wrong `-arch` can silently produce wrong results on some devices.
