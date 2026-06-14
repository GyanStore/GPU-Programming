#!/usr/bin/env python3
"""
plot_results.py
───────────────────────────────────────────────────────────────────────────────
Parse the CSV block printed by E5 and generate:
  1. Operating-map heatmap (GPU speedup vs N and message length)
  2. Crossover line plot (kernel time vs H2D time vs message length, E3)
  3. Scalability plot (B0 vs B2 end-to-end time vs N, E4)

Usage:
  1. Run: ./sha256_bench 5 | tee e5_output.txt
     Then: python3 plot_results.py --e5 e5_output.txt

  2. Or paste E3/E4 numbers directly into the arrays at the bottom.

Author : Shruthi Chinnasamy (G25AIT1165)
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─── Default illustrative data (replace with real measured values) ─────────────
# E3: Length crossover — B0, N=1M
E3_lengths    = [16,   64,   256,  1024, 4096]
E3_h2d_ms     = [2.1,  2.3,  3.8,  9.1,  34.2]  # REPLACE with measured
E3_kernel_ms  = [0.9,  1.1,  3.2,  12.5, 49.8]   # REPLACE with measured
E3_d2h_ms     = [0.4,  0.4,  0.5,  0.8,  2.1]    # REPLACE with measured

# E4: Scalability — B0 vs B2, len=128
E4_N          = [10000, 50000, 100000, 500000, 1000000, 5000000]
E4_b0_ms      = [1.2,   1.8,   2.9,    8.1,    15.3,    74.1]   # REPLACE
E4_b2_ms      = [1.0,   1.4,   2.1,    5.3,    9.8,     47.2]   # REPLACE

# E5: Operating map — rows = N, cols = len
E5_N          = [10000, 100000, 500000, 1000000, 5000000]
E5_lens       = [16,    64,     256,    1024,    4096]
# Rows = N, Cols = len.  Values = GPU end-to-end speedup vs single-core CPU.
E5_speedup    = np.array([
    [0.3,  0.5,  0.9,  1.1,  1.4],   # N=10k
    [0.8,  1.4,  3.2,  7.1,  14.2],  # N=100k
    [1.1,  3.2,  9.8,  22.1, 38.4],  # N=500k
    [1.3,  5.1,  14.2, 31.0, 52.3],  # N=1M
    [1.5,  7.8,  22.1, 44.8, 71.2],  # N=5M
])  # REPLACE all values with your measured data


# ─── Plot 1: Operating Map ──────────────────────────────────────────────────
def plot_operating_map(speedup, N_vals, len_vals):
    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.RdYlGn   # red = CPU better, green = GPU better
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=speedup.max())
    im   = ax.imshow(speedup, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(len(len_vals)))
    ax.set_xticklabels([str(l) for l in len_vals])
    ax.set_yticks(range(len(N_vals)))
    ax.set_yticklabels([f'{n:,}' for n in N_vals])

    ax.set_xlabel("Average Message Length (bytes)", fontsize=12)
    ax.set_ylabel("Number of Messages (N)", fontsize=12)
    ax.set_title("GPU End-to-End Speedup vs Single-Core CPU\n(SHA-256 Bulk Fingerprinting — B0 Baseline)", fontsize=12)

    # Annotate cells
    for i in range(len(N_vals)):
        for j in range(len(len_vals)):
            v    = speedup[i, j]
            col  = 'black' if 0.4 < v < 3.0 else 'white'
            text = f'{v:.1f}×'
            ax.text(j, i, text, ha='center', va='center', color=col, fontsize=9, fontweight='bold')

    # Draw contour at speedup = 1.0 (GPU/CPU parity line)
    ax.contour(speedup, levels=[1.0], colors=['black'], linewidths=2,
               linestyles='--')

    fig.colorbar(im, ax=ax, label='GPU speedup (×)', shrink=0.85)
    plt.tight_layout()
    plt.savefig("operating_map.pdf", bbox_inches='tight', dpi=150)
    plt.savefig("operating_map.png", bbox_inches='tight', dpi=150)
    print("[Plot] Saved operating_map.pdf / .png")


# ─── Plot 2: Crossover (E3) ─────────────────────────────────────────────────
def plot_crossover():
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(E3_lengths, E3_h2d_ms,    'o-b',  label='H2D Transfer',   linewidth=2)
    ax.plot(E3_lengths, E3_kernel_ms,  's-g',  label='Kernel Execution', linewidth=2)
    ax.plot(E3_lengths, E3_d2h_ms,    '^-r',  label='D2H Transfer',   linewidth=2)

    # Mark crossover (H2D == kernel)
    for i in range(len(E3_lengths) - 1):
        if (E3_h2d_ms[i] > E3_kernel_ms[i]) != (E3_h2d_ms[i+1] > E3_kernel_ms[i+1]):
            xc = (E3_lengths[i] + E3_lengths[i+1]) / 2
            ax.axvline(xc, color='gray', linestyle=':', linewidth=1.5)
            ax.text(xc + 20, max(E3_h2d_ms) * 0.7,
                    f'Crossover\n~{xc:.0f} B', color='gray', fontsize=9)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Average Message Length (bytes)", fontsize=12)
    ax.set_ylabel("Time (ms)  [log scale]", fontsize=12)
    ax.set_title("Transfer vs. Compute Time — B0, N = 1,000,000", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("crossover.pdf", bbox_inches='tight', dpi=150)
    plt.savefig("crossover.png", bbox_inches='tight', dpi=150)
    print("[Plot] Saved crossover.pdf / .png")


# ─── Plot 3: Scalability (E4) ────────────────────────────────────────────────
def plot_scalability():
    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.plot(E4_N, E4_b0_ms, 'o-b', label='B0 Sync', linewidth=2)
    ax1.plot(E4_N, E4_b2_ms, 's-g', label='B2 Pinned+Streams', linewidth=2)
    ax1.set_xscale('log')
    ax1.set_xlabel("Number of Messages (N)  [log scale]", fontsize=12)
    ax1.set_ylabel("End-to-End Time (ms)", fontsize=12, color='black')
    ax1.set_title("Scalability: B0 vs B2 — Average Length = 128 B", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)

    # Secondary axis: B2/B0 speedup ratio
    ax2 = ax1.twinx()
    ratio = [b0 / b2 for b0, b2 in zip(E4_b0_ms, E4_b2_ms)]
    ax2.plot(E4_N, ratio, 'd--k', label='B2/B0 speedup', linewidth=1.5, alpha=0.6)
    ax2.set_ylabel("Pipeline speedup (B0/B2)", fontsize=11, color='gray')
    ax2.tick_params(axis='y', colors='gray')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig("scalability.pdf", bbox_inches='tight', dpi=150)
    plt.savefig("scalability.png", bbox_inches='tight', dpi=150)
    print("[Plot] Saved scalability.pdf / .png")


# ─── Parse E5 CSV from benchmark output ──────────────────────────────────────
def parse_e5_csv(filepath):
    """
    Parses the CSV block from E5 output. Example format:
      N,len=16,len=64,...
      10000,0.31,0.52,...
    """
    with open(filepath) as f:
        lines = f.readlines()

    in_csv  = False
    rows    = []
    headers = None
    for line in lines:
        line = line.strip()
        if line.startswith("GPU_SPEEDUP_CSV"):
            in_csv = True; continue
        if not in_csv: continue
        if not line or line.startswith("(Values"): break
        parts = line.lstrip().split(',')
        if headers is None:
            headers = parts; continue
        try:
            vals = [float(p) for p in parts]
            rows.append(vals)
        except ValueError:
            continue

    N_vals   = [int(r[0]) for r in rows]
    len_vals = [int(h.split('=')[1]) for h in headers[1:]]
    data     = np.array([[v for v in r[1:]] for r in rows])
    return data, N_vals, len_vals


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot SHA-256 GPU benchmark results")
    parser.add_argument("--e5", help="Path to file containing E5 CSV output")
    args = parser.parse_args()

    speedup_data  = E5_speedup
    n_vals        = E5_N
    len_vals      = E5_lens

    if args.e5:
        try:
            speedup_data, n_vals, len_vals = parse_e5_csv(args.e5)
            print(f"[Info] Parsed E5 data: {speedup_data.shape[0]} N values, "
                  f"{speedup_data.shape[1]} length values")
        except Exception as ex:
            print(f"[Warn] Could not parse {args.e5}: {ex} — using defaults")

    plot_operating_map(speedup_data, n_vals, len_vals)
    plot_crossover()
    plot_scalability()
    print("\nAll plots saved. Insert operating_map.pdf and crossover.pdf into paper.")
