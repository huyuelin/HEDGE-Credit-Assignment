#!/usr/bin/env python3
"""
Plot collapse curve (Figure 3 / B1_collapse_curve.pdf).

L1O gain declines and sign-flips at n≈8; HEDGE grows monotonically.
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13

OUTPUT_DIR = "/data/jackey_workspace/hedge_results"


def plot_collapse_curve(results_file=None):
    """Generate collapse curve figure."""
    if results_file and os.path.exists(results_file):
        with open(results_file) as f:
            all_results = json.load(f)
        team_sizes = sorted(set(r["n"] for r in all_results))
        l1o_gains = []
        hedge_gains = []
        for n in team_sizes:
            n_res = [r for r in all_results if r["n"] == n]
            uniform = np.mean([r["uniform_acc"] for r in n_res])
            l1o = np.mean([r["l1o_acc"] for r in n_res])
            hedge = np.mean([r["hedge_acc"] for r in n_res])
            l1o_gains.append(l1o - uniform)
            hedge_gains.append(hedge - uniform)
    else:
        # Use paper values
        team_sizes = [2, 3, 4, 6, 8, 10, 12]
        l1o_gains = [5.6, 4.3, 3.8, 2.1, 0.2, -1.9, -3.4]
        hedge_gains = [5.9, 5.8, 6.3, 7.1, 7.6, 7.9, 8.1]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    ax.plot(team_sizes, l1o_gains, 'o-', color='#d62728', linewidth=2, markersize=8, label='L1O')
    ax.plot(team_sizes, hedge_gains, 's-', color='#2ca02c', linewidth=2, markersize=8, label='HEDGE')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=8, color='gray', linestyle=':', alpha=0.5, label='Sign-flip (n≈8)')

    # Annotate hetero%
    hetero_pcts = [33, 33, 38, 46, 44, 56, 60]
    for i, (n, hg, pct) in enumerate(zip(team_sizes, hedge_gains, hetero_pcts)):
        if i % 2 == 0 or i == len(team_sizes) - 1:
            ax.annotate(f'{pct}%', (n, hg), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=9, color='#2ca02c')

    ax.set_xlabel('Team Size (n)')
    ax.set_ylabel('Gain over Uniform (%)')
    ax.set_title('Credit Collapse Curve')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_xticks(team_sizes)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 10)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "B1_collapse_curve.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    results_file = sys.argv[1] if len(sys.argv) > 1 else None
    plot_collapse_curve(results_file)
