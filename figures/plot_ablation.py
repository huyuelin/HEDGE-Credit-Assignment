#!/usr/bin/env python3
"""
Plot ablation waterfall (Figure 6 / B4_ablation_waterfall.pdf).
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 11

OUTPUT_DIR = "/data/jackey_workspace/hedge_results"


def plot_ablation():
    """Generate ablation waterfall figure."""
    levels = ["L1O\n(base)", "+Entropy\nFilter", "+Inv-Var\nWeight",
              "Uniform\nShrink", "Var-Weight\nShrink", "HEDGE", "Oracle\nHEDGE"]
    accuracies = [53.9, 55.4, 56.9, 57.9, 60.3, 64.2, 65.1]
    deltas = [0] + [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = ['#1f77b4'] + ['#2ca02c' if d > 0 else '#d62728' for d in deltas[1:]]
    bars = ax.bar(range(len(levels)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add delta annotations
    for i, (acc, delta) in enumerate(zip(accuracies, deltas)):
        ax.text(i, acc + 0.3, f'{acc:.1f}', ha='center', fontsize=9, fontweight='bold')
        if i > 0:
            ax.text(i, acc - 1.5, f'+{delta:.1f}', ha='center', fontsize=8, color='white')

    # Connection lines
    for i in range(len(accuracies) - 1):
        ax.plot([i + 0.4, i + 0.6], [accuracies[i], accuracies[i]], 'k-', alpha=0.3, linewidth=0.5)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Six-Level Ablation Waterfall (n=12, MathChat)')
    ax.set_ylim(52, 67)
    ax.grid(True, axis='y', alpha=0.3)

    # Total gain annotation
    total_gain = accuracies[-2] - accuracies[0]
    ax.annotate(f'Total: +{total_gain:.1f}', xy=(5, 64.5), fontsize=11, fontweight='bold',
               color='#2ca02c', ha='center')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "B4_ablation_waterfall.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_ablation()
