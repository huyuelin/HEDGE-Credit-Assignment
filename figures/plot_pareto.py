#!/usr/bin/env python3
"""
Plot compute-performance Pareto front (Figure 5 / B3_compute_pareto.pdf).
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 11

OUTPUT_DIR = "/data/jackey_workspace/hedge_results"


def plot_pareto():
    """Generate Pareto front figure."""
    methods = ["L1O", "HEDGE-inv-H", "HEDGE", "VRRL", "MCTS (k=4)", "Math-Shepherd", "HEDGE+PRM"]
    compute = [1.0, 1.0, 1.02, 2.3, 8.4, 12.0, 13.0]
    accuracy = [49.8, 53.1, 55.7, 52.7, 53.3, 53.0, 57.1]
    colors = ['#d62728', '#9467bd', '#2ca02c', '#8c564b', '#e377c2', '#7f7f7f', '#17becf']
    markers = ['o', 'D', 's', '^', 'v', 'p', '*']

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    for i, (method, comp, acc) in enumerate(zip(methods, compute, accuracy)):
        ax.scatter(comp, acc, s=120, c=colors[i], marker=markers[i], zorder=5, label=method)

    # Pareto front line
    pareto_x = [1.0, 1.02, 13.0]
    pareto_y = [53.1, 55.7, 57.1]
    ax.plot(pareto_x, pareto_y, 'k--', alpha=0.3, linewidth=1)

    # Annotations
    offsets = [(10, -15), (10, 10), (-10, 12), (10, -12), (-15, -15), (10, 10), (-15, 12)]
    for i, (method, comp, acc) in enumerate(zip(methods, compute, accuracy)):
        ax.annotate(method, (comp, acc), textcoords="offset points",
                   xytext=offsets[i], fontsize=8.5, ha='center')

    ax.set_xlabel('Relative Compute (×)')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Compute-Performance Pareto Front (n=8)')
    ax.set_xscale('log')
    ax.set_xlim(0.8, 20)
    ax.set_ylim(48, 58.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "B3_compute_pareto.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_pareto()
