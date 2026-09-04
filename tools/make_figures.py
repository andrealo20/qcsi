#!/usr/bin/env python3
"""
Regenerate the figures used in the README.

Every number plotted here was measured, not modelled: the quantisation sweep
comes from tools/compare_quantised.py on SignFi lab_150, and the phase
statistics from tools/check_dataset.py on the same capture. The script exists
so the figures can be rebuilt when the measurements change, and so that where
each figure comes from is written down rather than remembered.

Usage:
    python3 tools/make_figures.py [output_dir]

Requires: numpy, matplotlib.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

INK = "#1b1f24"
ACCENT = "#c2410c"
MUTED = "#94a3b8"
GRID = "#e2e8f0"

# --- measured: tools/compare_quantised.py, SignFi lab_150, within-subject,
#     2250 test samples, 150 classes, antennas 0 and 1 -----------------------
BITS = [15, 12, 10, 8, 7, 6, 5, 4]
ACCURACY = [42.27, 42.27, 42.18, 42.04, 41.78, 41.33, 39.29, 32.31]
DISAGREE = [0, 2, 19, 71, 143, 332, 642, 1080]
LOW_MARGIN = [0.00, 0.18, 1.69, 6.31, 12.71, 29.07, 49.33, 66.13]
HIGH_MARGIN = [0.00, 0.00, 0.00, 0.00, 0.00, 0.44, 7.73, 29.87]
FLOAT_ACC = 42.27

# --- measured: tools/check_dataset.py, same capture ------------------------
PAIRS = ["rx0-rx1", "rx0-rx2", "rx1-rx2"]
OFFSET_REDUCTION = [2.5, 1.1, 1.1]
SLOPE_REDUCTION = [1.52, 0.97, 0.97]


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def fig_quantisation(outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(BITS))

    ax1.axhline(FLOAT_ACC, color=MUTED, linestyle="--", linewidth=1.2,
                label=f"float ({FLOAT_ACC:.2f}%)", zorder=1)
    ax1.plot(x, ACCURACY, "o-", color=ACCENT, linewidth=2, markersize=6,
             zorder=3, label="fixed point")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Q{b}" for b in BITS])
    ax1.set_ylabel("accuracy (%)")
    ax1.set_title("Accuracy against word length", color=INK, fontsize=11,
                  loc="left", pad=10)
    ax1.set_ylim(min(ACCURACY) - 2.5, FLOAT_ACC + 1.4)
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    ax1.annotate("bit-exact with float", xy=(0.05, ACCURACY[0]),
                 xytext=(1.15, ACCURACY[0] - 3.2), fontsize=8.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    style(ax1)

    width = 0.38
    ax2.bar(x - width / 2, LOW_MARGIN, width, color=ACCENT,
            label="narrow margin", zorder=3)
    ax2.bar(x + width / 2, HIGH_MARGIN, width, color=MUTED,
            label="wide margin", zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Q{b}" for b in BITS])
    ax2.set_ylabel("predictions changed (%)")
    ax2.set_title("Where the disagreements sit", color=INK, fontsize=11,
                  loc="left", pad=10)
    ax2.set_ylim(0, max(LOW_MARGIN) * 1.22)
    # Q15 and Q12 are zero or near-zero, so a note is needed: an absent bar
    # and a missing measurement look identical on a bar chart.
    ax2.text(0.02, 0.94, "Q15 and Q12 bars are zero, not missing",
             transform=ax2.transAxes, fontsize=8.5, color=MUTED, va="top")
    ax2.legend(frameon=False, fontsize=9, loc="upper left",
               bbox_to_anchor=(0.0, 0.88))
    style(ax2)

    fig.suptitle("SignFi lab_150, within-subject, 2250 test samples, "
                 "150 classes", color=MUTED, fontsize=9, y=1.04, x=0.005,
                 ha="left")
    fig.tight_layout()
    path = os.path.join(outdir, "quantisation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_antenna_pairs(outdir):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(PAIRS))
    width = 0.38

    b1 = ax.bar(x - width / 2, OFFSET_REDUCTION, width, color=ACCENT,
                label="phase offset", zorder=3)
    b2 = ax.bar(x + width / 2, SLOPE_REDUCTION, width, color=MUTED,
                label="residual slope", zorder=3)
    # The unity line is what the bars mean: above it the impairment shrank,
    # below it the operation made things worse.
    ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.1, zorder=4,
               label="no change")

    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}",
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8, color=INK, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(PAIRS)
    ax.set_ylabel("reduction factor")
    ax.set_ylim(0, max(OFFSET_REDUCTION) * 1.30)
    # The note goes inside the axes: placing it above pushed it past the
    # title, which read as a caption sitting on top of its own heading.
    ax.text(0.5, 0.965, "above the dashed line: impairment reduced",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, ha="center",
            va="top")
    ax.set_title("Effect of the conjugate product, by antenna pair",
                 color=INK, fontsize=11, loc="left", pad=10)
    # Upper right: the tallest bar and its value label are on the left, and
    # an automatic placement put the legend straight on top of them.
    ax.legend(frameon=False, fontsize=9, loc="upper right",
              bbox_to_anchor=(1.0, 0.90))
    style(ax)

    fig.tight_layout()
    path = os.path.join(outdir, "antenna_pairs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_phase_illustration(outdir):
    """Illustrative, not measured: what the conjugate product does.

    Two antennas are built explicitly, each carrying its own physical phase
    plus a shared per-packet impairment, and the conjugate product is then
    computed rather than asserted. An earlier version plotted the physical
    phase directly and labelled it "after the conjugate product", which was
    true only because the second antenna had been given zero physical phase,
    a coincidence stated in prose instead of shown by the arithmetic.

    Generated data is legitimate here because the figure explains an
    identity. It would not be legitimate as evidence, and the caption says
    so.
    """
    rng = np.random.default_rng(4)
    n_sub, n_pkt = 30, 12
    k = np.arange(n_sub)

    # Distinct physical responses: what the two antennas would see with no
    # impairment at all. Their difference is the quantity to recover.
    phys_a = 0.6 * np.sin(0.25 * k)
    phys_b = 0.25 * np.cos(0.18 * k + 0.7)
    target = np.angle(np.exp(1j * (phys_a - phys_b)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)

    for _ in range(n_pkt):
        # One oscillator, one sampling clock: the offset and slope are
        # identical on both antennas, which is the whole premise.
        cfo = rng.uniform(-np.pi, np.pi)
        sfo = rng.uniform(-0.08, 0.08)
        common = cfo + sfo * k

        h_a = np.exp(1j * (phys_a + common))
        h_b = np.exp(1j * (phys_b + common))

        ax1.plot(k, np.angle(h_a), color=MUTED, linewidth=1, alpha=0.8,
                 zorder=2)
        ax2.plot(k, np.angle(h_a * np.conj(h_b)), color=ACCENT, linewidth=1,
                 alpha=0.55, zorder=2)

    ax2.plot(k, target, color=INK, linewidth=1.4, linestyle="--", zorder=4,
             label=r"$\phi^{phys}_a - \phi^{phys}_b$")
    ax2.legend(frameon=False, fontsize=9, loc="upper right")

    ax1.set_title("Raw phase of one antenna", color=INK, fontsize=11,
                  loc="left", pad=10)
    ax1.set_xlabel("subcarrier")
    ax1.set_ylabel("phase (rad)")
    ax2.set_title(r"Phase of $H_a \cdot \overline{H_b}$", color=INK,
                  fontsize=11, loc="left", pad=10)
    ax2.set_xlabel("subcarrier")
    ax1.set_ylim(-np.pi - 0.3, np.pi + 0.3)
    for ax in (ax1, ax2):
        style(ax)

    fig.suptitle("12 packets of one unchanging scene. Left: the impairment "
                 "moves the phase anywhere. Right: it cancels, and all 12 "
                 "packets land on the same curve.\n"
                 "Illustrative, generated data.",
                 color=MUTED, fontsize=9, y=1.10, x=0.005, ha="left")
    fig.tight_layout()
    path = os.path.join(outdir, "phase_cancellation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "docs/images"
    os.makedirs(outdir, exist_ok=True)
    for fn in (fig_phase_illustration, fig_antenna_pairs, fig_quantisation):
        print("wrote", fn(outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
