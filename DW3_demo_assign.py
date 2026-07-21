"""Create a schematic figure of the point-correspondence assignment.

The extractor queries the nearest watermarked point independently for every
original point.  Consequently, uniqueness is not guaranteed: several original
points can select the same watermarked point (many-to-one in the query
direction, or one-to-many when viewed from the watermarked point).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


ORIGINAL_COLOR = "#2F6FB0"
WATERMARKED_COLOR = "#D1495B"
SHARED_COLOR = "#E18D24"
ORDINARY_COLOR = "#84909A"


def _arrow(ax, start, end, color, rad=0.0, linewidth=1.7, zorder=1):
    """Draw a curved correspondence arrow between two points."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=linewidth,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=11,
        shrinkB=13,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def create_assignment_figure(output: str = "DW3_demo_assign.png", show=False):
    """Generate and save a publication-ready correspondence diagram."""
    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    # Positions form a bipartite schematic rather than physical coordinates.
    original = {
        r"$p_1$": (1.25, 5.1),
        r"$p_2$": (1.25, 4.0),
        r"$p_3$": (1.25, 3.0),
        r"$p_4$": (1.25, 2.0),
        r"$p_5$": (1.25, 0.9),
    }
    watermarked = {
        r"$q_7$": (5.75, 4.9),
        r"$q_2$": (5.75, 3.0),
        r"$q_9$": (5.75, 1.1),
    }

    # Independent nearest-neighbor assignments.  p2, p3, and p4 share q2.
    _arrow(ax, original[r"$p_1$"], watermarked[r"$q_7$"], ORDINARY_COLOR)
    _arrow(ax, original[r"$p_2$"], watermarked[r"$q_2$"], SHARED_COLOR, 0.08, 2.2)
    _arrow(ax, original[r"$p_3$"], watermarked[r"$q_2$"], SHARED_COLOR, 0.00, 2.2)
    _arrow(ax, original[r"$p_4$"], watermarked[r"$q_2$"], SHARED_COLOR, -0.08, 2.2)
    _arrow(ax, original[r"$p_5$"], watermarked[r"$q_9$"], ORDINARY_COLOR)

    for label, (x, y) in original.items():
        ax.scatter(x, y, s=210, facecolor="white", edgecolor=ORIGINAL_COLOR,
                   linewidth=2.8, zorder=3)
        ax.text(x - 0.32, y, label, ha="right", va="center", fontsize=17)

    for label, (x, y) in watermarked.items():
        ax.scatter(x, y, s=215, marker="D", color=WATERMARKED_COLOR,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(x + 0.32, y, label, ha="left", va="center", fontsize=17)

    ax.text(1.25, 5.85, "Original point cloud", ha="center", va="center",
            fontsize=17, fontweight="bold", color=ORIGINAL_COLOR)
    ax.text(5.75, 5.85, "Watermarked point cloud", ha="center", va="center",
            fontsize=17, fontweight="bold", color=WATERMARKED_COLOR)

    legend_items = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=ORIGINAL_COLOR, markeredgewidth=2,
               markersize=9, label="Original point"),
        Line2D([0], [0], marker="D", linestyle="none",
               markerfacecolor=WATERMARKED_COLOR, markeredgecolor="white",
               markersize=9, label="Watermarked point"),
        Line2D([0], [0], color=SHARED_COLOR, linewidth=2.2,
               label="Shared correspondence"),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.015), frameon=False, fontsize=12,
              handlelength=2.0, columnspacing=1.5)

    ax.set_xlim(0.0, 7.0)
    ax.set_ylim(0.0, 6.35)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.5)

    output_path = Path(output)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved correspondence figure: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw the many-to-one point-correspondence assignment."
    )
    parser.add_argument(
        "--output",
        default="DW3_demo_assign.png",
        help="Output image path (default: DW3_demo_assign.png)",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_assignment_figure(args.output, args.show)
