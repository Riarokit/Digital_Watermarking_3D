"""Correspondence-distance calibration for the proposed method.

This script evaluates the two correspondence thresholds without embedding a
watermark.  The correspondence decision depends on the geometric relation
between the original and attacked point clouds; the small embedding
displacement is immaterial for this calibration.

It reports and saves:
  * index correspondence: fraction whose same-index distance exceeds each
    threshold (only attacks retaining the number of points), and
  * coordinate correspondence: fraction of original points accepted by a
    nearest-neighbour coordinate match.

The distances are normalized by the median nearest-neighbour spacing of the
attacked point cloud, as in DW1_X1_func.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

try:
    import matplotlib.pyplot as plt
except ImportError:  # CSV and terminal tables remain usable without plotting.
    plt = None

import DW2_func as DW2F


# Include the current values (20, 2) and the former common value (30).
FACTORS = (1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
INDEX_DISPLAY_FACTORS = (10.0, 15.0, 20.0, 30.0)
COORDINATE_DISPLAY_FACTORS = (1.0, 2.0, 3.0, 10.0, 30.0)
DEFAULT_INPUTS = ("C:/bun_zipper.ply", "C:/dragon.ply", "C:/Armadillo.ply")

# =========================================================
# Attack settings for this calibration
# Edit these lists for ordinary use; no command-line arguments are needed.
# The graph uses the last value of each list as its representative condition.
# =========================================================
NOISE_LEVELS = (2.0,)                 # Gaussian noise percentage
SMOOTHING_ITERATIONS = (50,)          # Laplacian-smoothing iteration count
REORDERING_RATIOS = (0.1, 0.5, 1.0)   # fraction of vertices reordered
CROP_KEEP_RATIOS = (0.3,)             # fraction retained by x-axis cropping
VOXEL_PERCENTS = (0.8, 1.0, 1.2)      # voxel side / bounding-box diagonal (%)


def read_and_normalize(path: str) -> np.ndarray:
    """Read vertices from a point cloud or a triangle mesh, then normalize."""
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.vertices) == 0:
            raise ValueError(f"No point-cloud or mesh vertices found: {path}")
        pcd.points = mesh.vertices
    normalized = DW2F.normalize_point_cloud(pcd)
    xyz = np.asarray(normalized.points, dtype=np.float64).copy()
    if len(xyz) < 2:
        raise ValueError(f"At least two points are required: {path}")
    return xyz


def attacked_spacing(xyz: np.ndarray) -> float:
    """Median nearest-neighbour distance used as the normalization scale."""
    distances, _ = cKDTree(xyz).query(xyz, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if len(nearest) == 0:
        raise ValueError("Cannot estimate spacing from duplicate-only point cloud.")
    return float(np.median(nearest))


def measure_distances(original: np.ndarray, attacked: np.ndarray) -> tuple[float, np.ndarray, Optional[np.ndarray]]:
    """Return spacing, coordinate distances, and same-index distances."""
    spacing = attacked_spacing(attacked)
    coordinate_distances, _ = cKDTree(attacked).query(original, k=1)
    coordinate_distances = coordinate_distances / spacing
    index_distances = None
    if len(original) == len(attacked):
        index_distances = np.linalg.norm(original - attacked, axis=1) / spacing
    return spacing, coordinate_distances, index_distances


def fraction_leq(distances: np.ndarray, factor: float) -> float:
    return float(np.mean(distances <= factor))


def fraction_gt(distances: np.ndarray, factor: float) -> float:
    return float(np.mean(distances > factor))


def format_factor(factor: float) -> str:
    return f"{factor:g}"


def parse_number_list(text: str, argument_name: str) -> list[float]:
    """Parse a non-empty comma-separated list used for attack parameters."""
    try:
        values = [float(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{argument_name} must be a comma-separated list of numbers."
        ) from error
    if not values:
        raise argparse.ArgumentTypeError(f"{argument_name} must not be empty.")
    return values


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_index_table(rows: list[dict[str, object]]) -> None:
    print("\n[Index correspondence: percentage judged inconsistent]")
    header = ["model", "attack", "parameter", "q99", "max"] + [f"> {f:g}" for f in INDEX_DISPLAY_FACTORS]
    print(" | ".join(header))
    print("-|-".join("---" for _ in header))
    for row in rows:
        values = [
            str(row["model"]), str(row["attack"]), str(row["parameter"]),
            f"{float(row['index_q99']):.2f}", f"{float(row['index_max']):.2f}",
        ]
        values += [f"{100.0 * float(row[f'index_bad_{f:g}']):.3f}%" for f in INDEX_DISPLAY_FACTORS]
        print(" | ".join(values))


def print_coordinate_table(rows: list[dict[str, object]]) -> None:
    print("\n[Coordinate correspondence: percentage accepted]")
    header = ["model", "attack", "parameter", "Natt/Norig", "q95", "max"] + [f"<= {f:g}" for f in COORDINATE_DISPLAY_FACTORS]
    print(" | ".join(header))
    print("-|-".join("---" for _ in header))
    for row in rows:
        values = [
            str(row["model"]), str(row["attack"]), str(row["parameter"]),
            f"{float(row['point_ratio']):.3f}", f"{float(row['coordinate_q95']):.2f}", f"{float(row['coordinate_max']):.2f}",
        ]
        values += [f"{100.0 * float(row[f'coordinate_accept_{f:g}']):.3f}%" for f in COORDINATE_DISPLAY_FACTORS]
        print(" | ".join(values))


def plot_profiles(
    index_rows: list[dict[str, object]],
    coordinate_rows: list[dict[str, object]],
    output_path: Path,
    index_selected: tuple[tuple[str, str], ...],
    coordinate_selected: tuple[tuple[str, str], ...],
) -> bool:
    """Plot the strongest representative attacks, averaged across models."""
    if plt is None:
        return False
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for attack, parameter in index_selected:
        selected = [r for r in index_rows if r["attack"] == attack and r["parameter"] == parameter]
        if not selected:
            continue
        values = [100.0 * np.mean([float(r[f"index_bad_{f:g}"]) for r in selected]) for f in FACTORS]
        axes[0].plot(FACTORS, values, marker="o", label=f"{attack} ({parameter})")
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1, label="allowed ratio 1%")
    axes[0].axvline(20.0, color="tab:green", linestyle=":", label="current index factor = 20")
    axes[0].axvline(30.0, color="tab:red", linestyle=":", label="former factor = 30")
    axes[0].set_title("Same-index inconsistency rate")
    axes[0].set_xlabel("index_distance_factor")
    axes[0].set_ylabel("inconsistent points (%)")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    for attack, parameter in coordinate_selected:
        selected = [r for r in coordinate_rows if r["attack"] == attack and r["parameter"] == parameter]
        if not selected:
            continue
        values = [100.0 * np.mean([float(r[f"coordinate_accept_{f:g}"]) for r in selected]) for f in FACTORS]
        axes[1].plot(FACTORS, values, marker="o", label=f"{attack} ({parameter})")
    axes[1].axvline(2.0, color="tab:green", linestyle=":", label="current coordinate factor = 2")
    axes[1].axvline(30.0, color="tab:red", linestyle=":", label="former factor = 30")
    axes[1].set_title("Coordinate-match acceptance rate")
    axes[1].set_xlabel("coordinate_distance_factor")
    axes[1].set_ylabel("original points accepted (%)")
    axes[1].set_ylim(-2, 102)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate index and coordinate correspondence thresholds.")
    parser.add_argument(
        "--input", action="append", dest="inputs", metavar="PATH",
        help="Input model path. Repeat for multiple models. Defaults to the three standard models.",
    )
    parser.add_argument("--output-prefix", default="DW3_demo_corrdis", help="Prefix for CSV and PNG reports.")
    parser.add_argument("--noise-levels", metavar="PERCENTAGES",
                        help="Optional comma-separated override of NOISE_LEVELS.")
    parser.add_argument("--smoothing-iterations", metavar="COUNTS",
                        help="Optional comma-separated override of SMOOTHING_ITERATIONS.")
    parser.add_argument("--reordering-ratios", metavar="RATIOS",
                        help="Optional comma-separated override of REORDERING_RATIOS.")
    parser.add_argument("--crop-keep-ratios", metavar="RATIOS",
                        help="Optional comma-separated override of CROP_KEEP_RATIOS.")
    parser.add_argument("--voxel-percents", metavar="PERCENTAGES",
                        help="Optional comma-separated override of VOXEL_PERCENTS.")
    args = parser.parse_args()

    inputs = args.inputs if args.inputs else list(DEFAULT_INPUTS)
    noise_levels = parse_number_list(args.noise_levels, "--noise-levels") if args.noise_levels else list(NOISE_LEVELS)
    smoothing_iterations = parse_number_list(args.smoothing_iterations, "--smoothing-iterations") if args.smoothing_iterations else list(SMOOTHING_ITERATIONS)
    reordering_ratios = parse_number_list(args.reordering_ratios, "--reordering-ratios") if args.reordering_ratios else list(REORDERING_RATIOS)
    crop_keep_ratios = parse_number_list(args.crop_keep_ratios, "--crop-keep-ratios") if args.crop_keep_ratios else list(CROP_KEEP_RATIOS)
    voxel_percents = parse_number_list(args.voxel_percents, "--voxel-percents") if args.voxel_percents else list(VOXEL_PERCENTS)
    if any(not 0.0 < ratio <= 1.0 for ratio in reordering_ratios + crop_keep_ratios):
        parser.error("Values of --reordering-ratios and --crop-keep-ratios must be in (0, 1].")
    if any(value <= 0.0 for value in noise_levels + smoothing_iterations + voxel_percents):
        parser.error("Noise levels, smoothing iterations, and voxel percentages must be positive.")

    # The graph plots the last supplied value for each attack. Thus its
    # condition can be selected simply by supplying the desired value last.
    index_selected = (
        ("noise", f"{format_factor(noise_levels[-1])}%"),
        ("smoothing", format_factor(smoothing_iterations[-1])),
        ("reordering", f"{format_factor(reordering_ratios[-1] * 100.0)}%"),
    )
    coordinate_selected = (
        ("cropping", f"{format_factor(crop_keep_ratios[-1] * 100.0)}% retained"),
        ("downsampling", f"{format_factor(voxel_percents[-1])}%"),
    )
    prefix = Path(args.output_prefix)
    index_rows: list[dict[str, object]] = []
    coordinate_rows: list[dict[str, object]] = []

    for model_path in inputs:
        source = Path(model_path)
        model = source.stem
        print(f"\nLoading {source} ...")
        original = read_and_normalize(str(source))

        # (attack name, parameter label, attack operation, examine same indices)
        experiments: list[tuple[str, str, Callable[[np.ndarray], np.ndarray], bool]] = []
        experiments += [
            ("noise", f"{format_factor(value)}%", lambda x, value=value: DW2F.noise_addition_attack(x, value, "gaussian", seed=42), True)
            for value in noise_levels
        ]
        experiments += [
            ("smoothing", format_factor(value), lambda x, value=value: DW2F.smoothing_attack(x, lambda_val=0.5, iterations=int(value), k=6, verbose=False), True)
            for value in smoothing_iterations
        ]
        experiments += [
            ("reordering", f"{format_factor(value * 100.0)}%", lambda x, value=value: DW2F.vertex_reordering_attack(x, value, seed=42), True)
            for value in reordering_ratios
        ]
        # Match the journal/experimental setting: crop along the x axis.
        experiments += [
            ("cropping", f"{format_factor(value * 100.0)}% retained", lambda x, value=value: DW2F.cropping_attack(x, keep_ratio=value, mode="axis", axis=0), False)
            for value in crop_keep_ratios
        ]
        experiments += [
            ("downsampling", f"{format_factor(value)}%", lambda x, value=value: DW2F.downsampling_attack(x, voxel_size_percent=value), False)
            for value in voxel_percents
        ]

        for attack_name, parameter, operation, check_index in experiments:
            print(f"  {attack_name}: {parameter}")
            attacked = np.asarray(operation(original), dtype=np.float64)
            spacing, coordinate_distances, index_distances = measure_distances(original, attacked)
            common = {
                "model": model,
                "attack": attack_name,
                "parameter": parameter,
                "original_points": len(original),
                "attacked_points": len(attacked),
                "point_ratio": len(attacked) / len(original),
                "attacked_spacing": spacing,
                "coordinate_q95": float(np.quantile(coordinate_distances, 0.95)),
                "coordinate_max": float(np.max(coordinate_distances)),
            }
            coordinate_row = dict(common)
            for factor in FACTORS:
                coordinate_row[f"coordinate_accept_{factor:g}"] = fraction_leq(coordinate_distances, factor)
            coordinate_rows.append(coordinate_row)

            if check_index and index_distances is not None:
                index_row = dict(common)
                index_row["index_q99"] = float(np.quantile(index_distances, 0.99))
                index_row["index_max"] = float(np.max(index_distances))
                for factor in FACTORS:
                    index_row[f"index_bad_{factor:g}"] = fraction_gt(index_distances, factor)
                index_rows.append(index_row)

    index_fields = list(index_rows[0].keys()) if index_rows else []
    coordinate_fields = list(coordinate_rows[0].keys()) if coordinate_rows else []
    index_csv = prefix.with_name(prefix.name + "_index.csv")
    coordinate_csv = prefix.with_name(prefix.name + "_coordinate.csv")
    figure_path = prefix.with_name(prefix.name + ".png")
    if index_rows:
        write_csv(index_csv, index_rows, index_fields)
        print_index_table(index_rows)
    if coordinate_rows:
        write_csv(coordinate_csv, coordinate_rows, coordinate_fields)
        print_coordinate_table(coordinate_rows)
    figure_written = plot_profiles(
        index_rows, coordinate_rows, figure_path, index_selected, coordinate_selected
    )

    print("\nSaved reports:")
    print(f"  {index_csv}")
    print(f"  {coordinate_csv}")
    if figure_written:
        print(f"  {figure_path}")
    else:
        print("  PNG was skipped (matplotlib is not installed).")
    print("\nInterpretation: choose the smallest factor that retains intended matches.")
    print("For cropping, acceptance substantially above the retained-point ratio indicates false coordinate matches.")


if __name__ == "__main__":
    main()
