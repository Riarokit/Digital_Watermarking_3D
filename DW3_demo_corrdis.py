"""Correspondence calibration for the current DW2 mesh-attack experiment.

No watermark is embedded.  This script measures the correspondence path used
by the proposed and El Zein extractors under the attacks in ``DW2_exp``.
Count-preserving attacks use bidir's cluster-local one-to-one assignment;
cropping and voxel downsampling use the intended coordinate many-to-one
fallback because their vertex count changes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import DW1_X1_func as DW1X1
import DW2_func as DW2F


# Threshold candidates, including the current DW2 setting (30).
FACTORS = (1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
DISPLAY_FACTORS = (1.0, 2.0, 3.0, 10.0, 30.0)
DEFAULT_INPUTS = ("C:/bun_zipper.ply", "C:/dragon.ply", "C:/Armadillo.ply")

# =========================================================
# Same attack ranges and options as DW2_exp.py.
# =========================================================
NOISE_LEVELS = (0.5, 0.8, 1.0)
SMOOTHING_ITERATIONS = (10, 30, 50)
REORDERING_RATIOS = (0.1, 0.5, 1.0)
CROP_KEEP_RATIOS = (0.7, 0.5, 0.3)
VOXEL_PERCENTS = (0.8, 1.0, 1.2)
NOISE_MODE = "gaussian"
SMOOTHING_LAMBDA = 0.1
CROPPING_MODE = "axis"
CROPPING_AXIS = 0
DOWNSAMPLING_MODE = "voxel"
ATTACK_SEED = 42

# Same bidir settings as DW2_exp.py's proposed method.
CLUSTER_POINTS = 2000
COORDINATE_DISTANCE_FACTOR = 30.0
ONE_TO_ONE_CANDIDATES = DW1X1.ONE_TO_ONE_CANDIDATES
ONE_TO_ONE_WORKERS = 0


def read_and_normalize_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a triangle mesh and apply the project's standard normalization."""
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(
            f"{path} must be a triangle mesh with vertices and triangles; "
            "DW2 mesh attacks cannot be calibrated from a point cloud alone."
        )
    # Match DW2_exp.load_mesh(): several supplied PLY files contain vertices
    # which are not referenced by any face.  They are harmless for embedding,
    # but mesh Laplacian smoothing correctly rejects them because they have no
    # 1-ring neighbours.
    vertices, triangles, _ = DW2F.remove_unreferenced_vertices(
        np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    normalized = DW2F.normalize_point_cloud(pcd)
    return (
        np.asarray(normalized.points, dtype=np.float64).copy(),
        triangles,
    )


def attacked_spacing(xyz: np.ndarray) -> float:
    distances, _ = cKDTree(xyz).query(xyz, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if len(nearest) == 0:
        raise ValueError("Cannot estimate spacing from duplicate-only point cloud.")
    return float(np.median(nearest))


def normalized_coordinate_distances(
    original: np.ndarray, attacked: np.ndarray
) -> tuple[float, np.ndarray]:
    """Distances used by the coordinate many-to-one fallback."""
    spacing = attacked_spacing(attacked)
    distances, _ = cKDTree(attacked).query(original, k=1)
    return spacing, np.asarray(distances, dtype=np.float64) / spacing


def normalized_bidir_distances(
    original: np.ndarray, attacked: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray, str]:
    """Run bidir when possible, otherwise the same coordinate fallback as DW1."""
    if len(original) != len(attacked):
        spacing, distances = normalized_coordinate_distances(original, attacked)
        return spacing, distances, "coordinate many-to-one (count-change fallback)"
    try:
        _, _matched, distances, _threshold = DW1X1.match_point_cloud_one_to_one(
            attacked,
            original,
            labels,
            distance_factor=COORDINATE_DISTANCE_FACTOR,
            candidate_counts=ONE_TO_ONE_CANDIDATES,
            workers=ONE_TO_ONE_WORKERS,
            verbose=False,
            mode="bidir",
        )
        spacing = attacked_spacing(attacked)
        return spacing, np.asarray(distances, dtype=np.float64) / spacing, "bidir one-to-one"
    except (RuntimeError, ValueError) as error:
        spacing, distances = normalized_coordinate_distances(original, attacked)
        return spacing, distances, f"coordinate many-to-one (bidir failed: {error})"


def fraction_leq(distances: np.ndarray, factor: float) -> float:
    return float(np.mean(np.isfinite(distances) & (distances <= factor)))


def format_factor(factor: float) -> str:
    return f"{factor:g}"


def parse_number_list(text: str, argument_name: str) -> list[float]:
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


def print_table(rows: list[dict[str, object]]) -> None:
    print("\n[Correspondence acceptance: coordinate many-to-one vs bidir path]")
    header = ["model", "attack", "parameter", "path", "Natt/Norig", "bidir@30"]
    header += [f"coord <= {factor:g}" for factor in DISPLAY_FACTORS]
    print(" | ".join(header))
    print("-|-".join("---" for _ in header))
    for row in rows:
        values = [
            str(row["model"]), str(row["attack"]), str(row["parameter"]),
            str(row["bidir_path"]), f"{float(row['point_ratio']):.3f}",
            f"{100.0 * float(row['bidir_accept_30']):.2f}%",
        ]
        values += [
            f"{100.0 * float(row[f'coordinate_accept_{factor:g}']):.2f}%"
            for factor in DISPLAY_FACTORS
        ]
        print(" | ".join(values))


def plot_profiles(rows: list[dict[str, object]], output_path: Path) -> bool:
    if plt is None:
        return False
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for attack in ("noise", "smoothing", "reordering", "cropping", "downsampling"):
        selected = [row for row in rows if row["attack"] == attack]
        if not selected:
            continue
        coordinate = [
            100.0 * np.mean([float(row[f"coordinate_accept_{factor:g}"]) for row in selected])
            for factor in FACTORS
        ]
        bidir = [
            100.0 * np.mean([float(row[f"bidir_accept_{factor:g}"]) for row in selected])
            for factor in FACTORS
        ]
        axes[0].plot(FACTORS, coordinate, marker="o", label=attack)
        axes[1].plot(FACTORS, bidir, marker="o", label=attack)
    for axis, title in zip(
        axes, ("Coordinate many-to-one acceptance", "bidir path acceptance")
    ):
        axis.axvline(
            COORDINATE_DISTANCE_FACTOR,
            color="tab:red",
            linestyle=":",
            label=f"DW2 factor = {COORDINATE_DISTANCE_FACTOR:g}",
        )
        axis.set_title(title)
        axis.set_xlabel("coordinate_distance_factor")
        axis.set_ylabel("original points accepted (%)")
        axis.set_ylim(-2, 102)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate current DW2 mesh attacks and bidir/fallback correspondence."
    )
    parser.add_argument(
        "--input", action="append", dest="inputs", metavar="PATH",
        help="Triangle-mesh path; repeat for multiple models.",
    )
    parser.add_argument("--output-prefix", default="figures/DW3_demo_corrdis")
    parser.add_argument("--noise-levels", metavar="PERCENTAGES")
    parser.add_argument("--smoothing-iterations", metavar="COUNTS")
    parser.add_argument("--reordering-ratios", metavar="RATIOS")
    parser.add_argument("--crop-keep-ratios", metavar="RATIOS")
    parser.add_argument("--voxel-percents", metavar="PERCENTAGES")
    args = parser.parse_args()

    inputs = args.inputs if args.inputs else list(DEFAULT_INPUTS)
    noise = parse_number_list(args.noise_levels, "--noise-levels") if args.noise_levels else list(NOISE_LEVELS)
    smoothing = parse_number_list(args.smoothing_iterations, "--smoothing-iterations") if args.smoothing_iterations else list(SMOOTHING_ITERATIONS)
    reorder = parse_number_list(args.reordering_ratios, "--reordering-ratios") if args.reordering_ratios else list(REORDERING_RATIOS)
    crop = parse_number_list(args.crop_keep_ratios, "--crop-keep-ratios") if args.crop_keep_ratios else list(CROP_KEEP_RATIOS)
    voxel = parse_number_list(args.voxel_percents, "--voxel-percents") if args.voxel_percents else list(VOXEL_PERCENTS)
    if any(not 0.0 < value <= 1.0 for value in reorder + crop):
        parser.error("Reordering and crop ratios must be in (0, 1].")
    if any(value <= 0.0 for value in noise + smoothing + voxel):
        parser.error("Noise, smoothing, and voxel values must be positive.")

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for model_path in inputs:
        source = Path(model_path)
        print(f"\nLoading {source} ...")
        original, triangles = read_and_normalize_mesh(str(source))
        labels = DW1X1.kmeans_cluster_points(
            original, cluster_point=CLUSTER_POINTS, seed=ATTACK_SEED
        )
        experiments: list[tuple[str, str, Callable[[], tuple[np.ndarray, np.ndarray]]]] = []
        for value in noise:
            experiments.append((
                "noise", f"{format_factor(value)}%",
                lambda value=value: DW2F.apply_mesh_attack(
                    original, triangles, "noise", value, ATTACK_SEED, noise_mode=NOISE_MODE
                ),
            ))
        for value in smoothing:
            experiments.append((
                "smoothing", format_factor(value),
                lambda value=value: DW2F.apply_mesh_attack(
                    original, triangles, "smoothing", value, ATTACK_SEED,
                    smoothing_lambda=SMOOTHING_LAMBDA,
                ),
            ))
        for value in reorder:
            experiments.append((
                "reordering", f"{format_factor(value * 100)}%",
                lambda value=value: DW2F.apply_mesh_attack(
                    original, triangles, "reordering", value, ATTACK_SEED
                ),
            ))
        for value in crop:
            experiments.append((
                "cropping", f"{format_factor(value * 100)}% retained",
                lambda value=value: DW2F.apply_mesh_attack(
                    original, triangles, "cropping", value, ATTACK_SEED,
                    cropping_mode=CROPPING_MODE, cropping_axis=CROPPING_AXIS,
                ),
            ))
        for value in voxel:
            experiments.append((
                "downsampling", f"{format_factor(value)}%",
                lambda value=value: DW2F.apply_mesh_attack(
                    original, triangles, "downsampling", value, ATTACK_SEED,
                    downsampling_mode=DOWNSAMPLING_MODE,
                ),
            ))

        for attack, parameter, operation in experiments:
            print(f"  {attack}: {parameter}")
            attacked, _ = operation()
            coord_spacing, coordinate = normalized_coordinate_distances(original, attacked)
            bidir_spacing, bidir, bidir_path = normalized_bidir_distances(
                original, attacked, labels
            )
            row: dict[str, object] = {
                "model": source.stem, "attack": attack, "parameter": parameter,
                "original_points": len(original), "attacked_points": len(attacked),
                "point_ratio": len(attacked) / len(original),
                "coordinate_spacing": coord_spacing, "bidir_spacing": bidir_spacing,
                "coordinate_q95": float(np.quantile(coordinate, 0.95)),
                "coordinate_max": float(np.max(coordinate)),
                "bidir_q95": float(np.quantile(bidir, 0.95)),
                "bidir_max": float(np.max(bidir)), "bidir_path": bidir_path,
            }
            for factor in FACTORS:
                row[f"coordinate_accept_{factor:g}"] = fraction_leq(coordinate, factor)
                row[f"bidir_accept_{factor:g}"] = fraction_leq(bidir, factor)
            rows.append(row)

    if not rows:
        raise RuntimeError("No calibration rows were produced.")
    csv_path = prefix.with_name(prefix.name + "_correspondence.csv")
    png_path = prefix.with_name(prefix.name + ".png")
    write_csv(csv_path, rows, list(rows[0].keys()))
    print_table(rows)
    figure_written = plot_profiles(rows, png_path)
    print("\nSaved reports:")
    print(f"  {csv_path}")
    print(f"  {png_path}" if figure_written else "  PNG was skipped (matplotlib is not installed).")
    print("\nNote: cropping/downsampling always show the coordinate many-to-one fallback;")
    print("they cannot use bidir because the received vertex count is lower.")


if __name__ == "__main__":
    main()
