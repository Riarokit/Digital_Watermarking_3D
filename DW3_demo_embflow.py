"""Create four explanatory figures for the proposed embedding flow.

The figures use the same KMeans clusters, PCA pseudo-plane, KNN graph, and
GFT as the proposed method.  They are intended for a paper/presentation, not
for embedding or extraction:

1. complete mesh vertices coloured by cluster;
2. one selected cluster and its PCA pseudo-plane;
3. the selected cluster's graph with signed plane heights as colours; and
4. the resulting GFT coefficients.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter

import DW1_X1_func as DW1X1
import DW2_func as DW2F


DEFAULT_INPUT = "C:/bun_zipper.ply"
DEFAULT_CLUSTER_POINTS = 1500
DEFAULT_GRAPH_MODE = "knn"
DEFAULT_K = 24
DEFAULT_RADIUS = 0.03
DEFAULT_SEED = 42
# A side-oriented Bunny view with a small elevation: the horizontal
# pseudo-plane remains visible instead of becoming an edge-on line.
DEFAULT_VIEW_ELEV = -5.0
DEFAULT_VIEW_AZIM = -90.0


def display_coordinates(points: np.ndarray) -> np.ndarray:
    """Map Bunny's y-up coordinates to Matplotlib's z-up display convention."""
    return np.asarray(points)[:, [0, 2, 1]]


def load_mesh_vertices(path: str) -> np.ndarray:
    """Load the same normalized, referenced vertices as the experiment code."""
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError("Input must be a triangle mesh containing vertices and faces.")
    vertices, _triangles, _ = DW2F.remove_unreferenced_vertices(
        np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    normalized = DW2F.normalize_point_cloud(pcd)
    return np.asarray(normalized.points, dtype=np.float64).copy()


def set_equal_3d_axes(axis, points: np.ndarray) -> None:
    """Use equal spatial scale so planes and cluster shape are not distorted."""
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_axis_off()


def cluster_colours(labels: np.ndarray) -> np.ndarray:
    """Stable distinct colours for every KMeans label."""
    count = int(np.max(labels)) + 1
    return plt.get_cmap("hsv", count)(labels)[:, :3]


def select_horizontal_cluster(points: np.ndarray, labels: np.ndarray) -> int:
    """Choose the cluster whose PCA plane is closest to horizontal (normal || y)."""
    best_label = -1
    best_score = -np.inf
    # bun_zipper.ply is y-up; display_coordinates maps this to plotted z-up.
    vertical = np.array([0.0, 1.0, 0.0])
    for label in np.unique(labels):
        cluster = points[labels == label]
        if len(cluster) < 3:
            continue
        _centroid, normal = DW1X1.compute_pseudoplane_pca(cluster)
        score = abs(float(np.dot(normal, vertical)))
        if score > best_score:
            best_label = int(label)
            best_score = score
    if best_label < 0:
        raise ValueError("No cluster contains enough points to construct a pseudo-plane.")
    return best_label


def plane_grid(
    points: np.ndarray, centroid: np.ndarray, normal: np.ndarray, scale: float = 1.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a rectangular PCA plane spanning the selected cluster."""
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, reference)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    relative = points - centroid
    u_half = max(np.max(np.abs(relative @ u)), 1e-6) * scale
    v_half = max(np.max(np.abs(relative @ v)), 1e-6) * scale
    uu, vv = np.meshgrid(np.linspace(-u_half, u_half, 16), np.linspace(-v_half, v_half, 16))
    plane = centroid[None, None, :] + uu[..., None] * u + vv[..., None] * v
    displayed = display_coordinates(plane.reshape(-1, 3)).reshape(plane.shape)
    return displayed[..., 0], displayed[..., 1], displayed[..., 2]


def unique_graph_edges(weights: np.ndarray, maximum: int) -> np.ndarray:
    """Return deterministic upper-triangle graph edges for visualisation only."""
    rows, cols = np.where(np.triu(weights > 0.0, k=1))
    edges = np.column_stack((rows, cols))
    if maximum > 0 and len(edges) > maximum:
        # Evenly spaced selection preserves coverage without random figures.
        edges = edges[np.linspace(0, len(edges) - 1, maximum, dtype=int)]
    return edges


def save_cluster_figure(
    points: np.ndarray, labels: np.ndarray, selected_label: int, output: Path,
    point_size: float, elevation: float, azimuth: float,
) -> None:
    figure = plt.figure(figsize=(8, 8), constrained_layout=True)
    axis = figure.add_subplot(projection="3d")
    displayed = display_coordinates(points)
    axis.scatter(
        displayed[:, 0], displayed[:, 1], displayed[:, 2],
        c=cluster_colours(labels), s=point_size, depthshade=False,
    )
    selected = displayed[labels == selected_label]
    # The gold overlay makes the cluster used in figures 2--4 unambiguous.
    axis.scatter(
        selected[:, 0], selected[:, 1], selected[:, 2],
        c="#ffd600", edgecolors="#212121", linewidths=0.15,
        s=point_size * 2.8, depthshade=False,
    )
    set_equal_3d_axes(axis, displayed)
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_title(
        f"(1) KMeans clusters; selected cluster {selected_label} highlighted"
        f" (about {len(points) / len(np.unique(labels)):.0f} points / cluster)"
    )
    figure.savefig(output, dpi=250, facecolor="white")
    plt.close(figure)


def save_plane_figure(
    cluster: np.ndarray, centroid: np.ndarray, normal: np.ndarray,
    selected_label: int, output: Path, point_size: float,
    elevation: float, azimuth: float,
) -> None:
    figure = plt.figure(figsize=(8, 8), constrained_layout=True)
    axis = figure.add_subplot(projection="3d")
    displayed = display_coordinates(cluster)
    axis.scatter(displayed[:, 0], displayed[:, 1], displayed[:, 2], c="#1976d2", s=point_size, depthshade=False)
    xx, yy, zz = plane_grid(cluster, centroid, normal)
    axis.plot_surface(xx, yy, zz, color="#ff9800", alpha=0.35, linewidth=0, shade=False)
    arrow_length = 0.25 * max(np.ptp(cluster, axis=0))
    axis.quiver(
        *display_coordinates(centroid[None, :])[0],
        *display_coordinates(normal[None, :])[0],
        length=arrow_length, color="#d32f2f", linewidth=2,
    )
    set_equal_3d_axes(axis, displayed)
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_title(f"(2) Selected cluster {selected_label} and PCA pseudo-plane")
    figure.savefig(output, dpi=250, facecolor="white")
    plt.close(figure)


def save_height_graph_figure(
    cluster: np.ndarray, heights: np.ndarray, weights: np.ndarray,
    output: Path, point_size: float, maximum_edges: int,
    elevation: float, azimuth: float,
) -> None:
    figure = plt.figure(figsize=(8, 8), constrained_layout=True)
    axis = figure.add_subplot(projection="3d")
    displayed = display_coordinates(cluster)
    normalizer = Normalize(vmin=float(np.min(heights)), vmax=float(np.max(heights)))
    cmap = plt.get_cmap("coolwarm")
    for start, end in unique_graph_edges(weights, maximum_edges):
        segment = displayed[[start, end]]
        axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="black", alpha=0.15, linewidth=0.35)
    dots = axis.scatter(
        displayed[:, 0], displayed[:, 1], displayed[:, 2], c=heights,
        cmap=cmap, norm=normalizer, s=point_size, depthshade=False,
    )
    colorbar = figure.colorbar(dots, ax=axis, shrink=0.64, pad=0.02)
    colorbar.set_label("signed height from fitted plane")
    set_equal_3d_axes(axis, displayed)
    axis.view_init(elev=elevation, azim=azimuth)
    suffix = "all edges" if maximum_edges <= 0 else f"{min(maximum_edges, len(unique_graph_edges(weights, 0)))} displayed edges"
    axis.set_title(f"(3) Graph vertices coloured by height ({suffix})")
    figure.savefig(output, dpi=250, facecolor="white")
    plt.close(figure)


def save_gft_figure(coefficients: np.ndarray, output: Path, bins: int = 50) -> None:
    """Use the same binned log-magnitude expression as DW3_demo_coef.py."""
    magnitude = np.abs(coefficients)
    bucket = np.minimum((np.arange(len(magnitude)) * bins) // len(magnitude), bins - 1)
    sums = np.bincount(bucket, weights=magnitude, minlength=bins)
    counts = np.bincount(bucket, minlength=bins)
    distribution = sums / np.maximum(counts, 1)
    # Log scale cannot draw a zero.  This only affects exactly-zero bins.
    positive = distribution[distribution > 0]
    floor = float(np.min(positive)) * 0.5 if len(positive) else 1e-15
    distribution = np.maximum(distribution, floor)
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.plot(np.linspace(0.0, 1.0, bins), distribution, color="black", linewidth=2)
    axis.set_yscale("log")
    axis.set_xlabel("Normalized frequency")
    axis.set_ylabel("Coefficient magnitude (log scale)")
    axis.set_title("(4) GFT coefficient distribution of the height signal")
    axis.grid(True, which="both", linestyle="--", alpha=0.5)
    figure.savefig(output, dpi=250, facecolor="white")
    plt.close(figure)


def illustrative_gft_modulation(
    cluster: np.ndarray, normal: np.ndarray, basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a smooth, visible low-frequency explanation-only perturbation."""
    point_count = len(cluster)
    # Skip the DC component but modify only the first 1--10% of the spectrum.
    # This makes the inverse-GFT displacement smooth over the whole cluster.
    start, end = max(1, int(point_count * 0.01)), max(2, int(point_count * 0.10))
    targets = np.arange(start, end, dtype=int)
    coefficient_change = np.zeros(point_count, dtype=np.float64)
    # Modulate the complete low-frequency band with a smooth coefficient
    # envelope, rather than perturbing a few isolated spectral samples.
    coefficient_change[targets] = np.cos(np.linspace(0.0, np.pi, len(targets)))
    unit_height_change = basis @ coefficient_change
    # Explanation figure: a visible 6% peak shift relative to the selected
    # cluster's bounding-box diagonal, while retaining a smooth deformation.
    target_peak = 0.06 * float(np.linalg.norm(np.ptp(cluster, axis=0)))
    scale = target_peak / max(float(np.max(np.abs(unit_height_change))), 1e-12)
    height_change = unit_height_change * scale
    modulated = cluster + height_change[:, None] * normal[None, :]
    return modulated, height_change


def save_modulated_cluster_figure(
    cluster: np.ndarray, modulated: np.ndarray, height_change: np.ndarray,
    output: Path, point_size: float, elevation: float, azimuth: float,
) -> None:
    """Show the inverse-GFT result overlaid on the original selected cluster."""
    figure = plt.figure(figsize=(8, 8), constrained_layout=True)
    axis = figure.add_subplot(projection="3d")
    displayed_original = display_coordinates(cluster)
    displayed_modulated = display_coordinates(modulated)
    axis.scatter(
        displayed_original[:, 0], displayed_original[:, 1], displayed_original[:, 2],
        c="#9e9e9e", s=point_size * 0.9, alpha=0.40, depthshade=False,
    )
    # Use a robust display range rather than the single largest displacement.
    # This deliberately saturates the largest 30% so the smooth, typical
    # positive/negative movement is not compressed into the central green.
    color_peak = max(float(np.quantile(np.abs(height_change), 0.70)), 1e-12)
    signed_displacement_cmap = LinearSegmentedColormap.from_list(
        "signed_displacement", ["#542788", "#1b9e77", "#fde725"]
    )
    dots = axis.scatter(
        displayed_modulated[:, 0], displayed_modulated[:, 1], displayed_modulated[:, 2],
        c=height_change, cmap=signed_displacement_cmap,
        norm=TwoSlopeNorm(vmin=-color_peak, vcenter=0.0, vmax=color_peak),
        s=point_size * 1.25, depthshade=False,
    )
    colorbar = figure.colorbar(
        dots, ax=axis, orientation="horizontal", shrink=0.72, pad=0.08
    )
    colorbar.set_ticks((-color_peak, 0.0, color_peak))
    colorbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    colorbar.set_label("signed normal-direction displacement (color range clipped)")
    set_equal_3d_axes(axis, displayed_original)
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_title("(5) Smooth low-frequency GFT modulation (original: gray)")
    figure.savefig(output, dpi=250, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate five explanatory proposed-method embedding-flow figures.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input triangle mesh.")
    parser.add_argument("--cluster-points", type=int, default=DEFAULT_CLUSTER_POINTS, help="Target number of points per KMeans cluster.")
    parser.add_argument("--cluster-id", type=int, default=None, help="Cluster ID to focus on; omit for the most-horizontal pseudo-plane.")
    parser.add_argument("--graph-mode", choices=("knn", "radius", "hybrid"), default=DEFAULT_GRAPH_MODE)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="KNN degree for the cluster graph.")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--max-graph-edges", type=int, default=5000, help="Maximum edges drawn in figure 3; 0 draws all edges.")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--view-elev", type=float, default=DEFAULT_VIEW_ELEV)
    parser.add_argument("--view-azim", type=float, default=DEFAULT_VIEW_AZIM)
    parser.add_argument("--output-dir", default="figures/embflow")
    parser.add_argument("--prefix", default="embflow")
    args = parser.parse_args()
    if args.cluster_points < 3 or args.k < 1 or args.max_graph_edges < 0:
        parser.error("cluster-points must be >= 3, k >= 1, and max-graph-edges >= 0.")

    points = load_mesh_vertices(args.input)
    if args.cluster_points > len(points):
        parser.error(f"--cluster-points={args.cluster_points} exceeds vertex count {len(points)}.")
    labels = DW1X1.kmeans_cluster_points(points, cluster_point=args.cluster_points, seed=DEFAULT_SEED)
    selected_label = (
        select_horizontal_cluster(points, labels)
        if args.cluster_id is None else args.cluster_id
    )
    if selected_label not in set(np.unique(labels)):
        parser.error(f"--cluster-id must be one of {np.unique(labels).tolist()}.")
    selected = points[labels == selected_label]
    if len(selected) <= args.k:
        parser.error(f"Selected cluster has only {len(selected)} points; --k must be smaller.")

    centroid, normal = DW1X1.compute_pseudoplane_pca(selected)
    heights = DW1X1.cluster_height_signal(selected, centroid, normal)
    graph = DW1X1.build_graph(selected, graph_mode=args.graph_mode, k=args.k, radius=args.radius)
    # Keep the basis for the final inverse-GFT visualisation as well.
    basis, _eigenvalues = DW1X1.gft_basis_auto(graph)
    coefficients = DW1X1.gft(heights, basis)
    modulated_cluster, height_change = illustrative_gft_modulation(
        selected, normal, basis
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_dir / f"{args.prefix}_1_clusters.png",
        output_dir / f"{args.prefix}_2_pseudoplane.png",
        output_dir / f"{args.prefix}_3_height_graph.png",
        output_dir / f"{args.prefix}_4_gft.png",
        output_dir / f"{args.prefix}_5_modulated_cluster.png",
    )
    save_cluster_figure(
        points, labels, selected_label, outputs[0], args.point_size,
        args.view_elev, args.view_azim,
    )
    save_plane_figure(
        selected, centroid, normal, selected_label, outputs[1], args.point_size,
        args.view_elev, args.view_azim,
    )
    save_height_graph_figure(
        selected, heights, graph, outputs[2], args.point_size,
        args.max_graph_edges, args.view_elev, args.view_azim,
    )
    save_gft_figure(coefficients, outputs[3])
    save_modulated_cluster_figure(
        selected, modulated_cluster, height_change, outputs[4], args.point_size,
        args.view_elev, args.view_azim,
    )

    print(f"Selected cluster: id={selected_label}, points={len(selected)}")
    print(f"Graph: mode={args.graph_mode}, k={args.k}, edges={len(unique_graph_edges(graph, 0))}")
    print("Saved figures:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
