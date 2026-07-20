"""Hu et al. (2023) の表面 EMD に基づく 3D 透かし法.

論文: J. Hu et al., "Robust 3D watermarking with high
imperceptibility based on EMD on surfaces", The Visual Computer, 2023.

埋め込み時は入力メッシュの三角形接続をそのまま利用する。切り取りや
ダウンサンプリングで接続情報が失われた場合、抽出時に攻撃後点群だけから
メッシュを再構築し、その表面上で EMD を計算する。
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.spatial import cKDTree
import warnings


@dataclass
class HuWatermarkKey:
    """Algorithm 2 で必要な補助情報。

    ``embedding_positions`` は論文で保存される埋め込み極値位置に対応する。
    抽出時はこの極値位置だけを用いて対応頂点を特定する。
    """

    embedding_indices: np.ndarray
    embedding_positions: np.ndarray
    matching_threshold: float
    bit_indices: np.ndarray
    repetitions: np.ndarray
    watermark_size: int
    circular_count: int
    arnold_iterations: int
    extreme_parameter: float
    epsilon: float


def _validate_mesh(vertices, triangles):
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("vertices must have shape (N, 3) and contain at least one vertex.")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("Hu 法には三角形面が必要です。triangles must have shape (M, 3).")
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("triangles contains an out-of-range vertex index.")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contains NaN or infinity.")
    return vertices, triangles


def _reconstruct_mesh_from_points(vertices):
    """Reconstruct the attacked surface required by Hu's surface EMD."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError(
            "At least four attacked points are required for mesh reconstruction."
        )

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    distances = np.asarray(point_cloud.compute_nearest_neighbor_distance())
    distances = distances[np.isfinite(distances) & (distances > 1e-12)]
    if len(distances) == 0:
        raise ValueError(
            "Cannot estimate a reconstruction radius from the attacked points."
        )
    average_distance = float(np.mean(distances))
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=6.0 * average_distance, max_nn=30
        )
    )
    if len(vertices) > 20:
        try:
            point_cloud.orient_normals_consistent_tangent_plane(20)
        except RuntimeError:
            pass

    reconstructed = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud,
        o3d.utility.DoubleVector(
            [3.0 * average_distance, 6.0 * average_distance]
        ),
    )
    reconstructed.remove_degenerate_triangles()
    reconstructed.remove_duplicated_triangles()
    reconstructed.remove_duplicated_vertices()
    reconstructed.remove_unreferenced_vertices()
    reconstructed_vertices = np.asarray(reconstructed.vertices).copy()
    reconstructed_triangles = np.asarray(reconstructed.triangles).copy()
    if len(reconstructed_vertices) == 0 or len(reconstructed_triangles) == 0:
        raise ValueError(
            "Ball-pivoting failed to reconstruct the attacked triangle mesh."
        )
    return reconstructed_vertices, reconstructed_triangles


def _prepare_extraction_mesh(vertices, original_triangles):
    """Use original connectivity only while the original vertex set remains."""
    vertices = np.asarray(vertices, dtype=float)
    original_triangles = np.asarray(original_triangles, dtype=np.int64)
    original_vertex_count = (
        int(original_triangles.max()) + 1 if len(original_triangles) else 0
    )
    if len(vertices) != original_vertex_count:
        return _reconstruct_mesh_from_points(vertices)
    return _validate_mesh(vertices, original_triangles)


def get_mesh_laplacian(vertices, triangles):
    """自己ループを含まない組合せ Laplacian ``D - W`` を構築する。"""
    n = len(vertices)
    rows, cols = [], []
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            rows.extend((u, v))
            cols.extend((v, u))
    adjacency = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    adjacency.data[:] = 1.0
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if np.any(degree == 0):
        raise ValueError("孤立頂点を含むメッシュでは表面 EMD を計算できません。")
    return sparse.diags(degree) - adjacency


def _one_ring_neighbors(n, triangles):
    neighbors = [set() for _ in range(n)]
    for a, b, c in triangles:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def _find_relaxed_extrema(triangles, signal, t):
    """論文式 (4) の緩和局所極値。EMD の包絡面には符号を問わず使う。"""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional. Use imfs[0], not imfs.")
    if not 0 < t <= 1:
        raise ValueError("t must be in (0, 1].")

    neighbors = _one_ring_neighbors(len(signal), triangles)
    maxima, minima = [], []
    for i, nbs_set in enumerate(neighbors):
        if not nbs_set:
            continue
        # set の反復順序に依存させず、同一入力での EMD を完全に再現可能にする。
        nbs = np.asarray(sorted(nbs_set), dtype=np.int64)
        required = t * len(nbs)
        if np.count_nonzero(signal[i] >= signal[nbs]) >= required:
            maxima.append(i)
        if np.count_nonzero(signal[i] <= signal[nbs]) >= required:
            minima.append(i)
    return np.asarray(maxima, dtype=np.int64), np.asarray(minima, dtype=np.int64)


def find_stable_extreme_points_mesh(triangles, signal, t=0.8, epsilon=1e-4):
    """論文式 (4) と 4.2 節に従い、安定した正極大・負極小を選ぶ。"""
    signal = np.asarray(signal, dtype=float)
    maxima, minima = _find_relaxed_extrema(triangles, signal, t)
    return maxima[signal[maxima] > epsilon], minima[signal[minima] < -epsilon]


def solve_biharmonic_interpolation(laplacian, boundary_indices, boundary_values):
    """Dirichlet 条件付き二重調和場を解き、上下包絡面を得る。"""
    n = laplacian.shape[0]
    boundary_indices = np.asarray(boundary_indices, dtype=np.int64)
    boundary_values = np.asarray(boundary_values, dtype=float)
    if len(boundary_indices) == 0:
        raise ValueError("Biharmonic interpolation needs at least one boundary point.")

    # L.T @ L は半正定値かつ対称。非対称な L @ L より安定して Dirichlet 問題を解ける。
    operator = (laplacian.T @ laplacian).tocsr()
    is_boundary = np.zeros(n, dtype=bool)
    is_boundary[boundary_indices] = True
    unknown = np.flatnonzero(~is_boundary)
    result = np.empty(n, dtype=float)
    result[boundary_indices] = boundary_values
    if len(unknown) == 0:
        return result

    a_uu = operator[unknown][:, unknown]
    rhs = -operator[unknown][:, boundary_indices] @ boundary_values
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            result[unknown] = spsolve(a_uu, rhs)
        except (MatrixRankWarning, RuntimeError) as exc:
            raise ValueError("二重調和補間が特異です。各連結成分に極値が必要です。") from exc
    if not np.isfinite(result).all():
        raise ValueError("二重調和補間で有限値を得られませんでした。")
    return result


def surface_emd_sifting(vertices, triangles, signal, max_iter=50, stop_sd=0.2, t=0.8):
    """最初の IMF と残差を求める表面 EMD の sifting。

    論文の Algorithm 1 は IMF1 のみを利用するため、一つの IMF を収束するまで
    sifting する。``stop_sd`` は連続反復の標準偏差型停止条件である。
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) != len(vertices):
        raise ValueError("signal must have one value per mesh vertex.")
    laplacian = get_mesh_laplacian(vertices, triangles)
    h = signal.copy()
    for _ in range(max_iter):
        max_idx, min_idx = _find_relaxed_extrema(triangles, h, t=t)
        if len(max_idx) < 4 or len(min_idx) < 4:
            break
        upper = solve_biharmonic_interpolation(laplacian, max_idx, h[max_idx])
        lower = solve_biharmonic_interpolation(laplacian, min_idx, h[min_idx])
        mean = (upper + lower) / 2.0
        next_h = h - mean
        sd = np.sum((h - next_h) ** 2) / (np.sum(h ** 2) + 1e-15)
        h = next_h
        if sd < stop_sd:
            break
    return h, signal - h


def _rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x, y = n - 1 - x, n - 1 - y
        x, y = y, x
    return x, y


def _hilbert_coordinates(size):
    """Hilbert 曲線の訪問順で (row, column) を返す。size は 2 の冪。"""
    if size < 2 or size & (size - 1):
        raise ValueError("Hilbert curve requires a square watermark whose size is a power of two.")
    coords = []
    for distance in range(size * size):
        x = y = 0
        value, step = distance, 1
        while step < size:
            rx = (value // 2) & 1
            ry = (value ^ rx) & 1
            x, y = _rot(step, x, y, rx, ry)
            x += step * rx
            y += step * ry
            value //= 4
            step *= 2
        coords.append((y, x))
    return np.asarray(coords, dtype=np.int64)


def _arnold_transform(image, iterations):
    n = image.shape[0]
    output = np.empty_like(image)
    for _ in range(iterations):
        output.fill(0)
        for y in range(n):
            for x in range(n):
                output[(x + y) % n, (x + 2 * y) % n] = image[y, x]
        image = output.copy()
    return image


def _inverse_arnold_transform(image, iterations):
    n = image.shape[0]
    output = np.empty_like(image)
    for _ in range(iterations):
        output.fill(0)
        for y in range(n):
            for x in range(n):
                # Forward は output[x+y, x+2y] = input[y, x]。
                # 従って元の (y, x) は変換後の同じ位置から復元できる。
                output[y, x] = image[(x + y) % n, (x + 2 * y) % n]
        image = output.copy()
    return image


def watermark_to_hilbert_signal(watermark_bits, watermark_size, arnold_iterations=20):
    bits = np.asarray(watermark_bits, dtype=np.uint8)
    if len(bits) != watermark_size * watermark_size or not np.isin(bits, (0, 1)).all():
        raise ValueError("watermark_bits must be a binary square image flattened in row-major order.")
    scrambled = _arnold_transform(bits.reshape(watermark_size, watermark_size), arnold_iterations)
    coords = _hilbert_coordinates(watermark_size)
    return scrambled[coords[:, 0], coords[:, 1]].astype(np.uint8)


def hilbert_signal_to_watermark(signal, watermark_size, arnold_iterations=20):
    signal = np.asarray(signal, dtype=np.uint8)
    if len(signal) != watermark_size * watermark_size:
        raise ValueError("Invalid Hilbert signal length.")
    scrambled = np.empty((watermark_size, watermark_size), dtype=np.uint8)
    coords = _hilbert_coordinates(watermark_size)
    scrambled[coords[:, 0], coords[:, 1]] = signal
    return _inverse_arnold_transform(scrambled, arnold_iterations).ravel()


def _make_embedding_schedule(max_idx, min_idx, signal, repetitions):
    """式 (6) の符号規則に従い、各ビットを T 個の別々の極値へ割り当てる。"""
    one_bits = np.flatnonzero(signal == 1)
    zero_bits = np.flatnonzero(signal == 0)
    needed_max = len(one_bits) * repetitions
    needed_min = len(zero_bits) * repetitions
    if len(max_idx) < needed_max or len(min_idx) < needed_min:
        raise ValueError(
            "安定極値が不足しています: "
            f"positive maxima {len(max_idx)}/{needed_max}, negative minima {len(min_idx)}/{needed_min}. "
            "水印サイズ、T、または極値パラメータ t を調整してください。"
        )

    indices, bit_indices, reps = [], [], []
    max_cursor = min_cursor = 0
    for rep in range(repetitions):
        for bit_index, bit in enumerate(signal):
            if bit:
                indices.append(max_idx[max_cursor])
                max_cursor += 1
            else:
                indices.append(min_idx[min_cursor])
                min_cursor += 1
            bit_indices.append(bit_index)
            reps.append(rep)
    return (np.asarray(indices, dtype=np.int64), np.asarray(bit_indices, dtype=np.int64),
            np.asarray(reps, dtype=np.int64))


def _nearest_neighbor_scale(points):
    """欠損した極値キャリアを除外するための代表点間隔を返す。"""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    distances, _ = cKDTree(points).query(points, k=2)
    spacing = distances[:, 1]
    spacing = spacing[np.isfinite(spacing) & (spacing > 1e-12)]
    return float(np.median(spacing)) if len(spacing) else 0.0


def embed_watermark_hu_mesh(vertices, triangles, watermark_bits, FideP=115.0, T=25,
                            watermark_size=32, arnold_iterations=20, extreme_parameter=0.8,
                            epsilon=1e-4):
    """論文 Algorithm 1 および式 (2)--(11) に基づき透かしを埋め込む。"""
    vertices, triangles = _validate_mesh(vertices, triangles)
    if T < 1:
        raise ValueError("T must be at least 1.")
    centered = vertices - np.mean(vertices, axis=0)
    rho = np.linalg.norm(centered, axis=1)
    rho_max = float(np.max(rho))
    if rho_max <= 1e-15:
        raise ValueError("全頂点が同一点のため、正規化モジュラス信号を作れません。")
    f_signal = rho / rho_max

    imf1, residue = surface_emd_sifting(vertices, triangles, f_signal, t=extreme_parameter)
    max_idx, min_idx = find_stable_extreme_points_mesh(
        triangles, imf1, t=extreme_parameter, epsilon=epsilon
    )
    watermark_signal = watermark_to_hilbert_signal(watermark_bits, watermark_size, arnold_iterations)
    embedding_indices, bit_indices, reps = _make_embedding_schedule(max_idx, min_idx, watermark_signal, T)

    # 論文式 (9)。実際に埋込む L 個の極値に対して評価する。
    sum_sq = float(np.sum(imf1[embedding_indices] ** 2))
    if sum_sq <= 1e-20:
        raise ValueError("埋込極値の IMF エネルギーが小さすぎます。")
    alpha = 10.0 ** ((-FideP / 10.0 + np.log10(len(vertices)) - np.log10(sum_sq)) / 2.0)

    # 式 (6): bit=1 は正極大、bit=0 は負極小を選んだ上で、どちらも (1+alpha)。
    imf1_modified = imf1.copy()
    imf1_modified[embedding_indices] *= (1.0 + alpha)
    f_new = imf1_modified + residue
    rho_new = f_new * rho_max
    if np.any(rho_new < -1e-10):
        raise ValueError("再構成半径が負になりました。EMD 設定を見直してください。")
    rho_new = np.maximum(rho_new, 0.0)

    scale = np.divide(rho_new, rho, out=np.zeros_like(rho_new), where=rho > 1e-15)
    watermarked = centered * scale[:, None] + np.mean(vertices, axis=0)
    raw_extent = np.linalg.norm(np.ptp(watermarked, axis=0))
    key = HuWatermarkKey(
        embedding_indices=embedding_indices,
        embedding_positions=watermarked[embedding_indices].copy(),
        matching_threshold=max(
            0.75 * _nearest_neighbor_scale(watermarked), raw_extent * 1e-5
        ),
        bit_indices=bit_indices,
        repetitions=reps,
        watermark_size=watermark_size,
        circular_count=T,
        arnold_iterations=arnold_iterations,
        extreme_parameter=extreme_parameter,
        epsilon=epsilon,
    )
    return watermarked, key, alpha


def _greedy_unique_match(reference, candidate, distance_threshold, neighbors=16):
    """保存極値と攻撃後頂点を1対1対応し、遠い極値は欠損扱いにする。"""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    matched = np.full(len(reference), -1, dtype=np.int64)
    matched_distances = np.full(len(reference), np.inf, dtype=float)
    if len(reference) == 0 or len(candidate) == 0:
        return matched, matched_distances

    k = min(int(neighbors), len(candidate))
    distances, indices = cKDTree(candidate).query(reference, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    carrier_ids = np.repeat(np.arange(len(reference)), k)
    candidate_ids = indices.reshape(-1)
    edge_distances = distances.reshape(-1)
    allowed = np.isfinite(edge_distances) & (edge_distances <= distance_threshold)
    order = np.argsort(edge_distances[allowed], kind="stable")
    carrier_ids = carrier_ids[allowed][order]
    candidate_ids = candidate_ids[allowed][order]
    edge_distances = edge_distances[allowed][order]

    candidate_used = np.zeros(len(candidate), dtype=bool)
    for carrier, vertex, distance in zip(
        carrier_ids, candidate_ids, edge_distances
    ):
        if matched[carrier] < 0 and not candidate_used[vertex]:
            matched[carrier] = vertex
            matched_distances[carrier] = distance
            candidate_used[vertex] = True
    return matched, matched_distances


def _match_embedding_positions(vertices, key):
    """論文で保存する埋め込み極値位置から対応頂点を取得する。"""
    indices, _ = _greedy_unique_match(
        key.embedding_positions,
        vertices,
        key.matching_threshold,
    )
    valid = indices >= 0
    return np.asarray(indices, dtype=np.int64), np.asarray(valid, dtype=bool)


def extract_watermark_hu_mesh(vertices, triangles, key_info):
    """論文 Algorithm 2、式 (12)、式 (13) に基づいて透かしを抽出する。"""
    if not isinstance(key_info, HuWatermarkKey):
        raise TypeError("key_info must be the HuWatermarkKey returned by embed_watermark_hu_mesh.")
    vertices, triangles = _prepare_extraction_mesh(vertices, triangles)
    centered = vertices - np.mean(vertices, axis=0)
    rho = np.linalg.norm(centered, axis=1)
    rho_max = float(np.max(rho))
    if rho_max <= 1e-15:
        raise ValueError("全頂点が同一点のため、抽出できません。")
    imf1, _ = surface_emd_sifting(vertices, triangles, rho / rho_max, t=key_info.extreme_parameter)
    matched, valid_carriers = _match_embedding_positions(vertices, key_info)

    wm_len = key_info.watermark_size ** 2
    vote_count = np.zeros(wm_len, dtype=np.int64)
    one_count = np.zeros(wm_len, dtype=np.int64)
    # 式 (12): 正なら 1、負なら 0。各 repetition は完全な 1D 透かし列となる。
    valid_bits = key_info.bit_indices[valid_carriers]
    valid_vertices = matched[valid_carriers]
    np.add.at(vote_count, valid_bits, 1)
    np.add.at(
        one_count,
        valid_bits,
        (imf1[valid_vertices] > 0).astype(np.int64),
    )
    # 式 (13): T 枚を画素ごとに多数決する。
    # Missing carriers cast no vote. No surviving vote and an exact tie are
    # represented as -1 so BER evaluation counts an undecodable bit as an error.
    known_signal = (vote_count > 0) & (2 * one_count != vote_count)
    recovered_signal = (2 * one_count > vote_count).astype(np.uint8)
    recovered = hilbert_signal_to_watermark(
        recovered_signal,
        key_info.watermark_size,
        key_info.arnold_iterations,
    ).astype(np.int8)
    known = hilbert_signal_to_watermark(
        known_signal.astype(np.uint8),
        key_info.watermark_size,
        key_info.arnold_iterations,
    ).astype(bool)
    recovered[~known] = -1
    return recovered
