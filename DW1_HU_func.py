"""Hu et al. (2023) の表面 EMD に基づく 3D 透かし法.

論文: J. Hu et al., "Robust 3D watermarking with high
imperceptibility based on EMD on surfaces", The Visual Computer, 2023.

この実装は点群を再メッシュ化しない。入力メッシュの頂点順序と三角形接続を
そのまま利用することが、表面 EMD と再現可能な抽出の前提になる。
"""

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.spatial import cKDTree
import warnings


@dataclass
class HuWatermarkKey:
    """Algorithm 2 で必要な補助情報。

    ``embedding_positions`` は生の頂点番号ではなく埋込時の座標も保存する。
    これにより頂点並べ替え後も最近傍対応を取り直せる。
    """

    embedding_indices: np.ndarray
    embedding_positions: np.ndarray
    embedding_signatures: np.ndarray
    bit_indices: np.ndarray
    repetitions: np.ndarray
    watermark_size: int
    circular_count: int
    arnold_iterations: int
    extreme_parameter: float
    epsilon: float


def remove_unreferenced_vertices(vertices, triangles):
    """面に一度も現れない頂点を除去し、面インデックスを再番号付けする。

    これは面接続を再構成・変更しないため、Hu 法が要求する元メッシュのトポロジーを
    保ったまま、PLY に含まれる孤立頂点だけを除外できる。
    """
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (M, 3).")
    used = np.unique(triangles.ravel())
    if np.any(used < 0) or np.any(used >= len(vertices)):
        raise ValueError("triangles contains an out-of-range vertex index.")
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used].copy(), remap[triangles], used


def find_unreferenced_vertex_indices(vertices, triangles):
    """どの三角形面にも含まれない頂点のインデックスを返す。"""
    vertices = np.asarray(vertices)
    _, _, used = remove_unreferenced_vertices(vertices, triangles)
    mask = np.ones(len(vertices), dtype=bool)
    mask[used] = False
    return np.flatnonzero(mask)


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


def _canonical_coordinates(points):
    """平行移動・一様拡大・回転を除いた PCA 座標を返す。"""
    centered = points - np.mean(points, axis=0)
    scale = np.linalg.norm(centered, axis=1).max()
    if scale <= 1e-15:
        raise ValueError("対応付け用の座標正規化に失敗しました。")
    centered /= scale
    _, vectors = np.linalg.eigh(centered.T @ centered)
    basis = vectors[:, ::-1]
    return centered @ basis


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
    key = HuWatermarkKey(
        embedding_indices=embedding_indices,
        embedding_positions=watermarked[embedding_indices].copy(),
        embedding_signatures=_canonical_coordinates(watermarked)[embedding_indices].copy(),
        bit_indices=bit_indices,
        repetitions=reps,
        watermark_size=watermark_size,
        circular_count=T,
        arnold_iterations=arnold_iterations,
        extreme_parameter=extreme_parameter,
        epsilon=epsilon,
    )
    return watermarked, key, alpha


def _match_embedding_positions(vertices, key):
    """保存座標を使い、頂点順序変更後の対応頂点を取得する。"""
    # 頂点順序が保持され、座標も近い通常ケースをまず使う。
    direct = key.embedding_indices
    if len(vertices) > int(np.max(direct)):
        direct_error = np.median(np.linalg.norm(vertices[direct] - key.embedding_positions, axis=1))
        extent = np.linalg.norm(np.ptp(vertices, axis=0)) + 1e-15
        if direct_error < extent * 1e-5:
            return direct

    # 頂点並べ替えには座標ベースで対応付ける。さらに PCA 座標の符号・軸置換を
    # 全探索して、平行移動・回転・一様スケールの similarity transform に対応する。
    target_signature = _canonical_coordinates(vertices)
    reference = key.embedding_signatures
    best_indices, best_error = None, np.inf
    for permutation in permutations(range(3)):
        permuted = target_signature[:, permutation]
        for signs in product((-1.0, 1.0), repeat=3):
            candidate = permuted * np.asarray(signs)
            tree = cKDTree(candidate)
            distances, indices = tree.query(reference, k=1)
            error = float(np.median(distances))
            if error < best_error:
                best_error, best_indices = error, indices
    return np.asarray(best_indices, dtype=np.int64)


def extract_watermark_hu_mesh(vertices, triangles, key_info):
    """論文 Algorithm 2、式 (12)、式 (13) に基づいて透かしを抽出する。"""
    if not isinstance(key_info, HuWatermarkKey):
        raise TypeError("key_info must be the HuWatermarkKey returned by embed_watermark_hu_mesh.")
    vertices, triangles = _validate_mesh(vertices, triangles)
    centered = vertices - np.mean(vertices, axis=0)
    rho = np.linalg.norm(centered, axis=1)
    rho_max = float(np.max(rho))
    if rho_max <= 1e-15:
        raise ValueError("全頂点が同一点のため、抽出できません。")
    imf1, _ = surface_emd_sifting(vertices, triangles, rho / rho_max, t=key_info.extreme_parameter)
    matched = _match_embedding_positions(vertices, key_info)

    wm_len = key_info.watermark_size ** 2
    repeated_signals = np.zeros((key_info.circular_count, wm_len), dtype=np.uint8)
    # 式 (12): 正なら 1、負なら 0。各 repetition は完全な 1D 透かし列となる。
    repeated_signals[key_info.repetitions, key_info.bit_indices] = (imf1[matched] > 0).astype(np.uint8)
    recovered_images = np.asarray([
        hilbert_signal_to_watermark(row, key_info.watermark_size, key_info.arnold_iterations)
        for row in repeated_signals
    ])
    # 式 (13): T 枚を画素ごとに多数決する。
    return (np.sum(recovered_images, axis=0) >= int(np.ceil(key_info.circular_count / 2.0))).astype(np.uint8)
