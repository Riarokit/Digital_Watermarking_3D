"""Luke et al. のランダムベクトル場の面積分（Flux）に基づく3Dメッシュ透かし法。

参照実装:
    "3D Model Watermarking Using Surface Integrals of Generated Random
    Vector Fields" の Code Ocean 実装に含まれる ``flux.py`` を、Open3D等で扱う
    ``vertices: (N, 3)`` と ``triangles: (M, 3)`` の関数型APIへ整理したもの。

重要:
    * 対象は三角形メッシュであり、面を持たない点群には直接適用できない。
    * 元論文の透かしは正規分布から生成される実数列である。本実装では比較実験用に、
      0/1列を渡した場合は内部で-1/+1へ変換する二値拡張にも対応する。
    * 抽出には、埋め込み前のFluxなどを保存した ``LukeWatermarkKey`` が必要で、
      完全ブラインド抽出ではない。
    * 面順序変更への対応は、面番号ではなくFluxの順位を鍵に保存して行う。

公開API:
    embed_watermark_luke_mesh(...)
    extract_watermark_luke_mesh(...)
    compute_flux_luke(...)
    compute_flux_gradient_luke(...)
    evaluate_luke_robustness(...)
    evaluate_luke_distortion(...)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class LukeWatermarkKey:
    """Luke法の抽出に必要な補助情報。

    ``flux_rank_indices`` は生の面番号ではなく、透かし埋め込み後の全Fluxを
    昇順に並べたときの順位である。これにより三角形面の格納順が変更されても、
    Fluxの順位関係が保存される範囲では抽出できる。
    """

    original_flux: np.ndarray
    flux_rank_indices: np.ndarray
    watermark: np.ndarray
    watermark_bits: Optional[np.ndarray]
    binary_watermark: bool
    watermark_element_indices: np.ndarray
    target_pca: np.ndarray
    centroid: np.ndarray
    selected_facet_indices: np.ndarray
    seed: int
    random_vector_count: int
    alpha: float
    requested_watermark_size: int
    embedded_watermark_size: int
    learning_rate: float
    iterations: int
    distortion_threshold: float
    mean_w: float
    variance_w: float
    center_before_embedding: bool


# -----------------------------------------------------------------------------
# 入力・メッシュ補助
# -----------------------------------------------------------------------------


def _validate_mesh(vertices: np.ndarray, triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """頂点配列と三角形接続を検証し、型を揃える。"""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("vertices must have shape (N, 3) and contain at least one vertex.")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("Luke法には三角形面が必要です。triangles must have shape (M, 3).")
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("triangles contains an out-of-range vertex index.")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contains NaN or infinity.")

    # 面積0の三角形はFluxと勾配の数式を不安定にしやすいため除外せず警告対象とする。
    p1 = vertices[triangles[:, 0]]
    p2 = vertices[triangles[:, 1]]
    p3 = vertices[triangles[:, 2]]
    double_area = np.linalg.norm(np.cross(p2 - p1, p3 - p1), axis=1)
    if np.any(double_area <= 1e-15):
        raise ValueError("退化した三角形面が含まれています。面積0の面を除去してください。")

    return vertices, triangles


def _facets_from_indexed_mesh(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """indexed meshをCode Ocean実装と同じ ``(M, 9)`` の面配列へ変換する。"""
    return vertices[triangles].reshape(-1, 9)


def _coordinate_groups(vertices: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """完全に同じ座標を持つ頂点を同一グループとしてまとめる。

    STLでは同じ幾何頂点が面ごとに重複して保存される。元実装は座標タプルを鍵に
    これらを同時更新しているため、indexed meshでも同じ挙動になるようグループ化する。
    """
    key_to_group: Dict[Tuple[float, float, float], int] = {}
    members: List[List[int]] = []
    group_ids = np.empty(len(vertices), dtype=np.int64)

    for vertex_index, vertex in enumerate(vertices):
        key = tuple(vertex.tolist())
        group_id = key_to_group.get(key)
        if group_id is None:
            group_id = len(members)
            key_to_group[key] = group_id
            members.append([])
        members[group_id].append(vertex_index)
        group_ids[vertex_index] = group_id

    return group_ids, [np.asarray(group, dtype=np.int64) for group in members]


def _select_noninterfering_facets(
    triangles: np.ndarray,
    vertex_group_ids: np.ndarray,
    watermark_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """互いに頂点を共有しない面を貪欲法で選択する。

    使用済み頂点へ追加するのは、実際に候補として選択した面の頂点だけである。
    選択されなかった走査済み面は、後続候補を排除しない。
    """
    if watermark_size < 1:
        raise ValueError("watermark_size must be at least 1.")

    seen_groups = set()
    candidates = []
    for facet_index, triangle in enumerate(triangles):
        groups = vertex_group_ids[triangle]
        marked = all(int(group) not in seen_groups for group in groups)
        if marked:
            candidates.append(facet_index)
            seen_groups.update(int(group) for group in groups)

    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) == 0:
        raise ValueError("埋め込み可能な非干渉三角形面を取得できませんでした。")
    if len(candidates) > watermark_size:
        candidates = rng.choice(candidates, size=watermark_size, replace=False)
    return candidates


# -----------------------------------------------------------------------------
# ランダムベクトル場・Flux・勾配
# -----------------------------------------------------------------------------


def generate_random_vectors_luke(
    N: int,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Code Ocean実装と同じ3組のランダムベクトル ``k, zeta, xi`` を生成する。

    戻り値の形状は ``(3, N, 3)``。
    """
    if N < 1:
        raise ValueError("N must be at least 1.")
    if rng is None:
        rng = np.random.mtrand._rand

    sigma = np.identity(3)
    mu = np.zeros(3)
    return np.asarray(
        [
            rng.multivariate_normal(mu, np.pi / 2.0 * sigma, N),
            rng.multivariate_normal(mu, sigma, N),
            rng.multivariate_normal(mu, sigma, N),
        ],
        dtype=np.float64,
    )


def compute_flux_luke(facets: np.ndarray, random_vectors: np.ndarray) -> np.ndarray:
    """各三角形面に対する生成ランダムベクトル場のFluxを解析式で計算する。

    Parameters
    ----------
    facets:
        ``(M, 9)``。各行は ``[x1,y1,z1,x2,y2,z2,x3,y3,z3]``。
    random_vectors:
        ``generate_random_vectors_luke`` が返す ``(3, N, 3)``。
    """
    facets = np.asarray(facets, dtype=np.float64)
    random_vectors = np.asarray(random_vectors, dtype=np.float64)
    if facets.ndim != 2 or facets.shape[1] != 9:
        raise ValueError("facets must have shape (M, 9).")
    if random_vectors.ndim != 3 or random_vectors.shape[0] != 3 or random_vectors.shape[2] != 3:
        raise ValueError("random_vectors must have shape (3, N, 3).")

    k = random_vectors[0]
    zeta = random_vectors[1]
    xi = random_vectors[2]

    p1 = facets[:, 0:3]
    p2 = facets[:, 3:6]
    p3 = facets[:, 6:9]

    a = (np.cross(p1, p2) - np.cross(p1, p3) + np.cross(p2, p3)).T
    b = np.array(
        [
            np.dot(np.cross(k, zeta), a),
            np.dot(np.cross(k, xi), a),
            np.dot(k, (p2 + p3 - 2.0 * p1).T),
            np.dot(k, (2.0 * p1 - p3).T),
            np.dot(k, p2.T),
            np.dot(k, p1.T),
            np.dot(k, (p1 - p3).T),
            np.dot(k, (p1 - p2).T),
        ]
    )

    # 元実装では分母が厳密に0の要素をNaNにしている。
    b[2][b[2] == 0] = np.nan
    b[6][b[6] == 0] = np.nan
    b[7][b[7] == 0] = np.nan
    c = np.array([b[6] * b[2], b[7] * b[6], np.cos(b[4]), np.sin(b[4])])

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        flux = np.sum(
            b[0]
            * (
                (np.cos(b[3]) - c[2]) / c[0]
                + (np.cos(b[5]) - c[2]) / c[1]
            )
            + b[1]
            * (
                (np.sin(b[3]) - c[3]) / c[0]
                + (np.sin(b[5]) - c[3]) / c[1]
            ),
            axis=0,
        )
    return flux


def compute_flux_gradient_luke(
    facets: np.ndarray,
    target_flux: np.ndarray,
    random_vectors: np.ndarray,
) -> np.ndarray:
    """目標Fluxに対する頂点更新用勾配を解析式で計算する。

    Code Ocean版 ``Flux.compute_gradient`` の式をそのまま関数化している。
    戻り値は各面の9座標に対応する ``(M, 9)``。
    """
    facets = np.asarray(facets, dtype=np.float64)
    target_flux = np.asarray(target_flux, dtype=np.float64)
    random_vectors = np.asarray(random_vectors, dtype=np.float64)
    if facets.ndim != 2 or facets.shape[1] != 9:
        raise ValueError("facets must have shape (M, 9).")
    if target_flux.ndim != 1 or len(target_flux) != len(facets):
        raise ValueError("target_flux must have one value per facet.")

    N = random_vectors.shape[1]
    watermark_size = facets.shape[0]

    k = random_vectors[0]
    zeta = random_vectors[1]
    xi = random_vectors[2]

    p1 = facets[:, 0:3]
    p2 = facets[:, 3:6]
    p3 = facets[:, 6:9]
    p = [p1, p2, p3]

    det = np.array([[np.zeros((N, 3, 3))]]).repeat(18, axis=1).repeat(watermark_size, axis=0)
    for i in range(3):
        pi = 1 + i // -3
        pj = 2 - i // 2
        det[:, i::3, :, 1, :] = k
        det[:, i::6, :, 2, :] = zeta
        det[:, i + 3::6, :, 2, :] = xi
        for j in range(0, 6, 3):
            det[:, [6 * i + j, 6 * i + j + 1], :, 0, [1, 0]] = np.repeat(
                [-(p[pi][:, 2] - p[pj][:, 2])], N, axis=0
            ).T
            det[:, [6 * i + j, 6 * i + j + 2], :, 0, [2, 0]] = np.repeat(
                [p[pi][:, 1] - p[pj][:, 1]], N, axis=0
            ).T
            det[:, 6 * i + j + 1, :, 0, 2] = np.repeat(
                [p[pi][:, 0] - p[pj][:, 0]], N, axis=0
            ).T
            det[:, 6 * i + j + 2, :, 0, 1] = np.repeat(
                [-(p[pi][:, 0] - p[pj][:, 0])], N, axis=0
            ).T
    det = np.linalg.det(det)

    a = (np.cross(p1, p2) - np.cross(p1, p3) + np.cross(p2, p3)).T
    b = np.array(
        [
            np.dot(np.cross(k, zeta), a),
            np.dot(np.cross(k, xi), a),
            np.dot(k, (p2 + p3 - 2.0 * p1).T),
            np.dot(k, (2.0 * p1 - p3).T),
            np.dot(k, p2.T),
            np.dot(k, p1.T),
            np.dot(k, (p1 - p3).T),
            np.dot(k, (p1 - p2).T),
        ]
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        c = np.array(
            [
                b[6] * b[2],
                b[7] * b[6],
                np.cos(b[4]),
                np.sin(b[4]),
                np.cos(b[5]),
                np.sin(b[5]),
                np.cos(b[3]),
                np.sin(b[3]),
            ]
        )
        d = np.array(
            [
                (c[4] - c[2]) / c[1],
                (c[6] - c[2]) / c[0],
                (c[5] - c[3]) / c[1],
                (c[7] - c[3]) / c[0],
            ]
        )
        e = np.array([d[0] + d[1], d[2] + d[3]])
        f = np.array(
            [
                d[0] / b[6] + d[0] / b[7] + d[1] / b[6] - 2.0 * d[1] / b[2] + c[5] / c[1] + 2.0 * c[7] / c[0],
                d[2] / b[6] + d[2] / b[7] + d[3] / b[6] - 2.0 * d[3] / b[2] - c[4] / c[1] - 2.0 * c[6] / c[0],
                d[0] / b[7] - d[1] / b[2] + c[3] / c[1] + c[3] / c[0],
                -d[2] / b[7] + d[3] / b[2] + c[2] / c[1] + c[2] / c[0],
                d[0] / b[6] + d[1] / b[6] - d[1] / b[2] + c[7] / c[0],
                d[2] / b[6] + d[3] / b[6] - d[3] / b[2] - c[6] / c[0],
            ]
        )
        g = np.array(
            [
                b[0] * f[0] + b[1] * f[1],
                b[0] * f[2] - b[1] * f[3],
                b[0] * f[4] + b[1] * f[5],
            ]
        )

        grad_components = np.sum(
            np.array(
                [
                    b[0] * e[0] + b[1] * e[1],
                    np.repeat([k[:, 0]], watermark_size, axis=0).T * g[0] - e[0] * det[:, 0].T - e[1] * det[:, 3].T,
                    np.repeat([k[:, 1]], watermark_size, axis=0).T * g[0] + e[0] * det[:, 1].T + e[1] * det[:, 4].T,
                    np.repeat([k[:, 2]], watermark_size, axis=0).T * g[0] + e[0] * det[:, 2].T + e[1] * det[:, 5].T,
                    -(np.repeat([k[:, 0]], watermark_size, axis=0).T * g[1] - e[0] * det[:, 6].T - e[1] * det[:, 9].T),
                    -(np.repeat([k[:, 1]], watermark_size, axis=0).T * g[1] + e[0] * det[:, 7].T + e[1] * det[:, 10].T),
                    -(np.repeat([k[:, 2]], watermark_size, axis=0).T * g[1] + e[0] * det[:, 8].T + e[1] * det[:, 11].T),
                    -(np.repeat([k[:, 0]], watermark_size, axis=0).T * g[2] + e[0] * det[:, 12].T + e[1] * det[:, 15].T),
                    -(np.repeat([k[:, 1]], watermark_size, axis=0).T * g[2] - e[0] * det[:, 13].T - e[1] * det[:, 16].T),
                    -(np.repeat([k[:, 2]], watermark_size, axis=0).T * g[2] - e[0] * det[:, 14].T - e[1] * det[:, 17].T),
                ]
            ),
            axis=1,
        )

    h = target_flux - grad_components[0, :]
    return np.array(
        [
            h * grad_components[1, :],
            h * grad_components[2, :],
            h * grad_components[3, :],
            h * grad_components[4, :],
            h * grad_components[5, :],
            h * grad_components[6, :],
            h * grad_components[7, :],
            h * grad_components[8, :],
            h * grad_components[9, :],
        ]
    ).T


def encode_flux_luke(flux: np.ndarray, watermark: np.ndarray, alpha: float) -> np.ndarray:
    """式 ``phi_w = phi + alpha * w`` により目標Fluxを作る。"""
    if alpha == 0:
        raise ValueError("alpha must be non-zero.")
    return np.asarray(flux, dtype=np.float64) + alpha * np.asarray(watermark, dtype=np.float64)


def decode_flux_luke(original_flux: np.ndarray, extracted_flux: np.ndarray, alpha: float) -> np.ndarray:
    """式 ``w_hat = (phi_attacked - phi_original) / alpha`` で実数透かしを復元する。"""
    if alpha == 0:
        raise ValueError("alpha must be non-zero.")
    return (np.asarray(extracted_flux, dtype=np.float64) - np.asarray(original_flux, dtype=np.float64)) / alpha


# -----------------------------------------------------------------------------
# 姿勢補正
# -----------------------------------------------------------------------------


def compute_pca_axes_luke(vertices: np.ndarray) -> np.ndarray:
    """元実装と同じ主成分軸と符号選択を計算する。"""
    vertices = np.unique(np.asarray(vertices, dtype=np.float64), axis=0)
    if len(vertices) < 3:
        raise ValueError("PCAには少なくとも3個の異なる頂点が必要です。")

    covariance_matrix = np.cov(vertices.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    eigen_indices = eigen_values.argsort()[::-1]
    sorted_eigen_vectors = np.real(eigen_vectors[:, eigen_indices])

    vertices_pca = vertices.dot(sorted_eigen_vectors)
    orientations = np.array([1.0, -1.0])
    orientation_indices = np.argmax(
        [
            np.sum(np.abs(vertices_pca - vertices_pca.max(axis=0)), axis=0),
            np.sum(np.abs(vertices_pca - vertices_pca.min(axis=0)), axis=0),
        ],
        axis=0,
    )
    return sorted_eigen_vectors * orientations[orientation_indices]


def rotation_align_luke(
    vertices: np.ndarray,
    source_eigenvectors: np.ndarray,
    target_eigenvectors: np.ndarray,
) -> np.ndarray:
    """PCA軸を用いて攻撃後モデルを埋め込み時の姿勢へ回転整列する。"""
    rotation, _ = Rotation.align_vectors(
        np.asarray(target_eigenvectors, dtype=np.float64).T,
        np.asarray(source_eigenvectors, dtype=np.float64).T,
    )
    return rotation.apply(np.asarray(vertices, dtype=np.float64))


def translate_vertices_luke(vertices: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """全頂点へ平行移動を適用する。"""
    translation = np.asarray(translation, dtype=np.float64)
    if translation.shape != (3,):
        raise ValueError("translation must have shape (3,).")
    return np.asarray(vertices, dtype=np.float64) + translation


# -----------------------------------------------------------------------------
# 勾配降下・埋め込み対象の精製
# -----------------------------------------------------------------------------


def _gradient_descent_luke(
    vertices: np.ndarray,
    triangles: np.ndarray,
    selected_facet_indices: np.ndarray,
    target_flux: np.ndarray,
    random_vectors: np.ndarray,
    vertex_group_ids: np.ndarray,
    group_members: List[np.ndarray],
    iterations: int,
    learning_rate: float,
    gradient_batch_size: Optional[int] = 2048,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fluxが目標値へ近づくよう、共有幾何頂点をまとめて更新する。"""
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if gradient_batch_size is not None and gradient_batch_size < 1:
        raise ValueError("gradient_batch_size must be positive or None.")

    updated = np.asarray(vertices, dtype=np.float64).copy()
    selected_groups = vertex_group_ids[triangles[selected_facet_indices]]
    active_groups = np.unique(selected_groups)
    errors = []

    for iteration in range(iterations):
        selected_facets = _facets_from_indexed_mesh(updated, triangles[selected_facet_indices])
        face_gradient = np.empty_like(selected_facets)

        if gradient_batch_size is None:
            face_gradient[:] = compute_flux_gradient_luke(selected_facets, target_flux, random_vectors)
        else:
            for start in range(0, len(selected_facets), gradient_batch_size):
                end = min(start + gradient_batch_size, len(selected_facets))
                face_gradient[start:end] = compute_flux_gradient_luke(
                    selected_facets[start:end], target_flux[start:end], random_vectors
                )

        if not np.isfinite(face_gradient).all():
            raise FloatingPointError(
                "Flux勾配にNaNまたは無限大が発生しました。モデルの退化面、座標スケール、"
                "N、alpha、learning_rateを確認してください。"
            )

        # 各面の局所3頂点勾配を、元実装の座標タプル単位の共有頂点へ集約する。
        group_gradient = np.zeros((len(group_members), 3), dtype=np.float64)
        np.add.at(group_gradient, selected_groups.ravel(), face_gradient.reshape(-1, 3))
        for group_id in active_groups:
            updated[group_members[int(group_id)]] -= learning_rate * group_gradient[int(group_id)]

        current_flux = compute_flux_luke(
            _facets_from_indexed_mesh(updated, triangles[selected_facet_indices]),
            random_vectors,
        )
        if not np.isfinite(current_flux).all():
            raise FloatingPointError("勾配更新後のFluxにNaNまたは無限大が発生しました。")
        error = float(np.mean((target_flux - current_flux) ** 2))
        errors.append(error)

        if verbose:
            print(f"[Luke] Gradient descent {iteration + 1}/{iterations}, mse={error:.6e}")

        check_interval = 10
        if iteration > 1 and iteration % check_interval == 0:
            if errors[iteration] > errors[iteration - check_interval]:
                if verbose:
                    print("[Luke] Early stopping: 10反復前より誤差が増加しました。")
                break

    return updated, np.asarray(errors, dtype=np.float64)


def _refine_luke_indexed_mesh(
    original_vertices: np.ndarray,
    watermarked_vertices: np.ndarray,
    triangles: np.ndarray,
    selected_facet_indices: np.ndarray,
    watermark: np.ndarray,
    watermark_element_indices: np.ndarray,
    vertex_group_ids: np.ndarray,
    group_members: List[np.ndarray],
    distortion_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """歪み閾値を超えた埋め込み面を復元し、その透かし要素を除外する。

    Code Ocean版はSTLの面配列を直接復元するが、indexed meshで同じ処理をすると
    共有頂点の不整合を表現できない。本実装ではメッシュ接続を壊さないよう、閾値を
    超えた「埋め込み対象面」の3頂点グループを元へ戻す。
    """
    if distortion_threshold < 0:
        raise ValueError("distortion_threshold must be non-negative.")

    refined = watermarked_vertices.copy()
    original_selected_facets = _facets_from_indexed_mesh(original_vertices, triangles[selected_facet_indices])
    current_selected_facets = _facets_from_indexed_mesh(refined, triangles[selected_facet_indices])
    displacement = (
        original_selected_facets - current_selected_facets
    ).reshape(-1, 3, 3)
    vertex_displacement = np.linalg.norm(displacement, axis=2)
    max_vertex_displacement = np.max(vertex_displacement, axis=1)
    keep = max_vertex_displacement <= distortion_threshold

    rejected_facets = selected_facet_indices[~keep]
    if len(rejected_facets):
        rejected_groups = np.unique(vertex_group_ids[triangles[rejected_facets]])
        for group_id in rejected_groups:
            members = group_members[int(group_id)]
            refined[members] = original_vertices[members]

    return (
        refined,
        selected_facet_indices[keep],
        watermark[keep],
        watermark_element_indices[keep],
    )


# -----------------------------------------------------------------------------
# 高水準の埋め込み・抽出API
# -----------------------------------------------------------------------------


def embed_watermark_luke_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    watermark_size: int,
    seed: int,
    N: int,
    alpha: float,
    learning_rate: float,
    iterations: int = 100,
    distortion_threshold: float = 0.001,
    mean_w: float = 0.0,
    variance_w: float = 1.0,
    center_before_embedding: bool = False,
    watermark: Optional[np.ndarray] = None,
    gradient_batch_size: Optional[int] = 2048,
    verbose: bool = False,
) -> Tuple[np.ndarray, LukeWatermarkKey, Dict[str, object]]:
    """LukeらのFlux方式で三角形メッシュへ実数透かしを埋め込む。

    Parameters
    ----------
    vertices, triangles:
        indexed triangle mesh。
    watermark_size:
        埋め込みを試みる実数透かし要素数。利用可能な非干渉面が少ない場合は減る。
    seed, N, alpha, learning_rate, iterations, distortion_threshold, mean_w, variance_w:
        Code Ocean版 ``Flux.watermark`` と同じ意味のパラメータ。
    center_before_embedding:
        Trueなら重心を原点へ移して埋め込み、最後に元位置へ戻す。
    watermark:
        省略時は元実装どおり正規分布から生成する。指定時は1次元実数列を利用する。
        指定列が候補面数より長い場合は、実際の選択面数へ切り詰める。
    gradient_batch_size:
        複雑な解析勾配のメモリ使用量を抑えるバッチサイズ。Noneなら一括計算。

    Returns
    -------
    watermarked_vertices, key_info, details
        ``details`` には勾配誤差、自己抽出値、歪み・頑健性評価を含む。
    """
    vertices, triangles = _validate_mesh(vertices, triangles)
    if alpha == 0:
        raise ValueError("alpha must be non-zero.")
    if variance_w < 0:
        raise ValueError("variance_w must be non-negative.")

    original_output_coordinates = vertices.copy()
    working_vertices = vertices.copy()
    original_centroid = np.mean(working_vertices, axis=0)
    if center_before_embedding:
        working_vertices -= original_centroid

    original_working_vertices = working_vertices.copy()
    vertex_group_ids, group_members = _coordinate_groups(working_vertices)

    # 元実装と同じ乱数系列: ベクトル場生成 → 面の乱択 → 正規透かし生成。
    rng = np.random.RandomState(int(seed))
    random_vectors = generate_random_vectors_luke(N, rng)
    selected_facet_indices = _select_noninterfering_facets(
        triangles, vertex_group_ids, watermark_size, rng
    )

    actual_size = len(selected_facet_indices)
    binary_watermark = False
    if watermark is None:
        watermark_values = rng.normal(mean_w, variance_w, size=actual_size)
    else:
        supplied_watermark = np.asarray(watermark, dtype=np.float64)
        if supplied_watermark.ndim != 1:
            raise ValueError("watermark must be a one-dimensional real-valued array.")
        if not np.isfinite(supplied_watermark).all():
            raise ValueError("watermark contains NaN or infinity.")
        if len(supplied_watermark) < actual_size:
            # 指定透かし長を優先し、選択面側を縮める。
            selected_facet_indices = selected_facet_indices[: len(supplied_watermark)]
            actual_size = len(selected_facet_indices)
        supplied_watermark = supplied_watermark[:actual_size].copy()

        # 0/1入力は内部で対称な-1/+1信号へ変換する。
        binary_watermark = bool(np.isin(supplied_watermark, (0.0, 1.0)).all())
        if binary_watermark:
            watermark_values = 2.0 * supplied_watermark - 1.0
        else:
            watermark_values = supplied_watermark
    if actual_size == 0:
        raise ValueError("埋め込む透かし要素がありません。")
    if not np.isfinite(watermark_values).all():
        raise ValueError("watermark contains NaN or infinity.")
    # 歪み閾値で一部要素が除外されても、元透かし列上の位置を保持する。
    watermark_element_indices = np.arange(actual_size, dtype=np.int64)

    original_facets = _facets_from_indexed_mesh(working_vertices, triangles)
    original_flux_all = compute_flux_luke(original_facets, random_vectors)
    if not np.isfinite(original_flux_all).all():
        raise FloatingPointError(
            "初期FluxにNaNまたは無限大が発生しました。座標原点・モデルスケール・"
            "退化面を確認してください。center_before_embedding=Trueも検討してください。"
        )

    target_flux = encode_flux_luke(
        original_flux_all[selected_facet_indices], watermark_values, alpha
    )
    selected_size_before_refinement = len(selected_facet_indices)
    watermarked_working, gradient_error = _gradient_descent_luke(
        working_vertices,
        triangles,
        selected_facet_indices,
        target_flux,
        random_vectors,
        vertex_group_ids,
        group_members,
        iterations,
        learning_rate,
        gradient_batch_size=gradient_batch_size,
        verbose=verbose,
    )

    # Record the displacement before the distortion refinement.  This makes it
    # possible to distinguish an overly strict threshold from facet-selection
    # or extraction failures.
    original_selected_before_refinement = _facets_from_indexed_mesh(
        original_working_vertices, triangles[selected_facet_indices]
    ).reshape(-1, 3, 3)
    watermarked_selected_before_refinement = _facets_from_indexed_mesh(
        watermarked_working, triangles[selected_facet_indices]
    ).reshape(-1, 3, 3)
    pre_refinement_max_displacement = np.max(
        np.linalg.norm(
            original_selected_before_refinement
            - watermarked_selected_before_refinement,
            axis=2,
        ),
        axis=1,
    )

    (
        watermarked_working,
        selected_facet_indices,
        watermark_values,
        watermark_element_indices,
    ) = _refine_luke_indexed_mesh(
        original_working_vertices,
        watermarked_working,
        triangles,
        selected_facet_indices,
        watermark_values,
        watermark_element_indices,
        vertex_group_ids,
        group_members,
        distortion_threshold,
    )
    if len(selected_facet_indices) == 0:
        raise ValueError(
            "distortion_thresholdにより全埋め込み面が除外されました。"
            "閾値、alpha、learning_rateを見直してください。"
        )

    final_facets = _facets_from_indexed_mesh(watermarked_working, triangles)
    final_flux_all = compute_flux_luke(final_facets, random_vectors)
    if not np.isfinite(final_flux_all).all():
        raise FloatingPointError("埋め込み後FluxにNaNまたは無限大が発生しました。")

    original_flux_selected = original_flux_all[selected_facet_indices]
    self_extracted = decode_flux_luke(
        original_flux_selected,
        final_flux_all[selected_facet_indices],
        alpha,
    )

    # 各選択面の「埋め込み後Fluxの順位」を保存し、面順序変更へ対応する。
    flux_rank_indices = np.argsort(np.argsort(final_flux_all))[selected_facet_indices]
    embedded_centroid = np.mean(watermarked_working, axis=0)
    target_pca = compute_pca_axes_luke(watermarked_working)

    if center_before_embedding:
        output_vertices = watermarked_working + original_centroid
    else:
        output_vertices = watermarked_working

    key = LukeWatermarkKey(
        original_flux=original_flux_selected.copy(),
        flux_rank_indices=np.asarray(flux_rank_indices, dtype=np.int64),
        watermark=watermark_values.copy(),
        watermark_bits=(
            (watermark_values > 0.0).astype(np.uint8)
            if binary_watermark else None
        ),
        binary_watermark=binary_watermark,
        watermark_element_indices=watermark_element_indices.copy(),
        target_pca=target_pca.copy(),
        centroid=embedded_centroid.copy(),
        selected_facet_indices=selected_facet_indices.copy(),
        seed=int(seed),
        random_vector_count=int(N),
        alpha=float(alpha),
        requested_watermark_size=int(watermark_size),
        embedded_watermark_size=int(len(watermark_values)),
        learning_rate=float(learning_rate),
        iterations=int(iterations),
        distortion_threshold=float(distortion_threshold),
        mean_w=float(mean_w),
        variance_w=float(variance_w),
        center_before_embedding=bool(center_before_embedding),
    )

    details: Dict[str, object] = {
        "gradient_descent_error": gradient_error,
        "self_extracted_watermark": self_extracted,
        "binary_watermark": binary_watermark,
        "distortion": evaluate_luke_distortion(original_output_coordinates, output_vertices),
        "robustness": evaluate_luke_robustness(watermark_values, self_extracted),
        "selected_facet_indices": selected_facet_indices.copy(),
        "watermark_element_indices": watermark_element_indices.copy(),
        "selected_size_before_refinement": int(selected_size_before_refinement),
        "pre_refinement_max_displacement": pre_refinement_max_displacement.copy(),
        "rejected_by_distortion": int(
            selected_size_before_refinement - len(selected_facet_indices)
        ),
        "requested_watermark_size": int(watermark_size),
        "embedded_watermark_size": int(len(watermark_values)),
    }
    return output_vertices, key, details


def extract_watermark_luke_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    key_info: LukeWatermarkKey,
    translation_align: Optional[bool] = None,
    rotation_align: bool = False,
    return_details: bool = False,
):
    """Luke法の鍵を用いて透かしを抽出する。

    実数透かしでは実数列を返す。0/1透かしを埋め込んだ鍵では、抽出した
    -1/+1信号を0で判定した0/1ビット列を返す。

    Parameters
    ----------
    translation_align:
        Trueなら攻撃後モデルの重心を、鍵に保存された埋め込み時重心へ合わせる。
        Noneの場合、埋め込み時に中心化した鍵では自動的にTrueとする。
    rotation_align:
        TrueならPCA軸を鍵の軸へ整列する。対称形状ではPCA軸が不安定な場合がある。
    return_details:
        Trueなら ``(extracted_watermark, details)`` を返す。
    """
    if not isinstance(key_info, LukeWatermarkKey):
        raise TypeError("key_info must be the LukeWatermarkKey returned by embed_watermark_luke_mesh.")
    vertices, triangles = _validate_mesh(vertices, triangles)
    working = vertices.copy()

    if translation_align is None:
        translation_align = key_info.center_before_embedding

    if translation_align:
        current_centroid = np.mean(working, axis=0)
        working += key_info.centroid - current_centroid

    if rotation_align:
        source_pca = compute_pca_axes_luke(working)
        working = rotation_align_luke(working, source_pca, key_info.target_pca)

    rng = np.random.RandomState(int(key_info.seed))
    random_vectors = generate_random_vectors_luke(key_info.random_vector_count, rng)
    attacked_flux = compute_flux_luke(_facets_from_indexed_mesh(working, triangles), random_vectors)

    if not np.isfinite(attacked_flux).all():
        extracted_signal = np.zeros_like(key_info.watermark)
        valid = False
    elif len(attacked_flux) == 0 or np.max(key_info.flux_rank_indices) >= len(attacked_flux):
        extracted_signal = np.zeros_like(key_info.watermark)
        valid = False
    else:
        sorted_flux = np.sort(attacked_flux)
        extracted_signal = decode_flux_luke(
            key_info.original_flux,
            sorted_flux[key_info.flux_rank_indices],
            key_info.alpha,
        )
        valid = True

    if key_info.binary_watermark:
        if valid:
            extracted = (extracted_signal >= 0.0).astype(np.uint8)
        else:
            extracted = np.zeros(len(key_info.watermark), dtype=np.uint8)
    else:
        extracted = extracted_signal

    if return_details:
        details = {
            "valid_rank_lookup": valid,
            "robustness": evaluate_luke_robustness(
                key_info.watermark, extracted_signal
            ),
            "extracted_watermark_signal": extracted_signal,
            "attacked_flux": attacked_flux,
        }
        return extracted, details
    return extracted


# -----------------------------------------------------------------------------
# 評価関数（元実装metric.py相当）
# -----------------------------------------------------------------------------


def _mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))


def evaluate_luke_robustness(
    original_watermark: np.ndarray,
    extracted_watermark: np.ndarray,
) -> Dict[str, float]:
    """実数透かし向けに相関、MSE、NMSE、similarityを返す。

    Code Ocean版の ``similarity`` は原透かしのノルムで正規化しない定義を使用する。
    ``bit_error_rate`` は浮動小数点表現のビット比較となり意味が特殊なため省略する。
    """
    original = np.asarray(original_watermark, dtype=np.float64)
    extracted = np.asarray(extracted_watermark, dtype=np.float64)
    if original.shape != extracted.shape:
        raise ValueError("original_watermark and extracted_watermark must have the same shape.")

    mse = _mean_squared_error(original, extracted)
    value_range = float(np.max(original) - np.min(original)) if len(original) else 0.0
    nmse = mse / value_range if value_range > 0 else float("inf")

    extracted_norm = float(np.linalg.norm(extracted))
    similarity = float(np.dot(original, extracted) / extracted_norm) if extracted_norm > 0 else 0.0
    if original.size > 1 and np.any(original) and np.isfinite(extracted).all():
        correlation = float(np.corrcoef(original, extracted)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
    else:
        correlation = 0.0

    return {
        "similarity": similarity,
        "correlation": correlation,
        "mean_squared_error": mse,
        "normalized_mean_squared_error": float(nmse),
    }


def evaluate_luke_distortion(
    original_vertices: np.ndarray,
    embedded_vertices: np.ndarray,
) -> Dict[str, float]:
    """元実装と同じ頂点歪み指標を返す。"""
    original = np.asarray(original_vertices, dtype=np.float64)
    embedded = np.asarray(embedded_vertices, dtype=np.float64)
    if original.shape != embedded.shape:
        raise ValueError("original_vertices and embedded_vertices must have the same shape.")

    delta = original - embedded
    mse = float(np.mean(delta ** 2))
    rmse = float(np.sqrt(mse))
    distances = np.linalg.norm(delta, axis=1)
    centered = original - np.mean(original, axis=0)
    signal_energy = float(np.sum(centered ** 2))
    noise_energy = float(np.sum(delta ** 2))
    if noise_energy == 0:
        snr = float("inf")
    elif signal_energy == 0:
        snr = float("-inf")
    else:
        snr = float(10.0 * np.log10(signal_energy / noise_energy))

    return {
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "average_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
        "signal_to_noise_ratio": snr,
    }
