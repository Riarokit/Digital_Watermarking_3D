"""El Zein et al. の Method II に準拠する非冗長な HARD 実装。

透かしを 5:3 で二分し、Portion 1 は 1-ring の平均と標準偏差、
Portion 2 は式 (6) ``V*=V+a*w`` で埋め込む。各透かし要素には異なる
moderate 頂点を1個だけ割り当て、反復埋め込み、グループ多数決、最近傍
同期、欠損補完は行わない。

原論文は Portion 1 の抽出許容値 delta を定めておらず、式 (9), (10) の
条件にも曖昧さがある。本実装では受信メッシュの 1-ring から得た2候補
``mu-2*sigma`` と ``mu+2*sigma`` のうち、受信頂点に近い方を選ぶ。
Portion 2 の二値入力は実数 -1/+1 に写像して式 (6) を適用する。
"""

from dataclasses import dataclass

import numpy as np

from DW1_ELZ_SOFT_func import (
    _mesh_topology,
    _validate_mesh,
    local_feature_clustering,
)


@dataclass(frozen=True)
class ElZeinHardKey:
    """埋め込み時に確定した Method II の最小限の設定。"""

    watermark_length: int
    portion1_length: int
    portion2_length: int
    scaling_factor: float
    seed: int


def split_watermark_5_to_3(watermark_length):
    """任意長を5:3に最も近い整数長へ round-half-up で分割する。"""
    length = int(watermark_length)
    if length < 2:
        raise ValueError("Method II requires at least two watermark bits.")
    portion1_length = (5 * length + 4) // 8
    portion2_length = length - portion1_length
    if portion1_length == 0 or portion2_length == 0:
        raise ValueError("Both Method II portions must contain at least one bit.")
    return portion1_length, portion2_length


def _validate_bits(watermark_bits):
    raw_bits = np.asarray(watermark_bits).reshape(-1)
    if np.any((raw_bits != 0) & (raw_bits != 1)):
        raise ValueError("watermark_bits must contain only 0 and 1.")
    bits = raw_bits.astype(np.uint8)
    split_watermark_5_to_3(len(bits))
    return bits


def _neighbor_statistics(vertices, neighbors, vertex_index):
    neighbor_indices = np.fromiter(
        sorted(neighbors[vertex_index]), dtype=np.int64, count=6
    )
    if len(neighbor_indices) != 6:
        raise ValueError(
            f"Carrier vertex {vertex_index} does not have a degree-6 1-ring."
        )
    ring = vertices[neighbor_indices]
    return ring.mean(axis=0), ring.std(axis=0)


def _select_carriers(vertices, triangles, watermark_length, verbose, seed):
    selected = local_feature_clustering(
        vertices, triangles, verbose=verbose, seed=seed
    )
    if len(selected) < watermark_length:
        raise ValueError(
            f"Watermark needs {watermark_length} distinct carriers, but the "
            f"moderate cluster contains only {len(selected)}."
        )
    return selected[:watermark_length]


def embed_watermark_elzein_hard_mesh(
    vertices,
    triangles,
    watermark_bits,
    a=0.01,
    verbose=True,
    seed=42,
):
    """Method II を5:3分割し、各要素を異なる1頂点へ埋め込む。"""
    vertices, triangles = _validate_mesh(vertices, triangles)
    bits = _validate_bits(watermark_bits)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("a must be a positive finite scaling factor.")

    portion1_length, portion2_length = split_watermark_5_to_3(len(bits))
    carriers = _select_carriers(
        vertices, triangles, len(bits), verbose=verbose, seed=seed
    )
    neighbors, _, _ = _mesh_topology(vertices, triangles)
    marked = vertices.copy()

    for bit, vertex_index in zip(bits[:portion1_length], carriers[:portion1_length]):
        mean, standard_deviation = _neighbor_statistics(
            vertices, neighbors, int(vertex_index)
        )
        direction = 1.0 if bit else -1.0
        marked[vertex_index] = mean + direction * 2.0 * standard_deviation

    portion2_bits = bits[portion1_length:]
    portion2_carriers = carriers[portion1_length:]
    real_watermark = 2.0 * portion2_bits.astype(float) - 1.0
    marked[portion2_carriers] += a * real_watermark[:, None]

    key = ElZeinHardKey(
        watermark_length=len(bits),
        portion1_length=portion1_length,
        portion2_length=portion2_length,
        scaling_factor=float(a),
        seed=int(seed),
    )
    return marked, key


def extract_watermark_elzein_hard_mesh(
    marked_vertices,
    original_vertices,
    triangles,
    key_info,
    verbose=False,
):
    """同期や補完を行わず、Method II の二経路から透かしを抽出する。"""
    if not isinstance(key_info, ElZeinHardKey):
        raise TypeError(
            "key_info must be the ElZeinHardKey returned by the HARD embedder."
        )
    original_vertices, triangles = _validate_mesh(original_vertices, triangles)
    marked_vertices = np.asarray(marked_vertices, dtype=float)
    if marked_vertices.shape != original_vertices.shape:
        raise ValueError(
            "HARD extraction requires unchanged vertex count and ordering; "
            "point-cloud synchronization is intentionally disabled."
        )

    carriers = _select_carriers(
        marked_vertices,
        triangles,
        key_info.watermark_length,
        verbose=verbose,
        seed=key_info.seed,
    )
    neighbors, _, _ = _mesh_topology(marked_vertices, triangles)
    extracted = []

    for vertex_index in carriers[:key_info.portion1_length]:
        mean, standard_deviation = _neighbor_statistics(
            marked_vertices, neighbors, int(vertex_index)
        )
        zero_prototype = mean - 2.0 * standard_deviation
        one_prototype = mean + 2.0 * standard_deviation
        vertex = marked_vertices[vertex_index]
        zero_distance = np.linalg.norm(vertex - zero_prototype)
        one_distance = np.linalg.norm(vertex - one_prototype)
        extracted.append(int(one_distance < zero_distance))

    portion2_carriers = carriers[key_info.portion1_length:]
    coordinate_difference = (
        marked_vertices[portion2_carriers] - original_vertices[portion2_carriers]
    )
    recovered_real_values = coordinate_difference.mean(axis=1) / key_info.scaling_factor
    extracted.extend((recovered_real_values >= 0.0).astype(np.uint8).tolist())
    return extracted


def embed_watermark_elzein_mesh(
    vertices, triangles, watermark_bits, n_points=None, a=0.01, verbose=True, seed=42
):
    """他方式と揃えた名前で HARD 埋め込みを呼び出す。"""
    if n_points is not None and int(n_points) != len(watermark_bits):
        raise ValueError("n_points must equal the number of watermark bits.")
    return embed_watermark_elzein_hard_mesh(
        vertices, triangles, watermark_bits, a=a, verbose=verbose, seed=seed
    )


def extract_watermark_elzein_mesh(
    marked_vertices, original_vertices, triangles, key_info, verbose=False
):
    """他方式と揃えた名前で HARD 抽出を呼び出す。"""
    return extract_watermark_elzein_hard_mesh(
        marked_vertices, original_vertices, triangles, key_info, verbose=verbose
    )


__all__ = [
    "ElZeinHardKey",
    "split_watermark_5_to_3",
    "embed_watermark_elzein_hard_mesh",
    "extract_watermark_elzein_hard_mesh",
    "embed_watermark_elzein_mesh",
    "extract_watermark_elzein_mesh",
]
