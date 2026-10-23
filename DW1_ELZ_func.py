"""El Zein et al. の Portion 2 を基にした比較手法。

FCMのmoderateクラスタに属する全キャリアをビットごとのグループに
分割し、各座標へ一律に正負の変位を与える。抽出はグループ内多数決で
行い、キャリアインデックスは埋め込み時のキーから再利用する。
"""

from dataclasses import dataclass
import hashlib

import numpy as np
import skfuzzy as fuzz

from DW1_X1_func import align_attacked_vertices_to_original

@dataclass(frozen=True)
class ElZeinWatermarkKey:
    """埋め込み時のキャリア位置と抽出設定。"""

    mode: str
    watermark_length: int
    scaling_factor: float
    seed: int
    carrier_indices: tuple
    original_vertex_count: int
    topology_signature: str


def _validate_mesh(vertices, triangles):
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3).")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (M, 3).")
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("A non-empty triangle mesh is required.")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("triangles contain an invalid vertex index.")
    return vertices, triangles


def _validate_bits(watermark_bits):
    raw_bits = np.asarray(watermark_bits).reshape(-1)
    if len(raw_bits) == 0:
        raise ValueError("watermark_bits must contain at least one bit.")
    if np.any((raw_bits != 0) & (raw_bits != 1)):
        raise ValueError("watermark_bits must contain only 0 and 1.")
    return raw_bits.astype(np.uint8)


def _validate_strength(a):
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("a must be a positive finite scaling factor.")
    return float(a)


def _topology_signature(triangles):
    """Return an order-sensitive signature of triangle connectivity."""
    canonical = np.ascontiguousarray(triangles, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _mesh_topology(vertices, triangles):
    """頂点ごとの隣接頂点、接続面、単位面法線を返す。"""
    neighbors = [set() for _ in range(len(vertices))]
    incident_faces = [[] for _ in range(len(vertices))]
    face_vectors = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    lengths = np.linalg.norm(face_vectors, axis=1)
    valid_faces = lengths > 1e-12
    face_normals = np.zeros_like(face_vectors)
    face_normals[valid_faces] = (
        face_vectors[valid_faces] / lengths[valid_faces, None]
    )

    for face_index, (i, j, k) in enumerate(triangles):
        neighbors[i].update((j, k))
        neighbors[j].update((i, k))
        neighbors[k].update((i, j))
        if valid_faces[face_index]:
            incident_faces[i].append(face_index)
            incident_faces[j].append(face_index)
            incident_faces[k].append(face_index)
    return neighbors, incident_faces, face_normals


def local_feature_clustering(vertices, triangles, verbose=False, seed=42):
    """論文Sec. 3.1に従い、moderate形状の次数6頂点を返す。"""
    vertices, triangles = _validate_mesh(vertices, triangles)
    neighbors, incident_faces, face_normals = _mesh_topology(vertices, triangles)

    candidate_indices = []
    features = []
    for vertex_index, faces in enumerate(incident_faces):
        if len(neighbors[vertex_index]) != 6 or len(faces) != 6:
            continue
        normals = face_normals[faces]
        average_normal = normals.mean(axis=0)
        average_length = np.linalg.norm(average_normal)
        if average_length <= 1e-12:
            continue
        average_normal /= average_length
        angles = np.arccos(np.clip(normals @ average_normal, -1.0, 1.0))
        candidate_indices.append(vertex_index)
        features.append(np.sort(angles))

    if len(candidate_indices) < 3:
        raise ValueError(
            "FCM requires at least three valid degree-6 vertices; "
            f"found {len(candidate_indices)}."
        )

    feature_matrix = np.asarray(features, dtype=float)
    centers, memberships, *_ = fuzz.cluster.cmeans(
        feature_matrix.T, c=3, m=2.0, error=1e-5, maxiter=300, seed=seed
    )
    labels = np.argmax(memberships, axis=0)
    roughness = centers.mean(axis=1)
    ordered_clusters = np.argsort(roughness)
    moderate_cluster = ordered_clusters[1]
    selected = np.asarray(candidate_indices, dtype=np.int64)[
        labels == moderate_cluster
    ]
    selected.sort()

    if verbose:
        names = ("low", "moderate", "high")
        print("\n--- El Zein FCM clustering (degree-6 vertices) ---")
        for name, cluster_id in zip(names, ordered_clusters):
            count = int(np.count_nonzero(labels == cluster_id))
            print(
                f"{name:<8}: vertices={count:<6} "
                f"mean angle={roughness[cluster_id]:.6f} rad"
            )
        print("--------------------------------------------------")
    return selected


def _saved_carriers(key_info):
    if not isinstance(key_info, ElZeinWatermarkKey):
        raise TypeError("key_info must be returned by the El Zein embedder.")
    if key_info.mode != "elzein":
        raise ValueError(f"Invalid El Zein key mode: {key_info.mode!r}.")
    carriers = np.asarray(key_info.carrier_indices, dtype=np.int64)
    if len(carriers) < key_info.watermark_length:
        raise ValueError("The saved carrier list is shorter than the watermark.")
    return carriers


def embed_watermark_elzein_mesh(
    vertices,
    triangles,
    watermark_bits,
    a,
    verbose=True,
    seed=42,
    carrier_indices=None,
):
    """Portion 2を全moderate頂点へ冗長に埋め込む。"""
    vertices, triangles = _validate_mesh(vertices, triangles)
    bits = _validate_bits(watermark_bits)
    a = _validate_strength(a)

    if carrier_indices is None:
        carriers = local_feature_clustering(
            vertices, triangles, verbose=verbose, seed=seed
        )
    else:
        carriers = np.asarray(carrier_indices, dtype=np.int64).reshape(-1)
        if np.any(carriers < 0) or np.any(carriers >= len(vertices)):
            raise ValueError("carrier_indices contains an invalid vertex index.")
    if len(carriers) < len(bits):
        raise ValueError(
            f"Watermark needs {len(bits)} groups, but the moderate cluster "
            f"contains only {len(carriers)} vertices."
        )

    marked = vertices.copy()
    groups = np.array_split(carriers, len(bits))
    for bit, group in zip(bits, groups):
        marked[group] += a if bit else -a

    key = ElZeinWatermarkKey(
        mode="elzein",
        watermark_length=len(bits),
        scaling_factor=a,
        seed=int(seed),
        carrier_indices=tuple(int(index) for index in carriers),
        original_vertex_count=len(vertices),
        topology_signature=_topology_signature(triangles),
    )
    return marked, key


def extract_watermark_elzein_mesh(
    marked_vertices,
    original_vertices,
    triangles,
    key_info,
    verbose=False,
    index_distance_factor=20.0,
    coordinate_distance_factor=2.0,
    max_index_inconsistency_ratio=0.01,
):
    """同一頂点インデックスの座標差をグループ内多数決して抽出する。

    El Zein Method II の Portion 2 は透かし入りモデルと原モデルの
    対応頂点差を前提とする。汎用座標同期は行わず、頂点数または
    頂点順序・接続情報が変化した場合は全ビットを抽出不能として返す。
    """
    original_vertices = np.asarray(original_vertices, dtype=float)
    if original_vertices.ndim != 2 or original_vertices.shape[1] != 3:
        raise ValueError("original_vertices must have shape (N, 3).")
    if not np.isfinite(original_vertices).all():
        raise ValueError("original_vertices contains NaN or infinity.")
    marked_vertices, triangles = _validate_mesh(marked_vertices, triangles)
    carriers = _saved_carriers(key_info)
    if len(original_vertices) != key_info.original_vertex_count:
        raise ValueError(
            "original_vertices does not match the mesh used for embedding."
        )
    aligned_marked_vertices, matched_vertex_mask = (
        align_attacked_vertices_to_original(
            marked_vertices,
            original_vertices,
            index_distance_factor=index_distance_factor,
            coordinate_distance_factor=coordinate_distance_factor,
            max_index_inconsistency_ratio=max_index_inconsistency_ratio,
            verbose=verbose,
        )
    )

    groups = np.array_split(carriers, key_info.watermark_length)
    extracted = []
    for group in groups:
        # Match the proposed method's complete-cluster rule: a bit group is
        # unusable when even one of its original carriers has no valid match.
        if not np.all(matched_vertex_mask[group]):
            extracted.append(-1)
            continue
        coordinate_sums = (
            aligned_marked_vertices[group] - original_vertices[group]
        ).sum(axis=1)
        ones = int(np.count_nonzero(coordinate_sums > 0))
        zeros = len(group) - ones
        extracted.append(-1 if ones == zeros else int(ones > zeros))
    return extracted


__all__ = [
    "ElZeinWatermarkKey",
    "local_feature_clustering",
    "embed_watermark_elzein_mesh",
    "extract_watermark_elzein_mesh",
]
