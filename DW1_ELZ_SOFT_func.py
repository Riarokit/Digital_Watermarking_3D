"""El Zein Method II の Portion 2 を二値・冗長化した従来 SOFT 実装。

moderate クラスタをビット数個のグループへ分割し、各グループへ同じ
ビットを反復埋め込みして、抽出時に多数決する。頂点数が変わった場合は
最近傍による点群同期も行う。これらは原論文に明記されていない拡張で、
過去の実験結果を再現するために残す。
"""

import numpy as np
import skfuzzy as fuzz

import DW2_func as DW2F


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


def _mesh_topology(vertices, triangles):
    """頂点ごとの隣接頂点と接続面、および単位面法線を返す。"""
    neighbors = [set() for _ in range(len(vertices))]
    incident_faces = [[] for _ in range(len(vertices))]
    face_vectors = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    lengths = np.linalg.norm(face_vectors, axis=1)
    valid_faces = lengths > 1e-12
    face_normals = np.zeros_like(face_vectors)
    face_normals[valid_faces] = face_vectors[valid_faces] / lengths[valid_faces, None]

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
    """論文 Sec. 3.1 に従い、中程度の局所形状を持つ次数6頂点を返す。"""
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
    selected = np.asarray(candidate_indices, dtype=np.int64)[labels == moderate_cluster]
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


def embed_watermark_elzein_mesh(
    vertices, triangles, watermark_bits, n_points, a, verbose=True
):
    """Portion 2を二値化し、頂点グループへ冗長に埋め込む。"""
    vertices, triangles = _validate_mesh(vertices, triangles)
    bits = np.asarray(watermark_bits, dtype=np.uint8).reshape(-1)
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("watermark_bits must contain only 0 and 1.")
    if n_points != len(bits):
        raise ValueError("n_points must equal the number of watermark bits.")

    selected = local_feature_clustering(vertices, triangles, verbose=verbose)
    if len(selected) < n_points:
        raise ValueError(
            f"Watermark needs {n_points} groups, but the moderate cluster "
            f"contains only {len(selected)}."
        )
    marked = vertices.copy()
    groups = np.array_split(selected, n_points)
    for bit, group in zip(bits, groups):
        displacement = a if bit else -a
        marked[group] += displacement
    return marked


def extract_watermark_elzein_mesh(
    marked_vertices, original_vertices, triangles, n_points, verbose=False
):
    """座標差を各頂点で判定し、グループ内多数決で抽出する。"""
    original_vertices, triangles = _validate_mesh(original_vertices, triangles)
    marked_vertices = np.asarray(marked_vertices, dtype=float)
    if marked_vertices.shape != original_vertices.shape:
        marked_vertices = DW2F.synchronize_point_cloud(
            marked_vertices, original_vertices, verbose=verbose
        )
    selected = local_feature_clustering(original_vertices, triangles, verbose=verbose)
    if len(selected) < n_points:
        raise ValueError("Not enough moderate degree-6 vertices during extraction.")
    groups = np.array_split(selected, n_points)
    extracted = []
    for group in groups:
        coordinate_sums = (
            marked_vertices[group] - original_vertices[group]
        ).sum(axis=1)
        extracted.append(
            int(np.count_nonzero(coordinate_sums > 0) > len(group) / 2)
        )
    return extracted


def embed_watermark_elzein(
    vertices, triangles, watermark_bits, n_points, a, k=6, verbose=True
):
    """旧関数名との互換ラッパー。"""
    if k != 6:
        raise ValueError("The paper defines feature vectors only for degree 6.")
    return embed_watermark_elzein_mesh(
        vertices, triangles, watermark_bits, n_points, a, verbose
    )


def extract_watermark_elzein(
    marked_vertices, original_vertices, triangles, n_points, k=6, verbose=False
):
    """旧関数名との互換ラッパー。"""
    if k != 6:
        raise ValueError("The paper defines feature vectors only for degree 6.")
    return extract_watermark_elzein_mesh(
        marked_vertices, original_vertices, triangles, n_points, verbose
    )


__all__ = [
    "local_feature_clustering",
    "embed_watermark_elzein_mesh",
    "extract_watermark_elzein_mesh",
    "embed_watermark_elzein",
    "extract_watermark_elzein",
]
