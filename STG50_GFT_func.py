import numpy as np
import open3d as o3d
import random
import string
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
import scipy.sparse as sp
import zlib
from PIL import Image
import copy
# import cupy as cp

# =========================================================
#  前処理関数群
# =========================================================

def generate_random_string(length):
    """
    指定した長さのランダムな英数字文字列を生成する関数。

    Parameters:
    length (int): 生成する文字列の長さ

    Returns:
    str: 生成されたランダムな英数字文字列
    """
    random.seed(42)
    if length <= 0:
        raise ValueError("文字列の長さは正の整数で指定してください。")

    characters = string.ascii_letters + string.digits  # 英字（大小）と数字を含む
    random_string = ''.join(random.choices(characters, k=length))  # ランダムな文字列を生成
    return random_string

def string_to_binary(input_string):
    """
    文字列をUTF-8バイナリビット列（int型リスト）に変換する関数。
    """
    return [int(bit) for byte in input_string.encode('utf-8') for bit in format(byte, '08b')]

def binary_to_string(extracted_binary_list):
    if isinstance(extracted_binary_list, list):
        extracted_binary_string = ''.join(str(b) for b in extracted_binary_list)
    else:
        extracted_binary_string = extracted_binary_list
    chars = []
    for i in range(0, len(extracted_binary_string), 8):
        byte = extracted_binary_string[i:i+8]
        if len(byte) < 8:
            break
        try:
            chars.append(chr(int(byte, 2)))
        except:
            chars.append('?')  # デコード不可な場合は?にする
    return ''.join(chars)

def image_to_bitarray(image_path, n=32):
    """
    画像ファイルをn×nの2値ビット配列に変換
    """
    img = Image.open(image_path).convert('L')  # グレースケール
    img = img.resize((n, n), Image.LANCZOS)
    arr = np.array(img)
    # しきい値で2値化
    arr = (arr > 127).astype(np.uint8)
    return arr.flatten().tolist()  # 1次元ビットリスト


def bitarray_to_image(bitarray, n=32, save_path=None):
    """
    1次元ビット配列をn×n画像に復元
    """
    arr = np.array(bitarray, dtype=np.uint8).reshape((n, n)) * 255
    img = Image.fromarray(arr, mode='L')
    if save_path:
        img.save(save_path)
    return img

def add_colors(pcd_before, color="grad"):
    """
    色情報を追加する関数。

    Parameters:
    pcd_before (pcd): 埋め込み前点群
    color (str): "grad" = グラデーション、"black" = 全部黒(視認用)

    Returns:
    pcd_before (pcd): 色情報がついた埋め込み前点群
    """
    if color == "grad":
        # 分岐OP. 点群に色情報を追加
        points = np.asarray(pcd_before.points)  # 点の座標を取得
        x_values = points[:, 0]  # x軸に基づいて色を生成
        y_values = points[:, 1]  # y軸に基づいて色を生成
        z_values = points[:, 2]  # z軸に基づいて色を生成
        x_min, x_max = x_values.min(), x_values.max()  # x軸の最小値と最大値を取得
        y_min, y_max = y_values.min(), y_values.max()  # y軸の最小値と最大値を取得
        z_min, z_max = z_values.min(), z_values.max()  # z軸の最小値と最大値を取得
        colors = np.zeros_like(points)
        colors[:, 0] = (x_values - x_min) / (x_max - x_min)  # 赤色のグラデーション
        colors[:, 1] = (y_values - y_min) / (y_max - y_min)  # 緑色のグラデーション
        colors[:, 2] = (z_values - z_min) / (z_max - z_min)  # 青色のグラデーション

    if color == "black":
        # 分岐OP. 全ての色を黒に設定 (視認用)
        points = np.asarray(pcd_before.points)  # 点の座標を取得
        colors = np.zeros_like(points)  #全点を黒にする
    
    pcd_before.colors = o3d.utility.Vector3dVector(colors)
    return pcd_before

def normalize_point_cloud(pcd):
    points = np.asarray(pcd.points)
    max_bound_before = pcd.get_max_bound()
    min_bound_before = pcd.get_min_bound()
    np.set_printoptions(precision=4, suppress=True)
    print(f"[Normalize] Before: {min_bound_before} ~ {max_bound_before}")

    # 1. 重心を原点へ移動 (Translation)
    centroid = np.mean(points, axis=0)
    points = points - centroid

    # 2. 原点からの最大距離でスケーリング (Uniform Scaling)
    # 各点の原点からの距離を計算
    distances = np.linalg.norm(points, axis=1)
    max_distance = np.max(distances)

    # 最大距離が0（点が1つしかない等）でなければ割る
    if max_distance > 0:
        points = points / max_distance

    # 結果を書き戻す
    pcd.points = o3d.utility.Vector3dVector(points)
    
    max_bound_after = pcd.get_max_bound()
    min_bound_after = pcd.get_min_bound()
    print(f"[Normalize]  After: {min_bound_after} ~ {max_bound_after}")
    
    return pcd

def estimate_normals_xyz(xyz, knn=30, orient_knn=30, make_outward=True):
    """
    xyz: (N,3)
    return normals: (N,3) unit vectors
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    # 近傍で向きを揃える（符号反転がバラけると埋め込み/抽出が不安定になる）
    pcd.orient_normals_consistent_tangent_plane(orient_knn)

    normals = np.asarray(pcd.normals)

    if make_outward:
        # “外向きっぽく”揃える簡易策：重心からのベクトルと内積が負なら反転
        centroid = np.mean(xyz, axis=0)
        v = xyz - centroid
        flip = np.sum(normals * v, axis=1) < 0
        normals[flip] *= -1.0

    # 念のため正規化
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    return normals

# =========================================================
#  クラスタリング関数群
# =========================================================

def kmeans_cluster_points(xyz, cluster_point=1000, seed=42):
    num_clusters = len(xyz) // cluster_point
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
    labels = kmeans.fit_predict(xyz)
    
    # 各クラスタごとの点数を集計
    unique, counts = np.unique(labels, return_counts=True)
    min_count = np.min(counts)

    print(f"[KMeans] 検出されたクラスタ数: {len(unique)}")
    print(f"[KMeans] 最も点数が少ないクラスタの点数: {min_count}")
    
    return labels

def region_growing_cluster_points(
    xyz,
    distance_thresh=None,
    angle_thresh_deg=4.0,
    min_cluster_size=500,
    knn_normal=15,
    knn_region=15
):
    """
    距離制限 + 法線類似度によるRegion Growingクラスタリング（クラス風の明示的実装）

    Parameters:
    - xyz: Nx3座標（np.ndarray）
    - distance_thresh: 空間距離の最大閾値 [m]
    - angle_thresh_deg: 法線の最大角度差 [deg] (Bunny,Armadillo: 4.0)
    - min_cluster_size: クラスタ最小点数
    - knn_normal: 法線推定用の近傍点数
    - knn_region: クラスタ拡張用の近傍点数

    Returns:
    - labels: 各点のクラスタ番号（np.ndarray, shape[N]）
    """
    # 自動パラメータ設定
    scale = np.linalg.norm(np.max(xyz, axis=0) - np.min(xyz, axis=0))
    if distance_thresh is None:
        distance_thresh = scale * 0.01  # 全体のx%
    print(f"[RegionGrowing] 距離閾値: {distance_thresh:.5f}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_normal))
    pcd.orient_normals_consistent_tangent_plane(knn_normal)
    normals = np.asarray(pcd.normals)

    N = len(xyz)
    visited = np.zeros(N, dtype=bool)
    labels = np.full(N, -1, dtype=int)
    label = 0
    cos_thresh = np.cos(np.deg2rad(angle_thresh_deg))

    # 空間近傍検索用
    neighbors = NearestNeighbors(n_neighbors=knn_region).fit(xyz)

    for i in range(N):
        if visited[i]:
            continue

        seed_queue = [i]
        visited[i] = True
        cluster = [i]

        while seed_queue:
            idx = seed_queue.pop()
            nbrs = neighbors.kneighbors([xyz[idx]], return_distance=False)[0]

            for n in nbrs:
                if visited[n]:
                    continue
                dot = np.dot(normals[idx], normals[n])
                dist = np.linalg.norm(xyz[idx] - xyz[n])
                if dot >= cos_thresh and dist <= distance_thresh:
                    visited[n] = True
                    seed_queue.append(n)
                    cluster.append(n)

        if len(cluster) >= min_cluster_size:
            labels[cluster] = label
            label += 1

    # 集計して最小点数表示
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique) > 0:
        min_count = np.min(counts)
        print(f"[RegionGrowing] 検出されたクラスタ数: {label}")
        print(f"[RegionGrowing] 最も点数が少ないクラスタの点数: {min_count}")
    else:
        print("[RegionGrowing] 有効なクラスタがありません")
    return labels

def ransac_cluster_points(xyz, distance_threshold=None, min_cluster_size=500, max_planes=1000):
    """
    RANSACによる平面クラスタ抽出（繰り返し）

    Parameters:
    distance_threshold: 平面からの最大距離。この距離以内の点を「平面に乗っている」とみなす(Bunny: 0.01, Armadillo: 0.001)
    min_cluster_size: クラスタとして採用するための最小点数
    max_planes: 検出する平面クラスタの最大数（ループ上限）

    Returns:
    - labels: 各点のクラスタ番号（未分類は -1）
    """
    # 自動スケール対応
    if distance_threshold is None:
        scale = np.linalg.norm(np.max(xyz, axis=0) - np.min(xyz, axis=0))
        distance_threshold = scale * 0.01  # 全体スケールのx%
        print(f"[RANSAC] 自動設定: distance_threshold = {distance_threshold:.6f}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    labels = np.full(len(xyz), -1)
    current_label = 0

    remaining_pcd = pcd
    remaining_indices = np.arange(len(xyz))  # xyz中のインデックスを追跡

    while current_label < max_planes and len(remaining_pcd.points) > 0:
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        if len(inliers) < min_cluster_size:
            break

        # 元のxyz上のインデックスに変換してlabel付け
        global_inliers = remaining_indices[inliers]
        labels[global_inliers] = current_label

        # 残りの点群とインデックスを更新
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        remaining_indices = np.delete(remaining_indices, inliers)
        current_label += 1

    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique) > 0:
        min_count = np.min(counts)
        print(f"[RANSAC] 検出されたクラスタ数: {current_label}")
        print(f"[RANSAC] 最も点数が少ないクラスタの点数: {min_count}")
    else:
        print("[RANSAC] クラスタが検出されませんでした")
    return labels

def split_large_clusters(xyz, labels, limit_points=3000, seed=42):
    """
    指定したクラスタラベル群をもとに、点数が limit_points を超えるクラスタを再クラスタリングし、
    labels を更新して返す。

    Parameters:
    - xyz: Nx3 点群座標
    - labels: 長さNのクラスタ番号配列（int）
    - limit_points: 1クラスタの最大点数上限
    - seed: 再クラスタリング時のKMeansシード

    Returns:
    - new_labels: 長さNの更新後クラスタ番号配列
    """
    new_labels = labels.copy()
    current_max_label = labels.max()
    label_offset = current_max_label + 1

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) <= limit_points:
            continue

        pts = xyz[idx]
        n_subclusters = int(np.ceil(len(idx) / limit_points))
        kmeans = KMeans(n_clusters=n_subclusters, random_state=seed)
        sub_labels = kmeans.fit_predict(pts)

        # ラベルを更新
        for sub in range(n_subclusters):
            sub_idx = idx[sub_labels == sub]
            new_labels[sub_idx] = label_offset
            label_offset += 1

        print(f"[Split] クラスタ {c}（{len(idx)}点）を {n_subclusters} 分割 → 新ラベル {label_offset - n_subclusters}〜{label_offset - 1}")

    return new_labels

def visualize_clusters(xyz, labels):
    """
    xyz: Nx3 numpy配列（点群座標）
    labels: クラスタラベル配列（長さN, int型）
    return: 各クラスタに割り当てたRGBカラー（リスト, shape=[n_clusters,3])
    """
    COLORS = np.array([
        [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1],
        [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5], [0.5,0.2,0.7], [0.7,0.7,0.7]
    ])
    n_clusters = np.max(labels) + 1
    if n_clusters > len(COLORS):
        rng = np.random.RandomState(42)
        COLORS = rng.rand(n_clusters, 3)
    cluster_colors = COLORS[:n_clusters]

    # ラベルごとに色を割り当て
    color_array = cluster_colors[labels]

    # 可視化用の一時点群
    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(xyz)
    tmp_pcd.colors = o3d.utility.Vector3dVector(color_array)
    o3d.visualization.draw_geometries([tmp_pcd])

    # 各クラスタ番号に対する色のリストを返す
    color_list = [list(cluster_colors[i]) for i in range(n_clusters)]
    return color_list

def estimate_cluster_flatness(xyz, labels, k_neighbors=20):
    """
    各クラスタごとに「平面らしさ（平坦度）」指標を算出（平均曲率を使用）
    - xyz: Nx3配列
    - labels: クラスタラベル（N要素、int）
    - k_neighbors: 局所曲率推定の近傍点数
    - return: dict{クラスタ番号: 平坦度指標値}
    """
    cluster_flatness = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        pts = xyz[idx]
        # 法線＋曲率推定
        nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(pts)-1)).fit(pts)
        _, indices = nn.kneighbors(pts)
        curvatures = []
        for i, neighbors in enumerate(indices):
            p = pts[neighbors]
            cov = np.cov(p.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)
            # 曲率：最小固有値/総和（標準的な点群局所曲率定義）
            curvature = eigvals[0] / np.sum(eigvals) if np.sum(eigvals)>0 else 0
            curvatures.append(curvature)
        cluster_flatness[c] = np.mean(curvatures)
    return cluster_flatness

def compute_cluster_weights(flatness_dict, flatness_weighting=0):
    """
    flatness_weighting:
      0 → 重みなし（全クラスタ重み1.0）
      1 → 平坦クラスタほど重み大（傾斜）
      2 → 曲面クラスタほど重み大（傾斜）
      3 → 平坦クラスタ上位半分を重み最大化、下位半分を重み最小化
      4 → 曲面クラスタ上位半分を重み最大化、下位半分を重み最小化
    """
    if flatness_weighting == 0:
        # 重みなし
        return {c: 1.0 for c in flatness_dict.keys()}

    vals = np.array(list(flatness_dict.values()))
    clusters = list(flatness_dict.keys())
    min_f, max_f = np.min(vals), np.max(vals)
    
    if max_f == min_f:
        # 曲率全て同じ時は全クラスタ重み1.0
        return {c: 1.0 for c in flatness_dict.keys()}
    
    if flatness_weighting == 1:
        # 平坦度が大きいほど重み大
        scaled = (max_f - vals) / (max_f - min_f)
        scaled *= (1.0 / np.mean(scaled)) # 平均1に調整
        weights = {c: w for c, w in zip(clusters, scaled)}
    elif flatness_weighting == 2:
        # 平坦度が小さいほど重み大
        scaled = (vals - min_f) / (max_f - min_f)
        scaled *= (1.0 / np.mean(scaled)) # 平均1に調整
        weights = {c: w for c, w in zip(clusters, scaled)}
    elif flatness_weighting == 3:
        # 平坦クラスタ上位半分をmax_weight、下位半分をmin_weight
        sorted_indices = np.argsort(vals)  # 昇順ソート
        n_clusters = len(clusters)
        half = n_clusters // 2
        weights = {}
        for i, idx in enumerate(sorted_indices):
            cluster_id = clusters[idx]
            if i < half:
                weights[cluster_id] = 2  # 上位半分
            else:
                weights[cluster_id] = 0  # 下位半分
    elif flatness_weighting == 4:
        # 曲面クラスタ上位半分をmax_weight、下位半分をmin_weight
        sorted_indices = np.argsort(vals)[::-1]  # 降順ソート
        n_clusters = len(clusters)
        half = n_clusters // 2
        weights = {}
        for i, idx in enumerate(sorted_indices):
            cluster_id = clusters[idx]
            if i < half:
                weights[cluster_id] = 2  # 上位半分
            else:
                weights[cluster_id] = 0  # 下位半分
    else:
        raise ValueError("flatness_weightingは0（重みなし）, 1（平坦優遇傾斜）, 2（曲面優遇傾斜）, 3（平坦二分）, 4（曲面二分）で指定")
    
    return weights

def compute_pseudoplane_pca(pts: np.ndarray):
    """
    クラスタ点群 pts (Nc,3) から疑似平面をPCAで推定する。
    平面: 重心 centroid を通る
    法線: 共分散の最小固有値に対応する固有ベクトル（=最小分散方向）
    さらに、法線の符号は決定規則で固定して再現性を担保する。

    Returns:
        centroid: (3,)
        normal: (3,) unit vector (sign-fixed)
    """
    centroid = np.mean(pts, axis=0)
    X = pts - centroid

    # 共分散（対称行列）
    C = (X.T @ X) / max(len(pts), 1)

    # 対称行列の固有分解（eighは昇順で返る）
    eigvals, eigvecs = np.linalg.eigh(C)

    # 最小固有値の固有ベクトル = 平面法線
    normal = eigvecs[:, 0]

    # 正規化
    nrm = np.linalg.norm(normal)
    if nrm < 1e-12:
        # 退化（全点ほぼ同一点など）: 適当な法線
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        normal = normal / nrm

    # 法線符号の決定（再現性のための規則）
    # |成分|が最大の軸の成分が + になるようにする
    j = int(np.argmax(np.abs(normal)))
    if normal[j] < 0:
        normal = -normal

    return centroid, normal

def cluster_height_signal(pts: np.ndarray, centroid: np.ndarray, normal: np.ndarray):
    """
    平面（centroid, normal）に対する各点の符号付き高さ（距離）:
        h_i = (p_i - centroid) dot normal
    Returns:
        h: (Nc,)
    """
    return np.sum((pts - centroid) * normal[None, :], axis=1)

def make_pseudoplane_lineset(
    centroid: np.ndarray,
    normal: np.ndarray,
    pts_cluster: np.ndarray,
    scale: float = 1.2
):
    """
    疑似平面（centroid, normal）を可視化するためのLineSet（四角い枠）を作る
    """
    c = centroid.astype(float)
    n = normal.astype(float)
    n = n / max(np.linalg.norm(n), 1e-12)

    # 平面内の直交基底 u, v
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])

    u = np.cross(n, a)
    u = u / max(np.linalg.norm(u), 1e-12)
    v = np.cross(n, u)
    v = v / max(np.linalg.norm(v), 1e-12)

    # クラスタ点群を平面内へ射影してサイズ決定
    X = pts_cluster - c[None, :]
    su = X @ u
    sv = X @ v

    half_u = 0.5 * (np.max(su) - np.min(su))
    half_v = 0.5 * (np.max(sv) - np.min(sv))
    half_u = max(half_u, 1e-6) * scale
    half_v = max(half_v, 1e-6) * scale

    p0 = c + (-half_u) * u + (-half_v) * v
    p1 = c + ( half_u) * u + (-half_v) * v
    p2 = c + ( half_u) * u + ( half_v) * v
    p3 = c + (-half_u) * u + ( half_v) * v

    points = np.vstack([p0, p1, p2, p3])
    lines = np.array([[0,1],[1,2],[2,3],[3,0]], dtype=np.int32)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)

    return ls

def visualize_clusters_with_pseudoplanes(
    xyz: np.ndarray,
    labels: np.ndarray,
    color_list,
    scale: float = 1.2,
    max_planes: int = None,
    show_frame: bool = False,
    window_name: str = "Clusters + Pseudo-planes"
):
    """
    visualize_clusters() の返り値 color_list（クラスタ番号→RGB）に同期して、
    各クラスタの疑似平面（PCA）を同じ色で可視化する。

    - visualize_clusters 自体は変更しない
    - この関数は「点群（クラスタ色） + 疑似平面（同色）」を1つのウィンドウで表示する

    Parameters
    ----------
    xyz : (N,3) np.ndarray
    labels : (N,) np.ndarray (クラスタID)
    color_list : list[list[float]]  # visualize_clusters の返り値
    scale : float  # 平面枠の大きさ係数
    max_planes : int or None  # 表示する平面数上限（重いとき用）
    show_frame : bool  # 座標軸フレーム表示
    window_name : str
    """
    # --- 点群（クラスタ色）を作る ---
    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(xyz)

    labels = np.asarray(labels)
    n_clusters = len(color_list)

    # labels に -1 が混じるケースにも一応対応（-1はグレー）
    color_array = np.zeros((len(labels), 3), dtype=float)
    gray = np.array([0.6, 0.6, 0.6], dtype=float)

    for i, lb in enumerate(labels):
        if lb < 0 or lb >= n_clusters:
            color_array[i] = gray
        else:
            color_array[i] = np.array(color_list[lb], dtype=float)

    tmp_pcd.colors = o3d.utility.Vector3dVector(color_array)

    # --- 疑似平面（LineSet）をクラスタごとに生成して同色をつける ---
    plane_geoms = []
    unique_clusters = np.unique(labels[labels >= 0])  # -1は除外

    count = 0
    for c in unique_clusters:
        if max_planes is not None and count >= max_planes:
            break

        idx = np.where(labels == c)[0]
        if len(idx) < 3:
            continue

        pts = xyz[idx]

        centroid, normal = compute_pseudoplane_pca(pts)
        plane_ls = make_pseudoplane_lineset(centroid, normal, pts, scale=scale)

        # LineSetの colors は「線の本数」と同じ長さが推奨（4本）
        col = np.array(color_list[int(c)], dtype=float)
        n_lines = len(plane_ls.lines)
        plane_ls.colors = o3d.utility.Vector3dVector(np.tile(col[None, :], (n_lines, 1)))

        plane_geoms.append(plane_ls)
        count += 1

    geoms = [tmp_pcd] + plane_geoms

    if show_frame:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]))

    o3d.visualization.draw_geometries(geoms, window_name=window_name)
    return plane_geoms

# =========================================================
#  グラフ構築・埋め込み・抽出関数群
# =========================================================

def build_graph(xyz, graph_mode='knn', k=12, radius=0.05):
    """
    点群からグラフ（重み付き隣接行列）を構築する関数

    Parameters:
    - xyz: 点群座標 (N, 3)
    - mode: グラフ構築モード
        - 'knn': k近傍法 (推奨: k=12~16)
        - 'radius': 半径法 (radius以内の点を接続)
        - 'hybrid': k近傍かつ、radius以内の点のみ接続 (遠すぎる接続を排除)
    - k: k近傍法のk
    - radius: 半径法の閾値 (データのスケールに依存するため注意)

    Returns:
    - W: 重み付き隣接行列 (scipy.sparse.csr_matrix or dense ndarray)
    """
    
    # 1. 隣接関係の構築 (adjacency matrix)
    if graph_mode == 'knn':
        # include_self=Falseで自分自身へのループを排除
        adj = kneighbors_graph(xyz, k, mode='distance', include_self=False)
        
    elif graph_mode == 'radius':
        adj = radius_neighbors_graph(xyz, radius, mode='distance', include_self=False)
        
    elif graph_mode == 'hybrid':
        # まずk近傍で大きめに取る
        adj = kneighbors_graph(xyz, k, mode='distance', include_self=False)
        # 疎行列の構造を保ったまま、radiusより大きい距離のエッジを削除（0にする）
        adj.data[adj.data > radius] = 0
        adj.eliminate_zeros() # 0になったエッジを構造から削除
        
    else:
        raise ValueError("mode must be 'knn', 'radius', or 'hybrid'")

    # 2. 重みの計算 (Gaussian Kernel)
    # 行列形式を扱いやすい形に変換
    W = adj.toarray() 
    
    # 距離が存在するエッジのみ抽出
    mask = W > 0
    dists = W[mask]
    
    # 孤立点対策: エッジが一つもない場合はsigmaを1.0にしてエラー回避
    if len(dists) > 0:
        sigma = np.mean(dists)
        # ガウス重み: 近いほど1、遠いほど0に近づく
        W[mask] = np.exp(-W[mask]**2 / (sigma**2))
    else:
        # 万が一エッジがゼロの場合（radiusが小さすぎる時など）
        print("Warning: No edges found. Check your radius or k.")
    
    # 3. 対称化 (無向グラフにするため)
    # GFTでは通常、無向グラフ（対称行列）が望ましいため、W = (W + W.T) / 2 などをする場合が多いですが
    # ここでは元の実装に合わせてそのまま返します（あるいは以下のように最大値を取って対称化も可）
    W = np.maximum(W, W.T) 

    return W

def gft_basis(W):
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvecs, eigvals

# def gft_basis_gpu(W):
#     # 入力Wはnumpy配列（CPU）、ここでGPUに転送
#     W_gpu = cp.asarray(W)
#     D_gpu = cp.diag(W_gpu.sum(axis=1))
#     L_gpu = D_gpu - W_gpu
#     eigvals_gpu, eigvecs_gpu = cp.linalg.eigh(L_gpu)
#     # 必要ならCPU（numpy配列）に戻す
#     eigvals = cp.asnumpy(eigvals_gpu)
#     eigvecs = cp.asnumpy(eigvecs_gpu)
#     return eigvecs, eigvals

def gft(signal, basis):
    return basis.T @ signal

def igft(gft_coeffs, basis):
    return basis @ gft_coeffs

def repeat_bits_blockwise(bits, n_repeat, total_length):
    """
    bits: 1次元配列, n_repeat: 各ビットごとの繰り返し数, total_length: 必要な全体長
    return: [w1,w1,...,w1,w2,w2,...,w2,...]の合計total_lengthの配列
    """
    rep = np.repeat(bits, n_repeat)
    # 足りなければ余りのビットを追加
    if len(rep) < total_length:
        extra = bits[:(total_length - len(rep))]
        rep = np.concatenate([rep, extra])
    elif len(rep) > total_length:
        rep = rep[:total_length]
    return rep


### 関数間違い注意！！
def embed_watermark_xyz(
    xyz, labels, embed_bits, beta=0.01,
    graph_mode='knn', k=10, radius=0.05,
    split_mode=0, flatness_weighting=0, k_neighbors=20, 
    min_spectre=0.0, max_spectre=1.0
):
    """
    各クラスタでembed_bitsを
      - split_mode=0: 3チャネル全てに同じ情報（冗長化）
      - split_mode=1: 3分割してx/y/zにそれぞれ独立情報
    としてGFT係数のmin_spectre - max_spectre分の周波数帯域にだけ埋め込む
    """
    xyz_after = xyz.copy()
    cluster_ids = np.unique(labels)
    # color_list = visualize_clusters(xyz, labels)
    flatness_dict = estimate_cluster_flatness(xyz, labels, k_neighbors=k_neighbors)
    weights = compute_cluster_weights(flatness_dict, flatness_weighting)
    phi = max(
        np.max(xyz[:, 0]) - np.min(xyz[:, 0]),
        np.max(xyz[:, 1]) - np.min(xyz[:, 1]),
        np.max(xyz[:, 2]) - np.min(xyz[:, 2])
    )

    # --- ビット列の用意 ---
    if split_mode == 0:
        # 全チャネル同じ情報
        embed_bits_per_channel = [embed_bits] * 3
        skip_threshold = len(embed_bits)/2
    elif split_mode == 1:
        # 3チャネルで異なる情報（3分割）
        embed_bits_per_channel = np.array_split(embed_bits, 3)
        skip_threshold = len(embed_bits)/5
    else:
        raise ValueError("split_modeは0（冗長化）か1（3分割）のみ指定可能です")

    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        if len(idx) <= skip_threshold:
                continue  # 点数が少なすぎるクラスタはスキップ
        pts = xyz[idx]
        W = build_graph(pts, graph_mode=graph_mode, k=k, radius=radius)
        basis, eigvals = gft_basis(W)
        for channel in range(3):  # 0:x, 1:y, 2:z
            bits = embed_bits_per_channel[channel]
            bits_len = len(bits)
            if bits_len == 0:
                continue  # このチャネルには何も埋め込まない
            signal = pts[:, channel]
            gft_coeffs = gft(signal, basis)
            Q_ = len(gft_coeffs)
            q_start = int(Q_ * min_spectre)
            q_end   = int(Q_ * max_spectre)
            Q_embed = q_end - q_start
            n_repeat = Q_embed // bits_len if bits_len > 0 else 1
            redundant_bits = repeat_bits_blockwise(bits, n_repeat, Q_embed)
            for i in range(Q_embed):
                w = redundant_bits[i] * 2 - 1
                gft_coeffs[q_start + i] += w * beta * weights[c] * phi
            embed_signal = igft(gft_coeffs, basis)
            xyz_after[idx, channel] = embed_signal
    return xyz_after

def extract_watermark_xyz(
    xyz_emb, xyz_orig, labels, embed_bits_length, 
    graph_mode='knn', k=10, radius=0.05,
    split_mode=0, min_spectre=0.0, max_spectre=1.0
):
    if split_mode == 0:
        skip_threshold = embed_bits_length/2
        bit_lists = [[] for _ in range(embed_bits_length)]
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) <= skip_threshold:
                continue  # 点数が少なすぎるクラスタはスキップ

            pts_emb = xyz_emb[idx]
            pts_orig = xyz_orig[idx]
            if len(pts_emb) != len(pts_orig):
                continue  # 点数不一致クラスタはスキップ

            W = build_graph(pts_orig, graph_mode=graph_mode, k=k, radius=radius)
            basis, eigvals = gft_basis(W)
            Q_ = len(basis)
            q_start = int(Q_ * min_spectre)
            q_end   = int(Q_ * max_spectre)
            Q_extract = q_end - q_start
            n_repeat = Q_extract // embed_bits_length if embed_bits_length > 0 else 1

            for channel in range(3):
                gft_coeffs_emb = gft(pts_emb[:, channel], basis)
                gft_coeffs_orig = gft(pts_orig[:, channel], basis)
                for bit_idx in range(embed_bits_length):
                    for rep in range(n_repeat):
                        i = bit_idx * n_repeat + rep
                        if i < Q_extract:
                            diff = gft_coeffs_emb[q_start + i] - gft_coeffs_orig[q_start + i]
                            bit = 1 if diff > 0 else 0
                            bit_lists[bit_idx].append(bit)
                for i in range(n_repeat * embed_bits_length, Q_extract):
                    bit_idx = i % embed_bits_length
                    diff = gft_coeffs_emb[q_start + i] - gft_coeffs_orig[q_start + i]
                    bit = 1 if diff > 0 else 0
                    bit_lists[bit_idx].append(bit)

        extracted_bits = []
        for bits in bit_lists:
            counts = {0: bits.count(0), 1: bits.count(1)}
            extracted_bit = 1 if counts[1] > counts[0] else 0
            extracted_bits.append(extracted_bit)
        return extracted_bits

    elif split_mode == 1:
        skip_threshold = embed_bits_length / 5
        split_sizes = [len(arr) for arr in np.array_split(np.zeros(embed_bits_length), 3)]

        # チャネルごとのbit_listsを初期化（クラスタループの外に1つだけ作っておく）
        channel_bit_lists = [
            [[] for _ in range(split_sizes[0])],  # x
            [[] for _ in range(split_sizes[1])],  # y
            [[] for _ in range(split_sizes[2])]   # z
        ]

        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) <= skip_threshold:
                continue

            pts_emb = xyz_emb[idx]
            pts_orig = xyz_orig[idx]
            if len(pts_emb) != len(pts_orig):
                continue  # 点数不一致クラスタはスキップ

            actual_k = min(6, len(idx) - 1)
            W = build_graph(pts_orig, k=actual_k)
            basis, eigvals = gft_basis(W)
            Q_ = len(basis)
            q_start = int(Q_ * min_spectre)
            q_end = int(Q_ * max_spectre)
            Q_extract = q_end - q_start

            for channel in range(3):  # 0:x, 1:y, 2:z
                bits_len = split_sizes[channel]
                if bits_len == 0:
                    continue
                n_repeat = Q_extract // bits_len if bits_len > 0 else 1

                gft_coeffs_emb = gft(pts_emb[:, channel], basis)
                gft_coeffs_orig = gft(pts_orig[:, channel], basis)

                for bit_idx in range(bits_len):
                    for rep in range(n_repeat):
                        i = bit_idx * n_repeat + rep
                        if i < Q_extract:
                            diff = gft_coeffs_emb[q_start + i] - gft_coeffs_orig[q_start + i]
                            bit = 1 if diff > 0 else 0
                            channel_bit_lists[channel][bit_idx].append(bit)

                for i in range(n_repeat * bits_len, Q_extract):
                    bit_idx = i % bits_len
                    diff = gft_coeffs_emb[q_start + i] - gft_coeffs_orig[q_start + i]
                    bit = 1 if diff > 0 else 0
                    channel_bit_lists[channel][bit_idx].append(bit)

        # 各チャネルのビットを多数決で確定
        extracted_bits_channel = []
        for bit_lists in channel_bit_lists:
            extracted = []
            for bits in bit_lists:
                counts = {0: bits.count(0), 1: bits.count(1)}
                bit = 1 if counts[1] > counts[0] else 0
                extracted.append(bit)
            extracted_bits_channel.extend(extracted)

        extracted_bits = np.array(extracted_bits_channel).astype(int).tolist()
        return extracted_bits

    else:
        raise ValueError("split_modeは0（冗長化）か1（3分割）のみ指定可能です")



### 関数間違い注意！！
def embed_watermark_xyz_check(
    xyz, labels, watermark_bits, beta=0.01,
    split_mode=0, flatness_weighting=0, k_neighbors=20,
    error_correction="none"
):
    """
    各クラスタでwatermark_bitsを
      - split_mode=0: 3チャネル全てに同じ情報（冗長化）
      - split_mode=1: 3分割してx/y/zにそれぞれ独立情報
    としてGFT係数に埋め込む

    error_correction: "none", "parity", "hamming" のいずれか
    """
    xyz_after = xyz.copy()
    cluster_ids = np.unique(labels)
    # color_list = visualize_clusters(xyz, labels)
    flatness_dict = estimate_cluster_flatness(xyz, labels, k_neighbors=k_neighbors)
    weights = compute_cluster_weights(flatness_dict, flatness_weighting)
    phi = max(
        np.min(xyz[:, 0]) + np.max(xyz[:, 0]),
        np.min(xyz[:, 1]) + np.max(xyz[:, 1]),
        np.min(xyz[:, 2]) + np.max(xyz[:, 2])
    )

    # 誤り訂正を事前に適用
    if error_correction == "none":
        pass
    elif error_correction == "parity":
        watermark_bits = add_parity_code(watermark_bits)
    elif error_correction == "hamming":
        watermark_bits = hamming74_encode(watermark_bits)
    else:
        print("誤り訂正符号の指定が不適切です。")
        return -1
    checked_bits_length = len(watermark_bits)

    # 分割モードに応じてビット列を準備
    if split_mode == 0:
        embed_bits_per_channel = [watermark_bits] * 3
        skip_threshold = len(watermark_bits) / 2
    elif split_mode == 1:
        embed_bits_per_channel = np.array_split(watermark_bits, 3)
        skip_threshold = len(watermark_bits) / 5
    else:
        raise ValueError("split_modeは0（冗長化）か1（3分割）のみ指定可能です")

    # 各クラスタで処理
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        if len(idx) <= skip_threshold:
            continue

        pts = xyz[idx]

        for channel in range(3):
            embed_bits = embed_bits_per_channel[channel]
            embed_bits_length = len(embed_bits)
            if embed_bits_length == 0:
                continue  # 空のビット列には埋め込まない

            signal = pts[:, channel]
            W = build_graph(pts, k=6)
            basis, eigvals = gft_basis(W)
            gft_coeffs = gft(signal, basis)
            Q_ = len(gft_coeffs)

            # ビットをリピートしてQ_にあわせる
            n_repeat = Q_ // embed_bits_length if embed_bits_length > 0 else 1
            redundant_bits = repeat_bits_blockwise(embed_bits, n_repeat, Q_)

            # 埋め込み
            for i in range(Q_):
                w = redundant_bits[i] * 2 - 1  # 0/1 → -1/+1
                gft_coeffs[i] += w * beta * weights[c] * phi

            embed_signal = igft(gft_coeffs, basis)
            xyz_after[idx, channel] = embed_signal

    return xyz_after, checked_bits_length

def extract_watermark_xyz_check(
    xyz_emb, xyz_orig, labels, watermark_bits_length, checked_bits_length,
    split_mode=0, error_correction="none"
):  
    total_checks = 0
    passed_checks = 0
    successrate_list = []

    if split_mode == 0:
        skip_threshold = 600
        bits_from_clusters = [[] for _ in range(watermark_bits_length)]

        # 全クラスタを探索
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) <= skip_threshold:
                continue

            pts_emb = xyz_emb[idx]
            pts_orig = xyz_orig[idx]
            if len(pts_emb) != len(pts_orig):
                continue

            actual_k = min(6, len(idx) - 1)
            W = build_graph(pts_orig, k=actual_k)
            basis, eigvals = gft_basis(W)
            Q_ = len(basis)
            n_repeat = Q_ // checked_bits_length if checked_bits_length > 0 else 1

            # 各チャネルごとにビット列抽出
            for channel in range(3):
                gft_coeffs_emb = gft(pts_emb[:, channel], basis)
                gft_coeffs_orig = gft(pts_orig[:, channel], basis)

                bits_from_gftcoeffs = [[] for _ in range(checked_bits_length)]

                # GFT係数からビット列取得
                for bit_idx in range(checked_bits_length):
                    for rep in range(n_repeat):
                        i = bit_idx * n_repeat + rep
                        if i < Q_:
                            diff = gft_coeffs_emb[i] - gft_coeffs_orig[i]
                            bits_from_gftcoeffs[bit_idx].append(1 if diff > 0 else 0)
                for i in range(n_repeat * checked_bits_length, Q_):
                    bit_idx = i % checked_bits_length
                    diff = gft_coeffs_emb[i] - gft_coeffs_orig[i]
                    bits_from_gftcoeffs[bit_idx].append(1 if diff > 0 else 0)

                # 冗長ビットに対して多数決
                cluster_channel_bits = []
                for votes in bits_from_gftcoeffs:
                    counts = {0: votes.count(0), 1: votes.count(1)}
                    cluster_channel_bits.append(1 if counts[1] > counts[0] else 0)

                # エラー訂正チェック
                if error_correction == "none":
                    checked_cluster_channel_bits = cluster_channel_bits
                elif error_correction == "parity":
                    checked_cluster_channel_bits = check_parity_code(cluster_channel_bits)
                elif error_correction == "hamming":
                    checked_cluster_channel_bits= hamming74_decode(cluster_channel_bits)
                else:
                    print("誤り訂正符号の指定が不適切です。")
                    return -1
            
                # OKなビットだけクラスタ間多数決に参加
                count = 0.0
                success = 0.0
                if len(checked_cluster_channel_bits) != watermark_bits_length:
                    print(f"[Error] cluster_channel_bits length = {len(checked_cluster_channel_bits)} but expected {watermark_bits_length}")
                for i, b in enumerate(checked_cluster_channel_bits):
                    count += 1
                    if b is not None:
                        success += 1
                        bits_from_clusters[i].append(b)
                successrate = success / count
                successrate_list.append(successrate)
        
        # クラスタ間多数決
        extracted_bits = []
        num_empty_votes = 0  # 投票なしビット数カウント用
        for votes in bits_from_clusters:
            if len(votes) == 0:
                num_empty_votes += 1  # 投票が1つもないビット
                extracted_bits.append(0)  # デフォルト値として0（または None でもOK）
            else:
                counts = {0: votes.count(0), 1: votes.count(1)}
                extracted_bits.append(1 if counts[1] > counts[0] else 0)

        empty_ratio = num_empty_votes / watermark_bits_length
        if error_correction == "parity":
            print(f"[parity] 符号チェック通過率: {sum(successrate_list)/len(successrate_list):.3f}")
            print(f"[parity] 投票ゼロのビット数: {num_empty_votes} / {watermark_bits_length}（{empty_ratio:.2%}）")
        elif error_correction == "hamming":
            print(f"[hamming] 符号チェック通過率: {sum(successrate_list)/len(successrate_list):.3f}")
            print(f"[hamming] 投票ゼロのビット数: {num_empty_votes} / {watermark_bits_length}（{empty_ratio:.2%}）")
        return extracted_bits
    
    elif split_mode == 1:
        skip_threshold = 200
        split_sizes = [len(arr) for arr in np.array_split(np.zeros(checked_bits_length), 3)]

        bits_from_clusters = [[] for _ in range(watermark_bits_length)]

        # 全クラスタを探索
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) <= skip_threshold:
                continue

            pts_emb = xyz_emb[idx]
            pts_orig = xyz_orig[idx]
            if len(pts_emb) != len(pts_orig):
                continue

            actual_k = min(6, len(idx) - 1)
            W = build_graph(pts_orig, k=actual_k)
            basis, eigvals = gft_basis(W)
            Q_ = len(basis)

            cluster_bits = []

            # 各チャネルごとにビット列抽出
            for channel in range(3):
                bits_len = split_sizes[channel]
                if bits_len == 0:
                    continue

                gft_coeffs_emb = gft(pts_emb[:, channel], basis)
                gft_coeffs_orig = gft(pts_orig[:, channel], basis)

                bits_from_gftcoeffs = [[] for _ in range(bits_len)]
                n_repeat = Q_ // bits_len if bits_len > 0 else 1

                # GFT係数からビット列取得
                for bit_idx in range(bits_len):
                    for rep in range(n_repeat):
                        i = bit_idx * n_repeat + rep
                        if i < Q_:
                            diff = gft_coeffs_emb[i] - gft_coeffs_orig[i]
                            bits_from_gftcoeffs[bit_idx].append(1 if diff > 0 else 0)

                for i in range(n_repeat * bits_len, Q_):
                    bit_idx = i % bits_len
                    diff = gft_coeffs_emb[i] - gft_coeffs_orig[i]
                    bits_from_gftcoeffs[bit_idx].append(1 if diff > 0 else 0)

                # 冗長ビットに対して多数決
                bits_majority = []
                for votes in bits_from_gftcoeffs:
                    counts = {0: votes.count(0), 1: votes.count(1)}
                    bits_majority.append(1 if counts[1] > counts[0] else 0)

                cluster_bits.extend(bits_majority)

            # エラー訂正チェック
            if error_correction == "none":
                checked_cluster_bits = cluster_bits
            elif error_correction == "parity":
                checked_cluster_bits = check_parity_code(cluster_bits)
            elif error_correction == "hamming":
                checked_cluster_bits= hamming74_decode(cluster_bits)
            else:
                print("誤り訂正符号の指定が不適切です。")
                return -1
            
            # OKなビットだけクラスタ間多数決に参加
            count = 0.0
            success = 0.0
            if len(checked_cluster_bits) != watermark_bits_length:
                print(f"[Error] cluster_bits length = {len(checked_cluster_bits)} but expected {watermark_bits_length}")
            for i, b in enumerate(checked_cluster_bits):
                count += 1
                if b is not None:
                    success += 1
                    bits_from_clusters[i].append(b)
            successrate = success / count
            successrate_list.append(successrate)
        
        # クラスタ間多数決
        extracted_bits = []
        num_empty_votes = 0  # 投票なしビット数カウント用
        for votes in bits_from_clusters:
            if len(votes) == 0:
                num_empty_votes += 1  # 投票が1つもないビット
                extracted_bits.append(0)  # デフォルト値として0（または None でもOK）
            else:
                counts = {0: votes.count(0), 1: votes.count(1)}
                extracted_bits.append(1 if counts[1] > counts[0] else 0)

        empty_ratio = num_empty_votes / watermark_bits_length
        if error_correction == "parity":
            print(f"[parity] 符号チェック通過率: {sum(successrate_list)/len(successrate_list):.3f}")
            print(f"[parity] 投票ゼロのビット数: {num_empty_votes} / {watermark_bits_length}（{empty_ratio:.2%}）")
        elif error_correction == "hamming":
            print(f"[hamming] 符号チェック通過率: {sum(successrate_list)/len(successrate_list):.3f}")
            print(f"[hamming] 投票ゼロのビット数: {num_empty_votes} / {watermark_bits_length}（{empty_ratio:.2%}）")

        return extracted_bits
    
    else:
        raise ValueError("split_modeは0（冗長化）か1（3分割）のみ指定可能です")
    

### 関数間違い注意！！
def embed_watermark_normal(
    xyz, normals, labels, embed_bits, beta=0.01,
    graph_mode='knn', k=10, radius=0.05,
    flatness_weighting=0, k_neighbors=20,
    min_spectre=0.0, max_spectre=1.0
):
    """
    法線方向スカラー信号 s_i = <p_i, n_i> にGFT埋め込みし、
    p'_i = p_i + (s'_i - s_i) * n_i で更新する（法線方向にのみ変位）。
    """
    xyz_after = xyz.copy()
    cluster_ids = np.unique(labels)
    # color_list = visualize_clusters(xyz, labels)

    # 重み
    flatness_dict = estimate_cluster_flatness(xyz, labels, k_neighbors=k_neighbors)
    weights = compute_cluster_weights(flatness_dict, flatness_weighting)

    # スケール
    phi = max(
        np.max(xyz[:, 0]) - np.min(xyz[:, 0]),
        np.max(xyz[:, 1]) - np.min(xyz[:, 1]),
        np.max(xyz[:, 2]) - np.min(xyz[:, 2])
    )

    skip_threshold = len(embed_bits) / 2
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        if len(idx) <= skip_threshold:
            continue

        pts = xyz[idx]
        nrm = normals[idx]

        # 法線方向スカラー信号
        s = np.sum(pts * nrm, axis=1)  # (Nc,)

        # グラフ & GFT
        W = build_graph(pts, graph_mode=graph_mode, k=k, radius=radius)
        basis, eigvals = gft_basis(W)
        gft_coeffs = gft(s, basis)

        Q_ = len(gft_coeffs)
        q_start = int(Q_ * min_spectre)
        q_end   = int(Q_ * max_spectre)
        Q_embed = max(q_end - q_start, 0)
        if Q_embed <= 0:
            continue

        bits_len = len(embed_bits)
        n_repeat = Q_embed // bits_len if bits_len > 0 else 1
        redundant_bits = repeat_bits_blockwise(embed_bits, n_repeat, Q_embed)

        for i in range(Q_embed):
            w = redundant_bits[i] * 2 - 1  # 0/1 -> -1/+1
            gft_coeffs[q_start + i] += w * beta * weights[c] * phi

        s_emb = igft(gft_coeffs, basis)

        # 法線方向変位だけ適用
        delta_n = (s_emb - s)  # (Nc,)
        xyz_after[idx] = pts + delta_n[:, None] * nrm

    return xyz_after

def extract_watermark_normal(
    xyz_emb, xyz_orig, normals, labels, embed_bits_length,
    graph_mode='knn', k=10, radius=0.05,
    min_spectre=0.0, max_spectre=1.0
):
    """
    s_i = <p_i, n_i> を用いて、GFT係数差分の符号でビット抽出（多数決）。
    """
    skip_threshold = embed_bits_length / 2
    bit_lists = [[] for _ in range(embed_bits_length)]

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) <= skip_threshold:
            continue

        pts_emb  = xyz_emb[idx]
        pts_orig = xyz_orig[idx]
        if len(pts_emb) != len(pts_orig):
            continue

        nrm = normals[idx]

        s_emb  = np.sum(pts_emb  * nrm, axis=1)
        s_orig = np.sum(pts_orig * nrm, axis=1)

        W = build_graph(pts_orig, graph_mode=graph_mode, k=k, radius=radius)
        basis, eigvals = gft_basis(W)

        gft_emb  = gft(s_emb, basis)
        gft_orig = gft(s_orig, basis)

        Q_ = len(gft_emb)
        q_start = int(Q_ * min_spectre)
        q_end   = int(Q_ * max_spectre)
        Q_extract = max(q_end - q_start, 0)
        if Q_extract <= 0:
            continue

        n_repeat = Q_extract // embed_bits_length if embed_bits_length > 0 else 1

        # 既存extractのロジックを踏襲（冗長分をbitごとに集める）
        for bit_idx in range(embed_bits_length):
            for rep in range(n_repeat):
                i = bit_idx * n_repeat + rep
                if i < Q_extract:
                    diff = gft_emb[q_start + i] - gft_orig[q_start + i]
                    bit = 1 if diff > 0 else 0
                    bit_lists[bit_idx].append(bit)

        for i in range(n_repeat * embed_bits_length, Q_extract):
            bit_idx = i % embed_bits_length
            diff = gft_emb[q_start + i] - gft_orig[q_start + i]
            bit = 1 if diff > 0 else 0
            bit_lists[bit_idx].append(bit)

    extracted_bits = []
    for votes in bit_lists:
        if len(votes) == 0:
            extracted_bits.append(0)
        else:
            extracted_bits.append(1 if votes.count(1) > votes.count(0) else 0)
    return extracted_bits


### 関数間違い注意！！
def embed_watermark_pseudoplane(
    xyz: np.ndarray,
    labels: np.ndarray,
    embed_bits,
    beta: float = 0.01,
    graph_mode: str = 'knn',
    k: int = 10,
    radius: float = 0.05,
    flatness_weighting: int = 0,
    k_neighbors: int = 20,
    min_spectre: float = 0.0,
    max_spectre: float = 1.0,
    skip_threshold_mode: str = "half",  # "half" or "none"
):
    """
    各クラスタで疑似平面を推定し、その平面からの高さ（符号付き距離）hを信号としてGFT→係数変調→逆GFT。
    座標更新は平面法線方向のみ:
        p'_i = p_i + (h'_i - h_i) * normal_c

    注意:
    - 抽出でも同じ疑似平面が再構築できる必要があるため、平面推定は乱数に依存しないPCAを使用。
    - 既存の重み設計（flatness_weighting等）を流用。
    """
    xyz_after = xyz.copy()
    cluster_ids = np.unique(labels)
    color_list = visualize_clusters(xyz, labels)
    visualize_clusters_with_pseudoplanes(
        xyz, labels, color_list,
        scale=1.2,
        max_planes=30,      # 重ければ制限
        show_frame=False
    )

    # 既存のクラスタ平坦度 -> 重み（あなたの関数を流用）
    flatness_dict = estimate_cluster_flatness(xyz, labels, k_neighbors=k_neighbors)
    weights = compute_cluster_weights(flatness_dict, flatness_weighting)

    # 既存コードと同じスケール（phi）
    phi = max(
        np.max(xyz[:, 0]) - np.min(xyz[:, 0]),
        np.max(xyz[:, 1]) - np.min(xyz[:, 1]),
        np.max(xyz[:, 2]) - np.min(xyz[:, 2])
    )

    bits_len = len(embed_bits)
    if bits_len == 0:
        return xyz_after

    if skip_threshold_mode == "half":
        skip_threshold = bits_len / 2
    else:
        skip_threshold = -1  # 実質スキップしない

    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        if len(idx) <= skip_threshold:
            continue

        pts = xyz[idx]

        # 疑似平面（PCA）推定
        centroid, normal = compute_pseudoplane_pca(pts)

        # 高さ信号
        h = cluster_height_signal(pts, centroid, normal)

        # グラフ構築
        W = build_graph(pts, graph_mode=graph_mode, k=k, radius=radius)
        basis, eigvals = gft_basis(W)

        # 高さ信号をGFT
        gft_coeffs = gft(h, basis)

        Q_ = len(gft_coeffs)
        q_start = int(Q_ * min_spectre)
        q_end = int(Q_ * max_spectre)
        Q_embed = max(q_end - q_start, 0)
        if Q_embed <= 0:
            continue

        # 冗長化
        n_repeat = Q_embed // bits_len if bits_len > 0 else 1
        redundant_bits = repeat_bits_blockwise(embed_bits, n_repeat, Q_embed)

        # 係数変調
        w_c = weights.get(c, 1.0)
        for i in range(Q_embed):
            w = redundant_bits[i] * 2 - 1  # 0/1 -> -1/+1
            gft_coeffs[q_start + i] += w * beta * w_c * phi

        # 逆GFT
        h_emb = igft(gft_coeffs, basis)

        # 座標更新（法線方向のみ）
        delta_h = (h_emb - h)  # (Nc,)
        xyz_after[idx] = pts + delta_h[:, None] * normal[None, :]

    return xyz_after


def extract_watermark_pseudoplane(
    xyz_emb: np.ndarray,
    xyz_orig: np.ndarray,
    labels: np.ndarray,
    embed_bits_length: int,
    graph_mode: str = 'knn',
    k: int = 10,
    radius: float = 0.05,
    min_spectre: float = 0.0,
    max_spectre: float = 1.0,
    skip_threshold_mode: str = "half",  # "half" or "none"
):
    """
    埋め込み前点群 xyz_orig から疑似平面を再構築し、
    高さ信号 h_orig と h_emb を作り、GFT係数差分の符号でビット抽出（多数決）。

    重要:
    - 疑似平面推定は xyz_orig（埋め込み前）で行う（xyz_embで推定すると透かし変位が平面推定に混入する）。
    - グラフも xyz_orig（クラスタ内座標）で構築する（埋め込み前と一致させる）。
    """
    if embed_bits_length <= 0:
        return []

    if skip_threshold_mode == "half":
        skip_threshold = embed_bits_length / 2
    else:
        skip_threshold = -1

    bit_lists = [[] for _ in range(embed_bits_length)]

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) <= skip_threshold:
            continue

        pts_orig = xyz_orig[idx]
        pts_emb = xyz_emb[idx]
        if len(pts_orig) != len(pts_emb):
            continue

        # 疑似平面は必ずorigから
        centroid, normal = compute_pseudoplane_pca(pts_orig)

        # 高さ信号（orig/emb）
        h_orig = cluster_height_signal(pts_orig, centroid, normal)
        h_emb = cluster_height_signal(pts_emb, centroid, normal)

        # グラフはorigで
        W = build_graph(pts_orig, graph_mode=graph_mode, k=k, radius=radius)
        basis, eigvals = gft_basis(W)

        gft_orig = gft(h_orig, basis)
        gft_emb = gft(h_emb, basis)

        Q_ = len(gft_emb)
        q_start = int(Q_ * min_spectre)
        q_end = int(Q_ * max_spectre)
        Q_extract = max(q_end - q_start, 0)
        if Q_extract <= 0:
            continue

        n_repeat = Q_extract // embed_bits_length if embed_bits_length > 0 else 1

        # 既存extractの投票構造を踏襲
        # まず「ブロック繰り返し分」
        for bit_idx in range(embed_bits_length):
            for rep in range(n_repeat):
                i = bit_idx * n_repeat + rep
                if i < Q_extract:
                    diff = gft_emb[q_start + i] - gft_orig[q_start + i]
                    bit = 1 if diff > 0 else 0
                    bit_lists[bit_idx].append(bit)

        # 余り分（もしあれば）
        for i in range(n_repeat * embed_bits_length, Q_extract):
            bit_idx = i % embed_bits_length
            diff = gft_emb[q_start + i] - gft_orig[q_start + i]
            bit = 1 if diff > 0 else 0
            bit_lists[bit_idx].append(bit)

    # 多数決
    extracted_bits = []
    for votes in bit_lists:
        if len(votes) == 0:
            extracted_bits.append(0)
        else:
            extracted_bits.append(1 if votes.count(1) > votes.count(0) else 0)

    return extracted_bits



def _compute_cluster_capacity_pseudoplane(
    xyz: np.ndarray,
    labels: np.ndarray,
    graph_mode: str,
    k: int,
    radius: float,
    min_spectre: float,
    max_spectre: float,
    min_points_for_graph: int,
):
    """
    各クラスタの埋め込み可能長 Q_embed を計算して返す。
    - グラフ構築/固有分解が成立する最小点数未満は capacity=0 とする。
    Returns:
        caps: dict {cluster_id: Q_embed}
    """
    caps = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < min_points_for_graph:
            caps[int(c)] = 0
            continue

        pts = xyz[idx]
        W = build_graph(pts, graph_mode=graph_mode, k=k, radius=radius)
        basis, _ = gft_basis(W)

        Q_ = len(basis)
        q_start = int(Q_ * min_spectre)
        q_end = int(Q_ * max_spectre)
        Q_embed = max(q_end - q_start, 0)

        caps[int(c)] = Q_embed
    return caps


def _build_global_slot_map(
    xyz: np.ndarray,
    labels: np.ndarray,
    graph_mode: str,
    k: int,
    radius: float,
    min_spectre: float,
    max_spectre: float,
    min_points_for_graph: int,
    cluster_order: str = "id",  # "id" or "size_desc"
):
    """
    「全クラスタの埋め込み帯域係数」を1本のスロット列にしたときの対応表を作る。

    Returns:
      slot_map: list of tuples
        [
          (cluster_id, idx_points, pts_orig, basis, q_start, local_i),
          ...
        ]
      ※ idx_points / pts_orig / basis などをまとめて返すのは、embed/extract側で
        同じグラフ・同じbasisを使うため（origベース）。
    """
    # クラスタ順序を決める（抽出と一致させる必要があるため、決定的に）
    clusters = [int(c) for c in np.unique(labels)]
    if cluster_order == "size_desc":
        clusters.sort(key=lambda c: np.sum(labels == c), reverse=True)
    else:
        clusters.sort()

    slot_map = []
    cluster_cache = {}

    for c in clusters:
        idx = np.where(labels == c)[0]
        if len(idx) < min_points_for_graph:
            continue

        pts = xyz[idx]
        W = build_graph(pts, graph_mode=graph_mode, k=k, radius=radius)
        basis, _ = gft_basis(W)

        Q_ = len(basis)
        q_start = int(Q_ * min_spectre)
        q_end = int(Q_ * max_spectre)
        Q_embed = max(q_end - q_start, 0)
        if Q_embed <= 0:
            continue

        # キャッシュ（同クラスタ内で x/y/z も扱う等に拡張しやすい）
        cluster_cache[c] = (idx, pts, basis, q_start, Q_embed)

        for local_i in range(Q_embed):
            slot_map.append((c, local_i))

    return slot_map, cluster_cache


def embed_watermark_pseudoplane_multicluster(
    xyz: np.ndarray,
    labels: np.ndarray,
    embed_bits,
    beta: float = 0.01,
    graph_mode: str = 'knn',
    k: int = 10,
    radius: float = 0.05,
    flatness_weighting: int = 0,
    k_neighbors: int = 20,
    min_spectre: float = 0.0,
    max_spectre: float = 1.0,
    cluster_order: str = "id",         # "id" or "size_desc"
    min_points_for_graph: int = None,  # Noneなら max(k+1,3)
):
    """
    1つの embed_bits をクラスタをまたいで分割して埋め込む版（疑似平面）。
    - 全クラスタの埋め込み可能係数（帯域）を “global slot” として連結し、
      repeat_bits_blockwise で global slot 長に合わせて冗長化して埋め込む。
    - 座標更新は各クラスタの疑似平面法線方向のみ。

    期待する効果:
    - クラスタ数を増やして1クラスタ点数が減っても、全体としての容量を稼げる。
    """
    xyz_after = xyz.copy()
    bits_len = len(embed_bits)
    if bits_len == 0:
        return xyz_after

    if min_points_for_graph is None:
        min_points_for_graph = max(k + 1, 3)

    # 重み（既存ロジック流用）
    flatness_dict = estimate_cluster_flatness(xyz, labels, k_neighbors=k_neighbors)
    weights = compute_cluster_weights(flatness_dict, flatness_weighting)

    # スケール（既存ロジック踏襲）
    phi = max(
        np.max(xyz[:, 0]) - np.min(xyz[:, 0]),
        np.max(xyz[:, 1]) - np.min(xyz[:, 1]),
        np.max(xyz[:, 2]) - np.min(xyz[:, 2])
    )

    # global slot を作る（origベースで決定的に）
    slot_map, cluster_cache = _build_global_slot_map(
        xyz, labels,
        graph_mode=graph_mode, k=k, radius=radius,
        min_spectre=min_spectre, max_spectre=max_spectre,
        min_points_for_graph=min_points_for_graph,
        cluster_order=cluster_order,
    )
    total_slots = len(slot_map)
    if total_slots <= 0:
        print("[MultiCluster] No available slots. Check clustering / min_spectre~max_spectre / graph params.")
        return xyz_after

    # global slot 長に合わせて冗長化（bit列を全体に均等にばらまく）
    n_repeat = total_slots // bits_len if bits_len > 0 else 1
    redundant_bits = repeat_bits_blockwise(embed_bits, n_repeat, total_slots)

    # 各クラスタごとに「一回」GFTして、該当slot分だけ係数変調して戻す
    # （slot単位で毎回固有分解すると激重なので、クラスタ単位でまとめて処理）
    # まず、クラスタごとに “どのlocal_iに何ビットを入れるか” を集める
    assign = {}  # {cluster_id: list[(local_i, bit)]}
    for s, (c, local_i) in enumerate(slot_map):
        b = int(redundant_bits[s])
        assign.setdefault(c, []).append((local_i, b))

    for c, items in assign.items():
        idx, pts, basis, q_start, Q_embed = cluster_cache[c]

        # 疑似平面（PCA）と高さ信号（origから）
        centroid, normal = compute_pseudoplane_pca(pts)
        h = cluster_height_signal(pts, centroid, normal)

        # 高さ信号GFT
        gft_coeffs = gft(h, basis)

        # 係数変調（このクラスタに割り当てられたlocal_iのみ）
        w_c = weights.get(c, 1.0)
        for (local_i, bit01) in items:
            w = bit01 * 2 - 1
            gft_coeffs[q_start + local_i] += w * beta * w_c * phi

        # 逆GFT → 法線方向のみ更新
        h_emb = igft(gft_coeffs, basis)
        delta_h = (h_emb - h)
        xyz_after[idx] = pts + delta_h[:, None] * normal[None, :]

    return xyz_after


def extract_watermark_pseudoplane_multicluster(
    xyz_emb: np.ndarray,
    xyz_orig: np.ndarray,
    labels: np.ndarray,
    embed_bits_length: int,
    graph_mode: str = 'knn',
    k: int = 10,
    radius: float = 0.05,
    min_spectre: float = 0.0,
    max_spectre: float = 1.0,
    cluster_order: str = "id",         # embed側と必ず一致させる
    min_points_for_graph: int = None,  # Noneなら max(k+1,3)
):
    """
    embed_watermark_pseudoplane_multicluster の抽出版。
    - global slot を同じ規則で再構築
    - 各slotの差分符号を “そのslotが担当するbit番号” に投票
    - 最後にbitごと多数決
    """
    if embed_bits_length <= 0:
        return []

    if min_points_for_graph is None:
        min_points_for_graph = max(k + 1, 3)

    # global slot を再構築（origベースで決定的に）
    slot_map, cluster_cache = _build_global_slot_map(
        xyz_orig, labels,
        graph_mode=graph_mode, k=k, radius=radius,
        min_spectre=min_spectre, max_spectre=max_spectre,
        min_points_for_graph=min_points_for_graph,
        cluster_order=cluster_order,
    )
    total_slots = len(slot_map)
    if total_slots <= 0:
        print("[MultiCluster] No available slots for extraction.")
        return [0] * embed_bits_length

    # embed側と同じ “bitの並べ方” を再現するため、total_slots から n_repeat を決める
    n_repeat = total_slots // embed_bits_length if embed_bits_length > 0 else 1

    # bitごとの投票箱
    bit_lists = [[] for _ in range(embed_bits_length)]

    # まずクラスタ単位で GFT差分を作っておく
    # その上で slot_map に従って投票
    cluster_diffs = {}  # {c: diff_gft_band_array}
    for c, (idx, pts, basis, q_start, Q_embed) in cluster_cache.items():
        pts_o = xyz_orig[idx]
        pts_e = xyz_emb[idx]
        if len(pts_o) != len(pts_e):
            continue

        centroid, normal = compute_pseudoplane_pca(pts_o)
        h_orig = cluster_height_signal(pts_o, centroid, normal)
        h_emb  = cluster_height_signal(pts_e, centroid, normal)

        gft_orig = gft(h_orig, basis)
        gft_emb  = gft(h_emb, basis)

        # 帯域部分だけ取り出しておく（local_iで参照）
        band_diff = gft_emb[q_start:q_start + Q_embed] - gft_orig[q_start:q_start + Q_embed]
        cluster_diffs[c] = band_diff

    # slotごとに「どのbitに投票するか」を決めて投票
    # embed側の repeat_bits_blockwise と一致する投票割当:
    # 先頭から n_repeat 個ずつ bit0, bit1, ... と割り当て、余りは i%embed_bits_length
    # ※ slot_map の並び = embed側の並び と一致する必要がある
    for s, (c, local_i) in enumerate(slot_map):
        if c not in cluster_diffs:
            continue
        if local_i >= len(cluster_diffs[c]):
            continue

        diff = cluster_diffs[c][local_i]
        bit = 1 if diff > 0 else 0

        # slot index s が担当する bit index を再現
        if n_repeat > 0 and s < n_repeat * embed_bits_length:
            bit_idx = s // n_repeat
        else:
            bit_idx = s % embed_bits_length

        bit_lists[bit_idx].append(bit)

    # 多数決で確定
    extracted_bits = []
    for votes in bit_lists:
        if len(votes) == 0:
            extracted_bits.append(0)
        else:
            extracted_bits.append(1 if votes.count(1) > votes.count(0) else 0)

    return extracted_bits

# =========================================================
#  誤り訂正関数群
# =========================================================

def add_parity_code(bits, block_size=8):
    """
    偶数パリティ符号をブロックごとに追加
    例: 8bit → 9bit（パリティ1bit追加）
    """
    encoded = []
    for i in range(0, len(bits), block_size):
        block = bits[i:i+block_size]
        if len(block) < block_size:
            block += [0] * (block_size - len(block))
        parity = sum(block) % 2
        encoded.extend(block + [parity])
    return encoded

def check_parity_code(bits, block_size=8):
    """
    偶数パリティ符号を検証し、元の長さに合わせてビット列を返す。
    正しく復元できなかったブロックには None を入れる。
    """
    decoded = []
    for i in range(0, len(bits), block_size + 1):
        block = bits[i:i + block_size + 1]
        if len(block) < block_size + 1:
            decoded.extend([None] * block_size)
            continue
        data, parity = block[:-1], block[-1]
        if sum(data) % 2 == parity:
            decoded.extend(data)
        else:
            decoded.extend([None] * block_size)  # 復元失敗ブロック
    return decoded

def hamming74_encode(bits):
    """
    4bitごとにハミング(7,4)符号を追加して7bit出力に変換
    """
    G = [
        [1, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1],
    ]
    encoded = []
    for i in range(0, len(bits), 4):
        block = bits[i:i+4]
        if len(block) < 4:
            block += [0] * (4 - len(block))
        encoded_block = [sum([block[j] * G[j][k] for j in range(4)]) % 2 for k in range(7)]
        encoded.extend(encoded_block)
    return encoded

def hamming74_decode(bits):
    """
    ハミング(7,4)復号。正しく復元できなかったブロックには None を返す。
    """
    H = [
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1]
    ]
    decoded = []
    for i in range(0, len(bits), 7):
        block = bits[i:i+7]
        if len(block) < 7:
            decoded.extend([None] * 4)
            continue
        syndrome = [sum([block[j] * H[k][j] for j in range(7)]) % 2 for k in range(3)]
        syndrome_value = syndrome[0]*4 + syndrome[1]*2 + syndrome[2]*1
        if syndrome_value != 0:
            if 1 <= syndrome_value <= 7:
                block[syndrome_value - 1] ^= 1
            else:
                decoded.extend([None] * 4)
                continue
        decoded.extend(block[:4])
    return decoded

# =========================================================
#  評価関数群
# =========================================================

def evaluate_imperceptibility(pcd_before, pcd_after, reverse=False, by_index=False):
    """
    点群の評価指標を計算します。
    - pcd_before, pcd_after: open3d.geometry.PointCloud
    - reverse: Trueならafter→beforeも評価し、両方向平均
    - by_index: True の場合、点配列のインデックスを直接対応させて評価します。
                （点数と順序が一致している必要があります）
              False の場合は既存の最近傍対応（order-free）で評価します。

    戻り値: dict { 'mse':..., 'rmse':..., 'psnr':..., 'snr':..., 'max_range':..., 'method': 'index'|'nn' }
    """
    points_before = np.asarray(pcd_before.points)
    points_after = np.asarray(pcd_after.points)

    if by_index:
        # インデックス対応モード: 点数と並びが同じであることを要求
        if points_before.shape != points_after.shape:
            raise ValueError("by_index=True の場合、pcd_before と pcd_after は同じ点数・形状である必要があります")

        # そのまま差を計算
        diffs_sq = np.sum((points_before - points_after) ** 2, axis=1)
        sum_sq_diff = np.sum(diffs_sq)
        mse_fwd = np.mean(diffs_sq)

        if reverse:
            # reverse を index モードでも意味があるように、after->before の差を同様に計算して平均
            diffs_rev_sq = np.sum((points_after - points_before) ** 2, axis=1)
            mse_rev = np.mean(diffs_rev_sq)
            mse = (mse_fwd + mse_rev) / 2
        else:
            mse = mse_fwd

    else:
        # 既存の最近傍対応（order-free）
        tree = cKDTree(points_after)
        dists, idxs = tree.query(points_before, k=1)
        matched_after = points_after[idxs]
        
        diffs_sq = np.sum((points_before - matched_after) ** 2, axis=1)
        sum_sq_diff = np.sum(diffs_sq)
        mse_fwd = np.mean(diffs_sq)

        if reverse:
            tree_rev = cKDTree(points_before)
            dists_rev, idxs_rev = tree_rev.query(points_after, k=1)
            matched_before = points_before[idxs_rev]
            mse_rev = np.mean(np.sum((points_after - matched_before) ** 2, axis=1))
            mse = (mse_fwd + mse_rev) / 2
        else:
            mse = mse_fwd

    sum_sq_signal = np.sum(points_before ** 2)
    if sum_sq_diff == 0:
        snr_ratio = float('inf')
    else:
        snr_ratio = sum_sq_signal / sum_sq_diff
    # max_range（PSNR用スケール）
    xyz = points_before
    max_range = max(
        np.max(xyz[:,0]) - np.min(xyz[:,0]),
        np.max(xyz[:,1]) - np.min(xyz[:,1]),
        np.max(xyz[:,2]) - np.min(xyz[:,2])
    )
    # RMSE（√MSE）
    rmse = np.sqrt(mse)
    # SNR
    snr = 10 * np.log10(snr_ratio) if snr_ratio != float('inf') else float('inf')
    # VSNR
    vsnr = 20 * np.log10(snr_ratio) if snr_ratio != float('inf') else float('inf')
    # PSNR
    psnr = float('inf') if mse == 0 else 10 * np.log10((max_range ** 2) / mse)

    print("------------------- 評価 -------------------")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"SNR  : {snr:.2f} dB")
    print(f"PSNR : {psnr:.2f} dB (max_range={max_range:.4f})")
    print(f"VSNR : {vsnr:.2f} dB")

def evaluate_robustness(watermark_bits, extracted_bits):
    """
    Corr（ピアソン相関係数）とBER（ビット誤り率）を計算し表示する。
    """
    
    # numpy配列に変換
    w = np.array(watermark_bits, dtype=float)
    w_ = np.array(extracted_bits, dtype=float)

    # --- BER計算 ---
    # 単純な一致率から計算
    ber = 1.0 - np.mean(w == w_)

    # --- Corr計算 (論文の式5: ピアソン相関係数) ---
    # np.corrcoef は相関行列を返すので [0, 1] 成分を取得
    # 入力が定数（分散0）だとNaNになるので注意が必要だが、透かしビットなら通常大丈夫
    if np.std(w) == 0 or np.std(w_) == 0:
        corr = 0.0
    else:
        corr = np.corrcoef(w, w_)[0, 1]

    # --- 結果を表示 ---
    print(f"Corr : {corr:.4f}")
    print(f"BER  : {ber:.4f}")

# =========================================================
#  攻撃関数群
# =========================================================

def noise_addition_attack(xyz, noise_percent=1.0, mode='uniform', seed=None):
    """
    numpy配列(xyz)にノイズを加える
    - noise_percent: ノイズ振幅
    - mode: 'uniform'または'gaussian'
    - return: ノイズ加算後のnumpy配列
    """
    rng = np.random.RandomState(seed)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    ranges = xyz_max - xyz_min
    scale = ranges * noise_percent / 100
    print(f"[Attack] ノイズ振幅: {noise_percent:.2f}% (scale={scale})")
    if mode == 'uniform':
        noise = rng.uniform(low=-scale, high=scale, size=xyz.shape)
    elif mode == 'gaussian':
        noise = rng.normal(loc=0.0, scale=scale/2, size=xyz.shape)
    else:
        raise ValueError('modeは "uniform" か "gaussian"')
    xyz_noisy = xyz + noise
    return xyz_noisy

def cropping_attack(xyz_after, keep_ratio=0.5, mode='center', axis=1):
    """
    xyz_after に対して切り取り攻撃を行い、一部の点群のみを残し、表示する。

    Parameters:
    - xyz_after (np.ndarray): 埋め込み後の点群座標（N×3）
    - keep_ratio (float): 残す点の割合（0.0～1.0]
    - mode (str): 
        - 'center': 重心に近い点を残す（球状クロッピング）
        - 'edge': 重心から遠い点を残す（逆球状クロッピング）
        - 'axis': 指定した軸に沿って端から切断する
    - axis (int): 'axis' モードで使用する切断軸 (0:X軸, 1:Y軸, 2:Z軸)

    Returns:
    - xyz_cropped (np.ndarray): 切り取り後の点群座標
    """
    assert 0.0 < keep_ratio <= 1.0, "keep_ratioは (0, 1] で指定してください"
    N = xyz_after.shape[0]
    keep_n = int(N * keep_ratio)

    if mode == 'center':
        # 重心からの距離でソートし、近い順に残す
        center = np.mean(xyz_after, axis=0)
        dists = np.linalg.norm(xyz_after - center, axis=1)
        keep_indices = np.argsort(dists)[:keep_n]
        
    elif mode == 'edge':
        # 重心からの距離でソートし、遠い順に残す
        center = np.mean(xyz_after, axis=0)
        dists = np.linalg.norm(xyz_after - center, axis=1)
        keep_indices = np.argsort(dists)[-keep_n:]
        
    elif mode == 'axis':
        # 指定軸の値でソートし、片側を残す（平面切断）
        if axis not in [0, 1, 2]:
            raise ValueError("axisは 0(x), 1(y), 2(z) のいずれかを指定してください")
        
        # 軸の値に基づいてインデックスをソート
        indices = np.argsort(xyz_after[:, axis])
        
        # 値が小さいほうを残す（あるいは大きいほうを残す）
        # ここでは「値が小さい順」の上位 keep_n 個を残します
        keep_indices = indices[:keep_n]
        
    else:
        raise ValueError("modeは 'center', 'edge', 'axis' のいずれかを指定してください")

    xyz_cropped = xyz_after[keep_indices]

    print(f"[Attack] 切り取り攻撃 ({mode}): 元点数={N} → 残点数={keep_n} ({keep_ratio*100:.1f}%)")

    # 可視化
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(xyz_cropped)
    cropped_pcd.paint_uniform_color([1, 0.6, 0])  # オレンジ系で表示
    
    # 軸などのガイドを表示するためにフレームを追加
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([cropped_pcd, mesh_frame], window_name=f"Cropped Point Cloud ({mode})")

    return xyz_cropped

def smoothing_attack(xyz, lambda_val=0.1, iterations=5, k=6, verbose=True):
    """
    点群に対してラプラシアンスムージング攻撃を行う
    
    Parameters:
    - xyz: (N, 3) の点群座標配列
    - lambda_val: スムージングの強度係数 λ (0.0 < lambda <= 1.0)。
                  1.0に近いほど近傍点の重心へ強く移動します。
                  論文(El Zein et al.)では delta(relaxation) と呼ばれるパラメータに相当します。
    - iterations: 繰り返し回数。回数が多いほど平滑化が進みます。
    - k: 近傍点の数。点群の接続関係を定義するために使用します。
    - verbose: ログ表示の有無

    Returns:
    - xyz_smooth: スムージング後の点群座標
    """
    if verbose:
        print(f"[Attack] スムージング: lambda={lambda_val}, iterations={iterations}, k={k}")

    N = xyz.shape[0]
    xyz_smooth = xyz.copy()

    # 近傍探索のための学習 (一度だけ実行)
    # ※厳密には反復ごとに近傍が変わる可能性がありますが、
    #   微小な移動であれば固定した方が計算効率が良く、攻撃としても十分機能します。
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(xyz)
    
    # 各点の近傍インデックスを取得 (自分自身が含まれるため k+1 個取得し、自分を除く)
    _, indices = nbrs.kneighbors(xyz)
    neighbor_indices = indices[:, 1:] # (N, k)

    for i in range(iterations):
        # 全点の近傍点の座標を取得 (N, k, 3)
        neighbor_coords = xyz_smooth[neighbor_indices]
        
        # 近傍点の重心を計算 (N, 3)
        centroids = np.mean(neighbor_coords, axis=1)
        
        # 重心方向へ移動 (Laplacian smoothing update rule)
        # P_new = P_old + lambda * (Centroid - P_old)
        xyz_smooth = xyz_smooth + lambda_val * (centroids - xyz_smooth)
        
    return xyz_smooth

def reconstruct_point_cloud(xyz_after, xyz_orig, threshold=0.01, verbose=True):
    """
    欠損した点を元点群（xyz_orig）から復元して返す。

    Parameters:
    - xyz_after (np.ndarray): 攻撃後の点群
    - xyz_orig (np.ndarray): 元点群（完全版）
    - threshold (float): 一致判定の距離しきい値
    - verbose (bool): ログ表示有無

    Returns:
    - xyz_reconstructed (np.ndarray): 点が補完されたxyz_after
    """
    tree = cKDTree(xyz_after)
    dists, _ = tree.query(xyz_orig, k=1)
    missing_mask = dists > threshold
    missing_points = xyz_orig[missing_mask]
    xyz_reconstructed = np.vstack([xyz_after, missing_points])
    
    if verbose:
        print(f"[Reconstruction] 復元された点数: {len(missing_points)} / {len(xyz_orig)}")
        print("再構成後xyzのshape:", xyz_reconstructed.shape)
        print("距離の最小:", np.min(dists))
        print("距離の最大:", np.max(dists))
        print("平均距離:", np.mean(dists))
    return xyz_reconstructed

def reorder_point_cloud(xyz_after, xyz_orig, verbose=True):
    """
    xyz_afterの順番を、xyz_origと最も近い点で対応づけて並び替える。

    Parameters:
    - xyz_after (np.ndarray): 復元後点群（点数 = xyz_orig以上）
    - xyz_orig (np.ndarray): 元点群（基準順）

    Returns:
    - xyz_reordered (np.ndarray): 並べ替え後の点群（xyz_origと順序一致）
    """
    tree = cKDTree(xyz_after)
    _, indices = tree.query(xyz_orig, k=1)
    xyz_reordered = xyz_after[indices]

    if verbose:
        print(f"[Reorder] 順序を xyz_orig に再整列しました。")

    return xyz_reordered

# =========================================================
#  参考関数群
# =========================================================

def sort_and_shuffle_points(xyz, seed=42):
    norms = np.linalg.norm(xyz, axis=1)
    sort_idx = np.argsort(-norms)  # 降順
    xyz_sorted = xyz[sort_idx]
    rng = np.random.RandomState(seed)
    shuffle_idx = np.arange(xyz_sorted.shape[0])
    rng.shuffle(shuffle_idx)
    xyz_shuffled = xyz_sorted[shuffle_idx]
    return xyz_shuffled, sort_idx, shuffle_idx

def greedy_cluster_points(xyz, num_clusters=8, seed=42):
    rng = np.random.RandomState(seed)
    N = xyz.shape[0]
    idx_all = np.arange(N)
    cluster_labels = np.full(N, -1)
    cluster_sizes = [N // num_clusters] * num_clusters
    for i in range(N % num_clusters):
        cluster_sizes[i] += 1
    unassigned = set(idx_all)
    clusters = {i: [] for i in range(num_clusters)}
    seeds = rng.choice(list(unassigned), size=num_clusters, replace=False)
    for c, s in enumerate(seeds):
        clusters[c].append(s)
        cluster_labels[s] = c
        unassigned.remove(s)
    for c in range(num_clusters):
        while len(clusters[c]) < cluster_sizes[c]:
            last = clusters[c][-1]
            available = np.array(list(unassigned))
            dists = np.linalg.norm(xyz[available] - xyz[last], axis=1)
            next_idx = available[np.argmin(dists)]
            clusters[c].append(next_idx)
            cluster_labels[next_idx] = c
            unassigned.remove(next_idx)
    return cluster_labels