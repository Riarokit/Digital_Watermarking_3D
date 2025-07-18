import numpy as np
import open3d as o3d
import random
import string
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import NearestNeighbors
# import cupy as cp

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

def kmeans_cluster_points(xyz, num_clusters=None, seed=42):
    if num_clusters is None:
        num_clusters = len(xyz) // 1000
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
    min_cluster_size=100,
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


def ransac_cluster_points(xyz, distance_threshold=None, min_cluster_size=100, max_planes=1000):
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

def split_large_clusters(xyz, labels, limit_points=4000, seed=42):
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

def compute_cluster_weights(flatness_dict, min_weight=0.2, max_weight=1.8, flatness_weighting=0):
    """
    flatness_weighting:
      0 → 重みなし（全クラスタ重み1.0）
      1 → 平坦クラスタほど重み大
      2 → 曲面クラスタほど重み大
    """
    if flatness_weighting == 0:
        # 重みなし
        return {c: 1.0 for c in flatness_dict.keys()}

    vals = np.array(list(flatness_dict.values()))
    min_f, max_f = np.min(vals), np.max(vals)
    if max_f == min_f:
        # 曲率全て同じ時は全クラスタ重み1.0
        return {c: 1.0 for c in flatness_dict.keys()}
    if flatness_weighting == 1:
        # 平坦度が大きいほど重み大（平坦部優遇）
        norm = (vals - min_f) / (max_f - min_f)
    elif flatness_weighting == 2:
        # 平坦度が小さいほど重み大（曲面部優遇）
        norm = (max_f - vals) / (max_f - min_f)
    else:
        raise ValueError("flatness_weightingは0（重みなし）, 1（平坦優遇）, 2（曲面優遇）で指定")
    scaled = min_weight + (max_weight - min_weight) * norm
    weights = {c: w for c, w in zip(flatness_dict.keys(), scaled)}
    return weights


def build_graph(xyz, k=6):
    adj = kneighbors_graph(xyz, k, mode='distance', include_self=False) # k最近傍グラフ構築
    W = adj.toarray() # 距離行列に変換
    dists = W[W > 0] # 全エッジ距離抽出
    sigma = np.mean(dists) if len(dists) > 0 else 1.0 # 距離平均を計算
    W[W > 0] = np.exp(-W[W > 0]**2 / (sigma**2)) # 距離重み計算、重み付き隣接行列に変換
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

def embed_watermark_xyz(
    xyz, labels, watermark_bits, beta=0.01,
    split_mode=0, flatness_weighting=0, k_neighbors=20, 
    min_weight=0, max_weight=2.0
):
    """
    各クラスタでwatermark_bitsを
      - split_mode=0: 3チャネル全てに同じ情報（冗長化）
      - split_mode=1: 3分割してx/y/zにそれぞれ独立情報
    としてGFT係数に埋め込む
    """
    xyz_after = xyz.copy()
    cluster_ids = np.unique(labels)
    # color_list = visualize_clusters(xyz, labels)
    flatness_dict = estimate_cluster_flatness(xyz, labels, k_neighbors=k_neighbors)
    weights = compute_cluster_weights(flatness_dict, min_weight, max_weight, flatness_weighting)
    phi = max(
        np.min(xyz[:, 0]) + np.max(xyz[:, 0]),
        np.min(xyz[:, 1]) + np.max(xyz[:, 1]),
        np.min(xyz[:, 2]) + np.max(xyz[:, 2])
    )

    # --- ビット列の用意 ---
    if split_mode == 0:
        # 全チャネル同じ情報
        embed_bits_per_channel = [watermark_bits] * 3
        skip_threshold = len(watermark_bits)/2
    elif split_mode == 1:
        # 3チャネルで異なる情報（3分割）
        embed_bits_per_channel = np.array_split(watermark_bits, 3)
        skip_threshold = len(watermark_bits)/5
    else:
        raise ValueError("split_modeは0（冗長化）か1（3分割）のみ指定可能です")

    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        if len(idx) <= skip_threshold:
                continue  # 点数が少なすぎるクラスタはスキップ
        pts = xyz[idx]
        for channel in range(3):  # 0:x, 1:y, 2:z
            embed_bits = embed_bits_per_channel[channel]
            embed_bits_length = len(embed_bits)
            if embed_bits_length == 0:
                continue  # このチャネルには何も埋め込まない
            signal = pts[:, channel]
            W = build_graph(pts, k=6)
            basis, eigvals = gft_basis(W)
            gft_coeffs = gft(signal, basis)
            Q_ = len(gft_coeffs)
            n_repeat = Q_ // embed_bits_length if embed_bits_length > 0 else 1
            redundant_bits = repeat_bits_blockwise(embed_bits, n_repeat, Q_)
            for i in range(Q_):
                w = redundant_bits[i] * 2 - 1
                gft_coeffs[i] += w * beta * weights[c] * phi
            embed_signal = igft(gft_coeffs, basis)
            xyz_after[idx, channel] = embed_signal
    return xyz_after


def extract_watermark_xyz(
    xyz_emb, xyz_orig, labels, watermark_bits_length, split_mode=0
):
    # -----------------------------------------------
    # split_mode = 0: 冗長化（3チャネルすべてに同じビット列を埋め込んで抽出）
    # -----------------------------------------------
    if split_mode == 0:
        skip_threshold = watermark_bits_length / 2  # 小さすぎるクラスタは除外（閾値）
        bits_from_clusters = [[] for _ in range(watermark_bits_length)]  # 各ビット位置ごとに多数決用リストを用意

        for c in np.unique(labels):  # 各クラスタに対して処理
            idx = np.where(labels == c)[0]
            if len(idx) <= skip_threshold:
                continue  # 小さなクラスタはスキップ

            pts_emb = xyz_emb[idx]  # 埋め込み後のクラスタ点群
            pts_orig = xyz_orig[idx]  # 埋め込み前のクラスタ点群
            if len(pts_emb) != len(pts_orig):
                continue  # 点数が一致しない場合はスキップ

            actual_k = min(6, len(idx) - 1)  # k近傍の設定（クラスタサイズに依存）
            W = build_graph(pts_orig, k=actual_k)  # グラフ構築
            basis, eigvals = gft_basis(W)  # GFT基底を取得
            Q_ = len(basis)
            n_repeat = Q_ // watermark_bits_length if watermark_bits_length > 0 else 1  # 1ビットあたりの繰り返し数

            # チャネルごとに処理
            for channel in range(3):
                gft_coeffs_emb = gft(pts_emb[:, channel], basis)
                gft_coeffs_orig = gft(pts_orig[:, channel], basis)

                bits_from_gftcoeffs = [[] for _ in range(watermark_bits_length)]

                # 各ビットに対応する係数から符号を抽出
                for bit_idx in range(watermark_bits_length):
                    for rep in range(n_repeat):
                        i = bit_idx * n_repeat + rep
                        if i < Q_:
                            diff = gft_coeffs_emb[i] - gft_coeffs_orig[i]
                            bit = 1 if diff > 0 else 0
                            bits_from_gftcoeffs[bit_idx].append(bit)


                # 余剰係数をラウンドロビンで加える
                for i in range(n_repeat * watermark_bits_length, Q_):
                    bit_idx = i % watermark_bits_length
                    diff = gft_coeffs_emb[i] - gft_coeffs_orig[i]
                    bit = 1 if diff > 0 else 0
                    bits_from_gftcoeffs[bit_idx].append(bit)

                # 多数決（GFT係数内の冗長ビット）
                for bit_idx, votes in enumerate(bits_from_gftcoeffs):
                    counts = {0: votes.count(0), 1: votes.count(1)}
                    majority_bit = 1 if counts[1] > counts[0] else 0
                    bits_from_clusters[bit_idx].append(majority_bit)

        # クラスタ間多数決
        extracted_bits = []
        for votes in bits_from_clusters:
            counts = {0: votes.count(0), 1: votes.count(1)}
            extracted_bits.append(1 if counts[1] > counts[0] else 0)

        return extracted_bits

    # -----------------------------------------------
    # split_mode = 1: 3分割（各チャネルに別々のビット列を埋め込んで抽出）
    # -----------------------------------------------
    elif split_mode == 1:
        skip_threshold = watermark_bits_length / 5  # より小さなクラスタも対象

        # ビット長を3チャネルでどう分割したか（例：64bit → [22, 21, 21]）
        split_sizes = [len(arr) for arr in np.array_split(np.zeros(watermark_bits_length), 3)]
        total_bits_len = sum(split_sizes)

        bits_from_clusters = [[] for _ in range(total_bits_len)]  # 結合済みビット列を多数決用に保持

        for c in np.unique(labels):  # 各クラスタに対して処理
            idx = np.where(labels == c)[0]
            if len(idx) <= skip_threshold:
                continue

            pts_emb = xyz_emb[idx]
            pts_orig = xyz_orig[idx]
            if len(pts_emb) != len(pts_orig):
                continue  # 点数不一致はスキップ

            actual_k = min(6, len(idx) - 1)
            W = build_graph(pts_orig, k=actual_k)
            basis, eigvals = gft_basis(W)
            Q_ = len(basis)

            offset = 0

            # チャネルごとに処理
            for channel in range(3):
                bits_len = split_sizes[channel]
                if bits_len == 0:
                    continue

                gft_coeffs_emb = gft(pts_emb[:, channel], basis)
                gft_coeffs_orig = gft(pts_orig[:, channel], basis)

                bits_from_gftcoeffs = [[] for _ in range(bits_len)]
                n_repeat = Q_ // bits_len if bits_len > 0 else 1

                # 各ビットに対応する係数から符号を抽出
                for bit_idx in range(bits_len):
                    for rep in range(n_repeat):
                        i = bit_idx * n_repeat + rep
                        if i < Q_:
                            diff = gft_coeffs_emb[i] - gft_coeffs_orig[i]
                            bit = 1 if diff > 0 else 0
                            bits_from_gftcoeffs[bit_idx].append(bit)

                # 余剰係数をラウンドロビンで加える
                for i in range(n_repeat * bits_len, Q_):
                    bit_idx = i % bits_len
                    diff = gft_coeffs_emb[i] - gft_coeffs_orig[i]
                    bit = 1 if diff > 0 else 0
                    bits_from_gftcoeffs[bit_idx].append(bit)

                # 多数決（GFT係数内の冗長ビット）
                for bit_idx, votes in enumerate(bits_from_gftcoeffs):
                    counts = {0: votes.count(0), 1: votes.count(1)}
                    majority_bit = 1 if counts[1] > counts[0] else 0
                    bits_from_clusters[bit_idx+offset].append(majority_bit)
                
                offset += bits_len

        # クラスタ間多数決
        extracted_bits = []
        for votes in bits_from_clusters:
            counts = {0: votes.count(0), 1: votes.count(1)}
            extracted_bits.append(1 if counts[1] > counts[0] else 0)

        return extracted_bits

    else:
        raise ValueError("split_modeは0（冗長化）か1（3分割）のみ指定可能です")


def embed_watermark_xyz_check(
    xyz, labels, watermark_bits, beta=0.01,
    split_mode=0, flatness_weighting=0, k_neighbors=20,
    min_weight=0, max_weight=2.0,
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
    flatness_dict = estimate_cluster_flatness(xyz, labels, k_neighbors=k_neighbors)
    weights = compute_cluster_weights(flatness_dict, min_weight, max_weight, flatness_weighting)
    phi = max(
        np.min(xyz[:, 0]) + np.max(xyz[:, 0]),
        np.min(xyz[:, 1]) + np.max(xyz[:, 1]),
        np.min(xyz[:, 2]) + np.max(xyz[:, 2])
    )

    # --- 誤り訂正を事前に適用 ---
    if error_correction == "parity":
        watermark_bits = add_parity_code(watermark_bits)
    elif error_correction == "hamming":
        watermark_bits = hamming74_encode(watermark_bits)
    checked_bits_length = len(watermark_bits)

    # --- 分割モードに応じてビット列を準備 ---
    if split_mode == 0:
        embed_bits_per_channel = [watermark_bits] * 3
        skip_threshold = len(watermark_bits) / 2
    elif split_mode == 1:
        embed_bits_per_channel = np.array_split(watermark_bits, 3)
        skip_threshold = len(watermark_bits) / 5
    else:
        raise ValueError("split_modeは0（冗長化）か1（3分割）のみ指定可能です")

    # --- 各クラスタで処理 ---
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

    if split_mode == 0:
        skip_threshold = 600
        bits_from_clusters = [[] for _ in range(watermark_bits_length)]

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

            # --- 各チャネルごとにビット列抽出 ---
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
                majority_bits = []
                for votes in bits_from_gftcoeffs:
                    counts = {0: votes.count(0), 1: votes.count(1)}
                    majority_bits.append(1 if counts[1] > counts[0] else 0)

                # --- エラー訂正チェック ---
                passed = False
                if error_correction == "none":
                    passed = True
                elif error_correction == "parity":
                    passed = check_parity_code(majority_bits)
                    if passed:
                        # パリティ部分を除去（例：9bit中の1bit）
                        block_size = 8
                        clean_bits = []
                        for i in range(0, len(majority_bits), block_size + 1):
                            clean_bits.extend(majority_bits[i:i + block_size])
                        majority_bits = clean_bits[:watermark_bits_length]
                elif error_correction == "hamming":
                    decoded = hamming74_decode(majority_bits)
                    if decoded and len(decoded) >= watermark_bits_length:
                        passed = True
                        majority_bits = decoded[:watermark_bits_length]

                # OKなチャネルだけクラスタ間多数決に参加
                if passed:
                    for bit_idx, b in enumerate(majority_bits):
                        bits_from_clusters[bit_idx].append(b)
                    passed_checks += 1
                total_checks += 1
        if error_correction != "none":
            print(f"[Check] passed channels: {passed_checks} / {total_checks}")

        # クラスタ間多数決
        extracted_bits = []
        for votes in bits_from_clusters:
            counts = {0: votes.count(0), 1: votes.count(1)}
            extracted_bits.append(1 if counts[1] > counts[0] else 0)

        return extracted_bits
    
    elif split_mode == 1:
        skip_threshold = 200
        split_sizes = [len(arr) for arr in np.array_split(np.zeros(checked_bits_length), 3)]

        bits_from_clusters = [[] for _ in range(watermark_bits_length)]

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

            for channel in range(3):
                bits_len = split_sizes[channel]
                if bits_len == 0:
                    continue

                gft_coeffs_emb = gft(pts_emb[:, channel], basis)
                gft_coeffs_orig = gft(pts_orig[:, channel], basis)

                bits_from_gftcoeffs = [[] for _ in range(bits_len)]
                n_repeat = Q_ // bits_len if bits_len > 0 else 1

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

                # GFT係数内で多数決
                bits_majority = []
                for votes in bits_from_gftcoeffs:
                    counts = {0: votes.count(0), 1: votes.count(1)}
                    bits_majority.append(1 if counts[1] > counts[0] else 0)

                cluster_bits.extend(bits_majority)

            # --- エラー訂正チェック ---
            passed = False
            if error_correction == "none":
                passed = True
            elif error_correction == "parity":
                passed = check_parity_code(cluster_bits)
                if passed:
                    clean_bits = []
                    block_size = 8
                    for i in range(0, len(cluster_bits), block_size + 1):
                        clean_bits.extend(cluster_bits[i:i + block_size])
                    cluster_bits = clean_bits[:watermark_bits_length]
            elif error_correction == "hamming":
                decoded = hamming74_decode(cluster_bits)
                if decoded and len(decoded) >= watermark_bits_length:
                    passed = True
                    cluster_bits = decoded[:watermark_bits_length]

            if passed:
                for i, b in enumerate(cluster_bits):
                    bits_from_clusters[i].append(b)
                passed_checks += 1
            total_checks += 1
        if error_correction != "none":
            print(f"[Check] passed clusters: {passed_checks} / {total_checks}")

        # --- クラスタ間多数決 ---
        extracted_bits = []
        for votes in bits_from_clusters:
            counts = {0: votes.count(0), 1: votes.count(1)}
            extracted_bits.append(1 if counts[1] > counts[0] else 0)

        return extracted_bits
    
    else:
        raise ValueError("split_modeは0（冗長化）か1（3分割）のみ指定可能です")

    
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
    偶数パリティ符号を検証。全ブロックのパリティが正しいかを返す
    """
    for i in range(0, len(bits), block_size + 1):
        block = bits[i:i + block_size + 1]
        if len(block) < block_size + 1:
            return False
        data, parity = block[:-1], block[-1]
        if sum(data) % 2 != parity:
            return False
    return True

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
    ハミング(7,4)復号：誤り1bitを訂正し、4bitデータを返す
    エラーが訂正不可能な場合（重複エラーなど）は None を返す
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
            return None
        syndrome = [sum([block[j] * H[k][j] for j in range(7)]) % 2 for k in range(3)]
        syndrome_value = syndrome[0]*4 + syndrome[1]*2 + syndrome[2]*1
        if syndrome_value != 0:
            if 1 <= syndrome_value <= 7:
                block[syndrome_value - 1] ^= 1
            else:
                return None  # 不正なエラー位置
        decoded.extend(block[:4])
    return decoded



######################################## 評価用 ##########################################################

def calc_psnr_xyz(pcd_before, pcd_after, reverse=False):
    """
    最近傍点対応で点群PSNRを計算する（order-free）
    - pcd_before, pcd_after: open3d.geometry.PointCloud
    - max_range: PSNRの分母に使うスケール（未指定なら点群全体の最大距離幅）
    - reverse: Trueならafter→beforeも評価し、両方向平均
    """
    points_before = np.asarray(pcd_before.points)
    points_after = np.asarray(pcd_after.points)
    
    # before→after最近傍
    tree = cKDTree(points_after)
    dists, _ = tree.query(points_before, k=1)
    mse_fwd = np.mean(dists ** 2)
    
    if reverse:
        # after→beforeも計算して平均
        tree_rev = cKDTree(points_before)
        dists_rev, _ = tree_rev.query(points_after, k=1)
        mse_rev = np.mean(dists_rev ** 2)
        mse = (mse_fwd + mse_rev) / 2
    else:
        mse = mse_fwd

    # スケール計算
    xyz = np.asarray(pcd_before.points)
    max_range = max(
            np.max(xyz[:,0]) - np.min(xyz[:,0]),
            np.max(xyz[:,1]) - np.min(xyz[:,1]),
            np.max(xyz[:,2]) - np.min(xyz[:,2])
        )
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((max_range ** 2) / mse)
    
    print("------------------- 評価 -------------------")
    print(f"MSE: {mse:.6f}")
    print(f"PSNR: {psnr:.2f} dB (max_range={max_range:.4f})")
    return psnr

def add_noise(xyz, noise_percent=0.01, mode='uniform', seed=None, verbose=True):
    """
    numpy配列(xyz)にノイズを加える
    - noise_percent: ノイズ振幅（座標値最大幅の割合, 例: 0.01 = 1%）
    - mode: 'uniform'または'gaussian'
    - return: ノイズ加算後のnumpy配列
    """
    rng = np.random.RandomState(seed)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    ranges = xyz_max - xyz_min
    scale = ranges * noise_percent
    if verbose:
        print(f"ノイズ振幅: {noise_percent*100:.2f}% (scale={scale})")
    if mode == 'uniform':
        noise = rng.uniform(low=-scale, high=scale, size=xyz.shape)
    elif mode == 'gaussian':
        noise = rng.normal(loc=0.0, scale=scale/2, size=xyz.shape)
    else:
        raise ValueError('modeは "uniform" か "gaussian"')
    xyz_noisy = xyz + noise
    return xyz_noisy

def crop_point_cloud_xyz(xyz_after, crop_ratio=0.5, mode='center', verbose=True):
    """
    xyz_after に対して切り取り攻撃を行い、一部の点群のみを残し、表示する。

    Parameters:
    - xyz_after (np.ndarray): 埋め込み後の点群座標（N×3）
    - crop_ratio (float): 残す点の割合（0.0～1.0]
    - mode (str): 'center'（中心部を残す）または 'edge'（端部を残す）
    - verbose (bool): 情報表示の有無

    Returns:
    - xyz_cropped (np.ndarray): 切り取り後の点群座標
    """
    import open3d as o3d
    assert 0.0 < crop_ratio <= 1.0, "crop_ratioは (0, 1] で指定してください"
    N = xyz_after.shape[0]
    keep_n = int(N * crop_ratio)

    center = np.mean(xyz_after, axis=0)
    dists = np.linalg.norm(xyz_after - center, axis=1)

    if mode == 'center':
        keep_indices = np.argsort(dists)[:keep_n]
    elif mode == 'edge':
        keep_indices = np.argsort(dists)[-keep_n:]
    else:
        raise ValueError("modeは 'center' または 'edge' を指定してください")

    xyz_cropped = xyz_after[keep_indices]

    if verbose:
        print(f"切り取り攻撃 ({mode}): 元点数={N} → 残点数={keep_n} ({crop_ratio*100:.1f}%)")

    # 可視化
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(xyz_cropped)
    cropped_pcd.paint_uniform_color([1, 0.6, 0])  # オレンジ系で表示
    o3d.visualization.draw_geometries([cropped_pcd], window_name="Cropped Point Cloud")

    return xyz_cropped


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
        print(f"[Reordering] 順序を xyz_orig に再整列しました。")

    return xyz_reordered


##################################### 参考用 ##########################################

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

def embed_watermark_rgb(colors, xyz, labels, embed_bits, channel=0, beta=0.01):
    """
    各クラスタで全embed_bitsを指定チャネル(R,G,B)のGFT低次成分に冗長埋め込み
    channel: 0=R, 1=G, 2=B
    """
    colors_after = colors.copy()
    embed_bits_length = len(embed_bits)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        pts = xyz[idx]
        signal = colors[idx, channel]
        W = build_graph(pts, k=6)
        basis, eigvals = gft_basis(W)
        gft_coeffs = gft(signal, basis)
        # 各ビットを対応する低次GFT係数に埋め込む
        for bit_idx in range(min(embed_bits_length, len(gft_coeffs))):
            w = embed_bits[bit_idx] * 2 - 1  # 0→-1, 1→+1
            gft_coeffs[bit_idx] += beta * w
        embed_signal = igft(gft_coeffs, basis)
        colors_after[idx, channel] = np.clip(embed_signal, 0.0, 1.0)
    return colors_after

def extract_watermark_rgb(colors_emb, colors_orig, xyz, labels, embed_bits_length, channel=0):
    """
    各クラスタで埋め込まれた全ビットを抽出し、多数決で最終ビット列を復元（RGB）
    channel: 0=R, 1=G, 2=B
    """
    bit_lists = [[] for _ in range(embed_bits_length)]
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        pts = xyz[idx]
        signal_emb = colors_emb[idx, channel]
        signal_orig = colors_orig[idx, channel]
        W = build_graph(pts, k=6)
        basis, eigvals = gft_basis(W)
        gft_coeffs_emb = gft(signal_emb, basis)
        gft_coeffs_orig = gft(signal_orig, basis)
        for bit_idx in range(min(embed_bits_length, len(gft_coeffs_emb))):
            diff = gft_coeffs_emb[bit_idx] - gft_coeffs_orig[bit_idx]
            bit = 1 if diff > 0 else 0
            bit_lists[bit_idx].append(bit)
    # 各ビットごとに多数決
    extracted_bits = []
    for bits in bit_lists:
        counts = {0: bits.count(0), 1: bits.count(1)}
        extracted_bit = 1 if counts[1] > counts[0] else 0
        extracted_bits.append(extracted_bit)
    return extracted_bits

def embed_cluster_channel(args):
    c, xyz, labels, embed_bits, beta, weights, phi, k_neighbors, flatness_dict = args
    idx = np.where(labels == c)[0]
    pts = xyz[idx]
    cluster_xyz_after = np.copy(pts)
    for channel in range(3):  # x=0, y=1, z=2
        signal = pts[:, channel]
        W = build_graph(pts, k=6)
        basis, eigvals = gft_basis(W)
        gft_coeffs = gft(signal, basis)
        Q_ = len(gft_coeffs)
        n_repeat = Q_ // len(embed_bits) if len(embed_bits) > 0 else 1
        redundant_bits = repeat_bits_blockwise(embed_bits, n_repeat, Q_)
        for i in range(Q_):
            w = redundant_bits[i] * 2 - 1
            gft_coeffs[i] += w * beta * weights[c] * phi
        embed_signal = igft(gft_coeffs, basis)
        cluster_xyz_after[:, channel] = embed_signal
    return c, cluster_xyz_after

def embed_watermark_xyz_parallel(
    xyz, labels, embed_bits, beta=0.01,
    flatness_weighting=0, k_neighbors=20, min_weight=0.2, max_weight=1.8
):
    """
    並列処理により高速化させたバージョン。
    """
    xyz_after = xyz.copy()
    embed_bits_length = len(embed_bits)
    cluster_ids = np.unique(labels)
    color_list = visualize_clusters(xyz, labels)
    # クラスタごと平坦度計算＆重み
    flatness_dict = estimate_cluster_flatness(xyz, labels, k_neighbors=k_neighbors)
    weights = compute_cluster_weights(flatness_dict, min_weight, max_weight, flatness_weighting)
    phi = max(
        np.min(xyz[:,0]) + np.max(xyz[:,0]),
        np.min(xyz[:,1]) + np.max(xyz[:,1]),
        np.min(xyz[:,2]) + np.max(xyz[:,2])
    )
    print("\n------------ クラスタごとの重みリスト ------------")
    for c in cluster_ids:
        print(f"クラスタ{c}: 色={color_list[c]}  平坦度={flatness_dict[c]:.3e}  平坦重み={weights[c]:.3f}  重み={weights[c]*phi*beta:.3e}")

    # 並列実行の準備
    args_list = [
        (c, xyz, labels, embed_bits, beta, weights, phi, k_neighbors, flatness_dict)
        for c in cluster_ids
    ]
    with ProcessPoolExecutor() as executor:
        results = executor.map(embed_cluster_channel, args_list)

    # 各クラスタの埋め込み後xyzを集約
    for c, cluster_xyz_after in results:
        idx = np.where(labels == c)[0]
        xyz_after[idx] = cluster_xyz_after
    return xyz_after

def extract_cluster_bits(args):
    c, xyz_emb, xyz_orig, labels, embed_bits_length = args
    idx = np.where(labels == c)[0]
    pts_emb = xyz_emb[idx]
    pts_orig = xyz_orig[idx]
    W = build_graph(pts_orig, k=6)
    basis, eigvals = gft_basis(W)
    Q_ = len(basis)
    bit_lists = [[] for _ in range(embed_bits_length)]
    for channel in range(3):
        gft_coeffs_emb = gft(pts_emb[:, channel], basis)
        gft_coeffs_orig = gft(pts_orig[:, channel], basis)
        n_repeat = Q_ // embed_bits_length if embed_bits_length > 0 else 1
        for bit_idx in range(embed_bits_length):
            for rep in range(n_repeat):
                i = bit_idx * n_repeat + rep
                if i < Q_:
                    diff = gft_coeffs_emb[i] - gft_coeffs_orig[i]
                    bit = 1 if diff > 0 else 0
                    bit_lists[bit_idx].append(bit)
        for i in range(n_repeat * embed_bits_length, Q_):
            bit_idx = i % embed_bits_length
            diff = gft_coeffs_emb[i] - gft_coeffs_orig[i]
            bit = 1 if diff > 0 else 0
            bit_lists[bit_idx].append(bit)
    return bit_lists

def extract_watermark_xyz_parallel(xyz_emb, xyz_orig, labels, embed_bits_length, max_workers=2):
    """
    並列版：3チャネル(x,y,z)・全クラスタ・全冗長分のビットを合体し、ビットごとに最頻値多数決
    max_workers: 使用CPUコア数（PCが重くなりすぎないように2〜4推奨）
    """
    bit_lists = [[] for _ in range(embed_bits_length)]
    args_list = [(c, xyz_emb, xyz_orig, labels, embed_bits_length) for c in np.unique(labels)]
    # 並列実行
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(extract_cluster_bits, args_list)
    # 各クラスタごとの結果を合体
    for cluster_bits in results:
        for bit_idx in range(embed_bits_length):
            bit_lists[bit_idx].extend(cluster_bits[bit_idx])
    # 最頻値多数決で最終ビット列
    extracted_bits = []
    for bits in bit_lists:
        counts = {0: bits.count(0), 1: bits.count(1)}
        extracted_bit = 1 if counts[1] > counts[0] else 0
        extracted_bits.append(extracted_bit)
    return extracted_bits