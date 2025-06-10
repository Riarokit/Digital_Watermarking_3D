import numpy as np
import open3d as o3d
import random
import string
from sklearn.neighbors import kneighbors_graph

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

def sort_and_shuffle_points(xyz, seed=42):
    norms = np.linalg.norm(xyz, axis=1)
    sort_idx = np.argsort(-norms)  # 降順
    xyz_sorted = xyz[sort_idx]
    rng = np.random.RandomState(seed)
    shuffle_idx = np.arange(xyz_sorted.shape[0])
    rng.shuffle(shuffle_idx)
    xyz_shuffled = xyz_sorted[shuffle_idx]
    return xyz_shuffled, sort_idx, shuffle_idx


def cluster_points(xyz, num_clusters=8, points_per_cluster=300, seed=42):
    rng = np.random.RandomState(seed)
    N = xyz.shape[0]
    idx_all = np.arange(N)
    cluster_labels = np.full(N, -1)
    seeds = rng.choice(idx_all, size=num_clusters, replace=False)
    clusters = {i: [s] for i, s in enumerate(seeds)}
    assigned = set(seeds)
    for c in clusters:
        while len(clusters[c]) < points_per_cluster:
            last = clusters[c][-1]
            dists = np.linalg.norm(xyz - xyz[last], axis=1)
            mask = np.array([i not in assigned for i in idx_all])
            if not mask.any():
                break
            next_idx = idx_all[mask][np.argmin(dists[mask])]
            clusters[c].append(next_idx)
            assigned.add(next_idx)
    # クラスタごとの点インデックスリスト→ラベル配列
    for c, idxs in clusters.items():
        for idx in idxs:
            cluster_labels[idx] = c
    # 未割り当て点は強制的にどれかに入れる
    for idx in np.where(cluster_labels == -1)[0]:
        dists = [np.linalg.norm(xyz[idx] - xyz[clusters[c][0]]) for c in clusters]
        c_nearest = np.argmin(dists)
        cluster_labels[idx] = c_nearest
        clusters[c_nearest].append(idx)
    return cluster_labels

def build_graph(xyz, k=6):
    adj = kneighbors_graph(xyz, k, mode='distance', include_self=False)
    W = adj.toarray()
    dists = W[W > 0]
    sigma = np.mean(dists) if len(dists) > 0 else 1.0
    W[W > 0] = np.exp(-W[W > 0]**2 / (sigma**2))
    return W

def gft_basis(W):
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvecs, eigvals

def gft(signal, basis):
    return basis.T @ signal

def igft(gft_coeffs, basis):
    return basis @ gft_coeffs

def embed_watermark(xyz, labels, watermark_bits, coeff_idx=0, alpha=0.001, seed=42):
    """x座標で電子透かしを埋め込む"""
    xyz_after = xyz.copy()
    bit_idx = 0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        pts = xyz[idx]
        x_signal = pts[:, 0]
        W = build_graph(pts, k=6)
        basis, eigvals = gft_basis(W)
        gft_coeffs = gft(x_signal, basis)
        if bit_idx < len(watermark_bits):
            w = watermark_bits[bit_idx]*2 - 1  # 0→-1, 1→+1
            gft_coeffs[coeff_idx] += alpha * w
            bit_idx += 1
        x_embed = igft(gft_coeffs, basis)
        xyz_after[idx, 0] = x_embed
    return xyz_after

def extract_watermark(xyz_emb, xyz_orig, labels, coeff_idx=0):
    """埋め込まれた点群（xyz_emb）と元点群（xyz_orig）からビット抽出"""
    extracted_bits = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        pts_emb = xyz_emb[idx]
        pts_orig = xyz_orig[idx]
        W = build_graph(pts_emb, k=6)
        basis, eigvals = gft_basis(W)
        gft_coeffs_emb = gft(pts_emb[:, 0], basis)
        gft_coeffs_orig = gft(pts_orig[:, 0], basis)
        diff = gft_coeffs_emb[coeff_idx] - gft_coeffs_orig[coeff_idx]
        bit = 1 if diff > 0 else 0
        extracted_bits.append(bit)
    return extracted_bits
