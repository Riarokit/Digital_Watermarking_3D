import numpy as np
import open3d as o3d
import random
import string
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

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


def kmeans_cluster_points(xyz, num_clusters=8, seed=42):
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
    return kmeans.fit_predict(xyz)

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

def embed_watermark_xyz(xyz, labels, embed_bits, channel=0, beta=0.001):
    """
    各クラスタで全embed_bitsを冗長blockwiseでGFT係数に埋め込む（x/y/z指定可）
    """
    xyz_after = xyz.copy()
    n_bits = len(embed_bits)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        pts = xyz[idx]
        signal = pts[:, channel]
        W = build_graph(pts, k=6)
        basis, eigvals = gft_basis(W)
        gft_coeffs = gft(signal, basis)
        # phi = np.max(signal) + np.min(signal)
        Q_ = len(gft_coeffs)
        n_repeat = Q_ // n_bits if n_bits > 0 else 1
        # 論文式のblockwise冗長化
        redundant_bits = repeat_bits_blockwise(embed_bits, n_repeat, Q_)
        for i in range(Q_):
            w = redundant_bits[i] * 2 - 1
            gft_coeffs[i] += beta * w
        embed_signal = igft(gft_coeffs, basis)
        xyz_after[idx, channel] = embed_signal
    return xyz_after

def extract_watermark_xyz(xyz_emb, xyz_orig, labels, embed_bits_length, channel=0):
    """
    冗長化埋め込みに対応した抽出
    """
    bit_lists = [[] for _ in range(embed_bits_length)]
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        pts_emb = xyz_emb[idx]
        pts_orig = xyz_orig[idx]
        W = build_graph(pts_orig, k=6)
        basis, eigvals = gft_basis(W)
        gft_coeffs_emb = gft(pts_emb[:, channel], basis)
        gft_coeffs_orig = gft(pts_orig[:, channel], basis)
        Q_ = len(gft_coeffs_emb)
        n_repeat = Q_ // embed_bits_length if embed_bits_length > 0 else 1
        # blockwise分割でグループ化
        for bit_idx in range(embed_bits_length):
            for rep in range(n_repeat):
                i = bit_idx * n_repeat + rep
                if i < Q_:
                    diff = gft_coeffs_emb[i] - gft_coeffs_orig[i]
                    bit = 1 if diff > 0 else 0
                    bit_lists[bit_idx].append(bit)
        # 余り係数（Q_ % embed_bits_length）も対応
        for i in range(n_repeat * embed_bits_length, Q_):
            bit_idx = i % embed_bits_length
            diff = gft_coeffs_emb[i] - gft_coeffs_orig[i]
            bit = 1 if diff > 0 else 0
            bit_lists[bit_idx].append(bit)
    # 多数決で復元
    extracted_bits = []
    for bits in bit_lists:
        counts = {0: bits.count(0), 1: bits.count(1)}
        extracted_bit = 1 if counts[1] > counts[0] else 0
        extracted_bits.append(extracted_bit)
    return extracted_bits


# def embed_watermark_xyz(xyz, labels, embed_bits, channel=0, beta=0.001):
#     """
#     各クラスタで全embed_bitsをGFT低次成分に埋め込む（x/y/zを選択可能）
#     channel: 0=x, 1=y, 2=z
#     """
#     xyz_after = xyz.copy()
#     embed_bits_length = len(embed_bits)
#     for c in np.unique(labels):
#         idx = np.where(labels == c)[0]
#         pts = xyz[idx]
#         signal = pts[:, channel]
#         W = build_graph(pts, k=6)
#         basis, eigvals = gft_basis(W)
#         gft_coeffs = gft(signal, basis)
#         # print(f"クラスタ{c}: basis[0,:5]={basis[0,:5]}")
#         # print(f"クラスタ{c}: gft_coeffs[:5]={gft_coeffs[:5]}")
#         # phi = np.max(signal) + np.min(signal)
#         for bit_idx in range(min(embed_bits_length, len(gft_coeffs))):
#             w = embed_bits[bit_idx] * 2 - 1 # 0→-1, 1→+1
#             gft_coeffs[bit_idx] += beta * w
#         embed_signal = igft(gft_coeffs, basis)
#         xyz_after[idx, channel] = embed_signal
#     return xyz_after

# def extract_watermark_xyz(xyz_emb, xyz_orig, labels, embed_bits_length, channel=0):
#     bit_lists = [[] for _ in range(embed_bits_length)]
#     for c in np.unique(labels):
#         idx = np.where(labels == c)[0]
#         pts_emb = xyz_emb[idx]
#         pts_orig = xyz_orig[idx]
#         W = build_graph(pts_orig, k=6)
#         basis, eigvals = gft_basis(W)
#         gft_coeffs_emb = gft(pts_emb[:, channel], basis)
#         gft_coeffs_orig = gft(pts_orig[:, channel], basis)
#         # print(f"クラスタ{c}: basis_extract[0,:5]={basis[0,:5]}")
#         # print(f"クラスタ{c}: gft_coeffs_emb[:5]={gft_coeffs_emb[:5]}")
#         # print(f"クラスタ{c}: gft_coeffs_orig[:5]={gft_coeffs_orig[:5]}")
#         for bit_idx in range(min(embed_bits_length, len(gft_coeffs_emb))):
#             diff = gft_coeffs_emb[bit_idx] - gft_coeffs_orig[bit_idx]
#             bit = 1 if diff > 0 else 0
#             bit_lists[bit_idx].append(bit)
#     extracted_bits = []
#     for bits in bit_lists:
#         counts = {0: bits.count(0), 1: bits.count(1)}
#         extracted_bit = 1 if counts[1] > counts[0] else 0
#         extracted_bits.append(extracted_bit)
#     return extracted_bits

def calc_psnr_xyz(pcd_before, pcd_after, max_range=None, verbose=True, reverse=False):
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

    if max_range is None:
        # 2点群の範囲の最大幅
        all_points = np.vstack([points_before, points_after])
        max_range = np.max(all_points) - np.min(all_points)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((max_range ** 2) / mse)
    
    if verbose:
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
