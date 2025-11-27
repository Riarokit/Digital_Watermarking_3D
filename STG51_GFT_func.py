import numpy as np
import open3d as o3d
import random
import string
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import cKDTree
from PIL import Image

# ==========================================
# ユーティリティ
# ==========================================

def image_to_bitarray(image_path, n=32):
    """画像ファイルをn×nの2値ビット配列に変換"""
    img = Image.open(image_path).convert('L')
    img = img.resize((n, n), Image.LANCZOS)
    arr = np.array(img)
    arr = (arr > 127).astype(np.uint8)
    return arr.flatten().tolist()

def bitarray_to_image(bitarray, n=32, save_path=None):
    """1次元ビット配列をn×n画像に復元"""
    arr = np.array(bitarray, dtype=np.uint8).reshape((n, n)) * 255
    img = Image.fromarray(arr, mode='L')
    if save_path:
        img.save(save_path)
    return img

def add_colors(pcd_before, color="grad"):
    """可視化用に色付け"""
    points = np.asarray(pcd_before.points)
    if color == "grad":
        x_val = points[:, 0]
        y_val = points[:, 1]
        z_val = points[:, 2]
        colors = np.zeros_like(points)
        # 正規化してRGBに割り当て
        colors[:, 0] = (x_val - x_val.min()) / (x_val.max() - x_val.min() + 1e-8)
        colors[:, 1] = (y_val - y_val.min()) / (y_val.max() - y_val.min() + 1e-8)
        colors[:, 2] = (z_val - z_val.min()) / (z_val.max() - z_val.min() + 1e-8)
    else:
        colors = np.zeros_like(points)
    pcd_before.colors = o3d.utility.Vector3dVector(colors)
    return pcd_before

# ==========================================
# クラスタリング
# ==========================================

def voxel_grid_clustering(xyz, grid_size=1.0, guard_band=0.05):
    """
    ボクセルグリッドによる決定論的クラスタリング
    
    Parameters:
    - grid_size: ボクセルの一辺の長さ L
    - guard_band: 境界からの禁止領域の幅 epsilon
    
    Returns:
    - labels: 各点のクラスタID (境界付近の点は -1 となる)
    """
    # 1. 各点がどのボクセルインデックスに属するか計算
    #    (床関数をとって整数化)
    voxel_indices = np.floor(xyz / grid_size).astype(int)
    
    # 2. ボクセル内での局所座標 (0.0 ~ grid_size)
    local_coords = xyz - (voxel_indices * grid_size)
    
    # 3. ガードバンド判定
    #    境界 (0とgrid_size) から guard_band 以内にある場合は False (使用しない)
    #    x, y, z 全てで安全圏にある点のみ True
    is_safe = np.all(
        (local_coords >= guard_band) & (local_coords <= (grid_size - guard_band)),
        axis=1
    )
    
    # 4. ラベル生成
    #    ボクセルインデックス (ix, iy, iz) をユニークなハッシュ値(ラベル)に変換
    #    ここでは単純化のため、タプルとして扱い、ユニークIDを割り振る
    
    # 安全な点のみを取り出してユニークID作成
    safe_indices = voxel_indices[is_safe]
    if len(safe_indices) == 0:
        print("[Voxel] 有効な点がありません。grid_sizeを大きくするかguard_bandを小さくしてください。")
        return np.full(len(xyz), -1)

    # ユニークなボクセルを行ごとに抽出
    unique_voxels, inverse_indices = np.unique(safe_indices, axis=0, return_inverse=True)
    
    # 全体のラベル配列 (-1で初期化)
    labels = np.full(len(xyz), -1, dtype=int)
    
    # 安全な点にのみ、0から始まるクラスタIDを付与
    labels[is_safe] = inverse_indices
    
    print(f"[Voxel] Grid: {grid_size}, Guard: {guard_band}")
    print(f"[Voxel] Total Points: {len(xyz)}, Safe Points: {np.sum(is_safe)} ({np.sum(is_safe)/len(xyz):.1%})")
    print(f"[Voxel] Clusters: {len(unique_voxels)}")
    
    return labels

# ==========================================
# グラフ構築・GFT
# ==========================================

def build_graph(xyz, k=6):
    # 安全点のみで構成されたxyzが渡される前提
    adj = kneighbors_graph(xyz, k, mode='distance', include_self=False)
    W = adj.toarray()
    dists = W[W > 0]
    sigma = np.mean(dists) if len(dists) > 0 else 1.0
    # ガウスカーネル重み
    W[W > 0] = np.exp(-W[W > 0]**2 / (sigma**2))
    return W

def gft_basis(W):
    # 次数行列
    D = np.diag(W.sum(axis=1))
    # ラプラシアン行列
    L = D - W
    # 固有値分解
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvecs, eigvals

def gft(signal, basis):
    return basis.T @ signal

def igft(gft_coeffs, basis):
    return basis @ gft_coeffs

# ==========================================
# QIM (量子化インデックス変調) ロジック
# ==========================================

def qim_embed_scalar(val, bit, delta):
    """
    1つの値をQIMで埋め込み変調する
    - bit 0: 偶数グリッド (0, delta, 2delta...) に丸める
    - bit 1: 奇数グリッド (0.5delta, 1.5delta...) に丸める
    """
    # ステップ幅 delta で正規化
    scaled = val / delta
    
    if bit == 0:
        # 最寄りの整数(偶数グリッド相当)に丸める
        embedded_scaled = np.round(scaled)
    else:
        # 最寄りのX.5(奇数グリッド相当)に丸める
        embedded_scaled = np.floor(scaled) + 0.5 if (scaled - np.floor(scaled)) < 0.5 else np.ceil(scaled) - 0.5
        # 別解: np.round(scaled - 0.5) + 0.5 でも可
    
    return embedded_scaled * delta

def qim_extract_scalar(val, delta):
    """
    QIMで埋め込まれた値からビットを判定
    """
    scaled = val / delta
    # 偶数グリッド(整数)までの距離
    dist_0 = np.abs(scaled - np.round(scaled))
    # 奇数グリッド(X.5)までの距離
    dist_1 = np.abs(scaled - (np.round(scaled - 0.5) + 0.5))
    
    return 0 if dist_0 < dist_1 else 1

# ==========================================
# 埋め込み・抽出 (メイン処理)
# ==========================================

def embed_watermark_qim(
    xyz, labels, embed_bits, 
    delta=0.01,  # QIMのステップ幅 (これが強さと画質を決める)
    k_neighbors=6,
    min_spectre=0.0, max_spectre=1.0
):
    """
    QIMを用いたブラインド透かし埋め込み
    """
    xyz_after = xyz.copy()
    cluster_ids = np.unique(labels)
    cluster_ids = cluster_ids[cluster_ids != -1] # -1(ガードバンド)は除外

    embed_len = len(embed_bits)
    
    # 各クラスタで処理
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        # 点数が少なすぎる場合はグラフが作れないのでスキップ
        if len(idx) <= k_neighbors + 1:
            continue

        pts = xyz[idx]
        
        # 1. グラフ構築 & GFT
        W = build_graph(pts, k=k_neighbors)
        basis, eigvals = gft_basis(W)
        
        # 2. 埋め込み (3チャネルそれぞれに埋め込み)
        #    ここでは簡単のため、全チャネルに同じビットを繰り返し埋め込む(冗長化)
        for ch in range(3):
            signal = pts[:, ch]
            coeffs = gft(signal, basis)
            
            Q_ = len(coeffs)
            q_start = int(Q_ * min_spectre)
            q_end   = int(Q_ * max_spectre)
            
            # 埋め込み可能な係数の数
            target_coeffs_len = q_end - q_start
            if target_coeffs_len <= 0:
                continue
                
            # ビット列を繰り返して係数長に合わせる
            # [b0, b1, ..., bN, b0, b1...] の順
            current_bit_idx = 0
            
            for i in range(target_coeffs_len):
                coeff_idx = q_start + i
                bit = embed_bits[current_bit_idx]
                
                # QIM変調
                coeffs[coeff_idx] = qim_embed_scalar(coeffs[coeff_idx], bit, delta)
                
                current_bit_idx = (current_bit_idx + 1) % embed_len
            
            # IGFTで座標に戻す
            xyz_after[idx, ch] = igft(coeffs, basis)
            
    return xyz_after

def extract_watermark_qim(
    xyz_target, labels, embed_len,
    delta=0.01,
    k_neighbors=6,
    min_spectre=0.0, max_spectre=1.0
):
    """
    QIMを用いたブラインド透かし抽出
    """
    cluster_ids = np.unique(labels)
    cluster_ids = cluster_ids[cluster_ids != -1]

    # ビットごとの投票箱 [ [bit0の投票リスト], [bit1の投票リスト], ... ]
    votes_per_bit = [[] for _ in range(embed_len)]
    
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        if len(idx) <= k_neighbors + 1:
            continue
            
        pts = xyz_target[idx]
        
        # 1. グラフ構築 & GFT (埋め込み時と同じロジック)
        #    ※座標が微小変化していても、トポロジー(k近傍)が変わらなければ
        #      基底はほぼ同じになるはず、という前提
        W = build_graph(pts, k=k_neighbors)
        basis, eigvals = gft_basis(W)
        
        # 2. 抽出
        for ch in range(3):
            signal = pts[:, ch]
            coeffs = gft(signal, basis)
            
            Q_ = len(coeffs)
            q_start = int(Q_ * min_spectre)
            q_end   = int(Q_ * max_spectre)
            target_coeffs_len = q_end - q_start
            
            current_bit_idx = 0
            for i in range(target_coeffs_len):
                coeff_idx = q_start + i
                
                # QIM復号
                detected_bit = qim_extract_scalar(coeffs[coeff_idx], delta)
                
                votes_per_bit[current_bit_idx].append(detected_bit)
                current_bit_idx = (current_bit_idx + 1) % embed_len

    # 多数決でビット確定
    extracted_bits = []
    for votes in votes_per_bit:
        if len(votes) == 0:
            extracted_bits.append(0) # 投票なし
        else:
            count0 = votes.count(0)
            count1 = votes.count(1)
            extracted_bits.append(1 if count1 > count0 else 0)
            
    return extracted_bits

# ==========================================
# 評価・攻撃シミュレーション
# ==========================================

def calc_psnr_xyz(pcd_before, pcd_after):
    points_before = np.asarray(pcd_before.points)
    points_after = np.asarray(pcd_after.points)
    
    # 最近傍で対応付けして誤差計算
    tree = cKDTree(points_after)
    dists, _ = tree.query(points_before, k=1)
    mse = np.mean(dists ** 2)
    
    xyz = points_before
    max_range = max(
        np.max(xyz[:,0]) - np.min(xyz[:,0]),
        np.max(xyz[:,1]) - np.min(xyz[:,1]),
        np.max(xyz[:,2]) - np.min(xyz[:,2])
    )
    
    psnr = 10 * np.log10((max_range ** 2) / mse) if mse > 0 else float('inf')
    print(f"[Metric] MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")
    return psnr

def evaluate_watermark(original_bits, extracted_bits):
    arr_org = np.array(original_bits)
    arr_ext = np.array(extracted_bits)
    ber = np.mean(arr_org != arr_ext)
    print(f"[Metric] BER (Bit Error Rate): {ber:.4f}")
    return ber

def add_noise(xyz, noise_std=0.001):
    """ガウシアンノイズ付加"""
    noise = np.random.normal(0, noise_std, xyz.shape)
    return xyz + noise

def crop_point_cloud(xyz, keep_ratio=0.5):
    """単純なランダムクロップ"""
    N = len(xyz)
    keep_n = int(N * keep_ratio)
    indices = np.random.choice(N, keep_n, replace=False)
    return xyz[indices]