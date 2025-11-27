import numpy as np
import open3d as o3d
import random
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
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

def normalize_point_cloud(pcd, target_scale=100.0, verbose=True, visualize=False):
    """
    点群を原点中心に移動し、バウンディングボックスの最大幅が target_scale になるように正規化する。
    """
    xyz = np.asarray(pcd.points)
    
    # 1. 中心化 (Centering)
    centroid = np.mean(xyz, axis=0)
    xyz -= centroid
    
    # 2. スケーリング (Scaling)
    bbox_size = np.max(xyz, axis=0) - np.min(xyz, axis=0)
    max_width = np.max(bbox_size)
    
    if max_width > 0:
        scale_factor = target_scale / max_width
        xyz *= scale_factor
    else:
        scale_factor = 1.0
        print("[Warning] 点群の広がりが0です。スケーリングをスキップしました。")

    # 3. Open3Dオブジェクトへの反映
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    if verbose:
        new_bbox_size = np.max(xyz, axis=0) - np.min(xyz, axis=0)
        print(f"[Normalize] BBox Size: {new_bbox_size}")
        
    if visualize:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=target_scale * 0.2, origin=[0, 0, 0]
        )
        window_name = f"Normalized PC (MaxWidth={target_scale})"
        o3d.visualization.draw_geometries([pcd, coord_frame], window_name=window_name)
        
    return pcd, xyz

# ==========================================
# クラスタリング
# ==========================================

def voxel_grid_clustering(xyz, grid_size=1.0, guard_band=0.05, visualize=False):
    """
    ボクセルグリッドによる決定論的クラスタリング
    """
    # 1. 各点がどのボクセルインデックスに属するか計算
    voxel_indices = np.floor(xyz / grid_size).astype(int)
    
    # 2. ボクセル内での局所座標 (0.0 ~ grid_size)
    local_coords = xyz - (voxel_indices * grid_size)
    
    # 3. ガードバンド判定
    is_safe = np.all(
        (local_coords >= guard_band) & (local_coords <= (grid_size - guard_band)),
        axis=1
    )
    
    # 4. ラベル生成
    safe_indices = voxel_indices[is_safe]
    
    if len(safe_indices) == 0:
        print("[Voxel] 有効な点がありません。grid_sizeを大きくするかguard_bandを小さくしてください。")
        return np.full(len(xyz), -1)

    unique_voxels, inverse_indices = np.unique(safe_indices, axis=0, return_inverse=True)
    
    # 全体のラベル配列 (-1で初期化)
    labels = np.full(len(xyz), -1, dtype=int)
    labels[is_safe] = inverse_indices
    
    # --- 統計情報の表示 ---
    n_safe = np.sum(is_safe)
    n_clusters = len(unique_voxels)
    
    if n_clusters > 0:
        _, counts = np.unique(labels[is_safe], return_counts=True)
        avg_points = np.mean(counts)
        min_points = np.min(counts)
        max_points = np.max(counts)
    else:
        avg_points = 0
        min_points = 0
        max_points = 0

    print(f"[Voxel] Grid: {grid_size}, Guard: {guard_band}")
    print(f"[Voxel] Total: {len(xyz)}, Safe: {n_safe} ({n_safe/len(xyz):.1%}), Clusters: {n_clusters}")
    print(f"[Voxel] Points per Cluster -> Avg: {avg_points:.1f}, Min: {min_points}, Max: {max_points}")
    
    if visualize and n_clusters > 0:
        rng = np.random.RandomState(42)
        cluster_colors = rng.rand(n_clusters, 3)
        colors = np.zeros((len(xyz), 3))
        colors[is_safe] = cluster_colors[labels[is_safe]]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        window_name = f"Voxel Clustering (Grid={grid_size}, Avg pts={avg_points:.1f})"
        o3d.visualization.draw_geometries([pcd], window_name=window_name)

    return labels

# ==========================================
# グラフ構築・GFT
# ==========================================

def build_graph(xyz, method='knn', param=6):
    """
    グラフ構築
    method: 'knn' (k-Nearest Neighbors) or 'radius' (Radius Search)
    param: k (int) or radius (float)
    """
    if method == 'knn':
        k = int(param)
        adj = kneighbors_graph(xyz, k, mode='distance', include_self=False)
    elif method == 'radius':
        radius = float(param)
        adj = radius_neighbors_graph(xyz, radius, mode='distance', include_self=False)
    else:
        raise ValueError("Unknown graph method. Use 'knn' or 'radius'.")

    W = adj.toarray()
    
    # 重み計算
    dists = W[W > 0]
    if len(dists) > 0:
        sigma = np.mean(dists)
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
    scaled = val / delta
    if bit == 0:
        embedded_scaled = np.round(scaled)
    else:
        embedded_scaled = np.floor(scaled) + 0.5 if (scaled - np.floor(scaled)) < 0.5 else np.ceil(scaled) - 0.5
    return embedded_scaled * delta

def qim_extract_scalar(val, delta):
    scaled = val / delta
    dist_0 = np.abs(scaled - np.round(scaled))
    dist_1 = np.abs(scaled - (np.round(scaled - 0.5) + 0.5))
    return 0 if dist_0 < dist_1 else 1

# ==========================================
# 埋め込み・抽出 (メイン処理)
# ==========================================

def is_points_safe(xyz, grid_size, guard_band):
    """点群が安全地帯(ガードバンド外)にいるか判定"""
    voxel_indices = np.floor(xyz / grid_size).astype(int)
    local_coords = xyz - (voxel_indices * grid_size)
    is_safe = np.all(
        (local_coords >= guard_band) & (local_coords <= (grid_size - guard_band)),
        axis=1
    )
    return is_safe

def embed_watermark_qim(
    xyz, labels, embed_bits, 
    grid_size, guard_band, # チェック用に必要
    graph_method='knn', graph_param=6, # グラフ構築設定
    delta=0.01,
    min_spectre=0.0, max_spectre=1.0
):
    """QIMを用いたブラインド透かし埋め込み)"""
    xyz_after = xyz.copy().astype(np.float64) # 型を明示
    cluster_ids = np.unique(labels)
    cluster_ids = cluster_ids[cluster_ids != -1]

    embed_len = len(embed_bits)
    skipped_clusters = 0
    total_processed = 0
    # デバッグ用カウンタ
    debug_printed = False
    
    # 各クラスタで処理
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        # 点数が少なすぎる場合はスキップ
        if len(idx) < 3: 
            continue
        
        # 半径探索の場合、孤立点が出る可能性があるため、グラフ構築時にチェックが必要
        pts_original = xyz[idx].copy()
        
        # 1. グラフ構築
        try:
            W = build_graph(pts_original, method=graph_method, param=graph_param)
            # 連結成分チェックなどは省略(GFTは非連結でも計算可能)
        except Exception as e:
            # 半径探索で点が見つからない場合など
            continue

        basis, eigvals = gft_basis(W)
        pts_embedded = pts_original.copy()
        total_processed += 1

        # デバッグ: 最初のクラスタだけ詳細表示
        if not debug_printed:
            print(f"\n[Debug] Cluster {c} (Points: {len(idx)})")
            print(f"[Debug] Graph Method: {graph_method}, Param: {graph_param}")
            print(f"[Debug] Eigenvalues (Head): {eigvals[:5]}")
            print(f"[Debug] Eigenvalues (Tail): {eigvals[-5:]}")
        
        # 2. 埋め込み計算
        for ch in range(3):
            signal = pts_original[:, ch]
            coeffs = gft(signal, basis)
            
            Q_ = len(coeffs)
            q_start = int(Q_ * min_spectre)
            q_end   = int(Q_ * max_spectre)
            target_coeffs_len = q_end - q_start
            
            if target_coeffs_len <= 0:
                continue

            # デバッグ: 埋め込み前の係数確認
            if not debug_printed and ch == 0:
                print(f"[Debug] Coeffs (Before) [Range {q_start}:{q_end}]: {coeffs[q_start:q_start+5]}")
                
            current_bit_idx = 0
            for i in range(target_coeffs_len):
                coeff_idx = q_start + i
                bit = embed_bits[current_bit_idx]
                coeffs[coeff_idx] = qim_embed_scalar(coeffs[coeff_idx], bit, delta)
                current_bit_idx = (current_bit_idx + 1) % embed_len

            # デバッグ: 埋め込み後の係数確認
            if not debug_printed and ch == 0:
                print(f"[Debug] Coeffs (After)  [Range {q_start}:{q_end}]: {coeffs[q_start:q_start+5]}")
            
            pts_embedded[:, ch] = igft(coeffs, basis)

        # デバッグ: 座標変化量の確認
        if not debug_printed:
            diff = np.abs(pts_embedded - pts_original)
            max_diff = np.max(diff)
            print(f"[Debug] Max Coordinate Change: {max_diff:.6f} (Guard Band: {guard_band})")
            debug_printed = True
        
        # 3. 整合性チェック
        safety_check = is_points_safe(pts_embedded, grid_size, guard_band)
        
        if np.all(safety_check):
            # 採用
            xyz_after[idx] = pts_embedded
        else:
            # 更新しない(Rollback)
            skipped_clusters += 1

    print(f"[Embed] Processed: {total_processed}, Skipped(Rollback): {skipped_clusters} ({skipped_clusters/total_processed*100:.1f}%)")
    return xyz_after

def extract_watermark_qim(
    xyz_target, labels, embed_len,
    graph_method='knn', graph_param=6,
    delta=0.01,
    min_spectre=0.0, max_spectre=1.0
):
    """QIMを用いたブラインド透かし抽出"""
    cluster_ids = np.unique(labels)
    cluster_ids = cluster_ids[cluster_ids != -1]

    votes_per_bit = [[] for _ in range(embed_len)]
    
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        if len(idx) < 3:
            continue
            
        pts = xyz_target[idx]
        
        try:
            W = build_graph(pts, method=graph_method, param=graph_param)
            basis, eigvals = gft_basis(W)
        except:
            continue
        
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
                detected_bit = qim_extract_scalar(coeffs[coeff_idx], delta)
                votes_per_bit[current_bit_idx].append(detected_bit)
                current_bit_idx = (current_bit_idx + 1) % embed_len

    extracted_bits = []
    for votes in votes_per_bit:
        if len(votes) == 0:
            extracted_bits.append(0)
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
    noise = np.random.normal(0, noise_std, xyz.shape)
    return xyz + noise

def crop_point_cloud(xyz, keep_ratio=0.5):
    N = len(xyz)
    keep_n = int(N * keep_ratio)
    indices = np.random.choice(N, keep_n, replace=False)
    return xyz[indices]