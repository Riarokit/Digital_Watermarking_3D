import numpy as np
import open3d as o3d
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.spatial import cKDTree
from PIL import Image

# ==========================================
# ユーティリティ
# ==========================================

def image_to_bitarray(image_path, n=32):
    img = Image.open(image_path).convert('L')
    img = img.resize((n, n), Image.LANCZOS)
    arr = np.array(img)
    arr = (arr > 127).astype(np.uint8)
    return arr.flatten().tolist()

def bitarray_to_image(bitarray, n=32, save_path=None):
    arr = np.array(bitarray, dtype=np.uint8).reshape((n, n)) * 255
    img = Image.fromarray(arr, mode='L')
    if save_path:
        img.save(save_path)
    return img

def add_colors(pcd, color="grad"):
    points = np.asarray(pcd.points)
    if color == "grad":
        x_val = points[:, 0]; y_val = points[:, 1]; z_val = points[:, 2]
        colors = np.zeros_like(points)
        colors[:, 0] = (x_val - x_val.min()) / (x_val.max() - x_val.min() + 1e-8)
        colors[:, 1] = (y_val - y_val.min()) / (y_val.max() - y_val.min() + 1e-8)
        colors[:, 2] = (z_val - z_val.min()) / (z_val.max() - z_val.min() + 1e-8)
    else:
        colors = np.zeros_like(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def normalize_point_cloud(pcd, target_scale=100.0, verbose=True, visualize=False):
    """点群の最大幅をtarget_scaleに正規化"""
    xyz = np.asarray(pcd.points)
    centroid = np.mean(xyz, axis=0)
    xyz -= centroid
    
    bbox_size = np.max(xyz, axis=0) - np.min(xyz, axis=0)
    max_width = np.max(bbox_size)
    
    if max_width > 0:
        scale_factor = target_scale / max_width
        xyz *= scale_factor
    
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    if verbose:
        new_bbox = np.max(xyz, axis=0) - np.min(xyz, axis=0)
        print(f"[Normalize] Scale aligned to {target_scale}. BBox: {new_bbox}")
    
    if visualize:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=target_scale*0.2)
        o3d.visualization.draw_geometries([pcd, frame], window_name="Normalized")
        
    return pcd, xyz

# ==========================================
# ボクセル処理 (VGSP Core)
# ==========================================

def voxelize_and_get_centroids(xyz, grid_size, guard_band):
    """
    点群をボクセル化し、有効なボクセルごとの重心を計算する。
    
    Returns:
    - voxel_labels: 各点の所属ボクセルID (ガードバンド内は-1)
    - active_voxel_indices: 有効ボクセルのグリッド住所 (M, 3) [ix, iy, iz]
    - centroids: 有効ボクセルの重心座標 (M, 3) --> これがグラフ信号になる
    """
    # 1. グリッドインデックス計算
    voxel_idx = np.floor(xyz / grid_size).astype(int)
    
    # 2. ガードバンド判定 (各ボクセルの壁から guard_band 以内はNG)
    local_coords = xyz - (voxel_idx * grid_size)
    is_safe = np.all(
        (local_coords >= guard_band) & (local_coords <= (grid_size - guard_band)),
        axis=1
    )
    
    # 3. 有効なボクセルのリストアップ
    #    safeな点が含まれているボクセルのみを「有効」とする
    safe_voxel_idx = voxel_idx[is_safe]
    if len(safe_voxel_idx) == 0:
        return np.full(len(xyz), -1), np.array([]), np.array([])

    # ユニークなボクセル住所を取得
    active_voxel_indices, inverse = np.unique(safe_voxel_idx, axis=0, return_inverse=True)
    num_voxels = len(active_voxel_indices)
    
    # 4. 各点へのラベル付け
    #    まず全員 -1
    voxel_labels = np.full(len(xyz), -1, dtype=int)
    #    safeな点にのみ、0 ~ M-1 のボクセルIDを付与
    voxel_labels[is_safe] = inverse
    
    # 5. 重心 (Centroid) の計算 -> グラフ信号
    centroids = np.zeros((num_voxels, 3))
    # 点の座標配列 (safeなものだけ)
    safe_points = xyz[is_safe]
    
    # numpyで高速にグループ集計 (bincount利用)
    for dim in range(3):
        # 各ボクセルの座標和
        sums = np.bincount(inverse, weights=safe_points[:, dim], minlength=num_voxels)
        # 各ボクセルの点数
        counts = np.bincount(inverse, minlength=num_voxels)
        # 重心 = 和 / 点数
        centroids[:, dim] = sums / (counts + 1e-8) # ゼロ除算防止
        
    return voxel_labels, active_voxel_indices, centroids

def check_guard_band_violation(xyz_new, voxel_labels, grid_size, guard_band):
    """
    移動後の点がガードバンドを侵犯していないかチェック
    Returns:
        violation_mask: ボクセルIDごとの違反フラグ (M,) Trueなら違反あり
    """
    # -1 (元々除外されていた点) は無視
    mask = (voxel_labels != -1)
    if not np.any(mask):
        return np.array([])
        
    pts = xyz_new[mask]
    lbls = voxel_labels[mask]
    
    # 現在の座標でのローカル位置
    # 注意: ボクセルIDは変わらない前提 (移動量は小さいはず)
    # しかし念のため、元のボクセルインデックス基準でチェックする
    # (点が境界を超えて隣のボクセルに行ったらアウト)
    
    # 現在の座標から計算されるインデックス
    current_idx = np.floor(pts / grid_size).astype(int)
    local_pos = pts - (current_idx * grid_size)
    
    # 安全か？
    is_safe_now = np.all(
        (local_coords >= guard_band) & (local_coords <= (grid_size - guard_band)),
        axis=1
    )
    # これだけだと「隣のボクセルの安全圏」に移動した場合を検知できないので
    # 「インデックスが変わっていないか」もチェックすべきだが、
    # ガードバンドがあればインデックスが変わるには必ずガードバンドを通過するので
    # local_posのチェックだけで基本的には十分。
    # 厳密には:
    
    local_coords = pts - (np.floor(pts/grid_size) * grid_size) # 簡易計算
    # 0~Lの範囲内かつ、縁から遠いこと
    in_band = np.any((local_coords < guard_band) | (local_coords > grid_size - guard_band), axis=1)
    
    # ボクセルごとに集計 (1点でも違反があれば True)
    # np.bincount等は合計しか出せないので、違反があるかどうかは
    # 「違反数の合計 > 0」で判定
    num_voxels = np.max(lbls) + 1
    violation_counts = np.bincount(lbls, weights=in_band.astype(int), minlength=num_voxels)
    
    return violation_counts > 0

# ==========================================
# グラフ構築 (Grid Graph)
# ==========================================

def build_voxel_graph(active_voxel_indices, radius=1.9):
    """
    ボクセルの住所(整数)に基づいてグラフを構築。
    データ(重心座標)は見ないので、埋め込み前後でグラフは不変。
    
    radius=1.9 -> 斜め隣接(26近傍)まで含む (sqrt(1^2+1^2+1^2) = 1.73)
    radius=1.1 -> 上下左右前後(6近傍)のみ (sqrt(1^2) = 1)
    """
    # 整数座標間の距離で接続を決める
    adj = radius_neighbors_graph(active_voxel_indices, radius, mode='connectivity', include_self=False)
    W = adj.toarray()
    
    # 無向グラフ化
    W = np.maximum(W, W.T)
    return W

def gft_basis(W):
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # 符号の向き統一 (Sign Disambiguation)
    max_abs_idx = np.argmax(np.abs(eigvecs), axis=0)
    signs = np.sign(eigvecs[max_abs_idx, range(eigvecs.shape[1])])
    signs[signs == 0] = 1
    eigvecs = eigvecs * signs
    
    return eigvecs, eigvals

def gft(signal, basis):
    return basis.T @ signal

def igft(coeffs, basis):
    return basis @ coeffs

# ==========================================
# QIM
# ==========================================

def qim_embed_scalar(val, bit, delta):
    scaled = val / delta
    if bit == 0:
        embedded = np.round(scaled)
    else:
        embedded = np.floor(scaled) + 0.5 if (scaled - np.floor(scaled)) < 0.5 else np.ceil(scaled) - 0.5
    return embedded * delta

def qim_extract_scalar(val, delta):
    scaled = val / delta
    dist_0 = np.abs(scaled - np.round(scaled))
    dist_1 = np.abs(scaled - (np.round(scaled - 0.5) + 0.5))
    return 0 if dist_0 < dist_1 else 1

# ==========================================
# メイン処理: 埋め込み
# ==========================================

def embed_watermark_vgsp(
    xyz, embed_bits, 
    grid_size, guard_band, 
    qim_delta=0.1, 
    min_spectre=0.05, max_spectre=0.9
):
    """
    Voxel-based GSP Embedding
    """
    xyz_embedded = xyz.copy()
    
    # 1. ボクセル化 & 重心計算 (グラフ信号の生成)
    labels, voxel_indices, centroids = voxelize_and_get_centroids(xyz, grid_size, guard_band)
    num_voxels = len(voxel_indices)
    
    if num_voxels < 10:
        print("[Error] ボクセル数が少なすぎます。GRID_SIZEを小さくしてください。")
        return xyz
    
    print(f"[Embed] Active Voxels: {num_voxels} (Graph Nodes)")
    
    # 2. グラフ構築 (ボクセル住所ベース -> 不変)
    W = build_voxel_graph(voxel_indices, radius=1.9) # 26近傍
    basis, eigvals = gft_basis(W)
    
    # 3. 埋め込み (重心を移動させる)
    centroids_embedded = centroids.copy()
    embed_len = len(embed_bits)
    total_embedded_coeffs = 0
    
    for ch in range(3): # x, y, z
        signal = centroids[:, ch]
        coeffs = gft(signal, basis)
        
        # 帯域制限
        q_start = int(num_voxels * min_spectre)
        q_end   = int(num_voxels * max_spectre)
        
        bit_idx = 0
        for i in range(q_start, q_end):
            bit = embed_bits[bit_idx % embed_len]
            coeffs[i] = qim_embed_scalar(coeffs[i], bit, qim_delta)
            bit_idx += 1
            total_embedded_coeffs += 1
            
        centroids_embedded[:, ch] = igft(coeffs, basis)
        
    # 4. 座標への反映 (Broadcast) & Rollback
    #    各ボクセル内の全点に対して、重心の移動量ベクトルを加算する
    shift_vectors = centroids_embedded - centroids  # (M, 3)
    
    # 効率的な更新のために、各点にシフトベクトルを適用
    # labels == m の点に shift_vectors[m] を足す
    
    # -1の点には足さないので、0埋めした配列を作る
    full_shifts = np.zeros((np.max(labels) + 1, 3))
    full_shifts[:num_voxels] = shift_vectors
    
    # 各点の移動量 (labelsが-1の場所はindex errorになるので対処)
    # labelsの-1を、ダミーインデックス(最後)に逃がすテクニック
    # でも単純にループの方が安全確実
    
    # まず仮移動
    xyz_temp = xyz.copy()
    valid_mask = (labels != -1)
    
    # numpyのファンシーインデックスで一括加算
    # validな点のラベルに対応するシフトを取り出す
    point_shifts = full_shifts[labels[valid_mask]]
    xyz_temp[valid_mask] += point_shifts
    
    # 5. ガードバンド違反チェック (Rollback)
    #    ボクセルごとに「違反者が出たか」を確認
    #    簡易実装: もう一度ループを回して確認
    
    rollback_count = 0
    
    # xyz_temp の各点がガードバンドを守れているか？
    # -> 守れていないボクセルIDを特定
    # 簡易チェック:
    temp_idx = np.floor(xyz_temp / grid_size).astype(int)
    temp_local = xyz_temp - (temp_idx * grid_size)
    
    # 違反している点
    violation_mask = np.any((temp_local < guard_band) | (temp_local > grid_size - guard_band), axis=1)
    # 元々除外(-1)だった点は関係ない
    violation_mask = violation_mask & valid_mask
    
    # 違反が発生したボクセルIDのリスト
    violated_voxel_ids = np.unique(labels[violation_mask])
    
    # 反映
    final_shifts = full_shifts.copy()
    if len(violated_voxel_ids) > 0:
        # 違反ボクセルのシフトを0に戻す
        final_shifts[violated_voxel_ids] = 0.0
        rollback_count = len(violated_voxel_ids)
    
    # 最終適用
    xyz_embedded[valid_mask] += final_shifts[labels[valid_mask]]
    
    print(f"[Embed] Rollback Voxels: {rollback_count} / {num_voxels} ({rollback_count/num_voxels:.1%})")
    
    return xyz_embedded

# ==========================================
# メイン処理: 抽出
# ==========================================

def extract_watermark_vgsp(
    xyz, embed_len, 
    grid_size, guard_band, 
    qim_delta=0.1, 
    min_spectre=0.05, max_spectre=0.9
):
    # 1. ボクセル化 & 重心計算
    #    (埋め込み時と全く同じグリッド・ガードバンドを使うことで、
    #     Rollbackされたボクセルも含めて「同じボクセル構成」が再現される)
    _, voxel_indices, centroids = voxelize_and_get_centroids(xyz, grid_size, guard_band)
    num_voxels = len(voxel_indices)
    
    if num_voxels < 10:
        return [0]*embed_len
        
    # 2. グラフ構築 (住所ベース -> 完全一致)
    W = build_voxel_graph(voxel_indices, radius=1.9)
    basis, eigvals = gft_basis(W)
    
    # 3. 抽出
    votes = [[] for _ in range(embed_len)]
    
    for ch in range(3):
        signal = centroids[:, ch]
        coeffs = gft(signal, basis)
        
        q_start = int(num_voxels * min_spectre)
        q_end   = int(num_voxels * max_spectre)
        
        bit_idx = 0
        for i in range(q_start, q_end):
            val = coeffs[i]
            # QIM判定
            detected = qim_extract_scalar(val, qim_delta)
            votes[bit_idx % embed_len].append(detected)
            bit_idx += 1
            
    # 4. 多数決
    extracted_bits = []
    for v_list in votes:
        if len(v_list) == 0:
            extracted_bits.append(0)
        else:
            c0 = v_list.count(0)
            c1 = v_list.count(1)
            extracted_bits.append(1 if c1 > c0 else 0)
            
    return extracted_bits

# ==========================================
# 評価関数
# ==========================================

def calc_psnr(pcd_before, pcd_after):
    points_before = np.asarray(pcd_before.points)
    points_after = np.asarray(pcd_after.points)
    
    # MSE (平均二乗誤差) の計算
    tree = cKDTree(points_after)
    dists, _ = tree.query(points_before, k=1)
    mse = np.mean(dists ** 2)
    
    # PSNR (ピーク信号対雑音比) の計算
    # 基準: 点群のバウンディングボックスの最大幅 (Max Range)
    xyz = points_before
    max_range = np.max(np.max(xyz,0)-np.min(xyz,0))
    psnr = 10 * np.log10((max_range ** 2) / mse) if mse > 0 else float('inf')
    
    # SNR (信号対雑音比) の計算
    # 基準: 信号(座標値)そのもののパワー (原点からの距離の二乗平均)
    signal_power = np.mean(np.sum(points_before ** 2, axis=1))
    snr = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')
    
    print(f"[Metric] MSE: {mse:.6f}")
    print(f"[Metric] PSNR: {psnr:.2f} dB")
    print(f"[Metric] SNR:  {snr:.2f} dB")
    
    return psnr, snr

def calc_ber(original_bits, extracted_bits):
    ber = np.mean(np.array(original_bits) != np.array(extracted_bits))
    print(f"[Metric] BER: {ber:.4f}")
    return ber

# ==========================================
# 攻撃関数
# ==========================================

def noise_addition_attack(xyz, noise_percent=0.01, mode='uniform', seed=None, verbose=True):
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

def cropping_attack(xyz_after, keep_ratio=0.5, mode='center'):
    """
    xyz_after に対して切り取り攻撃を行い、一部の点群のみを残し、表示する。

    Parameters:
    - xyz_after (np.ndarray): 埋め込み後の点群座標（N×3）
    - keep_ratio (float): 残す点の割合（0.0～1.0]
    - mode (str): 'center'（中心部を残す）または 'edge'（端部を残す）
    - verbose (bool): 情報表示の有無

    Returns:
    - xyz_cropped (np.ndarray): 切り取り後の点群座標
    """
    import open3d as o3d
    assert 0.0 < keep_ratio <= 1.0, "keep_ratioは (0, 1] で指定してください"
    N = xyz_after.shape[0]
    keep_n = int(N * keep_ratio)

    center = np.mean(xyz_after, axis=0)
    dists = np.linalg.norm(xyz_after - center, axis=1)

    if mode == 'center':
        keep_indices = np.argsort(dists)[:keep_n]
    elif mode == 'edge':
        keep_indices = np.argsort(dists)[-keep_n:]
    else:
        raise ValueError("modeは 'center' または 'edge' を指定してください")

    xyz_cropped = xyz_after[keep_indices]

    print(f"切り取り攻撃 ({mode}): 元点数={N} → 残点数={keep_n} ({keep_ratio*100:.1f}%)")

    # 可視化
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(xyz_cropped)
    cropped_pcd.paint_uniform_color([1, 0.6, 0])  # オレンジ系で表示
    o3d.visualization.draw_geometries([cropped_pcd], window_name="Cropped Point Cloud")

    return xyz_cropped