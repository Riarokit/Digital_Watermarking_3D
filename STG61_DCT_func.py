import numpy as np
import open3d as o3d
from scipy.fftpack import dct, idct
from scipy.spatial import cKDTree
from PIL import Image

# ==============================================================================
# ユーティリティ (変更なし)
# ==============================================================================
def image_to_bitarray(image_path, n=32):
    img = Image.open(image_path).convert('L')
    img = img.resize((n, n), Image.LANCZOS)
    arr = np.array(img)
    arr = (arr > 127).astype(np.uint8)
    return arr.flatten().tolist()

def bitarray_to_image(bitarray, n=32, save_path=None):
    expected_len = n * n
    if len(bitarray) < expected_len: bitarray += [0]*(expected_len-len(bitarray))
    elif len(bitarray) > expected_len: bitarray = bitarray[:expected_len]
    arr = np.array(bitarray, dtype=np.uint8).reshape((n, n)) * 255
    img = Image.fromarray(arr, mode='L')
    if save_path: img.save(save_path)
    return img

def add_colors(pcd, color="grad"):
    points = np.asarray(pcd.points)
    if color == "grad":
        x, y, z = points[:,0], points[:,1], points[:,2]
        c = np.zeros_like(points)
        if x.max()>x.min():
            c[:,0]=(x-x.min())/(x.max()-x.min()); c[:,1]=(y-y.min())/(y.max()-y.min()); c[:,2]=(z-z.min())/(z.max()-z.min())
    else: c = np.zeros_like(points)
    pcd.colors = o3d.utility.Vector3dVector(c)
    return pcd

def normalize_point_cloud_exact(pcd, target_size=100.0, verbose=True, visualize=False):
    xyz = np.asarray(pcd.points)
    min_b, max_b = np.min(xyz,0), np.max(xyz,0)
    xyz -= (min_b + max_b)/2.0
    curr_size = np.max(max_b - min_b)
    if curr_size > 0: xyz *= (target_size / curr_size)
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if verbose: print(f"[Normalize] Max Width: {curr_size*(target_size/curr_size) if curr_size>0 else 0:.4f}")
    return pcd, xyz

# ==============================================================================
# 重み・重心計算 (変更なし)
# ==============================================================================
def get_custom_weight_curve(dists, voxel_len):
    r = dists / voxel_len
    return np.interp(r, [0.0, 0.5, 1.0], [1.0, 0.5, 0.0], left=1.0, right=0.0)

def compute_weights_in_octant(xyz_octant, grid_divs, octant_min, octant_max):
    N = len(xyz_octant)
    if N == 0: return None, None, 0
    width = octant_max[0] - octant_min[0]
    voxel_len = width / grid_divs
    local_xyz = xyz_octant - octant_min
    rng = np.linspace(voxel_len/2, width - voxel_len/2, grid_divs)
    gx, gy, gz = np.meshgrid(rng, rng, rng, indexing='ij')
    voxel_centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    
    ijk = np.floor(local_xyz / voxel_len).astype(int)
    ijk = np.clip(ijk, 0, grid_divs - 1)
    offsets = np.array([[dx,dy,dz] for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)])
    neighbor_ijk = ijk[:,None,:] + offsets[None,:,:]
    valid_mask = np.all((neighbor_ijk >= 0) & (neighbor_ijk < grid_divs), axis=2)
    flat_indices = (neighbor_ijk[:,:,0]*(grid_divs**2) + neighbor_ijk[:,:,1]*grid_divs + neighbor_ijk[:,:,2])
    flat_indices[~valid_mask] = 0
    
    centers = voxel_centers[flat_indices]
    dists = np.linalg.norm(local_xyz[:,None,:] - centers, axis=2)
    w = get_custom_weight_curve(dists, voxel_len)
    w[~valid_mask] = 0.0
    return w, flat_indices, len(voxel_centers)

def calc_soft_centroids(xyz, weights, neighbor_indices, num_voxels):
    num = np.zeros((num_voxels, 3)); den = np.zeros((num_voxels))
    w_flat = weights.ravel(); idx_flat = neighbor_indices.ravel()
    for d in range(3): np.add.at(num[:, d], idx_flat, (xyz[:, d][:,None]*weights).ravel())
    np.add.at(den, idx_flat, w_flat)
    valid = den > 1e-6
    centroids = np.zeros_like(num)
    centroids[valid] = num[valid] / den[valid][:,None]
    return centroids, den

# ==============================================================================
# 1D-DCT / QIM
# ==============================================================================
def apply_1d_dct(signal):
    return dct(signal, type=2, norm='ortho')

def apply_1d_idct(coeffs):
    return idct(coeffs, type=2, norm='ortho')

def qim_embed_scalar(val, bit, delta):
    scaled = val / delta
    if bit == 0: emb = np.round(scaled)
    else: emb = np.floor(scaled)+0.5 if (scaled-np.floor(scaled))<0.5 else np.ceil(scaled)-0.5
    return emb * delta

def qim_extract_scalar(val, delta):
    scaled = val / delta
    d0 = np.abs(scaled - np.round(scaled))
    d1 = np.abs(scaled - (np.round(scaled-0.5)+0.5))
    return 0 if d0 < d1 else 1

# ==============================================================================
# Top-K Selection Logic (New!)
# ==============================================================================
def get_stable_voxels(denom, min_weight=1e-3):
    """
    有効なボクセルを「空間インデックス順」で返す。
    密度順(Top-K)は埋め込みで順位が変動するため廃止。
    """
    # 重みが閾値以上のボクセルIDを取得
    valid_indices = np.where(denom > min_weight)[0]
    
    # 必ずインデックス順（昇順）にソートして返す
    # これにより、配列の並び順が埋め込み前後で100%一致する
    valid_indices.sort()
    
    return valid_indices

# ==============================================================================
# 埋め込み (Top-K 1D-DCT)
# ==============================================================================
def process_octant_embedding(
    xyz_octant, embed_bits, 
    octant_min, octant_max, grid_divs, 
    qim_delta, min_freq, max_freq, 
    iterations,
    block_id=0
):
    if len(xyz_octant) == 0: return xyz_octant
    
    total_bits = len(embed_bits)
    chunk = int(np.ceil(total_bits / 3))
    bits_xyz = [embed_bits[0:chunk], embed_bits[chunk:2*chunk], embed_bits[2*chunk:]]
    
    weights, neighbor_idx, num_voxels = compute_weights_in_octant(
        xyz_octant, grid_divs, octant_min, octant_max
    )
    
    point_weight_sum = np.sum(weights, axis=1, keepdims=True)
    point_weight_sum[point_weight_sum < 1e-9] = 1.0
    
    current_xyz = xyz_octant.copy()
    
    # DCTの固定長 (パディングサイズ)
    # 十分な大きさを確保 (例: 全ボクセル数)
    DCT_LEN = num_voxels 

    # デバッグ用
    debug_samples = [] 

    for itr in range(iterations):
        centroids, denom = calc_soft_centroids(current_xyz, weights, neighbor_idx, num_voxels)
        
        # 安定ボクセル取得 (インデックス順)
        target_indices = get_stable_voxels(denom)
        if len(target_indices) < 10: break
        
        diff_vectors = np.zeros_like(centroids)
        
        for ch in range(3):
            target_bits = bits_xyz[ch]
            if len(target_bits) == 0: continue
            
            # 信号生成 (有効ボクセルのみ抽出)
            raw_signal = centroids[target_indices, ch]
            
            # 長さを固定 (DCT_LEN) にパディング
            signal_padded = np.zeros(DCT_LEN)
            signal_padded[:len(raw_signal)] = raw_signal
            
            # 1D-DCT
            coeffs = apply_1d_dct(signal_padded)
            coeffs_target = coeffs.copy()
            
            bit_cursor = 0
            blen = len(target_bits)
            
            # 埋め込み (固定インデックスに対して行う)
            for i in range(min_freq, max_freq):
                if i >= DCT_LEN: break
                
                bit = target_bits[bit_cursor % blen]
                val_orig = coeffs[i]
                val_target = qim_embed_scalar(val_orig, bit, qim_delta)
                
                coeffs_target[i] = val_target
                
                # デバッグ記録
                if block_id==0 and ch==0 and itr==0 and i==min_freq:
                    debug_samples.append({'idx': i, 'orig': val_orig, 'target': val_target})
                    
                bit_cursor += 1
                
            # 差分IDCT
            diff_coeffs = coeffs_target - coeffs
            diff_signal_padded = apply_1d_idct(diff_coeffs) * 2.0 # ブースト
            
            # パディング除去 & 有効部分のみ取り出し
            diff_signal = diff_signal_padded[:len(raw_signal)]
            
            # 差分を全体配列に戻す
            diff_vectors[target_indices, ch] = diff_signal
            
        # 還元
        neighbor_deltas = diff_vectors[neighbor_idx]
        point_shifts = np.sum(weights[:, :, np.newaxis] * neighbor_deltas, axis=1)
        point_shifts /= point_weight_sum
        
        current_xyz += point_shifts
        
        # 収束確認 (Debug)
        if block_id==0 and itr==iterations-1 and len(debug_samples) > 0:
             c_fin, d_fin = calc_soft_centroids(current_xyz, weights, neighbor_idx, num_voxels)
             idx_fin = get_stable_voxels(d_fin)
             sig_fin = np.zeros(DCT_LEN)
             sig_fin[:len(idx_fin)] = c_fin[idx_fin, 0]
             coef_fin = apply_1d_dct(sig_fin)
             
             s = debug_samples[0]
             final_val = coef_fin[s['idx']]
             err = abs(final_val - s['target'])
             print(f"[Debug] Block0 Iter{itr}: Target={s['target']:.4f}, Final={final_val:.4f}, Error={err:.4f}")

    return current_xyz

def process_octant_extraction(
    xyz_octant, total_bits_len, octant_min, octant_max, grid_divs, qim_delta, min_freq, max_freq
):
    if len(xyz_octant) == 0: return None
    chunk = int(np.ceil(total_bits_len / 3))
    
    weights, neighbor_idx, num_voxels = compute_weights_in_octant(xyz_octant, grid_divs, octant_min, octant_max)
    centroids, denom = calc_soft_centroids(xyz_octant, weights, neighbor_idx, num_voxels)
    
    target_indices = get_stable_voxels(denom)
    if len(target_indices) < 10: return None
    
    # 固定長
    DCT_LEN = num_voxels 
    
    votes_list = [[], [], []]
    lengths = [chunk, chunk, total_bits_len - 2*chunk]
    
    for ch in range(3):
        tlen = lengths[ch]
        if tlen <= 0: continue
        
        raw_signal = centroids[target_indices, ch]
        signal_padded = np.zeros(DCT_LEN)
        signal_padded[:len(raw_signal)] = raw_signal
        
        coeffs = apply_1d_dct(signal_padded)
        
        bit_cursor = 0
        temp_votes = [[] for _ in range(tlen)]
        
        for i in range(min_freq, max_freq):
            if i >= DCT_LEN: break
            
            val = coeffs[i]
            res = qim_extract_scalar(val, qim_delta)
            temp_votes[bit_cursor % tlen].append(res)
            bit_cursor += 1
            
        for v in temp_votes:
            if not v: votes_list[ch].append(None)
            else: votes_list[ch].append(1 if v.count(1) > v.count(0) else 0)
            
    return votes_list

# ==============================================================================
# Embed/Extract Main Loop (変更なし)
# ==============================================================================
def embed_watermark_main(xyz, embed_bits, grid_divs=16, qim_delta=0.2, min_freq=1, max_freq=20, iterations=10):
    xyz_final = np.zeros_like(xyz)
    masks = []
    for x_s in [False, True]:
        for y_s in [False, True]:
            for z_s in [False, True]:
                mask = (xyz[:,0]>=0) if x_s else (xyz[:,0]<0)
                mask &= (xyz[:,1]>=0) if y_s else (xyz[:,1]<0)
                mask &= (xyz[:,2]>=0) if z_s else (xyz[:,2]<0)
                o_min = np.array([0.0 if x_s else -50.0, 0.0 if y_s else -50.0, 0.0 if z_s else -50.0])
                o_max = np.array([50.0 if x_s else 0.0, 50.0 if y_s else 0.0, 50.0 if z_s else 0.0])
                masks.append((mask, o_min, o_max))
    
    print(f"[Embed] 8 Octants, Top-K 1D-DCT...")
    for i, (mask, o_min, o_max) in enumerate(masks):
        pts = xyz[mask]
        if len(pts) == 0: continue
        pts_embedded = process_octant_embedding(pts, embed_bits, o_min, o_max, grid_divs, qim_delta, min_freq, max_freq, iterations, i)
        xyz_final[mask] = pts_embedded
    return xyz_final

def extract_watermark_main(xyz, embed_len, grid_divs=16, qim_delta=0.2, min_freq=1, max_freq=20):
    chunk = int(np.ceil(embed_len / 3))
    conts = [[[] for _ in range(chunk)], [[] for _ in range(chunk)], [[] for _ in range(embed_len - 2*chunk)]]
    masks = []
    for x_s in [False, True]:
        for y_s in [False, True]:
            for z_s in [False, True]:
                mask = (xyz[:,0]>=0) if x_s else (xyz[:,0]<0)
                mask &= (xyz[:,1]>=0) if y_s else (xyz[:,1]<0)
                mask &= (xyz[:,2]>=0) if z_s else (xyz[:,2]<0)
                o_min = np.array([0.0 if x_s else -50.0, 0.0 if y_s else -50.0, 0.0 if z_s else -50.0])
                o_max = np.array([50.0 if x_s else 0.0, 50.0 if y_s else 0.0, 50.0 if z_s else 0.0])
                masks.append((mask, o_min, o_max))
    
    for i, (mask, o_min, o_max) in enumerate(masks):
        pts = xyz[mask]
        if len(pts) == 0: continue
        res = process_octant_extraction(pts, embed_len, o_min, o_max, grid_divs, qim_delta, min_freq, max_freq)
        if res is None: continue
        for ch in range(3):
            bits = res[ch]
            for bi, bv in enumerate(bits):
                if bv is not None: conts[ch][bi].append(bv)
    
    final_bits = []
    for c in conts:
        for v in c:
            if not v: final_bits.append(0)
            else: final_bits.append(1 if v.count(1)>v.count(0) else 0)
    return final_bits

# 他の関数(calc_psnr等)は省略なしでそのままコピー
# visualize_hierarchy, calculate_capacity もそのままでOK
def calc_psnr(pcd_before, pcd_after):
    points_before = np.asarray(pcd_before.points)
    points_after = np.asarray(pcd_after.points)
    tree = cKDTree(points_after)
    dists, _ = tree.query(points_before, k=1)
    mse = np.mean(dists ** 2)
    xyz = points_before
    max_range = np.max(np.max(xyz,0)-np.min(xyz,0))
    psnr = 10 * np.log10((max_range ** 2) / mse) if mse > 0 else float('inf')
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

def noise_addition_attack(xyz, noise_percent=1.0, mode='uniform', seed=None, verbose=True):
    rng = np.random.RandomState(seed)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    ranges = xyz_max - xyz_min
    scale = ranges * noise_percent / 100.0
    if verbose: print(f"ノイズ振幅: {noise_percent:.2f}%")
    if mode == 'uniform': noise = rng.uniform(low=-scale, high=scale, size=xyz.shape)
    elif mode == 'gaussian': noise = rng.normal(loc=0.0, scale=scale/2, size=xyz.shape)
    else: raise ValueError('mode')
    return xyz + noise

def cropping_attack(xyz_after, keep_ratio=0.5, mode='center'):
    import open3d as o3d
    N = xyz_after.shape[0]
    keep_n = int(N * keep_ratio)
    center = np.mean(xyz_after, axis=0)
    dists = np.linalg.norm(xyz_after - center, axis=1)
    if mode == 'center': keep_indices = np.argsort(dists)[:keep_n]
    elif mode == 'edge': keep_indices = np.argsort(dists)[-keep_n:]
    else: raise ValueError("mode")
    xyz_cropped = xyz_after[keep_indices]
    print(f"切り取り攻撃 ({mode}): 元点数={N} → 残点数={keep_n} ({keep_ratio*100:.1f}%)")
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(xyz_cropped)
    cropped_pcd.paint_uniform_color([1, 0.6, 0])
    o3d.visualization.draw_geometries([cropped_pcd], window_name="Cropped Point Cloud")
    return xyz_cropped

def visualize_hierarchy(xyz, grid_divs_per_octant=16, range_max=50.0):
    geometries = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if not pcd.has_colors(): pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(pcd)
    
    box_a = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([-range_max]*3), 
        max_bound=np.array([range_max]*3)
    )
    box_a.color = (0, 0, 0)
    geometries.append(box_a)
    
    lines_b = []
    points_b = []
    points_b.extend([[-range_max, 0, 0], [range_max, 0, 0]])
    lines_b.append([0, 1])
    points_b.extend([[0, -range_max, 0], [0, range_max, 0]])
    lines_b.append([2, 3])
    points_b.extend([[0, 0, -range_max], [0, 0, range_max]])
    lines_b.append([4, 5])
    ls_b = o3d.geometry.LineSet()
    ls_b.points = o3d.utility.Vector3dVector(points_b)
    ls_b.lines = o3d.utility.Vector2iVector(lines_b)
    ls_b.paint_uniform_color([0, 0, 0])
    geometries.append(ls_b)
    
    print("[Visualizing] Creating Active Voxel C wireframes...")
    centers = []
    offset = range_max / 2.0
    for x in [-offset, offset]:
        for y in [-offset, offset]:
            for z in [-offset, offset]:
                centers.append(np.array([x, y, z]))
    lines_c_points = []
    lines_c_indices = []
    voxel_size = (range_max) / grid_divs_per_octant
    cube_v = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0],
                       [0,0,1], [1,0,1], [0,1,1], [1,1,1]]) * voxel_size
    cube_e = np.array([[0,1], [1,3], [3,2], [2,0],
                       [4,5], [5,7], [7,6], [6,4],
                       [0,4], [1,5], [2,6], [3,7]])
    for center in centers:
        oct_min = center - offset
        local_xyz = xyz - oct_min
        indices = np.floor(local_xyz / voxel_size).astype(int)
        mask = np.all((indices >= 0) & (indices < grid_divs_per_octant), axis=1)
        valid_indices = indices[mask]
        if len(valid_indices) == 0: continue
        active_voxels = np.unique(valid_indices, axis=0)
        base_idx = len(lines_c_points)
        for v_idx in active_voxels:
            origin = oct_min + v_idx * voxel_size
            lines_c_points.extend(origin + cube_v)
            current_base = base_idx
            lines_c_indices.extend(cube_e + current_base)
            base_idx += 8
    ls_c = o3d.geometry.LineSet()
    ls_c.points = o3d.utility.Vector3dVector(lines_c_points)
    ls_c.lines = o3d.utility.Vector2iVector(lines_c_indices)
    ls_c.paint_uniform_color([0, 0, 0])
    geometries.append(ls_c)
    print("[Visualizing] Opening window...")
    o3d.visualization.draw_geometries(geometries, window_name="Hierarchy: A(Outer), B(Cross), C(Grid)")