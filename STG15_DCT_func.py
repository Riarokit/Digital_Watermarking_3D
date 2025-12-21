import numpy as np
import open3d as o3d
from scipy.fftpack import dctn, idctn
from scipy.spatial import cKDTree
from PIL import Image

# ... (ユーティリティ関数: image_to_bitarray ~ normalize_point_cloud_exact は変更なし) ...
def image_to_bitarray(image_path, n=32):
    img = Image.open(image_path).convert('L')
    img = img.resize((n, n), Image.LANCZOS)
    arr = np.array(img)
    arr = (arr > 127).astype(np.uint8)
    return arr.flatten().tolist()

def bitarray_to_image(bitarray, n=32, save_path=None):
    expected_len = n * n
    if len(bitarray) < expected_len:
        bitarray = bitarray + [0] * (expected_len - len(bitarray))
    elif len(bitarray) > expected_len:
        bitarray = bitarray[:expected_len]
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
        if x_val.max() > x_val.min():
            colors[:, 0] = (x_val - x_val.min()) / (x_val.max() - x_val.min())
            colors[:, 1] = (y_val - y_val.min()) / (y_val.max() - y_val.min())
            colors[:, 2] = (z_val - z_val.min()) / (z_val.max() - z_val.min())
    else:
        colors = np.zeros_like(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def normalize_point_cloud_exact(pcd, target_size=100.0, verbose=True, visualize=False):
    xyz = np.asarray(pcd.points)
    min_bound = np.min(xyz, axis=0)
    max_bound = np.max(xyz, axis=0)
    center = (min_bound + max_bound) / 2.0
    xyz -= center
    current_size = np.max(max_bound - min_bound)
    if current_size > 0:
        scale = target_size / current_size
        xyz *= scale
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if verbose:
        new_min = np.min(xyz, axis=0)
        new_max = np.max(xyz, axis=0)
        print(f"[Normalize] Range: {new_min} ~ {new_max}")
        print(f"[Normalize] Max Width: {np.max(new_max - new_min):.4f}")
    return pcd, xyz

# ... (可視化関数: calculate_capacity, visualize_voxels_with_points も変更なし) ...
def calculate_capacity(grid_divs, min_freq_idx, max_freq_idx, num_blocks=8):
    count = 0
    for ix in range(grid_divs):
        for iy in range(grid_divs):
            for iz in range(grid_divs):
                if ix==0 and iy==0 and iz==0: continue 
                if (ix + iy + iz) >= min_freq_idx and \
                   ix < max_freq_idx and iy < max_freq_idx and iz < max_freq_idx:
                    count += 1
    total = count * 3
    print("--------------------------------------------------")
    print(f"【埋め込み容量計算 (Fixed 3D-DCT)】")
    print(f"  Grid: {grid_divs}^3 per block")
    print(f"  Freq Band: Sum>={min_freq_idx}, Index<{max_freq_idx}")
    print(f"  -> Capacity/Block: {total} bits")
    print(f"  -> Total ({num_blocks} blocks): {total * num_blocks} bits")
    print("--------------------------------------------------")
    return total * num_blocks # 冗長化しないなら合計を返す

def visualize_voxels_with_points(xyz, grid_divs=8, block_radius=30.0):
    # (省略: 以前のコードと同じでOKです)
    pass 

# ... (重み計算: compute_weights_in_octant, calc_soft_centroids も変更なし) ...
def get_custom_weight_curve(dists, voxel_len):
    r = dists / voxel_len
    d = [0.0, 0.5, 1.0]
    w = [1.0, 0.5, 0.0]
    return np.interp(r, d, w, left=1.0, right=0.0)

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
    offsets = np.array([[dx, dy, dz] for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)])
    neighbor_ijk = ijk[:, np.newaxis, :] + offsets[np.newaxis, :, :]
    valid_mask = np.all((neighbor_ijk >= 0) & (neighbor_ijk < grid_divs), axis=2)
    flat_indices = (neighbor_ijk[:,:,0]*(grid_divs**2) + neighbor_ijk[:,:,1]*grid_divs + neighbor_ijk[:,:,2])
    flat_indices[~valid_mask] = 0
    centers = voxel_centers[flat_indices]
    dists = np.linalg.norm(local_xyz[:, np.newaxis, :] - centers, axis=2)
    w = get_custom_weight_curve(dists, voxel_len)
    w[~valid_mask] = 0.0
    return w, flat_indices, len(voxel_centers)

def calc_soft_centroids(xyz, weights, neighbor_indices, num_voxels):
    numerator = np.zeros((num_voxels, 3))
    denominator = np.zeros((num_voxels))
    w_flat = weights.ravel()
    idx_flat = neighbor_indices.ravel()
    for dim in range(3):
        val_flat = (xyz[:, dim][:, np.newaxis] * weights).ravel()
        np.add.at(numerator[:, dim], idx_flat, val_flat)
    np.add.at(denominator, idx_flat, w_flat)
    valid = denominator > 1e-9
    centroids = np.zeros_like(numerator)
    centroids[valid] = numerator[valid] / denominator[valid][:, np.newaxis]
    return centroids, denominator

# ==============================================================================
# 3D-DCT / QIM (ここから変更)
# ==============================================================================
def apply_3d_dct(volume):
    return dctn(volume, axes=(0, 1, 2), norm='ortho')

def apply_3d_idct(coeffs):
    return idctn(coeffs, axes=(0, 1, 2), norm='ortho')

def qim_embed_scalar(val, bit, delta):
    scaled = val / delta
    if bit == 0: embedded = np.round(scaled)
    else: embedded = np.floor(scaled) + 0.5 if (scaled - np.floor(scaled)) < 0.5 else np.ceil(scaled) - 0.5
    return embedded * delta

def qim_extract_scalar(val, delta):
    scaled = val / delta
    dist_0 = np.abs(scaled - np.round(scaled))
    dist_1 = np.abs(scaled - (np.round(scaled - 0.5) + 0.5))
    return 0 if dist_0 < dist_1 else 1

# ==============================================================================
# 処理ロジック: 1つのオクタントに対する埋め込み (Fixed Grid 3D-DCT)
# ==============================================================================

def process_octant_embedding(
    xyz_octant, embed_bits, 
    octant_min, octant_max, grid_divs, 
    qim_delta, min_freq_idx, max_freq_idx, 
    iterations,
    block_id=0
):
    if len(xyz_octant) == 0: return xyz_octant
    
    total_bits = len(embed_bits)
    chunk_size = int(np.ceil(total_bits / 3))
    bits_xyz = [embed_bits[0:chunk_size], embed_bits[chunk_size:2*chunk_size], embed_bits[2*chunk_size:]]
    
    weights, neighbor_idx, num_voxels = compute_weights_in_octant(
        xyz_octant, grid_divs, octant_min, octant_max
    )
    point_weight_sum = np.sum(weights, axis=1, keepdims=True)
    point_weight_sum[point_weight_sum < 1e-9] = 1.0
    
    current_xyz = xyz_octant.copy()
    
    debug_samples = []
    debug_indices = []

    for itr in range(iterations):
        centroids, denom = calc_soft_centroids(current_xyz, weights, neighbor_idx, num_voxels)
        
        # 3Dグリッドに整形 (固定サイズ)
        centroids_grid = centroids.reshape((grid_divs, grid_divs, grid_divs, 3))
        diff_grid = np.zeros_like(centroids_grid)
        
        # 点が存在するボクセルのマスク (これ以外のボクセルは動かせない)
        valid_mask_grid = (denom > 1e-3).reshape((grid_divs, grid_divs, grid_divs))
        
        for ch in range(3):
            target_bits = bits_xyz[ch]
            if len(target_bits) == 0: continue
            
            # 3D-DCT (全ボクセル対象)
            vol = centroids_grid[:, :, :, ch]
            coeffs = apply_3d_dct(vol)
            coeffs_target = coeffs.copy()
            
            bit_cursor = 0
            bit_len = len(target_bits)
            
            # 低周波埋め込み
            for ix in range(grid_divs):
                for iy in range(grid_divs):
                    for iz in range(grid_divs):
                        if ix==0 and iy==0 and iz==0: continue
                        
                        if (ix + iy + iz) >= min_freq_idx and \
                           ix < max_freq_idx and iy < max_freq_idx and iz < max_freq_idx:
                            
                            bit = target_bits[bit_cursor % bit_len]
                            val_orig = coeffs[ix, iy, iz]
                            val_target = qim_embed_scalar(val_orig, bit, qim_delta)
                            coeffs_target[ix, iy, iz] = val_target
                            
                            if itr == 0 and block_id == 0 and ch == 0 and len(debug_indices) < 5:
                                debug_indices.append((ix, iy, iz))
                                debug_samples.append({'orig': val_orig, 'target': val_target})
                            
                            bit_cursor += 1
            
            # 差分IDCT
            diff_coeffs = coeffs_target - coeffs
            
            # 【重要】ブースト係数を大きくする (5.0)
            # 空ボクセル(0)が含まれる分、全体のエネルギーが下がるため、強く押す必要がある
            diff_vol = apply_3d_idct(diff_coeffs) * 1.5
            
            # 【重要】有効なボクセルのみ更新を許可
            # 空のボクセルに移動命令が出ても無視する (動かせないから)
            diff_vol[~valid_mask_grid] = 0.0
            
            diff_grid[:, :, :, ch] = diff_vol
            
        # 点への還元
        deltas_flat = diff_grid.reshape((num_voxels, 3))
        neighbor_deltas = deltas_flat[neighbor_idx]
        point_shifts = np.sum(weights[:, :, np.newaxis] * neighbor_deltas, axis=1)
        point_shifts /= point_weight_sum
        
        current_xyz += point_shifts
        
        # デバッグ表示
        if itr == iterations - 1 and block_id == 0 and len(debug_indices) > 0:
            centroids_fin, _ = calc_soft_centroids(current_xyz, weights, neighbor_idx, num_voxels)
            grid_fin = centroids_fin.reshape((grid_divs, grid_divs, grid_divs, 3))
            coeffs_fin = apply_3d_dct(grid_fin[:, :, :, 0])
            print(f"\n--- DCT Coefficients Debug (Block 0, 3D-Fixed) ---")
            print(f"Index     | Original  | Target    | Final     | Error")
            print("-" * 60)
            for k, idx_tuple in enumerate(debug_indices):
                s = debug_samples[k]
                val = coeffs_fin[idx_tuple]
                err = abs(val - s['target'])
                print(f"{str(idx_tuple):<10} | {s['orig']:8.4f}  | {s['target']:8.4f}  | {val:8.4f}  | {err:.4f}")
            print("-" * 60)

    return current_xyz

def process_octant_extraction(
    xyz_octant, total_bits_len, octant_min, octant_max, grid_divs, qim_delta, min_freq_idx, max_freq_idx
):
    if len(xyz_octant) == 0: return None
    chunk_size = int(np.ceil(total_bits_len / 3))
    weights, neighbor_idx, num_voxels = compute_weights_in_octant(xyz_octant, grid_divs, octant_min, octant_max)
    centroids, denom = calc_soft_centroids(xyz_octant, weights, neighbor_idx, num_voxels)
    
    # 固定サイズグリッド
    centroids_grid = centroids.reshape((grid_divs, grid_divs, grid_divs, 3))
    
    votes_list = [[], [], []]
    lengths = [chunk_size, chunk_size, total_bits_len - 2*chunk_size]
    
    for ch in range(3):
        target_len = lengths[ch]
        if target_len <= 0: continue
        
        # 3D-DCT
        vol = centroids_grid[:, :, :, ch]
        coeffs = apply_3d_dct(vol)
        
        bit_cursor = 0
        temp_votes = [[] for _ in range(target_len)]
        
        for ix in range(grid_divs):
            for iy in range(grid_divs):
                for iz in range(grid_divs):
                    if ix==0 and iy==0 and iz==0: continue
                    if (ix + iy + iz) >= min_freq_idx and \
                       ix < max_freq_idx and iy < max_freq_idx and iz < max_freq_idx:
                        
                        val = coeffs[ix, iy, iz]
                        res = qim_extract_scalar(val, qim_delta)
                        temp_votes[bit_cursor % target_len].append(res)
                        bit_cursor += 1
        
        for v in temp_votes:
            if not v: votes_list[ch].append(None)
            else: votes_list[ch].append(1 if v.count(1) > v.count(0) else 0)
            
    return votes_list

def embed_watermark_main(
    xyz, embed_bits, grid_divs=16, qim_delta=0.2, min_freq_idx=1, max_freq_idx=6, iterations=15
):
    xyz_final = np.zeros_like(xyz)
    masks = []
    for x_s in [False, True]:
        for y_s in [False, True]:
            for z_s in [False, True]:
                mask = (xyz[:, 0] >= 0) if x_s else (xyz[:, 0] < 0)
                mask &= (xyz[:, 1] >= 0) if y_s else (xyz[:, 1] < 0)
                mask &= (xyz[:, 2] >= 0) if z_s else (xyz[:, 2] < 0)
                o_min = np.array([0.0 if x_s else -50.0, 0.0 if y_s else -50.0, 0.0 if z_s else -50.0])
                o_max = np.array([50.0 if x_s else 0.0, 50.0 if y_s else 0.0, 50.0 if z_s else 0.0])
                masks.append((mask, o_min, o_max))
    
    print(f"[Embed] Splitting into 8 Octants (Independent Processing)...")
    for i, (mask, o_min, o_max) in enumerate(masks):
        pts = xyz[mask]
        if len(pts) == 0: continue
        pts_embedded = process_octant_embedding(
            pts, embed_bits, o_min, o_max, grid_divs, qim_delta, min_freq_idx, max_freq_idx, iterations,
            block_id=i # デバッグ用ID追加
        )
        xyz_final[mask] = pts_embedded
    return xyz_final

def extract_watermark_main(
    xyz, embed_len, grid_divs=16, qim_delta=0.2, min_freq_idx=1, max_freq_idx=6
):
    chunk_size = int(np.ceil(embed_len / 3))
    target_containers = [[[] for _ in range(chunk_size)], [[] for _ in range(chunk_size)], [[] for _ in range(embed_len - 2*chunk_size)]]
    masks = []
    for x_s in [False, True]:
        for y_s in [False, True]:
            for z_s in [False, True]:
                mask = (xyz[:, 0] >= 0) if x_s else (xyz[:, 0] < 0)
                mask &= (xyz[:, 1] >= 0) if y_s else (xyz[:, 1] < 0)
                mask &= (xyz[:, 2] >= 0) if z_s else (xyz[:, 2] < 0)
                o_min = np.array([0.0 if x_s else -50.0, 0.0 if y_s else -50.0, 0.0 if z_s else -50.0])
                o_max = np.array([50.0 if x_s else 0.0, 50.0 if y_s else 0.0, 50.0 if z_s else 0.0])
                masks.append((mask, o_min, o_max))
                
    for i, (mask, o_min, o_max) in enumerate(masks):
        pts = xyz[mask]
        if len(pts) == 0: continue
        res = process_octant_extraction(pts, embed_len, o_min, o_max, grid_divs, qim_delta, min_freq_idx, max_freq_idx)
        if res is None: continue
        for ch in range(3):
            bits = res[ch]
            for bit_i, bit_val in enumerate(bits):
                if bit_val is not None: target_containers[ch][bit_i].append(bit_val)
                    
    full_extracted_bits = []
    for container in target_containers:
        for votes in container:
            if not votes: full_extracted_bits.append(0)
            else: full_extracted_bits.append(1 if votes.count(1) > votes.count(0) else 0)
    return full_extracted_bits

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