import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse.linalg import spsolve
import math

# --- 1. 点群からメッシュへの変換 ---

def pcd_to_mesh(xyz):
    """
    点群(xyz)からBall Pivotingアルゴリズムを用いてメッシュを構築する [Source 9, 11]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals()
    
    # 半径を自動推定してBall Pivotingを実行
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2])
    )
    return mesh

# --- 2. 幾何学的計算と行列生成 ---

def get_mesh_laplacian(mesh):
    """
    メッシュの隣接関係からラプラシアン行列(L)を生成する。
    バイハーモニック場の求解に使用する。
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    n = len(vertices)
    
    # 隣接行列の構築
    adj = sparse.lil_matrix((n, n))
    for t in triangles:
        adj[t, t[1]] = adj[t[1], t] = 1
        adj[t[1], t[2]] = adj[t[2], t[1]] = 1
        adj[t[2], t] = adj[t, t[2]] = 1
    
    # 度数行列 D とラプラシアン L = D - W
    d = np.array(adj.sum(axis=1)).flatten()
    d_inv = sparse.diags(1.0 / np.where(d > 0, d, 1.0))
    l_matrix = sparse.eye(n) - d_inv @ adj.tocsr()
    return l_matrix

def solve_biharmonic_interpolation(l_matrix, boundary_indices, boundary_values):
    """
    【忠実再現の核】Dirichlet境界条件を用いたバイハーモニック場(Δ²u = 0)を解く [Source 9: Sect 3.1]
    極値点を補間してエンベロープを構築する。
    """
    n = l_matrix.shape[0]
    # Δ² 行列の計算
    biharmonic_op = l_matrix @ l_matrix
    
    # 方程式 Ax = b の構築
    a = biharmonic_op.tolil()
    b = np.zeros(n)
    
    # 境界条件の適用 (極値点の値を固定)
    for idx, val in zip(boundary_indices, boundary_values):
        a[idx, :] = 0
        a[idx, idx] = 1
        b[idx] = val
        
    return spsolve(a.tocsr(), b)

# --- 3. 表面上のEMDアルゴリズム ---

def surface_emd_sifting(mesh, signal, max_imfs=1, max_iter=5):
    """
    メッシュ表面でのSiftingプロセス [Source 9: Sect 3.1, Algorithm 1]
    """
    l_matrix = get_mesh_laplacian(mesh)
    vertices = np.asarray(mesh.vertices)
    
    res = signal.copy()
    imfs = []
    
    for _ in range(max_imfs):
        h = res.copy()
        for _ in range(max_iter):
            # 1. 極値点の検出 [Source 9: Eq.4]
            max_idx, min_idx = find_stable_extreme_points_mesh(mesh, h)
            
            if len(max_idx) < 4 or len(min_idx) < 4: break
            
            # 2. バイハーモニック場による包絡面の構築 [Source 9: Sect 3.1-Step3]
            upper_env = solve_biharmonic_interpolation(l_matrix, max_idx, h[max_idx])
            lower_env = solve_biharmonic_interpolation(l_matrix, min_idx, h[min_idx])
            
            # 3. 平均の除去
            mean_env = (upper_env + lower_env) / 2.0
            h = h - mean_env
            
        imf_k = h
        imfs.append(imf_k)
        res = res - imf_k
        
    return np.array(imfs), res

def find_stable_extreme_points_mesh(mesh, signal, t=0.8):
    """
    メッシュのトポロジーに基づいた極値点の検出 [Source 9: Eq.4]
    """
    triangles = np.asarray(mesh.triangles)
    n = len(signal)
    
    # 1-ring近傍の取得
    neighbors = [set() for _ in range(n)]
    for tri in triangles:
        for i in range(3):
            neighbors[tri[i]].update([tri[(i+1)%3], tri[(i+2)%3]])
            
    max_indices = []
    min_indices = []
    
    for i in range(n):
        nbs = list(neighbors[i])
        if not nbs: continue
        
        # 緩和された極値定義 [Source 9: Eq.4]
        count_greater = np.sum(signal[i] >= signal[nbs])
        if count_greater >= t * len(nbs) and signal[i] > 0.0001:
            max_indices.append(i)
            
        count_smaller = np.sum(signal[i] <= signal[nbs])
        if count_smaller >= t * len(nbs) and signal[i] < -0.0001:
            min_indices.append(i)
            
    return np.array(max_indices), np.array(min_indices)

# --- 4. 透かしの埋め込みと抽出 ---

def embed_watermark_hu_mesh(xyz, watermark_bits, FideP=115, T=25):
    """
    HuらのメッシュベースEMD透かし埋め込み [Source 9: Algorithm 1]
    """
    # 1. メッシュ化
    mesh = pcd_to_mesh(xyz)
    
    # 2. 球座標変換と正規化信号の生成 [Source 9: Eq.2, 3]
    rho = np.linalg.norm(xyz, axis=1)
    rho_max = np.max(rho)
    f_signal = rho / rho_max
    
    # 3. 表面上でのEMD実行
    imfs, res = surface_emd_sifting(mesh, f_signal, max_imfs=1)
    imf1 = imfs
    
    # 4. 安定した極値点の選択
    max_idx, min_idx = find_stable_extreme_points_mesh(mesh, imf1)
    
    # 5. 最適な埋め込み強度alphaの計算 [Source 9: Eq.9]
    all_extremes = np.concatenate([imf1[max_idx], imf1[min_idx]])
    sum_sq_imf = np.sum(all_extremes**2)
    alpha = 10**((-FideP / 10 + np.log10(len(xyz)) - np.log10(sum_sq_imf + 1e-12)) / 2)
    
    # 6. 円状埋め込み [Source 9: Sect 4.3.2]
    imf1_new = imf1.copy()
    wm_len = len(watermark_bits)
    for i, idx in enumerate(max_idx):
        bit = watermark_bits[i % wm_len]
        if bit == 1: imf1_new[idx] *= (1 + alpha)
        else: imf1_new[idx] *= (1 - alpha)
        
    # 7. 再構成 [Source 9: Eq.10, 11]
    f_new = imf1_new + res
    rho_new = f_new * rho_max
    
    # 角度成分を維持して直交座標に戻す
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])
    phi = np.arccos(np.clip(xyz[:, 2] / (rho + 1e-12), -1.0, 1.0))
    
    xyz_new = np.zeros_like(xyz)
    xyz_new[:, 0] = rho_new * np.sin(phi) * np.cos(theta)
    xyz_new[:, 1] = rho_new * np.sin(phi) * np.sin(theta)
    xyz_new[:, 2] = rho_new * np.cos(phi)
    
    return xyz_new, (max_idx, min_idx)

def extract_watermark_hu_mesh(xyz_w, key_info, wm_len):
    """
    投票戦略を用いた透かし抽出 [Source 9: Algorithm 2, Eq.13]
    """
    max_idx, _ = key_info
    mesh_w = pcd_to_mesh(xyz_w)
    rho_w = np.linalg.norm(xyz_w, axis=1)
    f_w = rho_w / np.max(rho_w)
    
    # EMD実行
    imfs_w, _ = surface_emd_sifting(mesh_w, f_w, max_imfs=1)
    imf1_w = imfs_w
    
    # 各ビットへの投票
    votes = [[] for _ in range(wm_len)]
    for i, idx in enumerate(max_idx):
        bit_idx = i % wm_len
        # 元の極値との相関判定（簡略化：正負や増減を判定）
        votes[bit_idx].append(1 if imf1_w[idx] > 0 else 0)
        
    # 多数決 [Source 9: Eq.13]
    extracted = [1 if sum(v) >= len(v)/2 else 0 for v in votes]
    return np.array(extracted)