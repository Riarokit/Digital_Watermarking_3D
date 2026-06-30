import numpy as np
import open3d as o3d
import random
import string
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
import scipy.sparse as sp
from PIL import Image

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

# =========================================================
#  評価関数群
# =========================================================

def visualize_embedded_points(xyz_orig, xyz_after, threshold=1e-8):
    """
    点群のうち情報が埋め込まれている点（座標が変化した点）を黄色で、
    埋め込まれていない点（座標が変化していない点）を灰色で表して可視化する関数。
    
    Parameters:
    - xyz_orig: 埋め込み前の点群座標 (N, 3)
    - xyz_after: 埋め込み後の点群座標 (N, 3)
    - threshold: 座標の変化を判定する閾値
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_after)
    
    # 変化量を計算
    diffs = np.linalg.norm(xyz_after - xyz_orig, axis=1)
    
    # 色配列を準備 (初期値は灰色: [0.6, 0.6, 0.6])
    colors = np.ones((len(xyz_after), 3)) * 0.6
    
    # 変化量が閾値以上の点を黄色: [1.0, 1.0, 0.0] に設定
    embedded_indices = diffs > threshold
    colors[embedded_indices] = [1.0, 1.0, 0.0]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"[Visualize] 総点数: {len(xyz_after)}, 埋め込み(変更)された点数: {np.sum(embedded_indices)}")
    o3d.visualization.draw_geometries([pcd], window_name="Embedded Points (Yellow) vs Non-embedded (Gray)")
    return pcd

def evaluate_psnr(pcd_before, pcd_after, reverse=True, by_index=False, verbose=True):
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

    mean_sq_signal = np.mean(points_before ** 2)
    if mse == 0:
        snr_ratio = float('inf')
    else:
        snr_ratio = mean_sq_signal / mse
    # max_range（PSNR用スケール）
    xyz = points_before
    p_max = np.max(xyz, axis=0)
    p_min = np.min(xyz, axis=0)
    max_range = np.linalg.norm(p_max - p_min)
    # RMSE（√MSE）
    rmse = np.sqrt(mse)
    # SNR
    snr = 10 * np.log10(snr_ratio) if snr_ratio != float('inf') else float('inf')
    # VSNR
    vsnr = 20 * np.log10(snr_ratio) if snr_ratio != float('inf') else float('inf')
    # PSNR
    psnr = float('inf') if mse == 0 else 10 * np.log10((max_range ** 2) / mse)

    if verbose:
        print("------------------- 評価 -------------------")
        print(f"MSE  : {mse:.6f}")
        print(f"RMSE : {rmse:.6f}")
        print(f"SNR  : {snr:.2f} dB")
        print(f"PSNR : {psnr:.2f} dB (max_range={max_range:.4f})")
        print(f"VSNR : {vsnr:.2f} dB")
        
    return {
        'mse': mse,
        'rmse': rmse,
        'snr': snr,
        'psnr': psnr,
        'vsnr': vsnr,
        'max_range': max_range
    }

def evaluate_robustness(watermark_bits, extracted_bits, verbose=True):
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
    if verbose:
        print(f"Corr : {corr:.4f}")
        print(f"BER  : {ber:.4f}")
        
    return corr, ber

def extract_surface_info(xyz, k=6):
    """
    点群の各点においてPCAを用いて近似接平面と2次曲面をフィッティングし、
    その係数と局所座標系（フレーム）を抽出します。
    """

    tree = cKDTree(xyz)
    _, idxs = tree.query(xyz, k=k)
    N = len(xyz)
    
    frames = np.zeros((N, 3, 3))
    coeffs = np.zeros((N, 6))
    curvatures = np.zeros(N)
    
    for i in range(N):
        pts = xyz[idxs[i]]
        p = xyz[i]
        
        centered = pts - p
        cov = np.cov(centered.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            frames[i] = np.eye(3)
            continue
            
        uz = eigvecs[:, 0]
        ux = eigvecs[:, 1]
        uy = eigvecs[:, 2]
        frames[i] = np.array([ux, uy, uz])
        
        x_i = centered.dot(ux)
        y_i = centered.dot(uy)
        z_i = centered.dot(uz)
        
        A = np.column_stack((x_i**2, y_i**2, x_i * y_i, x_i, y_i, np.ones(len(x_i))))
        
        try:
            theta, _, _, _ = np.linalg.lstsq(A, z_i, rcond=None)
            coeffs[i] = theta
            
            a, b, c, d, e, f_coeff = theta
            num = (1 + d**2)*a + (1 + e**2)*b - c*d*e
            den = (1 + d**2 + e**2)**1.5
            curvatures[i] = np.abs(num / den)
            
        except np.linalg.LinAlgError:
            pass
            
    return {'frames': frames, 'coeffs': coeffs, 'curvatures': curvatures}

def _compute_msdm_directional(xyz_source, xyz_target, info_source, info_target, tree_target, h):
    """
    一方向 (source -> target) の局所歪み(LD)を計算します。
    """

    _, idxs_closest = tree_target.query(xyz_source, k=1)
    
    curv_source = info_source['curvatures']
    curv_proj = np.zeros(len(xyz_source))
    
    frames_t = info_target['frames']
    coeffs_t = info_target['coeffs']
    
    for i in range(len(xyz_source)):
        p = xyz_source[i]
        q_idx = idxs_closest[i]
        q = xyz_target[q_idx]
        
        ux, uy, uz = frames_t[q_idx]
        a, b, c, d, e, f_coeff = coeffs_t[q_idx]
        
        x_p = np.dot(p - q, ux)
        y_p = np.dot(p - q, uy)
        
        Qx = 2*a*x_p + c*y_p + d
        Qy = 2*b*y_p + c*x_p + e
        num = (1 + Qx**2)*2*b - 2*Qx*Qy*c + (1 + Qy**2)*2*a
        den = 2 * (1 + Qx**2 + Qy**2)**1.5
        curv_proj[i] = np.abs(num / den)
        
    tree_source = cKDTree(xyz_source)
    # 球形近傍の取得
    nbrs_list = tree_source.query_ball_point(xyz_source, r=h)
    
    LDs = np.zeros(len(xyz_source))
    sigma_g = h / 3.0
    
    for i, nbrs in enumerate(nbrs_list):
        if len(nbrs) <= 1:
            LDs[i] = 0.0
            continue
            
        pts = xyz_source[nbrs]
        dists = np.linalg.norm(pts - xyz_source[i], axis=1)
        w = np.exp(-(dists**2) / (2 * sigma_g**2))
        w_sum = np.sum(w)
        if w_sum <= 0:
            LDs[i] = 0.0
            continue
        w /= w_sum
        
        c_p = curv_source[nbrs]
        c_hat = curv_proj[nbrs]
        
        mu_p = np.sum(w * c_p)
        mu_hat = np.sum(w * c_hat)
        
        sigma_p = np.sqrt(np.maximum(np.sum(w * (c_p - mu_p)**2), 0))
        sigma_hat = np.sqrt(np.maximum(np.sum(w * (c_hat - mu_hat)**2), 0))
        
        cov = np.sum(w * (c_p - mu_p) * (c_hat - mu_hat))
        
        K = 1e-5
        L = np.abs(mu_p - mu_hat) / (np.maximum(mu_p, mu_hat) + K)
        C = np.abs(sigma_p - sigma_hat) / (np.maximum(sigma_p, sigma_hat) + K)
        S = np.abs(sigma_p * sigma_hat - cov) / (sigma_p * sigma_hat + K)
        
        LDs[i] = (L + C + 0.5 * S) / 2.5
        
    return LDs

def evaluate_pc_msdm(pcd_before, pcd_after, k=6, m=2, verbose=True):
    """
    論文での定義に忠実なPC-MSDM。
    対応点探索時の曲面への投影、球形近傍でのガウス重み、ソートに依存しない正確な共分散、
    および対称性のすべての要件を含みます。
    ※ 2次曲面フィッティングの近傍点数として引数kを使用します。
    """
    
    xyz_ref = np.asarray(pcd_before.points)
    xyz_dist = np.asarray(pcd_after.points)
    
    # 1. 局所2次曲面情報の計算
    info_ref = extract_surface_info(xyz_ref, k=k)
    info_dist = extract_surface_info(xyz_dist, k=k)
    
    # 2. バウンディングボックスからスケール h の決定
    bb_min = np.min(xyz_ref, axis=0)
    bb_max = np.max(xyz_ref, axis=0)
    BB_length = np.linalg.norm(bb_max - bb_min)
    h = 0.02 * BB_length
    
    tree_ref = cKDTree(xyz_ref)
    tree_dist = cKDTree(xyz_dist)
    
    # 3. 双方向のLDを計算
    LD_dist_to_ref = _compute_msdm_directional(xyz_dist, xyz_ref, info_dist, info_ref, tree_ref, h)
    LD_ref_to_dist = _compute_msdm_directional(xyz_ref, xyz_dist, info_ref, info_dist, tree_dist, h)
    
    # Minkowski Pooling
    l_minkowski_d2r = np.mean(LD_dist_to_ref ** m) ** (1.0 / m)
    l_minkowski_r2d = np.mean(LD_ref_to_dist ** m) ** (1.0 / m)
    
    # 対称的なMSDMの歪みスコア
    symmetric_distortion = (l_minkowski_d2r + l_minkowski_r2d) / 2.0
    
    # 品質スコア（1.0に近いほど高評価）として返す
    final_score = np.clip(1.0 - symmetric_distortion, 0.0, 1.0)
    
    if verbose:
        print(f"PC-MSDM (k={k}, p={m}): {final_score:.4f}")
    return final_score

def evaluate_angular_similarity(pcd_before, pcd_after, k_normals=12, verbose=True):
    """
    論文 'Point Cloud Quality Assessment Metric Based on Angular Similarity' に基づく
    Angular Similarity 指標の実装。
    
    接平面（法線ベクトル）のなす角を計算し、幾何的な構造の類似度を評価します。
    """
    
    def _ensure_normals(pcd, k):
        """法線がない場合に推定を行う"""
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
        return np.asarray(pcd.normals)

    # 1. 法線の準備
    normals_ref = _ensure_normals(pcd_before, k_normals)
    normals_dist = _ensure_normals(pcd_after, k_normals)
    
    xyz_ref = np.asarray(pcd_before.points)
    xyz_dist = np.asarray(pcd_after.points)

    def _compute_directed_angular_sim(p_source, p_target, n_source, n_target):
        """片方向の Angular Similarity を計算 (Source -> Target)"""
        tree_target = cKDTree(p_target)
        # 相手側点群から最も近い点のインデックスを取得
        _, indices = tree_target.query(p_source, k=1)
        
        # 対応する点の法線ベクトルを取得
        # n_a: 参照側の法線, n_b: 歪み側の法線
        n_a = n_source
        n_b = n_target[indices]
        
        # 2. コサイン類似度の計算
        # x = (n_a · n_b) / (||n_a|| * ||n_b||)
        # 通常、法線は正規化されているためドット積のみで計算可能
        dot_product = np.sum(n_a * n_b, axis=1)
        cos_sim = np.clip(dot_product, -1.0, 1.0)
        
        # 3. 最小の角 theta_tilde の計算
        # theta_tilde = arccos(|x|)  範囲: [0, pi/2]
        theta_tilde = np.arccos(np.abs(cos_sim))
        
        # 4. 各点の Angular Similarity の計算
        # score = 1 - (2 * theta_tilde / pi)
        point_scores = 1.0 - (2.0 * theta_tilde / np.pi)
        
        # 5. プーリング: 平均をとる
        return np.mean(point_scores)

    # 双方向の類似度を計算
    sim_ref_to_dist = _compute_directed_angular_sim(xyz_ref, xyz_dist, normals_ref, normals_dist)
    sim_dist_to_ref = _compute_directed_angular_sim(xyz_dist, xyz_ref, normals_dist, normals_ref)

    # 6. 対称化: 2つの方向のうち「最小値」を最終スコアとする
    final_score = min(sim_ref_to_dist, sim_dist_to_ref)

    if verbose:
        print(f"Angular-Similarity (k={k_normals}): {final_score:.4f}")

    return final_score

def evaluate_p2d(pcd_before, pcd_after, k=40, eps=1e-8, verbose=True):
    """
    論文 'A Point-to-Distribution Joint Geometry and Color Metric...' に基づく
    幾何歪み評価指標 P2D (Point-to-Distribution) の実装。
    
    各点と、相手側点群の近傍 K 個の分布（平均・分散）とのマハラノビス距離を計算します。
    """
    xyz_ref = np.asarray(pcd_before.points)
    xyz_dist = np.asarray(pcd_after.points)

    def _compute_directed_p2d(p_source, p_target, k_nn):
        """片方向の P2D 距離を計算 (Source -> Target Distribution)"""
        tree_target = cKDTree(p_target)
        # 1. 相手側点群から K 個の近傍点を探索
        dist_nn, indices = tree_target.query(p_source, k=k_nn)
        
        # 近傍点の座標を取得
        neighbors = p_target[indices] # 形状: (n_points, k, 3)
        
        # 2. 近傍点分布の統計量（平均・分散）を計算
        # mu: 各点に対応する近傍 K 個の平均 (n_points, 3)
        mu = np.mean(neighbors, axis=1)
        # var: 各点に対応する近傍 K 個の分散 (n_points, 3)
        var = np.var(neighbors, axis=1)
        
        # 3. マハラノビス距離（標準化ユークリッド距離）の計算
        # 成分間の相関を無視する場合、距離は成分ごとの (差^2 / 分散) の和の平方根となる
        # ゼロ割り防止のため微小値 eps を加算
        diff_sq = (p_source - mu) ** 2
        m_dist_sq = np.sum(diff_sq / (var + eps), axis=1)
        m_dist = np.sqrt(m_dist_sq)
        
        # 5. プーリング: 全点の平均をとる
        return np.mean(m_dist)

    # 双方向の距離を計算
    d_ref_to_dist = _compute_directed_p2d(xyz_ref, xyz_dist, k)
    d_dist_to_ref = _compute_directed_p2d(xyz_dist, xyz_ref, k)

    # 対称化: 2つの方向のうち最大値を最終的な歪みスコアとする
    symmetric_distortion = max(d_ref_to_dist, d_dist_to_ref)
    
    # 品質スコアへの変換 (LogP2D): スコアが高いほど高品質
    # Q = log10(1 + 1/D)
    final_score = np.log10(1 + (1.0 / (symmetric_distortion + eps)))

    if verbose:
        print(f"LogP2D (k={k}, eps={eps}): {final_score:.4f}")

    return final_score

def evaluate_point_ssim(pcd_before, pcd_after, attribute='geometry', dispersion='variance', k=12, m=2, epsilon=None, verbose=True):
    """
    PointSSIM (Point Cloud Structural Similarity) 評価指標。
    論文: "PointSSIM: A Structural Similarity Index for Point Clouds" に基づく。
    
    Parameters:
    - pcd_before: 参照点群 (X)
    - pcd_after: 評価対象点群 (Y)
    - attribute: 'geometry', 'normals', 'curvature', 'colors' のいずれか
    - dispersion: 'variance', 'cov', 'muad', 'mad', 'qcd' のいずれか
    - k: 近傍点の数 (Neighborhood size)
    - m: プーリングパラメータ (論文中の Equation 6 の k。デフォルトはMSE相当の2)
    - epsilon: ゼロ除算回避のための定数 (Noneなら機械イプシロン)
    - verbose: ログ出力
    
    Returns:
    - final_score: 品質スコア (1.0 に近いほど高品質=類似度が高い)
    """
    
    if epsilon is None:
        epsilon = np.finfo(float).eps

    xyz_b = np.asarray(pcd_before.points)
    xyz_a = np.asarray(pcd_after.points)
    
    tree_b = cKDTree(xyz_b)
    tree_a = cKDTree(xyz_a)
    
    # 評価点群（A）から参照点群（B）への対応点を探索
    # Y(A) の各点 p に対して X(B) の最近傍 q を見つける
    _, idxs_a2b = tree_b.query(xyz_a, k=1) 
    
    # 近傍の取得 (Aの各点、および、Bの各点)
    _, nbrs_a = tree_a.query(xyz_a, k=k)
    _, nbrs_b = tree_b.query(xyz_b, k=k)

    def compute_attribute(pcd, xyz, nbrs, attr_type):
        N = xyz.shape[0]
        if attr_type == 'geometry':
            # ユークリッド距離
            diff = xyz[nbrs] - xyz[:, None, :] # (N, k, 3)
            dist = np.linalg.norm(diff, axis=2) # (N, k)
            return dist
        elif attr_type == 'normals':
            # 法線ベクトルの角度類似度
            if not pcd.has_normals():
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max(10, k)))
                pcd.orient_normals_consistent_tangent_plane(max(10, k))
            norms = np.asarray(pcd.normals)
            norms_center = norms[:, None, :] # (N, 1, 3)
            norms_nbrs = norms[nbrs]         # (N, k, 3)
            # 内積をとることで類似度を計算
            sim = np.sum(norms_center * norms_nbrs, axis=2) # (N, k)
            return sim
        elif attr_type == 'curvature':
            # 曲率
            info = extract_surface_info(xyz, k=max(6, k))
            curvs = info['curvatures'] # (N,)
            return curvs[nbrs] # (N, k)
        elif attr_type == 'colors':
            if not pcd.has_colors():
                raise ValueError("PointCloud does not have colors for the 'colors' attribute.")
            colors = np.asarray(pcd.colors)
            # RGB to Luminance
            luminance = np.dot(colors, [0.2989, 0.5870, 0.1140])
            return luminance[nbrs] # (N, k)
        else:
            raise ValueError(f"Unknown attribute: {attr_type}")

    def compute_dispersion(attr_array, disp_type):
        # attr_array: (N, k)
        if disp_type == 'variance':
            return np.var(attr_array, axis=1) # (N,)
        elif disp_type == 'cov':
            mu = np.mean(attr_array, axis=1)
            sigma = np.std(attr_array, axis=1)
            return np.divide(sigma, mu, out=np.zeros_like(sigma), where=mu!=0)
        elif disp_type == 'muad':
            # Mean Absolute Deviation
            mu = np.mean(attr_array, axis=1, keepdims=True)
            return np.mean(np.abs(attr_array - mu), axis=1)
        elif disp_type == 'mad':
            # Median Absolute Deviation
            med = np.median(attr_array, axis=1, keepdims=True)
            return np.mean(np.abs(attr_array - med), axis=1)
        elif disp_type == 'qcd':
            # Quartile Coefficient of Dispersion
            q1 = np.percentile(attr_array, 25, axis=1)
            q3 = np.percentile(attr_array, 75, axis=1)
            denom = q3 + q1
            return np.divide(q3 - q1, denom, out=np.zeros_like(denom), where=denom!=0)
        else:
            raise ValueError(f"Unknown dispersion type: {disp_type}")

    # 属性の取得
    attr_a = compute_attribute(pcd_after, xyz_a, nbrs_a, attribute)
    attr_b = compute_attribute(pcd_before, xyz_b, nbrs_b, attribute)

    # 特徴量(ばらつき統計量)の抽出
    F_a = compute_dispersion(attr_a, dispersion) # A(Y)の特徴量
    F_b = compute_dispersion(attr_b, dispersion) # B(X)の特徴量

    # 1. 評価対象A (Distorted) から参照B (Reference) への誤差
    # Aの各点 p に対して Bの最近傍 q = idxs_a2b[p]
    F_b_matched = F_b[idxs_a2b]
    diff_a2b = np.abs(F_b_matched - F_a)
    denom_a2b = np.maximum(np.abs(F_b_matched), np.abs(F_a)) + epsilon
    S_a2b_p = diff_a2b / denom_a2b
    # Equation 6 による Error pooling (指数の Root はとらない)
    error_a2b = np.mean(S_a2b_p ** m)

    # 2. 参照B (Reference) から評価対象A (Distorted) への誤差
    # Bの各点 p に対して Aの最近傍 q = idxs_b2a[p]
    _, idxs_b2a = tree_a.query(xyz_b, k=1) 
    F_a_matched = F_a[idxs_b2a]
    diff_b2a = np.abs(F_a_matched - F_b)
    denom_b2a = np.maximum(np.abs(F_a_matched), np.abs(F_b)) + epsilon
    S_b2a_p = diff_b2a / denom_b2a
    # Equation 6 による Error pooling (指数の Root はとらない)
    error_b2a = np.mean(S_b2a_p ** m)

    # 対称誤差(Symmetric Error)として、2つの非対称スコアの「最小値(Minimum)」を採用する
    error_sym = min(error_a2b, error_b2a)
    
    # 品質スコア化（1に近いほど高品質）
    final_score = np.clip(1.0 - error_sym, 0.0, 1.0)
    
    if verbose:
        print(f"PointSSIM (attr={attribute}, disp={dispersion}, m={m}): {final_score:.4f}")
    return final_score

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
    # o3d.visualization.draw_geometries([cropped_pcd, mesh_frame], window_name=f"Cropped Point Cloud ({mode})")

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

def downsampling_attack(xyz, keep_ratio=0.5, mode='voxel', voxel_size_percent=1.0, seed=None):
    """
    点群に対してダウンサンプリング攻撃を行う。
    
    Parameters:
    - xyz (np.ndarray): 入力点群座標（N×3）
    - keep_ratio (float): ランダム・FPSモードで残す点の割合 (0.0 < keep_ratio <= 1.0)
    - mode (str): ダウンサンプリングの方式
        - 'random': ランダムに点をサンプリング（高速）
        - 'voxel': Open3Dを用いたVoxel Gridダウンサンプリング（均一化）
        - 'fps': 最遠点サンプリング（計算量は多いが極めて均一）
    - voxel_size_percent (float): 'voxel' モード時のボクセルサイズ（対角線長に対するパーセンテージ）
    - seed (int): ランダムシード（再現性用）
    
    Returns:
    - xyz_downsampled (np.ndarray): ダウンサンプリング後の点群（M×3, M <= N）
    """
    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise ValueError("keep_ratioは (0, 1] の範囲で指定してください。")
        
    N = xyz.shape[0]
    
    if mode == 'random':
        rng = np.random.RandomState(seed)
        keep_n = max(1, int(N * keep_ratio))
        indices = rng.choice(N, keep_n, replace=False)
        indices.sort()  # 元の相対順序を維持するためにソート
        xyz_downsampled = xyz[indices]
        print(f"[Attack] ランダム・ダウンサンプリング: 元点数={N} → 残点数={keep_n} ({keep_ratio*100:.1f}%)")
        
    elif mode == 'voxel':
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
        # 対角線長を基準としてボクセルサイズを計算
        scale_base = np.linalg.norm(xyz_max - xyz_min)
        voxel_size = scale_base * voxel_size_percent / 100
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        xyz_downsampled = np.asarray(pcd_down.points)
        print(f"[Attack] ボクセル・ダウンサンプリング (voxel_size={voxel_size:.6f}, {voxel_size_percent:.2f}%): 元点数={N} → 残点数={xyz_downsampled.shape[0]}")
        
    elif mode == 'fps':
        keep_n = max(1, int(N * keep_ratio))
        
        # 最遠点サンプリング (FPS) の実装
        selected_indices = np.zeros(keep_n, dtype=int)
        rng = np.random.RandomState(seed)
        selected_indices[0] = rng.randint(N)
        
        # 各点から、選択済みの点集合への最小距離
        distances = np.full(N, np.inf)
        
        current_pt = xyz[selected_indices[0]]
        for i in range(1, keep_n):
            # 現在追加された点と全点との距離
            dist_to_current = np.linalg.norm(xyz - current_pt, axis=1)
            # 最小距離を更新
            distances = np.minimum(distances, dist_to_current)
            # 最小距離が最大となる点を選択
            next_idx = np.argmax(distances)
            selected_indices[i] = next_idx
            current_pt = xyz[next_idx]
            
        selected_indices.sort()  # 元の相対順序を維持するためにソート
        xyz_downsampled = xyz[selected_indices]
        print(f"[Attack] 最遠点サンプリング (FPS): 元点数={N} → 残点数={keep_n} ({keep_ratio*100:.1f}%)")
        
    else:
        raise ValueError("modeは 'random', 'voxel', 'fps' のいずれかを指定してください。")
        
    return xyz_downsampled