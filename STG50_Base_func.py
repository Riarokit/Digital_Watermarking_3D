import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import skfuzzy as fuzz
import STG50_GFT_func as STG50F

def local_feature_clustering(xyz, k=6, verbose=False):
    """
    ■ 1. 局所特徴量の算出とクラスタリング
    入力点群に対してk-NNで近傍点を取得し、法線ベクトルとその近傍平均法線ベクトルのなす角を計算する。
    1次元の角度データにFCM(c=3)を適用し、中間の粗さ(Medium roughness)のクラスタに属する頂点インデックスを返す。
    """
    # 1. & 2. 点群の法線推定
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # k傍近傍点で法線ベクトルを算出 (PCA)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    # 法線の向きを揃える（角度計算のばらつきを抑えるため）
    pcd.orient_normals_consistent_tangent_plane(k)
    normals = np.asarray(pcd.normals)
    
    # 近傍点のインデックスを取得するためのk-NN (自身を含めるため k+1)
    nn = NearestNeighbors(n_neighbors=k+1).fit(xyz)
    _, indices = nn.kneighbors(xyz)
    
    theta = np.zeros(len(xyz))
    
    # 各頂点について、その法線と近傍点の平均法線とのなす角θを計算
    for i in range(len(xyz)):
        n_v = normals[i]
        # indices[i, 1:] は自身を除いた近傍k個のインデックス
        neighbor_idx = indices[i, 1:]
        n_neighbors = normals[neighbor_idx]
        
        # 近傍点の平均法線ベクトル
        n_avg = np.mean(n_neighbors, axis=0)
        norm_avg = np.linalg.norm(n_avg)
        
        # ゼロベクトル割りを防ぐ
        if norm_avg > 1e-12:
            n_avg = n_avg / norm_avg
        else:
            n_avg = n_v
            
        # なす角θの計算
        cos_theta = np.dot(n_v, n_avg)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta[i] = np.arccos(cos_theta)
        
    # 3. 角度θに対してscikit-fuzzyを用いたFuzzy C-Means (FCM) を適用 (c=3)
    # skfuzzyは(n_features, n_samples)の形状を期待するためリシェイプする
    data = theta.reshape(1, -1)
    c = 3
    m = 2.0
    error = 1e-5
    maxiter = 100
    
    cntr, U, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, c, m, error, maxiter, init=None, seed=42
    )
    
    centroids = cntr.flatten()
    # Uは(c, n_samples)なので、各点のラベルは列方向の最大値インデックス
    labels = np.argmax(U, axis=0)
    
    # 4. 3つのクラスタの重心を比較し、「真ん中（Medium roughness）」を選択
    sorted_centroid_indices = np.argsort(centroids)
    medium_cluster_id = sorted_centroid_indices[1]  # 0:低, 1:中, 2:高
    
    if verbose:
        print("\n--- クラスタリング結果 (FCM, c=3) ---")
        names = ["Low roughness", "Medium roughness", "High roughness"]
        for i, c_idx in enumerate(sorted_centroid_indices):
            cluster_theta = theta[labels == c_idx]
            count = len(cluster_theta)
            if count > 0:
                t_mean = np.mean(cluster_theta)
                t_min = np.min(cluster_theta)
                t_max = np.max(cluster_theta)
                print(f"{names[i]:<17} : 点数={count:<5}, θ平均={t_mean:.4f}, 境界(min~max)=[{t_min:.4f} ~ {t_max:.4f}]")
            else:
                print(f"{names[i]:<17} : 点数=0")
        print("-------------------------------------")

    # 中間クラスタの頂点インデックスを返す
    medium_cluster_indices = np.where(labels == medium_cluster_id)[0]
    return medium_cluster_indices

def embed_watermark_baseline(xyz, watermark_bits, n_points, a, k=6, verbose=True):
    """
    ■ 2. 透かしの埋め込み
    中間クラスタの頂点をn個選び、各頂点座標に対してビットに応じて+aまたは-aを足し引きして透かしを埋め込む。
    """
    xyz_new = xyz.copy()
    
    # 1. 中間クラスタの頂点インデックスを取得する
    medium_indices = local_feature_clustering(xyz, k=k, verbose=verbose)
    
    # 2. 頂点数が足りない場合はエラー
    if len(medium_indices) < n_points:
        raise ValueError(f"Target points ({n_points}) exceeds available points in medium cluster ({len(medium_indices)}).")
        
    # 中間クラスタの頂点をn_points個のグループに分割
    groups = np.array_split(medium_indices, n_points)
    
    # 3. 各グループの全頂点に対して同じビットを埋め込む
    for i in range(n_points):
        bit = watermark_bits[i]
        group_indices = groups[i]
        
        for idx in group_indices:
            v = xyz_new[idx]
            if bit == 1:
                xyz_new[idx] = v + a
            elif bit == 0:
                xyz_new[idx] = v - a
                
    # 4. 透かしが埋め込まれた新しい点群を返す
    return xyz_new

def extract_watermark_baseline(xyz_after, xyz, n_points, k=6, verbose=False):
    """
    ■ 3. 透かしの抽出
    元の点群から中間クラスタを特定し、埋め込み後点群との座標の差分からビットを抽出する(完全ノンブラインド型)。
    """
    # 0. 攻撃を受けた点群をオリジナル点群に1対1対応させて再サンプリング（同期・補完）
    xyz_after = STG50F.synchronize_point_cloud(xyz_after, xyz, verbose=True)

    # 1. クラスタリングは元の点群(xyz)を用いて、埋め込み対象の頂点インデックスを特定
    medium_indices = local_feature_clustering(xyz, k=k, verbose=verbose)
    
    if len(medium_indices) < n_points:
        raise ValueError("Not enough points in medium cluster during extraction.")
        
    # 中間クラスタの頂点をn_points個のグループに再分割
    groups = np.array_split(medium_indices, n_points)
    extracted_bits = []
    
    # 2. 各グループ内で差分からビットを判定し、多数決で抽出
    for i in range(n_points):
        group_indices = groups[i]
        votes = []
        
        for idx in group_indices:
            v_w = xyz_after[idx]
            v_o = xyz[idx]
            
            # 差分ベクトル diff = v_w - v_o
            diff = v_w - v_o
            diff_sum = diff.sum()
            
            # 抽出
            if diff_sum > 0:
                votes.append(1)
            else:
                votes.append(0)
                
        # 3. グループ内多数決
        counts = {0: votes.count(0), 1: votes.count(1)}
        if counts[1] > counts[0]:
            extracted_bits.append(1)
        else:
            extracted_bits.append(0)
            
    # 4. 抽出されたnビットの配列を返す
    return extracted_bits
