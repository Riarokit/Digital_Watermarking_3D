import numpy as np
import open3d as o3d
import random
import string
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
import scipy.sparse as sp
import zlib
from PIL import Image
import copy
try:
    import cupy as cp
except ImportError:
    cp = None

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

def estimate_normals_xyz(xyz, knn=30, orient_knn=30, make_outward=True):
    """
    xyz: (N,3)
    return normals: (N,3) unit vectors
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    # 近傍で向きを揃える（符号反転がバラけると埋め込み/抽出が不安定になる）
    pcd.orient_normals_consistent_tangent_plane(orient_knn)

    normals = np.asarray(pcd.normals)

    if make_outward:
        # “外向きっぽく”揃える簡易策：重心からのベクトルと内積が負なら反転
        centroid = np.mean(xyz, axis=0)
        v = xyz - centroid
        flip = np.sum(normals * v, axis=1) < 0
        normals[flip] *= -1.0

    # 念のため正規化
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    return normals

# =========================================================
#  クラスタリング関数群
# =========================================================

def kmeans_cluster_points(xyz, cluster_point=1000, seed=42):
    num_clusters = len(xyz) // cluster_point
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
    labels = kmeans.fit_predict(xyz)
    
    # 各クラスタごとの点数を集計
    unique, counts = np.unique(labels, return_counts=True)
    min_count = np.min(counts)

    print(f"[KMeans] 検出されたクラスタ数: {len(unique)}")
    print(f"[KMeans] 最も点数が少ないクラスタの点数: {min_count}")
    
    return labels

def split_large_clusters(xyz, labels, limit_points=3000, seed=42):
    """
    指定したクラスタラベル群をもとに、点数が limit_points を超えるクラスタを再クラスタリングし、
    labels を更新して返す。

    Parameters:
    - xyz: Nx3 点群座標
    - labels: 長さNのクラスタ番号配列（int）
    - limit_points: 1クラスタの最大点数上限
    - seed: 再クラスタリング時のKMeansシード

    Returns:
    - new_labels: 長さNの更新後クラスタ番号配列
    """
    new_labels = labels.copy()
    current_max_label = labels.max()
    label_offset = current_max_label + 1

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) <= limit_points:
            continue

        pts = xyz[idx]
        n_subclusters = int(np.ceil(len(idx) / limit_points))
        kmeans = KMeans(n_clusters=n_subclusters, random_state=seed)
        sub_labels = kmeans.fit_predict(pts)

        # ラベルを更新
        for sub in range(n_subclusters):
            sub_idx = idx[sub_labels == sub]
            new_labels[sub_idx] = label_offset
            label_offset += 1

        print(f"[Split] クラスタ {c}（{len(idx)}点）を {n_subclusters} 分割 → 新ラベル {label_offset - n_subclusters}〜{label_offset - 1}")

    return new_labels

def visualize_clusters(xyz, labels):
    """
    xyz: Nx3 numpy配列（点群座標）
    labels: クラスタラベル配列（長さN, int型）
    return: 各クラスタに割り当てたRGBカラー（リスト, shape=[n_clusters,3])
    """
    COLORS = np.array([
        [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1],
        [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5], [0.5,0.2,0.7], [0.7,0.7,0.7]
    ])
    n_clusters = np.max(labels) + 1
    if n_clusters > len(COLORS):
        rng = np.random.RandomState(42)
        COLORS = rng.rand(n_clusters, 3)
    cluster_colors = COLORS[:n_clusters]

    # ラベルごとに色を割り当て
    color_array = cluster_colors[labels]

    # 可視化用の一時点群
    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(xyz)
    tmp_pcd.colors = o3d.utility.Vector3dVector(color_array)
    o3d.visualization.draw_geometries([tmp_pcd])

    # 各クラスタ番号に対する色のリストを返す
    color_list = [list(cluster_colors[i]) for i in range(n_clusters)]
    return color_list

def estimate_cluster_flatness(xyz, labels, k_neighbors=20):
    """
    各クラスタごとに「平面らしさ（平坦度）」指標を算出（平均曲率を使用）
    - xyz: Nx3配列
    - labels: クラスタラベル（N要素、int）
    - k_neighbors: 局所曲率推定の近傍点数
    - return: dict{クラスタ番号: 平坦度指標値}
    """
    cluster_flatness = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        pts = xyz[idx]
        # 法線＋曲率推定
        nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(pts)-1)).fit(pts)
        _, indices = nn.kneighbors(pts)
        curvatures = []
        for i, neighbors in enumerate(indices):
            p = pts[neighbors]
            cov = np.cov(p.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)
            # 曲率：最小固有値/総和（標準的な点群局所曲率定義）
            curvature = eigvals[0] / np.sum(eigvals) if np.sum(eigvals)>0 else 0
            curvatures.append(curvature)
        cluster_flatness[c] = np.mean(curvatures)
    return cluster_flatness

def compute_cluster_weights(flatness_dict, flatness_weighting=0):
    """
    flatness_weighting:
      0 → 重みなし（全クラスタ重み1.0）
      1 → 平坦クラスタほど重み大（傾斜）
      2 → 曲面クラスタほど重み大（傾斜）
      3 → 平坦クラスタ上位半分を重み最大化、下位半分を重み最小化
      4 → 曲面クラスタ上位半分を重み最大化、下位半分を重み最小化
    """
    if flatness_weighting == 0:
        # 重みなし
        return {c: 1.0 for c in flatness_dict.keys()}

    vals = np.array(list(flatness_dict.values()))
    clusters = list(flatness_dict.keys())
    min_f, max_f = np.min(vals), np.max(vals)
    
    if max_f == min_f:
        # 曲率全て同じ時は全クラスタ重み1.0
        return {c: 1.0 for c in flatness_dict.keys()}
    
    if flatness_weighting == 1:
        # 平坦度が大きいほど重み大
        scaled = (max_f - vals) / (max_f - min_f)
        scaled *= (1.0 / np.mean(scaled)) # 平均1に調整
        weights = {c: w for c, w in zip(clusters, scaled)}
    elif flatness_weighting == 2:
        # 平坦度が小さいほど重み大
        scaled = (vals - min_f) / (max_f - min_f)
        scaled *= (1.0 / np.mean(scaled)) # 平均1に調整
        weights = {c: w for c, w in zip(clusters, scaled)}
    elif flatness_weighting == 3:
        # 平坦クラスタ上位半分をmax_weight、下位半分をmin_weight
        sorted_indices = np.argsort(vals)  # 昇順ソート
        n_clusters = len(clusters)
        half = n_clusters // 2
        weights = {}
        for i, idx in enumerate(sorted_indices):
            cluster_id = clusters[idx]
            if i < half:
                weights[cluster_id] = 2  # 上位半分
            else:
                weights[cluster_id] = 0  # 下位半分
    elif flatness_weighting == 4:
        # 曲面クラスタ上位半分をmax_weight、下位半分をmin_weight
        sorted_indices = np.argsort(vals)[::-1]  # 降順ソート
        n_clusters = len(clusters)
        half = n_clusters // 2
        weights = {}
        for i, idx in enumerate(sorted_indices):
            cluster_id = clusters[idx]
            if i < half:
                weights[cluster_id] = 2  # 上位半分
            else:
                weights[cluster_id] = 0  # 下位半分
    else:
        raise ValueError("flatness_weightingは0（重みなし）, 1（平坦優遇傾斜）, 2（曲面優遇傾斜）, 3（平坦二分）, 4（曲面二分）で指定")
    
    return weights

def compute_pseudoplane_pca(pts: np.ndarray):
    """
    クラスタ点群 pts (Nc,3) から疑似平面をPCAで推定する。
    平面: 重心 centroid を通る
    法線: 共分散の最小固有値に対応する固有ベクトル（=最小分散方向）
    さらに、法線の符号は決定規則で固定して再現性を担保する。

    Returns:
        centroid: (3,)
        normal: (3,) unit vector (sign-fixed)
    """
    centroid = np.mean(pts, axis=0)
    X = pts - centroid

    # 共分散（対称行列）
    C = (X.T @ X) / max(len(pts), 1)

    # 対称行列の固有分解（eighは昇順で返る）
    eigvals, eigvecs = np.linalg.eigh(C)

    # 最小固有値の固有ベクトル = 平面法線
    normal = eigvecs[:, 0]

    # 正規化
    nrm = np.linalg.norm(normal)
    if nrm < 1e-12:
        # 退化（全点ほぼ同一点など）: 適当な法線
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        normal = normal / nrm

    # 法線符号の決定（再現性のための規則）
    # |成分|が最大の軸の成分が + になるようにする
    j = int(np.argmax(np.abs(normal)))
    if normal[j] < 0:
        normal = -normal

    return centroid, normal

def cluster_height_signal(pts: np.ndarray, centroid: np.ndarray, normal: np.ndarray):
    """
    平面（centroid, normal）に対する各点の符号付き高さ（距離）:
        h_i = (p_i - centroid) dot normal
    Returns:
        h: (Nc,)
    """
    return np.sum((pts - centroid) * normal[None, :], axis=1)

def make_pseudoplane_lineset(
    centroid: np.ndarray,
    normal: np.ndarray,
    pts_cluster: np.ndarray,
    scale: float = 1.2
):
    """
    疑似平面（centroid, normal）を可視化するためのLineSet（四角い枠）を作る
    """
    c = centroid.astype(float)
    n = normal.astype(float)
    n = n / max(np.linalg.norm(n), 1e-12)

    # 平面内の直交基底 u, v
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])

    u = np.cross(n, a)
    u = u / max(np.linalg.norm(u), 1e-12)
    v = np.cross(n, u)
    v = v / max(np.linalg.norm(v), 1e-12)

    # クラスタ点群を平面内へ射影してサイズ決定
    X = pts_cluster - c[None, :]
    su = X @ u
    sv = X @ v

    half_u = 0.5 * (np.max(su) - np.min(su))
    half_v = 0.5 * (np.max(sv) - np.min(sv))
    half_u = max(half_u, 1e-6) * scale
    half_v = max(half_v, 1e-6) * scale

    p0 = c + (-half_u) * u + (-half_v) * v
    p1 = c + ( half_u) * u + (-half_v) * v
    p2 = c + ( half_u) * u + ( half_v) * v
    p3 = c + (-half_u) * u + ( half_v) * v

    points = np.vstack([p0, p1, p2, p3])
    lines = np.array([[0,1],[1,2],[2,3],[3,0]], dtype=np.int32)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)

    return ls

def visualize_clusters_with_pseudoplanes(
    xyz: np.ndarray,
    labels: np.ndarray,
    color_list,
    scale: float = 1.2,
    max_planes: int = None,
    show_frame: bool = False,
    window_name: str = "Clusters + Pseudo-planes"
):
    """
    visualize_clusters() の返り値 color_list（クラスタ番号→RGB）に同期して、
    各クラスタの疑似平面（PCA）を同じ色で可視化する。

    - visualize_clusters 自体は変更しない
    - この関数は「点群（クラスタ色） + 疑似平面（同色）」を1つのウィンドウで表示する

    Parameters
    ----------
    xyz : (N,3) np.ndarray
    labels : (N,) np.ndarray (クラスタID)
    color_list : list[list[float]]  # visualize_clusters の返り値
    scale : float  # 平面枠の大きさ係数
    max_planes : int or None  # 表示する平面数上限（重いとき用）
    show_frame : bool  # 座標軸フレーム表示
    window_name : str
    """
    # --- 点群（クラスタ色）を作る ---
    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(xyz)

    labels = np.asarray(labels)
    n_clusters = len(color_list)

    # labels に -1 が混じるケースにも一応対応（-1はグレー）
    color_array = np.zeros((len(labels), 3), dtype=float)
    gray = np.array([0.6, 0.6, 0.6], dtype=float)

    for i, lb in enumerate(labels):
        if lb < 0 or lb >= n_clusters:
            color_array[i] = gray
        else:
            color_array[i] = np.array(color_list[lb], dtype=float)

    tmp_pcd.colors = o3d.utility.Vector3dVector(color_array)

    # --- 疑似平面（LineSet）をクラスタごとに生成して同色をつける ---
    plane_geoms = []
    unique_clusters = np.unique(labels[labels >= 0])  # -1は除外

    count = 0
    for c in unique_clusters:
        if max_planes is not None and count >= max_planes:
            break

        idx = np.where(labels == c)[0]
        if len(idx) < 3:
            continue

        pts = xyz[idx]

        centroid, normal = compute_pseudoplane_pca(pts)
        plane_ls = make_pseudoplane_lineset(centroid, normal, pts, scale=scale)

        # LineSetの colors は「線の本数」と同じ長さが推奨（4本）
        col = np.array(color_list[int(c)], dtype=float)
        n_lines = len(plane_ls.lines)
        plane_ls.colors = o3d.utility.Vector3dVector(np.tile(col[None, :], (n_lines, 1)))

        plane_geoms.append(plane_ls)
        count += 1

    geoms = [tmp_pcd] + plane_geoms

    if show_frame:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]))

    o3d.visualization.draw_geometries(geoms, window_name=window_name)
    return plane_geoms

# =========================================================
#  グラフ構築・埋め込み・抽出関数群
# =========================================================

def build_graph(xyz, graph_mode='knn', k=12, radius=0.05):
    """
    点群からグラフ（重み付き隣接行列）を構築する関数

    Parameters:
    - xyz: 点群座標 (N, 3)
    - mode: グラフ構築モード
        - 'knn': k近傍法 (推奨: k=12~16)
        - 'radius': 半径法 (radius以内の点を接続)
        - 'hybrid': k近傍かつ、radius以内の点のみ接続 (遠すぎる接続を排除)
    - k: k近傍法のk
    - radius: 半径法の閾値 (データのスケールに依存するため注意)

    Returns:
    - W: 重み付き隣接行列 (scipy.sparse.csr_matrix or dense ndarray)
    """
    
    # 1. 隣接関係の構築 (adjacency matrix)
    if graph_mode == 'knn':
        # include_self=Falseで自分自身へのループを排除
        adj = kneighbors_graph(xyz, k, mode='distance', include_self=False)
        
    elif graph_mode == 'radius':
        adj = radius_neighbors_graph(xyz, radius, mode='distance', include_self=False)
        
    elif graph_mode == 'hybrid':
        # まずk近傍で大きめに取る
        adj = kneighbors_graph(xyz, k, mode='distance', include_self=False)
        # 疎行列の構造を保ったまま、radiusより大きい距離のエッジを削除（0にする）
        adj.data[adj.data > radius] = 0
        adj.eliminate_zeros() # 0になったエッジを構造から削除
        
    else:
        raise ValueError("mode must be 'knn', 'radius', or 'hybrid'")

    # 2. 重みの計算 (Gaussian Kernel)
    # 行列形式を扱いやすい形に変換
    W = adj.toarray() 
    
    # 距離が存在するエッジのみ抽出
    mask = W > 0
    dists = W[mask]
    
    # 孤立点対策: エッジが一つもない場合はsigmaを1.0にしてエラー回避
    if len(dists) > 0:
        sigma = np.mean(dists)
        # ガウス重み: 近いほど1、遠いほど0に近づく
        W[mask] = np.exp(-W[mask]**2 / (sigma**2))
    else:
        # 万が一エッジがゼロの場合（radiusが小さすぎる時など）
        print("Warning: No edges found. Check your radius or k.")
    
    # 3. 対称化 (無向グラフにするため)
    # GFTでは通常、無向グラフ（対称行列）が望ましいため、W = (W + W.T) / 2 などをする場合が多いですが
    # ここでは元の実装に合わせてそのまま返します（あるいは以下のように最大値を取って対称化も可）
    W = np.maximum(W, W.T) 

    return W

def gft_basis(W):
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvecs, eigvals

def gft_basis_gpu(W):
    # 入力Wはnumpy配列（CPU）、ここでGPUに転送
    W_gpu = cp.asarray(W)
    D_gpu = cp.diag(W_gpu.sum(axis=1))
    L_gpu = D_gpu - W_gpu
    eigvals_gpu, eigvecs_gpu = cp.linalg.eigh(L_gpu)
    # 必要ならCPU（numpy配列）に戻す
    eigvals = cp.asnumpy(eigvals_gpu)
    eigvecs = cp.asnumpy(eigvecs_gpu)
    return eigvecs, eigvals

def gft(signal, basis):
    return basis.T @ signal

def igft(gft_coeffs, basis):
    return basis @ gft_coeffs

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

### 関数間違い注意！！
def embed_watermark_pseudoplane(
    xyz: np.ndarray,
    labels: np.ndarray,
    embed_bits,
    beta: float = 0.01,
    graph_mode: str = 'knn',
    k: int = 10,
    radius: float = 0.05,
    flatness_weighting: int = 0,
    k_neighbors: int = 20,
    min_spectre: float = 0.0,
    max_spectre: float = 1.0,
    skip_threshold_mode: str = "half",  # "half" or "none"
):
    """
    各クラスタで疑似平面を推定し、その平面からの高さ（符号付き距離）hを信号としてGFT→係数変調→逆GFT。
    座標更新は平面法線方向のみ:
        p'_i = p_i + (h'_i - h_i) * normal_c

    注意:
    - 抽出でも同じ疑似平面が再構築できる必要があるため、平面推定は乱数に依存しないPCAを使用。
    """
    xyz_after = xyz.copy()
    cluster_ids = np.unique(labels)
    # color_list = visualize_clusters(xyz, labels)
    # visualize_clusters_with_pseudoplanes(
    #     xyz, labels, color_list,
    #     scale=1.2,
    #     max_planes=30,      # 重ければ制限
    #     show_frame=False
    # )

    # 既存のクラスタ平坦度 -> 重み
    flatness_dict = estimate_cluster_flatness(xyz, labels, k_neighbors=k_neighbors)
    weights = compute_cluster_weights(flatness_dict, flatness_weighting)

    # 既存コードと同じスケール（phi）
    phi = max(
        np.max(xyz[:, 0]) - np.min(xyz[:, 0]),
        np.max(xyz[:, 1]) - np.min(xyz[:, 1]),
        np.max(xyz[:, 2]) - np.min(xyz[:, 2])
    )

    bits_len = len(embed_bits)
    if bits_len == 0:
        return xyz_after

    if skip_threshold_mode == "half":
        skip_threshold = bits_len / 2
    else:
        skip_threshold = -1  # 実質スキップしない

    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        if len(idx) <= skip_threshold:
            continue

        pts = xyz[idx]

        # 疑似平面（PCA）推定
        centroid, normal = compute_pseudoplane_pca(pts)

        # 高さ信号
        h = cluster_height_signal(pts, centroid, normal)

        # グラフ構築
        W = build_graph(pts, graph_mode=graph_mode, k=k, radius=radius)
        if cp is not None:
            basis, eigvals = gft_basis_gpu(W)
        else:
            basis, eigvals = gft_basis(W)

        # 高さ信号をGFT
        gft_coeffs = gft(h, basis)

        Q_ = len(gft_coeffs)
        q_start = int(Q_ * min_spectre)
        q_end = int(Q_ * max_spectre)
        Q_embed = max(q_end - q_start, 0)
        if Q_embed <= 0:
            continue

        # 冗長化
        n_repeat = Q_embed // bits_len if bits_len > 0 else 1
        redundant_bits = repeat_bits_blockwise(embed_bits, n_repeat, Q_embed)

        # 係数変調
        w_c = weights.get(c, 1.0)
        for i in range(Q_embed):
            w = redundant_bits[i] * 2 - 1  # 0/1 -> -1/+1
            gft_coeffs[q_start + i] += w * beta * w_c * phi

        # 逆GFT
        h_emb = igft(gft_coeffs, basis)

        # 座標更新（法線方向のみ）
        delta_h = (h_emb - h)  # (Nc,)
        xyz_after[idx] = pts + delta_h[:, None] * normal[None, :]

    return xyz_after


def extract_watermark_pseudoplane(
    xyz_emb: np.ndarray,
    xyz_orig: np.ndarray,
    labels: np.ndarray,
    embed_bits_length: int,
    graph_mode: str = 'knn',
    k: int = 10,
    radius: float = 0.05,
    min_spectre: float = 0.0,
    max_spectre: float = 1.0,
    skip_threshold_mode: str = "half",  # "half" or "none"
):
    """
    埋め込み前点群 xyz_orig から疑似平面を再構築し、
    高さ信号 h_orig と h_emb を作り、GFT係数差分の符号でビット抽出（多数決）。

    重要:
    - 疑似平面推定は xyz_orig（埋め込み前）で行う（xyz_embで推定すると透かし変位が平面推定に混入する）。
    - グラフも xyz_orig（クラスタ内座標）で構築する（埋め込み前と一致させる）。
    """
    xyz_emb = synchronize_point_cloud(xyz_emb, xyz_orig, verbose=True)
    if embed_bits_length <= 0:
        return []

    if skip_threshold_mode == "half":
        skip_threshold = embed_bits_length / 2
    else:
        skip_threshold = -1

    bit_lists = [[] for _ in range(embed_bits_length)]

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) <= skip_threshold:
            continue

        pts_orig = xyz_orig[idx]
        pts_emb = xyz_emb[idx]
        if len(pts_orig) != len(pts_emb):
            continue

        # 疑似平面はorigから
        centroid, normal = compute_pseudoplane_pca(pts_orig)

        # 高さ信号（orig/emb）
        h_orig = cluster_height_signal(pts_orig, centroid, normal)
        h_emb = cluster_height_signal(pts_emb, centroid, normal)

        # グラフはorigで
        W = build_graph(pts_orig, graph_mode=graph_mode, k=k, radius=radius)
        if cp is not None:
            basis, eigvals = gft_basis_gpu(W)
        else:
            basis, eigvals = gft_basis(W)

        gft_orig = gft(h_orig, basis)
        gft_emb = gft(h_emb, basis)

        Q_ = len(gft_emb)
        q_start = int(Q_ * min_spectre)
        q_end = int(Q_ * max_spectre)
        Q_extract = max(q_end - q_start, 0)
        if Q_extract <= 0:
            continue

        n_repeat = Q_extract // embed_bits_length if embed_bits_length > 0 else 1

        # ブロック繰り返し分
        for bit_idx in range(embed_bits_length):
            for rep in range(n_repeat):
                i = bit_idx * n_repeat + rep
                if i < Q_extract:
                    diff = gft_emb[q_start + i] - gft_orig[q_start + i]
                    bit = 1 if diff > 0 else 0
                    bit_lists[bit_idx].append(bit)

        # 余り分（もしあれば）
        for i in range(n_repeat * embed_bits_length, Q_extract):
            bit_idx = i % embed_bits_length
            diff = gft_emb[q_start + i] - gft_orig[q_start + i]
            bit = 1 if diff > 0 else 0
            bit_lists[bit_idx].append(bit)

    # 多数決
    extracted_bits = []
    for votes in bit_lists:
        if len(votes) == 0:
            extracted_bits.append(0)
        else:
            extracted_bits.append(1 if votes.count(1) > votes.count(0) else 0)

    return extracted_bits

def synchronize_point_cloud(xyz_att, xyz_orig, distance_threshold=None, verbose=True):
    """
    攻撃後点群をオリジナル点群と完全に同期（同じ点数・順序）させる。
    """
    if distance_threshold is None:
        max_bound = np.max(xyz_orig, axis=0)
        min_bound = np.min(xyz_orig, axis=0)
        scale = np.linalg.norm(max_bound - min_bound)
        distance_threshold = scale * 0.01

    tree = cKDTree(xyz_att)
    dists, indices = tree.query(xyz_orig, k=1)
    
    xyz_synced = xyz_att[indices].copy()
    
    missing_mask = dists > distance_threshold
    xyz_synced[missing_mask] = xyz_orig[missing_mask]
    
    if verbose:
        n_missing = np.sum(missing_mask)
        print(f"[Sync] 再サンプリング完了。補完された欠損点数: {n_missing} / {len(xyz_orig)}")

    return xyz_synced

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

# =========================================================
#  評価関数群
# =========================================================

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
        print(f"PC-MSDM (Original Paper eq, sym, p={m}): {final_score:.4f} (Raw Distortion={symmetric_distortion:.4f})")
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
        print(f"PointSSIM (attr={attribute}, disp={dispersion}, m={m}): {final_score:.4f} (error_sym={error_sym:.4f})")
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

def downsampling_attack(xyz, keep_ratio=0.5, mode='voxel', voxel_size=0.02, seed=None):
    """
    点群に対してダウンサンプリング攻撃を行う。
    
    Parameters:
    - xyz (np.ndarray): 入力点群座標（N×3）
    - keep_ratio (float): ランダム・FPSモードで残す点の割合 (0.0 < keep_ratio <= 1.0)
    - mode (str): ダウンサンプリングの方式
        - 'random': ランダムに点をサンプリング（高速）
        - 'voxel': Open3Dを用いたVoxel Gridダウンサンプリング（均一化）
        - 'fps': 最遠点サンプリング（計算量は多いが極めて均一）
    - voxel_size (float): 'voxel' モード時のボクセルサイズ（辺の長さ）
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
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        xyz_downsampled = np.asarray(pcd_down.points)
        print(f"[Attack] ボクセル・ダウンサンプリング (voxel_size={voxel_size}): 元点数={N} → 残点数={xyz_downsampled.shape[0]}")
        
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