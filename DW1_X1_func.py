import numpy as np
import open3d as o3d
import random
import string
from sklearn.cluster import KMeans
import DW2_func as DW2F
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

def embed_watermark_x1(
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

def extract_watermark_x1(
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
    xyz_emb = DW2F.synchronize_point_cloud(xyz_emb, xyz_orig, verbose=True)
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
