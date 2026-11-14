import numpy as np
import open3d as o3d
import os
import sys
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.csgraph import (
    connected_components,
    min_weight_full_bipartite_matching,
)
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import DW2_func as DW2F
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
from modules.assignment import hk_matching, auction_assignment, warmup_assignment
try:
    import cupy as cp
except ImportError as exc:
    cp = None
    _CUPY_IMPORT_ERROR = exc
else:
    _CUPY_IMPORT_ERROR = None


def _detect_gft_gpu():
    """CuPyが利用可能なCUDAデバイス上で実行できるか判定する。"""
    if cp is None:
        return False, "CuPyがインストールされていません"
    try:
        if not cp.is_available():
            return False, "CuPyが利用可能なCUDAデバイスを検出できませんでした"
        device_id = int(cp.cuda.runtime.getDevice())
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        device_name = properties["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode(errors="replace")
        # 最初の大規模なクラスタ固有値分解を行う前に、最小限のGPUメモリを
        # 確保し、CUDAランタイムやドライバの不足を検出する。
        cp.asarray([0.0])
        cp.cuda.Stream.null.synchronize()
        return True, f"CUDAデバイス {device_id}: {device_name}"
    except Exception as exc:
        return False, f"CuPy/CUDAの初期化に失敗しました: {exc}"


_GFT_GPU_AVAILABLE, _GFT_BACKEND_DETAIL = _detect_gft_gpu()
_GFT_BACKEND_REPORTED = False

# DualBand watermark / correspondence defaults.  The code key defines the
# deterministic spread chips and is not related to input point order.
DUAL_BAND_CODE_KEY = 20261116
DUAL_BAND_LOW_RANGE = (0.0, 0.2)
DUAL_BAND_HIGH_RANGE = (0.6, 1.0)
DUAL_BAND_LOW_ENERGY_SHARE = 0.30
ONE_TO_ONE_CANDIDATES = (8, 16, 32, 64, 128)
ONE_TO_ONE_AUTO_WORKER_THRESHOLD = 100_000
ONE_TO_ONE_MODES = ("hk", "auction", "bidir")
CORRESPONDENCE_MODES = ("auto", "index", "coordinate") + ONE_TO_ONE_MODES
BIDIRECTIONAL_CANDIDATES = (8, 16, 32)
# Temporary investigation switch.  The summary is deliberately emitted to
# stderr so DW2_exp's stdout suppression does not hide correspondence timing.
ONE_TO_ONE_PROFILE_LOGS = True

# =========================================================
#  クラスタリング関数群
# =========================================================

def kmeans_cluster_points(xyz, cluster_point=1000, seed=42):
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("xyz must have shape (N, 3) and be non-empty.")
    if cluster_point <= 0:
        raise ValueError("cluster_point must be positive.")
    num_clusters = len(xyz) // cluster_point
    if num_clusters < 1:
        raise ValueError(
            f"cluster_point={cluster_point} exceeds the vertex count {len(xyz)}."
        )
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
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
        kmeans = KMeans(
            n_clusters=n_subclusters, random_state=seed, n_init=10
        )
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
    if not _GFT_GPU_AVAILABLE:
        raise RuntimeError(_GFT_BACKEND_DETAIL)
    # 入力Wはnumpy配列（CPU）、ここでGPUに転送
    W_gpu = cp.asarray(W)
    D_gpu = cp.diag(W_gpu.sum(axis=1))
    L_gpu = D_gpu - W_gpu
    eigvals_gpu, eigvecs_gpu = cp.linalg.eigh(L_gpu)
    # 必要ならCPU（numpy配列）に戻す
    eigvals = cp.asnumpy(eigvals_gpu)
    eigvecs = cp.asnumpy(eigvecs_gpu)
    return eigvecs, eigvals


def gft_basis_auto(W):
    """CUDAが利用可能な場合はGPUでGFT基底の固有値分解を行う。"""
    global _GFT_BACKEND_REPORTED
    if not _GFT_BACKEND_REPORTED:
        backend = "GPU" if _GFT_GPU_AVAILABLE else "CPU"
        print(f"[GFT] 基底計算バックエンド: {backend} ({_GFT_BACKEND_DETAIL})")
        _GFT_BACKEND_REPORTED = True
    if _GFT_GPU_AVAILABLE:
        return gft_basis_gpu(W)
    return gft_basis(W)


def gft_analyze_signals_auto(W, signals):
    """固有基底をGPU上に保持したまま、1つ以上の信号のGFT係数を計算する。"""
    signals = np.asarray(signals, dtype=np.float64)
    was_vector = signals.ndim == 1
    if was_vector:
        signals = signals[:, None]
    if signals.ndim != 2 or signals.shape[0] != len(W):
        raise ValueError("signalsの形状は(N,)または(N, K)である必要があります。")

    global _GFT_BACKEND_REPORTED
    if not _GFT_BACKEND_REPORTED:
        backend = "GPU" if _GFT_GPU_AVAILABLE else "CPU"
        print(f"[GFT] 基底計算バックエンド: {backend} ({_GFT_BACKEND_DETAIL})")
        _GFT_BACKEND_REPORTED = True

    if _GFT_GPU_AVAILABLE:
        W_gpu = cp.asarray(W)
        laplacian_gpu = cp.diag(W_gpu.sum(axis=1)) - W_gpu
        eigvals_gpu, basis_gpu = cp.linalg.eigh(laplacian_gpu)
        coefficients_gpu = basis_gpu.T @ cp.asarray(signals)
        eigvals = cp.asnumpy(eigvals_gpu)
        coefficients = cp.asnumpy(coefficients_gpu)
    else:
        degree = np.diag(W.sum(axis=1))
        eigvals, basis = np.linalg.eigh(degree - W)
        coefficients = basis.T @ signals

    if was_vector:
        coefficients = coefficients[:, 0]
    return coefficients, eigvals


def gft_synthesize_changes_auto(W, coefficient_changes):
    """固有基底を返さずに、1つ以上のスペクトル変化を逆GFTする。"""
    coefficient_changes = np.asarray(coefficient_changes, dtype=np.float64)
    was_vector = coefficient_changes.ndim == 1
    if was_vector:
        coefficient_changes = coefficient_changes[:, None]
    if coefficient_changes.ndim != 2 or coefficient_changes.shape[0] != len(W):
        raise ValueError(
            "coefficient_changesの形状は(N,)または(N, K)である必要があります。"
        )

    global _GFT_BACKEND_REPORTED
    if not _GFT_BACKEND_REPORTED:
        backend = "GPU" if _GFT_GPU_AVAILABLE else "CPU"
        print(f"[GFT] 基底計算バックエンド: {backend} ({_GFT_BACKEND_DETAIL})")
        _GFT_BACKEND_REPORTED = True

    if _GFT_GPU_AVAILABLE:
        W_gpu = cp.asarray(W)
        laplacian_gpu = cp.diag(W_gpu.sum(axis=1)) - W_gpu
        _, basis_gpu = cp.linalg.eigh(laplacian_gpu)
        changes = cp.asnumpy(basis_gpu @ cp.asarray(coefficient_changes))
    else:
        degree = np.diag(W.sum(axis=1))
        _, basis = np.linalg.eigh(degree - W)
        changes = basis @ coefficient_changes

    if was_vector:
        changes = changes[:, 0]
    return changes


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


def _small_cluster_size_limit(watermark_length, small_cluster_policy):
    """Return the inclusive size limit used to reject small clusters.

    ``half_watermark`` preserves the existing rule: clusters containing at
    most half as many vertices as watermark bits are not used. ``none``
    disables only this size-based rejection; correspondence failures are
    handled separately during extraction.
    """
    if small_cluster_policy == "half_watermark":
        return watermark_length / 2
    if small_cluster_policy == "none":
        return -1
    raise ValueError(
        "small_cluster_policy must be 'half_watermark' or 'none'."
    )


def prepare_watermark_x1_band_displacements(
    xyz,
    labels,
    embed_bits,
    bands,
    graph_mode='knn',
    k=10,
    radius=0.05,
    flatness_weighting=0,
    k_neighbors=20,
    small_cluster_policy="half_watermark",
):
    """複数のGFT周波数帯について、beta=1の座標変位を一括計算する。

    ``bands``には``(名前, 最小周波数, 最大周波数)``のタプルを渡す。
    各クラスタのグラフを一度だけ固有値分解し、指定された全周波数帯の変位を
    同時に生成する。得られた変位は、グラフ構築や固有値分解をやり直さず、
    任意のbetaに比例させて利用できる。
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    labels = np.asarray(labels)
    embed_bits = list(embed_bits)
    band_specs = list(bands)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyzの形状は(N, 3)である必要があります。")
    if labels.shape != (len(xyz),):
        raise ValueError("labelsの形状は(N,)である必要があります。")
    if not embed_bits:
        return {name: np.zeros_like(xyz) for name, _, _ in band_specs}
    if not band_specs:
        return {}

    names = [name for name, _, _ in band_specs]
    if len(set(names)) != len(names):
        raise ValueError("周波数帯の名前は重複しないように指定してください。")
    for name, minimum, maximum in band_specs:
        if not 0.0 <= minimum <= maximum <= 1.0:
            raise ValueError(
                f"{name}: 周波数範囲は0 <= 最小値 <= 最大値 <= 1を"
                "満たす必要があります。"
            )

    cluster_ids = np.unique(labels)
    if flatness_weighting == 0:
        weights = {cluster_id: 1.0 for cluster_id in cluster_ids}
    else:
        flatness = estimate_cluster_flatness(
            xyz, labels, k_neighbors=k_neighbors
        )
        weights = compute_cluster_weights(flatness, flatness_weighting)
    phi = float(np.max(np.ptp(xyz, axis=0)))
    displacements = {name: np.zeros_like(xyz) for name in names}
    small_cluster_size_limit = _small_cluster_size_limit(
        len(embed_bits), small_cluster_policy
    )

    for cluster_id in cluster_ids:
        cluster_indices = np.flatnonzero(labels == cluster_id)
        if len(cluster_indices) <= small_cluster_size_limit:
            continue
        points = xyz[cluster_indices]
        _, normal = compute_pseudoplane_pca(points)
        graph = build_graph(points, graph_mode=graph_mode, k=k, radius=radius)

        coefficient_changes = np.zeros((len(points), len(band_specs)))
        cluster_weight = weights.get(cluster_id, 1.0)
        for column, (_, minimum, maximum) in enumerate(band_specs):
            start = int(len(points) * minimum)
            end = int(len(points) * maximum)
            count = max(end - start, 0)
            if count == 0:
                continue
            repeat_count = count // len(embed_bits)
            redundant = repeat_bits_blockwise(embed_bits, repeat_count, count)
            signs = np.asarray(redundant, dtype=np.float64) * 2.0 - 1.0
            coefficient_changes[start:end, column] = (
                signs * cluster_weight * phi
            )

        height_changes = gft_synthesize_changes_auto(
            graph, coefficient_changes
        )
        for column, name in enumerate(names):
            displacements[name][cluster_indices] = (
                height_changes[:, column, None] * normal[None, :]
            )

    return displacements


def _dual_band_balanced_chips(length, rng):
    """Return a deterministic, nearly balanced +/-1 chip sequence."""
    chips = np.ones(length, dtype=np.float64)
    chips[: length // 2] = -1.0
    rng.shuffle(chips)
    return chips


def _dual_band_layout(cluster_id, point_count, bit_length, band, band_id, code_key):
    start = int(point_count * band[0])
    end = int(point_count * band[1])
    count = end - start
    if count <= 0:
        raise ValueError(f"DualBand has an empty GFT band: {band}.")
    rng = np.random.RandomState(
        int(code_key) + 1009 * int(cluster_id) + 97 * int(band_id)
    )
    bit_index = np.arange(count, dtype=np.int64) % int(bit_length)
    rng.shuffle(bit_index)
    return {
        "start": start,
        "end": end,
        "bit_index": bit_index,
        "chips": _dual_band_balanced_chips(count, rng),
    }


def _dual_band_layouts(cluster_id, point_count, bit_length, low_band, high_band, code_key):
    return {
        "Low": _dual_band_layout(
            cluster_id, point_count, bit_length, low_band, 0, code_key
        ),
        "High": _dual_band_layout(
            cluster_id, point_count, bit_length, high_band, 1, code_key
        ),
    }


def prepare_watermark_x1_dualband_displacement(
    xyz,
    labels,
    embed_bits,
    graph_mode="knn",
    k=10,
    radius=0.05,
    flatness_weighting=0,
    k_neighbors=20,
    small_cluster_policy="half_watermark",
    low_band=DUAL_BAND_LOW_RANGE,
    high_band=DUAL_BAND_HIGH_RANGE,
    low_energy_share=DUAL_BAND_LOW_ENERGY_SHARE,
    code_key=DUAL_BAND_CODE_KEY,
):
    """Build the unit displacement for the proposed ``DualBand`` method.

    Low (default 0--20%) and High (default 60--100%) GFT regions use separate
    deterministic spread codes.  Each region is RMS-normalized before mixing,
    so ``low_energy_share=0.30`` gives 30% / 70% displacement energy while the
    caller remains free to set the final global MSE/PSNR.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    labels = np.asarray(labels)
    bits = np.asarray(embed_bits, dtype=np.uint8).ravel()
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3).")
    if labels.shape != (len(xyz),):
        raise ValueError("labels must have one entry per point.")
    if len(bits) == 0:
        return np.zeros_like(xyz)
    if not 0.0 <= low_energy_share <= 1.0:
        raise ValueError("low_energy_share must be in [0, 1].")
    for name, band in (("low_band", low_band), ("high_band", high_band)):
        if len(band) != 2 or not 0.0 <= band[0] < band[1] <= 1.0:
            raise ValueError(f"{name} must satisfy 0 <= start < end <= 1.")

    cluster_ids = np.unique(labels)
    if flatness_weighting == 0:
        weights = {cluster_id: 1.0 for cluster_id in cluster_ids}
    else:
        flatness = estimate_cluster_flatness(
            xyz, labels, k_neighbors=k_neighbors
        )
        weights = compute_cluster_weights(flatness, flatness_weighting)

    phi = float(np.max(np.ptp(xyz, axis=0)))
    signs = bits.astype(np.float64) * 2.0 - 1.0
    units = {"Low": np.zeros_like(xyz), "High": np.zeros_like(xyz)}
    small_cluster_size_limit = _small_cluster_size_limit(
        len(bits), small_cluster_policy
    )
    for cluster_id in cluster_ids:
        index = np.flatnonzero(labels == cluster_id)
        if len(index) <= small_cluster_size_limit:
            continue
        points = xyz[index]
        _, normal = compute_pseudoplane_pca(points)
        basis, _ = gft_basis_auto(
            build_graph(points, graph_mode=graph_mode, k=k, radius=radius)
        )
        layouts = _dual_band_layouts(
            cluster_id, len(index), len(bits), low_band, high_band, code_key
        )
        cluster_weight = weights.get(cluster_id, 1.0)
        for name, layout in layouts.items():
            coefficient_changes = np.zeros(len(index), dtype=np.float64)
            coefficient_changes[layout["start"]:layout["end"]] = (
                phi
                * cluster_weight
                * signs[layout["bit_index"]]
                * layout["chips"]
            )
            units[name][index] = (
                basis @ coefficient_changes
            )[:, None] * normal[None, :]

    pieces = []
    for name, share in (("Low", low_energy_share), ("High", 1.0 - low_energy_share)):
        if share <= 0.0:
            continue
        rms = float(np.sqrt(np.mean(np.sum(units[name] ** 2, axis=1))))
        if not np.isfinite(rms) or rms <= 0.0:
            raise RuntimeError(f"DualBand {name} produced zero displacement.")
        pieces.append(np.sqrt(share) * units[name] / rms)
    return np.sum(pieces, axis=0)

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
    small_cluster_policy: str = "half_watermark",
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
    if flatness_weighting == 0:
        weights = {cluster_id: 1.0 for cluster_id in cluster_ids}
    else:
        flatness_dict = estimate_cluster_flatness(
            xyz, labels, k_neighbors=k_neighbors
        )
        weights = compute_cluster_weights(flatness_dict, flatness_weighting)

    # 既存コードと同じスケール（phi）
    phi = max(
        np.max(xyz[:, 0]) - np.min(xyz[:, 0]),
        np.max(xyz[:, 1]) - np.min(xyz[:, 1]),
        np.max(xyz[:, 2]) - np.min(xyz[:, 2])
    )

    watermark_length = len(embed_bits)
    if watermark_length == 0:
        return xyz_after

    small_cluster_size_limit = _small_cluster_size_limit(
        watermark_length, small_cluster_policy
    )

    for cluster_id in cluster_ids:
        cluster_indices = np.flatnonzero(labels == cluster_id)
        if len(cluster_indices) <= small_cluster_size_limit:
            continue

        cluster_points = xyz[cluster_indices]

        # 疑似平面（PCA）推定
        centroid, normal = compute_pseudoplane_pca(cluster_points)

        # 高さ信号
        h = cluster_height_signal(cluster_points, centroid, normal)

        # グラフ構築
        W = build_graph(
            cluster_points, graph_mode=graph_mode, k=k, radius=radius
        )
        basis, eigvals = gft_basis_auto(W)

        # 高さ信号をGFT
        gft_coeffs = gft(h, basis)

        Q_ = len(gft_coeffs)
        q_start = int(Q_ * min_spectre)
        q_end = int(Q_ * max_spectre)
        Q_embed = max(q_end - q_start, 0)
        if Q_embed <= 0:
            continue

        # 冗長化
        n_repeat = Q_embed // watermark_length
        redundant_bits = repeat_bits_blockwise(embed_bits, n_repeat, Q_embed)

        # 係数変調
        w_c = weights.get(cluster_id, 1.0)
        for i in range(Q_embed):
            w = redundant_bits[i] * 2 - 1  # 0/1 -> -1/+1
            gft_coeffs[q_start + i] += w * beta * w_c * phi

        # 逆GFT
        h_emb = igft(gft_coeffs, basis)

        # 座標更新（法線方向のみ）
        delta_h = (h_emb - h)  # (Nc,)
        xyz_after[cluster_indices] = (
            cluster_points + delta_h[:, None] * normal[None, :]
        )

    return xyz_after

def match_point_cloud_to_original(
    xyz_att,
    xyz_orig,
    distance_threshold=None,
    distance_factor=30.0,
    verbose=True,
):
    """攻撃後点群を元頂点位置へ多対1対応させ、遠すぎる対応を棄却する。

    複数の元頂点が同じ攻撃後点を参照できるため、ダウンサンプリング後も
    元点群と同じ頂点数の信号として扱える。対応なし座標にはNaNを設定する。
    """
    xyz_att = np.asarray(xyz_att, dtype=np.float64)
    xyz_orig = np.asarray(xyz_orig, dtype=np.float64)
    if xyz_att.ndim != 2 or xyz_att.shape[1] != 3 or len(xyz_att) == 0:
        raise ValueError("xyz_att must have shape (N, 3) and be non-empty.")
    if xyz_orig.ndim != 2 or xyz_orig.shape[1] != 3:
        raise ValueError("xyz_orig must have shape (N, 3).")
    if distance_factor <= 0:
        raise ValueError("distance_factor must be positive.")

    attacked_tree = cKDTree(xyz_att)
    if distance_threshold is None:
        if len(xyz_att) < 2:
            raise ValueError(
                "At least two attacked points are required to estimate spacing."
            )
        neighbor_distances, _ = attacked_tree.query(xyz_att, k=2)
        spacing = neighbor_distances[:, 1]
        spacing = spacing[np.isfinite(spacing) & (spacing > 1e-12)]
        if len(spacing) == 0:
            raise ValueError(
                "Cannot estimate spacing from duplicate attacked points."
            )
        distance_threshold = float(distance_factor * np.median(spacing))
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be non-negative.")

    distances, indices = attacked_tree.query(xyz_orig, k=1)
    valid = np.isfinite(distances) & (distances <= distance_threshold)
    matched = xyz_att[indices].copy()
    matched[~valid] = np.nan
    if verbose:
        print(
            f"[Match] threshold={distance_threshold:.6e}, "
            f"unmatched={np.count_nonzero(~valid)} / {len(xyz_orig)}"
        )
    return matched, valid, distances, float(distance_threshold)


def _one_to_one_candidate_graph(distances, candidate_global):
    """Build the sparse assignment graph for one candidate prefix."""
    row_count, candidate_count = candidate_global.shape
    candidates = np.unique(candidate_global)
    if len(candidates) < row_count:
        return None, None
    graph = csr_matrix(
        (
            np.maximum(distances.ravel() ** 2, 1e-15),
            (
                np.repeat(np.arange(row_count, dtype=np.int64), candidate_count),
                np.searchsorted(candidates, candidate_global).ravel(),
            ),
        ),
        shape=(row_count, len(candidates)),
    )
    return graph, candidates


def _bipartite_component_capacity_ok(graph):
    """Fast necessary condition for a full row matching.

    If any disconnected bipartite component has more original rows than
    received-point columns, Hall's condition is already violated.  This is
    cheaper than a maximum matching and can only reject impossible prefixes;
    a positive result still proceeds to the exact matcher.
    """
    row_count, column_count = graph.shape
    bipartite = bmat(
        [[None, graph], [graph.T, None]],
        format="csr",
    )
    component_count, labels = connected_components(
        bipartite, directed=False, return_labels=True
    )
    row_counts = np.bincount(labels[:row_count], minlength=component_count)
    column_counts = np.bincount(
        labels[row_count:row_count + column_count], minlength=component_count
    )
    return bool(np.all(row_counts <= column_counts))


def _solve_one_to_one_cluster(
    tree, xyz_orig, cluster_id, rows, candidate_counts, mode="hk"
):
    """Solve a cluster using HK feasibility and exact or auction assignment."""
    if mode == "bidir":
        return _solve_bidirectional_cluster(
            tree, xyz_orig, cluster_id, rows, max(candidate_counts)
        )
    profile = {
        "query_seconds": 0.0,
        "graph_seconds": 0.0,
        "capacity_seconds": 0.0,
        "capacity_rejections": 0,
        "feasibility_seconds": 0.0,
        "min_cost_seconds": 0.0,
        "auction_seconds": 0.0,
        "attempts": [],
    }
    # Reuse each KD query for its candidate prefixes.  This preserves the fixed
    # 8,16,32,64,128 schedule while avoiding five independent neighbor queries.
    stages = (
        tuple(value for value in candidate_counts if value <= 16),
        tuple(value for value in candidate_counts if value > 16),
    )
    for stage in stages:
        if not stage:
            continue
        maximum = min(max(stage), tree.n)
        started = time.perf_counter()
        distances_all, candidates_all = tree.query(xyz_orig[rows], k=maximum)
        profile["query_seconds"] += time.perf_counter() - started
        if maximum == 1:
            distances_all = distances_all[:, None]
            candidates_all = candidates_all[:, None]
        for requested_count in stage:
            candidate_count = min(requested_count, maximum)
            started = time.perf_counter()
            graph, candidates = _one_to_one_candidate_graph(
                distances_all[:, :candidate_count],
                candidates_all[:, :candidate_count],
            )
            profile["graph_seconds"] += time.perf_counter() - started
            if graph is None:
                profile["attempts"].append((candidate_count, False, 0.0, 0.0))
                continue
            started = time.perf_counter()
            capacity_ok = _bipartite_component_capacity_ok(graph)
            profile["capacity_seconds"] += time.perf_counter() - started
            if not capacity_ok:
                profile["capacity_rejections"] += 1
                profile["attempts"].append((candidate_count, False, 0.0, 0.0))
                continue
            started = time.perf_counter()
            feasibility = hk_matching(graph.indptr, graph.indices, graph.shape[1])
            feasibility_seconds = time.perf_counter() - started
            profile["feasibility_seconds"] += feasibility_seconds
            if np.any(feasibility < 0):
                profile["attempts"].append(
                    (candidate_count, False, feasibility_seconds, 0.0)
                )
                continue
            if mode == "auction":
                started = time.perf_counter()
                costs = np.ascontiguousarray(np.maximum(
                    distances_all[:, :candidate_count] ** 2, 1e-15
                ))
                columns = np.ascontiguousarray(np.searchsorted(
                    candidates, candidates_all[:, :candidate_count]
                ), dtype=np.int64)
                epsilon = max(float(np.median(costs[:, 0])) * .01, 1e-15)
                local_columns, _ = auction_assignment(
                    costs, columns, len(candidates), epsilon
                )
                auction_seconds = time.perf_counter() - started
                profile["auction_seconds"] += auction_seconds
                profile["attempts"].append(
                    (candidate_count, True, feasibility_seconds, auction_seconds)
                )
                return rows, candidates[local_columns], candidate_count, profile
            try:
                started = time.perf_counter()
                local_rows, local_columns = min_weight_full_bipartite_matching(
                    graph
                )
                min_cost_seconds = time.perf_counter() - started
                profile["min_cost_seconds"] += min_cost_seconds
            except ValueError:
                min_cost_seconds = time.perf_counter() - started
                profile["min_cost_seconds"] += min_cost_seconds
                profile["attempts"].append(
                    (candidate_count, False, feasibility_seconds, min_cost_seconds)
                )
                continue
            if len(local_rows) != len(rows):
                profile["attempts"].append(
                    (candidate_count, False, feasibility_seconds, min_cost_seconds)
                )
                continue
            assigned = np.empty(len(rows), dtype=np.int64)
            assigned[local_rows] = candidates[local_columns]
            profile["attempts"].append(
                (candidate_count, True, feasibility_seconds, min_cost_seconds)
            )
            return rows, assigned, candidate_count, profile
    raise RuntimeError(
        f"{mode} could not match original cluster "
        f"{cluster_id} with candidates={candidate_counts}."
    )


def _solve_bidirectional_cluster(tree, xyz_orig, cluster_id, rows, pool_count):
    """Sparse exact assignment on forward/reverse neighbour edges.

    The column pool is the union of each row's ``pool_count`` nearest received
    points. Try 8, 16, and 32 forward AND reverse neighbours within that pool.
    Duplicate edges keep their cost once (sparse maximum, never a sum).
    """
    profile = {
        "query_seconds": 0.0, "graph_seconds": 0.0,
        "capacity_seconds": 0.0, "capacity_rejections": 0,
        "feasibility_seconds": 0.0, "min_cost_seconds": 0.0,
        "auction_seconds": 0.0, "attempts": [],
    }
    started = time.perf_counter()
    points = xyz_orig[rows]
    pool_count = min(pool_count, tree.n)
    distances, indices = tree.query(points, k=pool_count)
    distances = distances.reshape(len(rows), pool_count)
    indices = indices.reshape(len(rows), pool_count)
    pool = np.unique(indices)
    if len(pool) < len(rows):
        raise RuntimeError(f"bidir has too few received candidates for cluster {cluster_id}.")
    reverse_count = min(max(BIDIRECTIONAL_CANDIDATES), len(rows))
    reverse_distances, reverse_rows = cKDTree(points).query(
        tree.data[pool], k=reverse_count
    )
    reverse_distances = reverse_distances.reshape(len(pool), reverse_count)
    reverse_rows = reverse_rows.reshape(len(pool), reverse_count)
    profile["query_seconds"] += time.perf_counter() - started
    for count in BIDIRECTIONAL_CANDIDATES:
        forward_count = min(count, pool_count)
        backward_count = min(count, reverse_count)
        started = time.perf_counter()
        forward = csr_matrix(
            (np.maximum(distances[:, :forward_count].ravel() ** 2, 1e-15),
             (np.repeat(np.arange(len(rows)), forward_count),
              np.searchsorted(pool, indices[:, :forward_count]).ravel())),
            shape=(len(rows), len(pool)),
        )
        reverse = csr_matrix(
            (np.maximum(reverse_distances[:, :backward_count].ravel() ** 2, 1e-15),
             (reverse_rows[:, :backward_count].ravel(),
              np.repeat(np.arange(len(pool)), backward_count))),
            shape=forward.shape,
        )
        graph = forward.maximum(reverse)
        profile["graph_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        try:
            local_rows, local_columns = min_weight_full_bipartite_matching(graph)
        except ValueError:
            seconds = time.perf_counter() - started
            profile["min_cost_seconds"] += seconds
            profile["attempts"].append((count, False, 0.0, seconds))
            continue
        seconds = time.perf_counter() - started
        profile["min_cost_seconds"] += seconds
        assigned = np.empty(len(rows), dtype=np.int64)
        assigned[local_rows] = pool[local_columns]
        profile["attempts"].append((count, True, 0.0, seconds))
        return rows, assigned, count, profile
    raise RuntimeError(
        f"bidir could not match original cluster {cluster_id} "
        f"with pool={pool_count}, neighbours={BIDIRECTIONAL_CANDIDATES}."
    )


def _one_to_one_profile_summary(results):
    """Aggregate per-cluster timing without retaining large candidate arrays."""
    selected = {count: 0 for count in ONE_TO_ONE_CANDIDATES}
    rejected = {count: 0 for count in ONE_TO_ONE_CANDIDATES}
    timings = {
        "query_seconds": 0.0,
        "graph_seconds": 0.0,
        "capacity_seconds": 0.0,
        "capacity_rejections": 0,
        "feasibility_seconds": 0.0,
        "min_cost_seconds": 0.0,
        "auction_seconds": 0.0,
    }
    for _, _, chosen_count, profile in results:
        selected[chosen_count] = selected.get(chosen_count, 0) + 1
        for key in timings:
            timings[key] += profile[key]
        for count, feasible, _, _ in profile["attempts"]:
            if not feasible:
                rejected[count] = rejected.get(count, 0) + 1
    return selected, rejected, timings


def _one_to_one_worker_count(requested_workers, point_count, cluster_count):
    requested_workers = int(requested_workers)
    if requested_workers < 0:
        raise ValueError("workers must be zero (auto) or a positive integer.")
    if requested_workers == 0:
        if point_count < ONE_TO_ONE_AUTO_WORKER_THRESHOLD:
            return 1
        requested_workers = min(4, os.cpu_count() or 1)
    return max(1, min(requested_workers, cluster_count))


def match_point_cloud_one_to_one(
    xyz_att,
    xyz_orig,
    labels,
    distance_factor=30.0,
    candidate_counts=ONE_TO_ONE_CANDIDATES,
    workers=0,
    verbose=True,
    mode="hk",
):
    """Match original points to received points by a local geometry-only bijection.

    ``hk`` checks candidate prefixes with compiled Hopcroft-Karp, then uses
    the original sparse exact minimum-cost solver on the first feasible prefix.
    ``auction`` uses the same prefixes and HK check, then approximate auction.
    ``bidir`` uses 8/16/32 forward and reverse edges from a column pool defined
    by max(candidate_counts). No original/received row identity is used.

    Uniqueness is local to each GFT cluster; different clusters can reuse a
    received point. Arrays may have different row counts (El Zein carriers).
    The alignment wrapper handles count reduction by coordinate fallback.

    ``workers=0`` uses serial execution below 100,000 points and up to four
    threads above it. Explicit workers are capped by the number of clusters.
    HK/auction JIT warmup happens before starting cluster threads.
    """
    if mode not in ONE_TO_ONE_MODES:
        raise ValueError(f"mode must be one of {ONE_TO_ONE_MODES}, got {mode!r}.")
    xyz_att = np.asarray(xyz_att, dtype=np.float64)
    xyz_orig = np.asarray(xyz_orig, dtype=np.float64)
    labels = np.asarray(labels)
    if (
        xyz_att.ndim != 2
        or xyz_att.shape[1] != 3
        or xyz_orig.ndim != 2
        or xyz_orig.shape[1] != 3
    ):
        raise ValueError(f"{mode} correspondence requires (N, 3) point arrays.")
    if not np.isfinite(xyz_att).all() or not np.isfinite(xyz_orig).all():
        raise ValueError("Correspondence points must contain finite coordinates.")
    if labels.shape != (len(xyz_orig),):
        raise ValueError("labels must have one entry per original point.")
    if len(xyz_orig) == 0:
        raise ValueError(f"{mode} correspondence requires original points.")
    if len(xyz_att) < 2:
        raise ValueError(
            f"{mode} correspondence requires at least two received points."
        )
    if not np.isfinite(distance_factor) or distance_factor <= 0:
        raise ValueError("distance_factor must be finite and positive.")
    candidate_counts = tuple(int(value) for value in candidate_counts)
    if (
        not candidate_counts
        or any(value <= 0 for value in candidate_counts)
        or any(left >= right for left, right in zip(candidate_counts, candidate_counts[1:]))
    ):
        raise ValueError(
            "candidate_counts must be a strictly increasing sequence of positive integers."
        )
    tasks = [
        (int(cluster_id), np.flatnonzero(labels == cluster_id))
        for cluster_id in np.unique(labels)
    ]
    worker_count = _one_to_one_worker_count(workers, len(xyz_orig), len(tasks))
    warmup_assignment(mode)
    tree = cKDTree(xyz_att)
    def solve(task):
        cluster_id, rows = task
        return _solve_one_to_one_cluster(
            tree, xyz_orig, cluster_id, rows, candidate_counts, mode
        )
    if worker_count == 1:
        results = [solve(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(solve, tasks))

    output = np.full(len(xyz_orig), -1, dtype=np.int64)
    maximum_candidates = 0
    for rows, assigned, candidate_count, _ in results:
        output[rows] = assigned
        maximum_candidates = max(maximum_candidates, candidate_count)

    neighbor_distances = tree.query(xyz_att, k=2)[0][:, 1]
    spacing = neighbor_distances[
        np.isfinite(neighbor_distances) & (neighbor_distances > 1e-12)
    ]
    if len(spacing) == 0:
        raise ValueError("Cannot estimate spacing from received points.")
    threshold = float(distance_factor * np.median(spacing))
    matched = xyz_att[output].copy()
    distances = np.linalg.norm(matched - xyz_orig, axis=1)
    valid = np.isfinite(distances) & (distances <= threshold)
    matched[~valid] = np.nan
    if verbose:
        global_unique_ratio = len(np.unique(output)) / len(output)
        print(
            f"[{mode}] threshold={threshold:.6e}, "
            f"unmatched={np.count_nonzero(~valid)} / {len(xyz_orig)}, "
            "within_cluster_unique=100.00%, "
            f"global_received_unique={global_unique_ratio:.2%}, "
            f"max_candidates={maximum_candidates}, workers={worker_count}"
        )
    if ONE_TO_ONE_PROFILE_LOGS:
        selected, rejected, timings = _one_to_one_profile_summary(results)
        selected_text = ", ".join(
            f"{count}:{value}" for count, value in selected.items() if value
        ) or "none"
        rejected_text = ", ".join(
            f"{count}:{value}" for count, value in rejected.items() if value
        ) or "none"
        assignment_fingerprint = hashlib.blake2b(
            np.ascontiguousarray(output).view(np.uint8), digest_size=8
        ).hexdigest()
        print(
            f"[{mode} profile] "
            f"clusters={len(results)}, selected={{ {selected_text} }}, "
            f"infeasible={{ {rejected_text} }}, workers={worker_count}, "
            f"check={'HK' if mode != 'bidir' else 'min-cost'}, "
            f"query={timings['query_seconds']:.3f}s, "
            f"graph={timings['graph_seconds']:.3f}s, "
            f"capacity={timings['capacity_seconds']:.3f}s "
            f"(rejected={timings['capacity_rejections']}), "
            f"feasibility={timings['feasibility_seconds']:.3f}s, "
            f"min_cost={timings['min_cost_seconds']:.3f}s, "
            f"auction={timings['auction_seconds']:.3f}s",
            f"assignment={assignment_fingerprint}",
            file=sys.stderr,
        )
    return matched, valid, distances, threshold


def dualband_geometry_feature(points, normal_neighbors=24, normal_compare=6):
    """Order-invariant point-count and local-normal-variation feature."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("points must have at least three rows of xyz coordinates.")
    neighbor_count = min(int(normal_neighbors), len(points) - 1)
    compare_count = min(int(normal_compare), neighbor_count)
    if neighbor_count < 2 or compare_count < 1:
        raise ValueError("Not enough points for the DualBand geometry feature.")
    indices = cKDTree(points).query(points, k=neighbor_count + 1)[1]
    neighbors = points[indices[:, 1:]]
    centered = neighbors - np.mean(neighbors, axis=1, keepdims=True)
    covariance = np.einsum(
        "nki,nkj->nij", centered, centered, optimize=True
    ) / neighbor_count
    _, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    dot_products = np.sum(
        normals[:, None, :] * normals[indices[:, 1:compare_count + 1]], axis=2
    )
    return {
        "count": len(points),
        "normal_variation": float(np.median(1.0 - np.abs(dot_products))),
    }


def classify_dualband_attack(
    attacked_points,
    reference_feature,
    downsample_count_threshold=0.95,
    noise_normal_variation_threshold=1.50,
    smoothing_normal_variation_threshold=0.80,
):
    """Choose a DualBand decoder rule from geometry only, not watermark bits."""
    if reference_feature is None:
        return "neutral", float("nan"), float("nan")
    value = dualband_geometry_feature(attacked_points)
    count_ratio = value["count"] / max(float(reference_feature["count"]), 1.0)
    normal_ratio = value["normal_variation"] / max(
        float(reference_feature["normal_variation"]), 1e-15
    )
    if count_ratio < downsample_count_threshold:
        label = "downsampling"
    elif normal_ratio > noise_normal_variation_threshold:
        label = "noise"
    elif normal_ratio < smoothing_normal_variation_threshold:
        label = "smoothing"
    else:
        label = "neutral"
    return label, count_ratio, normal_ratio


def is_vertex_order_consistent(
    xyz_att,
    xyz_orig,
    distance_factor=20.0,
    max_bad_ratio=0.01,
    verbose=True,
):
    """全点の同一インデックス間の直接距離で順序整合性を調べる。

    頂点順序が維持されていれば、元点群で近傍だったインデックス対は
    ノイズやスムージング後も概ね近い。一方、順序変更後は同じ番号の
    頂点対が空間的に離れるため、攻撃後点間隔に対して長い辺が増える。
    """
    xyz_att = np.asarray(xyz_att, dtype=np.float64)
    xyz_orig = np.asarray(xyz_orig, dtype=np.float64)
    if xyz_att.ndim != 2 or xyz_att.shape[1] != 3:
        raise ValueError("xyz_att must have shape (N, 3).")
    if xyz_orig.ndim != 2 or xyz_orig.shape[1] != 3:
        raise ValueError("xyz_orig must have shape (N, 3).")
    if xyz_att.shape != xyz_orig.shape:
        return False
    if len(xyz_orig) < 2:
        return True
    if distance_factor <= 0 or not 0 <= max_bad_ratio <= 1:
        raise ValueError("Invalid vertex-order check parameters.")

    attacked_tree = cKDTree(xyz_att)
    attacked_nn, _ = attacked_tree.query(xyz_att, k=2)
    spacing_values = attacked_nn[:, 1]
    spacing_values = spacing_values[
        np.isfinite(spacing_values) & (spacing_values > 1e-12)
    ]
    if len(spacing_values) == 0:
        raise ValueError("Cannot estimate attacked point spacing.")
    attacked_spacing = float(np.median(spacing_values))

    index_distances = np.linalg.norm(xyz_att - xyz_orig, axis=1)
    normalized_distances = index_distances / attacked_spacing
    median_ratio = float(np.median(normalized_distances))
    bad_ratio = float(np.mean(normalized_distances > distance_factor))
    consistent = bad_ratio <= max_bad_ratio
    if verbose:
        mode = "index" if consistent else "coordinate"
        print(
            f"[Order] median_index_distance/spacing={median_ratio:.3f}, "
            f"inconsistent_indices={bad_ratio:.2%}, correspondence={mode}"
        )
    return consistent


def align_attacked_vertices_to_original(
    attacked_vertices,
    original_vertices,
    index_distance_factor=20.0,
    coordinate_distance_factor=2.0,
    max_index_inconsistency_ratio=0.01,
    correspondence_mode="auto",
    labels=None,
    one_to_one_candidates=ONE_TO_ONE_CANDIDATES,
    verbose=True,
    one_to_one_workers=0,
):
    """Align attacked vertices to original-vertex order using X1 matching.

    ``correspondence_mode="auto"`` retains index correspondence when the arrays
    have equal shape and the order-consistency test succeeds. ``"coordinate"``
    always uses coordinate-based nearest-neighbor matching, while ``"index"``
    requires equal array shapes. ``"hk"``, ``"auction"`` and ``"bidir"`` use
    geometry-only one-to-one assignment inside each original GFT cluster. The returned mask
    identifies original vertices with a match inside the distance threshold.
    """
    if correspondence_mode not in CORRESPONDENCE_MODES:
        raise ValueError(
            f"correspondence_mode must be one of {CORRESPONDENCE_MODES}, "
            f"got {correspondence_mode!r}."
        )
    attacked_vertices = np.asarray(attacked_vertices, dtype=float)
    original_vertices = np.asarray(original_vertices, dtype=float)
    if correspondence_mode in ONE_TO_ONE_MODES:
        if labels is None:
            raise ValueError(f"{correspondence_mode} correspondence requires original labels.")
        if len(attacked_vertices) == len(original_vertices):
            try:
                aligned, matched_mask, _, _ = match_point_cloud_one_to_one(
                    attacked_vertices,
                    original_vertices,
                    labels,
                    distance_factor=coordinate_distance_factor,
                    candidate_counts=one_to_one_candidates,
                    workers=one_to_one_workers,
                    verbose=verbose,
                    mode=correspondence_mode,
                )
                return aligned, matched_mask
            except (RuntimeError, ValueError) as exc:
                if verbose:
                    print(
                        f"[{correspondence_mode}] matching failed; falling back to "
                        f"coordinate many-to-one ({exc})"
                    )
        elif verbose:
            print(
                f"[{correspondence_mode}] vertex count changed: "
                f"{len(original_vertices)} -> "
                f"{len(attacked_vertices)}; correspondence=coordinate fallback"
            )
        aligned_vertices, matched_vertex_mask, _, _ = match_point_cloud_to_original(
            attacked_vertices,
            original_vertices,
            distance_factor=coordinate_distance_factor,
            verbose=verbose,
        )
        return aligned_vertices, matched_vertex_mask

    use_index = correspondence_mode == "index" or (
        correspondence_mode == "auto"
        and attacked_vertices.shape == original_vertices.shape
        and is_vertex_order_consistent(
            attacked_vertices,
            original_vertices,
            distance_factor=index_distance_factor,
            max_bad_ratio=max_index_inconsistency_ratio,
            verbose=verbose,
        )
    )
    if use_index:
        if attacked_vertices.shape != original_vertices.shape:
            raise ValueError(
                "index correspondence requires equal attacked/original shapes."
            )
        matched_vertex_mask = np.ones(len(original_vertices), dtype=bool)
        return attacked_vertices.copy(), matched_vertex_mask

    if verbose:
        if correspondence_mode == "coordinate":
            print("[Order] correspondence=coordinate (forced)")
        elif len(attacked_vertices) != len(original_vertices):
            print(
                f"[Order] vertex count changed: {len(original_vertices)} -> "
                f"{len(attacked_vertices)}; correspondence=coordinate"
            )
    aligned_vertices, matched_vertex_mask, _, _ = match_point_cloud_to_original(
        attacked_vertices,
        original_vertices,
        distance_factor=coordinate_distance_factor,
        verbose=verbose,
    )
    return aligned_vertices, matched_vertex_mask


def extract_watermark_x1(
    xyz_emb: np.ndarray,
    xyz_orig: np.ndarray,
    labels: np.ndarray,
    embed_bits_length: int,
    graph_mode: str = 'knn',
    k: int = 12,
    radius: float = 0.05,
    min_spectre: float = 0.0,
    max_spectre: float = 1.0,
    small_cluster_policy: str = "half_watermark",
    index_distance_factor: float = 20.0,
    coordinate_distance_factor: float = 2.0,
    max_index_inconsistency_ratio: float = 0.01,
    correspondence_mode: str = "auto",
    one_to_one_candidates=ONE_TO_ONE_CANDIDATES,
    one_to_one_workers: int = 0,
):
    """
    埋め込み前点群 xyz_orig から疑似平面を再構築し、
    高さ信号 h_orig と h_emb を作り、GFT係数差分の符号でビット抽出（多数決）。

    重要:
    - 疑似平面推定は xyz_orig（埋め込み前）で行う（xyz_embで推定すると透かし変位が平面推定に混入する）。
    - グラフも xyz_orig（クラスタ内座標）で構築する（埋め込み前と一致させる）。
    """
    xyz_emb = np.asarray(xyz_emb, dtype=float)
    xyz_orig = np.asarray(xyz_orig, dtype=float)
    aligned_attacked_vertices, matched_vertex_mask = (
        align_attacked_vertices_to_original(
            xyz_emb,
            xyz_orig,
            index_distance_factor=index_distance_factor,
            coordinate_distance_factor=coordinate_distance_factor,
            max_index_inconsistency_ratio=max_index_inconsistency_ratio,
            correspondence_mode=correspondence_mode,
            labels=labels,
            one_to_one_candidates=one_to_one_candidates,
            one_to_one_workers=one_to_one_workers,
            verbose=True,
        )
    )
    if embed_bits_length <= 0:
        return []

    small_cluster_size_limit = _small_cluster_size_limit(
        embed_bits_length, small_cluster_policy
    )

    bit_lists = [[] for _ in range(embed_bits_length)]

    for cluster_id in np.unique(labels):
        cluster_indices = np.flatnonzero(labels == cluster_id)
        if len(cluster_indices) <= small_cluster_size_limit:
            continue
        # GFT requires the original cluster membership to remain complete.
        # This erasure rule is independent of small_cluster_policy.
        if not np.all(matched_vertex_mask[cluster_indices]):
            continue

        pts_orig = xyz_orig[cluster_indices]
        pts_emb = aligned_attacked_vertices[cluster_indices]
        if len(pts_orig) != len(pts_emb):
            continue

        # 疑似平面はorigから
        centroid, normal = compute_pseudoplane_pca(pts_orig)

        # 高さ信号（orig/emb）
        h_orig = cluster_height_signal(pts_orig, centroid, normal)
        h_emb = cluster_height_signal(pts_emb, centroid, normal)

        # グラフはorigで
        W = build_graph(pts_orig, graph_mode=graph_mode, k=k, radius=radius)
        coefficients, eigvals = gft_analyze_signals_auto(
            W, np.column_stack((h_orig, h_emb))
        )
        gft_orig = coefficients[:, 0]
        gft_emb = coefficients[:, 1]

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
        ones = votes.count(1)
        zeros = votes.count(0)
        if len(votes) == 0 or ones == zeros:
            extracted_bits.append(-1)
        else:
            extracted_bits.append(1 if ones > zeros else 0)

    return extracted_bits


def extract_watermark_x1_dualband(
    xyz_emb,
    xyz_orig,
    labels,
    embed_bits_length,
    graph_mode="knn",
    k=12,
    radius=0.05,
    small_cluster_policy="half_watermark",
    index_distance_factor=20.0,
    coordinate_distance_factor=2.0,
    max_index_inconsistency_ratio=0.01,
    correspondence_mode="hk",
    one_to_one_candidates=ONE_TO_ONE_CANDIDATES,
    low_band=DUAL_BAND_LOW_RANGE,
    high_band=DUAL_BAND_HIGH_RANGE,
    code_key=DUAL_BAND_CODE_KEY,
    attack_reference_feature=None,
    one_to_one_workers=0,
):
    """Extract the ``DualBand`` spread watermark without using PCD row order.

    Low and High correlation scores are independently normalized.  Geometry
    alone selects ``Low + 0.5 High`` for noise, High for smoothing, Low for
    point-count reduction, and an equal normalized fusion otherwise.
    """
    xyz_emb = np.asarray(xyz_emb, dtype=np.float64)
    xyz_orig = np.asarray(xyz_orig, dtype=np.float64)
    labels = np.asarray(labels)
    if xyz_orig.ndim != 2 or xyz_orig.shape[1] != 3:
        raise ValueError("xyz_orig must have shape (N, 3).")
    if labels.shape != (len(xyz_orig),):
        raise ValueError("labels must have one entry per original point.")
    if embed_bits_length <= 0:
        return []

    aligned_vertices, matched_vertex_mask = align_attacked_vertices_to_original(
        xyz_emb,
        xyz_orig,
        index_distance_factor=index_distance_factor,
        coordinate_distance_factor=coordinate_distance_factor,
        max_index_inconsistency_ratio=max_index_inconsistency_ratio,
        correspondence_mode=correspondence_mode,
        labels=labels,
        one_to_one_candidates=one_to_one_candidates,
        one_to_one_workers=one_to_one_workers,
        verbose=True,
    )

    low_scores = np.zeros(embed_bits_length, dtype=np.float64)
    high_scores = np.zeros(embed_bits_length, dtype=np.float64)
    small_cluster_size_limit = _small_cluster_size_limit(
        embed_bits_length, small_cluster_policy
    )
    for cluster_id in np.unique(labels):
        index = np.flatnonzero(labels == cluster_id)
        if len(index) <= small_cluster_size_limit:
            continue
        if not np.all(matched_vertex_mask[index]):
            continue
        original_points = xyz_orig[index]
        attacked_points = aligned_vertices[index]
        centroid, normal = compute_pseudoplane_pca(original_points)
        original_height = cluster_height_signal(original_points, centroid, normal)
        attacked_height = cluster_height_signal(attacked_points, centroid, normal)
        coefficients, _ = gft_analyze_signals_auto(
            build_graph(original_points, graph_mode=graph_mode, k=k, radius=radius),
            np.column_stack((original_height, attacked_height)),
        )
        difference = coefficients[:, 1] - coefficients[:, 0]
        layouts = _dual_band_layouts(
            cluster_id,
            len(index),
            embed_bits_length,
            low_band,
            high_band,
            code_key,
        )
        for name, scores in (("Low", low_scores), ("High", high_scores)):
            layout = layouts[name]
            np.add.at(
                scores,
                layout["bit_index"],
                layout["chips"] * difference[layout["start"]:layout["end"]],
            )

    def normalize(scores):
        return scores / max(float(np.median(np.abs(scores))), 1e-12)

    label, count_ratio, normal_ratio = classify_dualband_attack(
        xyz_emb, attack_reference_feature
    )
    low_scores = normalize(low_scores)
    high_scores = normalize(high_scores)
    if label == "noise":
        scores, rule = low_scores + 0.5 * high_scores, "Low+0.5High"
    elif label == "smoothing":
        scores, rule = high_scores, "High"
    elif label == "downsampling":
        scores, rule = low_scores, "Low"
    else:
        scores, rule = low_scores + high_scores, "Low+High"
    print(
        f"[DualBand] attack={label}, rule={rule}, "
        f"count_ratio={count_ratio:.4f}, normal_ratio={normal_ratio:.4f}"
    )
    return (scores > 0.0).astype(int).tolist()
