import numpy as np
import open3d as o3d

# ----------------------------
# Graph utilities
# ----------------------------
def build_knn_graph(points: np.ndarray, k: int = 10, sigma: float | None = None):
    N = points.shape[0]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    all_d2 = []
    knn_idx, knn_d2 = [], []
    for i in range(N):
        _, idx, d2 = kdtree.search_knn_vector_3d(points[i], k + 1)  # includes self
        idx, d2 = idx[1:], d2[1:]
        knn_idx.append(idx)
        knn_d2.append(d2)
        all_d2.extend(d2)

    all_d2 = np.array(all_d2, dtype=np.float64)
    if sigma is None:
        sigma = np.sqrt(np.median(all_d2) + 1e-12)

    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        idx = np.array(knn_idx[i], dtype=int)
        d2 = np.array(knn_d2[i], dtype=np.float64)
        w = np.exp(-d2 / (2 * sigma * sigma))
        W[i, idx] = w
        W[idx, i] = np.maximum(W[idx, i], w)  # symmetric

    return W

def graph_laplacian(W: np.ndarray):
    D = np.diag(W.sum(axis=1))
    return D - W

# ----------------------------
# Demo surface + coloring
# ----------------------------
def make_surface_grid(n: int = 35, span: float = 1.0):
    xs = np.linspace(-span, span, n)
    ys = np.linspace(-span, span, n)
    X, Y = np.meshgrid(xs, ys)

    # 見栄え用の“ゆるい曲面”
    Z = 0.15 * np.exp(-(X**2 + Y**2) * 1.3) + 0.04 * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return pts

def colorize(z: np.ndarray):
    zmin, zmax = float(z.min()), float(z.max())
    t = (z - zmin) / (zmax - zmin + 1e-12)
    return np.stack([t, 0.2*np.ones_like(t), 1.0 - t], axis=1)

# ----------------------------
# Band mask by percentiles
# ----------------------------
def make_band_mask_percent(N: int, mode: str):
    def idx(p):  # percent -> index
        return int(np.floor(p * N))

    mask = np.zeros(N, dtype=np.float64)

    if mode == "LOW":       # 0-20%
        mask[0:idx(0.20)] = 1.0
    elif mode == "MID":     # 40-60%
        mask[idx(0.40):idx(0.60)] = 1.0
    elif mode == "HIGH":    # 80-100%
        mask[idx(0.80):N] = 1.0
    elif mode == "ALL":
        mask[:] = 1.0
    else:
        raise ValueError("Unknown mode")
    return mask

def band_label(mode: str):
    return {
        "LOW":  "LOW  (0–20%)   [press 1]",
        "MID":  "MID  (40–60%)  [press 2]",
        "HIGH": "HIGH (80–100%) [press 3]",
        "ALL":  "ALL  (0–100%)  [press 4]",
    }[mode]

# ----------------------------
# Main
# ----------------------------
def main():
    # ===== 調整パラメータ（ここだけ触ればOK） =====
    n = 35        # 点数: n*n。スライド向けに 30〜40 推奨
    k = 10        # kNN
    amp = 0.4    # 動きの大きさ（0.12〜0.30で調整）
    speed = 1.6   # 動き速度

    seed = 7      # 波の“形”を固定（スライド撮影向け）

    # 1) create points & signal (height from pseudo-plane)
    pts0 = make_surface_grid(n=n, span=1.0)
    x0 = pts0[:, 2].copy()

    # 2) build graph & laplacian
    print("Building kNN graph...")
    W = build_knn_graph(pts0, k=k)
    L = graph_laplacian(W)

    # 3) eigen decomposition (GFT)
    print("Eigen-decomposition (GFT basis)...")
    evals, U = np.linalg.eigh(L)
    xhat0 = U.T @ x0
    N = xhat0.shape[0]

    # 4) prepare "wave pattern" in spectral domain (fixed shape)
    rng = np.random.default_rng(seed)
    # 各周波数係数に±1の符号を割り当てて、帯域内に“形のある”変形を作る
    phase_pattern = rng.choice([-1.0, 1.0], size=N)

    # 5) visualize (key callbacks)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts0.copy())
    pcd.colors = o3d.utility.Vector3dVector(colorize(pts0[:, 2]))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="GFT Band Wave Demo", width=1100, height=780)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 4.0
    opt.background_color = np.asarray([0.05, 0.05, 0.06])

    state = {"mode": "LOW", "t": 0.0}

    def print_mode():
        print(f"[MODE] {band_label(state['mode'])}  |  amp={amp}, n={n*n}pts")

    def set_mode(m):
        state["mode"] = m
        print_mode()

    def cb1(_): set_mode("LOW");  return False
    def cb2(_): set_mode("MID");  return False
    def cb3(_): set_mode("HIGH"); return False
    def cb4(_): set_mode("ALL");  return False
    def cbr(_):
        state["t"] = 0.0
        print("[RESET] phase reset")
        return False

    vis.register_key_callback(ord("1"), cb1)
    vis.register_key_callback(ord("2"), cb2)
    vis.register_key_callback(ord("3"), cb3)
    vis.register_key_callback(ord("4"), cb4)
    vis.register_key_callback(ord("R"), cbr)
    vis.register_key_callback(ord("r"), cbr)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.85)

    print_mode()
    print("Click window, then press 1/2/3/4 to switch bands. Press R to reset phase.")

    # 6) animation
    while vis.poll_events():
        mask = make_band_mask_percent(N, state["mode"])
        active = np.count_nonzero(mask)
        # 係数数に応じて正規化（帯域ごとに変形量が揃う）
        scale = 1.0 / np.sqrt(active + 1e-12)

        wave = np.sin(speed * state["t"])
        # “形のある波”を帯域に注入
        delta_hat = (amp * wave * scale) * (mask * phase_pattern)

        xhat = xhat0 + delta_hat
        x = U @ xhat

        pts = np.asarray(pcd.points)
        pts[:, 2] = x
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colorize(pts[:, 2]))

        vis.update_geometry(pcd)
        vis.update_renderer()

        state["t"] += 0.03

    vis.destroy_window()

if __name__ == "__main__":
    main()
