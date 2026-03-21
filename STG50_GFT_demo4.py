import numpy as np
import open3d as o3d
import STG50_GFT_func as STG50F
import matplotlib.pyplot as plt
import os

"""
GFT係数の分布を可視化する
"""

def check_spectrum():
    # 1. データ準備
    input_file = "C:/dragon_vrip_res2.ply"
    if os.path.exists(input_file):
        pcd = o3d.io.read_point_cloud(input_file)
    else:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
    pcd = STG50F.normalize_point_cloud(pcd)
    xyz_orig = np.asarray(pcd.points)

    # クラスタリング
    labels = STG50F.kmeans_cluster_points(xyz_orig, cluster_point=500, seed=42)
    
    # ノイズ付加とスムージング付加
    rng = np.random.RandomState(42)
    noise_param = 0.05
    noise = rng.normal(0, noise_param * np.mean(np.linalg.norm(xyz_orig, axis=1)), xyz_orig.shape)
    xyz_noise = xyz_orig + noise
    xyz_smooth = STG50F.smoothing_attack(xyz_orig, lambda_val=0.05, iterations=100, k=6)

    # 周波数ごとのエネルギー蓄積用
    # 周波数を 0.0~1.0 に正規化し、例えば 50 ビンで平均をとる
    num_bins = 50
    orig_spectrum = np.zeros(num_bins)
    noise_spectrum = np.zeros(num_bins)
    smooth_spectrum = np.zeros(num_bins)
    counts = np.zeros(num_bins)

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 10:
            continue
            
        pts_orig = xyz_orig[idx]
        pts_noise = xyz_noise[idx]
        pts_smooth = xyz_smooth[idx]

        # 疑似平面は「元の点群」から推定
        centroid, normal = STG50F.compute_pseudoplane_pca(pts_orig)
        
        # 高さ信号
        h_orig = STG50F.cluster_height_signal(pts_orig, centroid, normal)
        h_noise = STG50F.cluster_height_signal(pts_noise, centroid, normal)
        h_smooth = STG50F.cluster_height_signal(pts_smooth, centroid, normal)

        # グラフ基底（元点群）
        W = STG50F.build_graph(pts_orig, graph_mode='knn', k=6, radius=0.03)
        basis, _ = STG50F.gft_basis(W)

        # GFT展開
        g_orig = STG50F.gft(h_orig, basis)
        g_noise = STG50F.gft(h_noise, basis)
        g_smooth = STG50F.gft(h_smooth, basis)

        # 差分（攻撃によるノイズ）
        diff_noise = g_noise - g_orig
        diff_smooth = g_smooth - g_orig

        # 各係数のエネルギー（絶対値または2乗）を正規化周波数ビンに足し込む
        Q_ = len(g_orig)
        for i in range(Q_):
            freq_norm = i / Q_
            bin_idx = min(int(freq_norm * num_bins), num_bins - 1)
            
            orig_spectrum[bin_idx] += np.abs(g_orig[i])
            noise_spectrum[bin_idx] += np.abs(diff_noise[i])
            smooth_spectrum[bin_idx] += np.abs(diff_smooth[i])
            counts[bin_idx] += 1

    # 平均化
    orig_spectrum = np.divide(orig_spectrum, counts, out=np.zeros_like(orig_spectrum), where=counts!=0)
    noise_spectrum = np.divide(noise_spectrum, counts, out=np.zeros_like(noise_spectrum), where=counts!=0)
    smooth_spectrum = np.divide(smooth_spectrum, counts, out=np.zeros_like(smooth_spectrum), where=counts!=0)

    # 描画
    x_axis = np.linspace(0, 1, num_bins)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, orig_spectrum, label='Original Component', color='black', linewidth=2)
    plt.plot(x_axis, noise_spectrum, label='Random Noise Error', color='red', linestyle='--')
    plt.plot(x_axis, smooth_spectrum, label='Smoothing Error', color='blue', linestyle='-.')
    
    plt.yscale('log')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Coefficient Magnitude (Log Scale)')
    plt.title('GFT Coefficient Distribution & Attack Noise Spread')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    # Artifactディレクトリ等に保存（エフェメラル用）
    save_dir = r"C:\Users\ryoi1\.gemini\antigravity\brain\f518c22c-410c-469c-bf13-5f418ad98397"
    save_path = os.path.join(save_dir, "gft_spectrum.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Spectrum plot saved to: {save_path}")

if __name__ == "__main__":
    check_spectrum()
