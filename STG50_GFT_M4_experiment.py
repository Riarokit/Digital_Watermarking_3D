import numpy as np
import open3d as o3d
import STG50_GFT_func as STG50F
import os

def evaluate_vsnr(pcd_before, pcd_after):
    xyz_before = np.asarray(pcd_before.points)
    xyz_after = np.asarray(pcd_after.points)
    mse = np.mean(np.linalg.norm(xyz_before - xyz_after, axis=1)**2)
    p_max = np.max(xyz_before, axis=0)
    p_min = np.min(xyz_before, axis=0)
    peak = np.linalg.norm(p_max - p_min)
    if mse == 0:
        return 0, float('inf')
    psnr = 10 * np.log10((peak**2) / mse)
    return mse, psnr

def eval_robustness(watermark_bits, extracted_bits):
    if len(watermark_bits) != len(extracted_bits) or len(watermark_bits) == 0:
        return 0.0
    correct = sum([1 for w, e in zip(watermark_bits, extracted_bits) if w == e])
    return correct / len(watermark_bits)

def run_embedding_only(pcd_before, watermark_bits, min_sp, max_sp, beta):
    """埋め込みのみを行い、MSEと埋め込み後点群を返す"""
    xyz = np.asarray(pcd_before.points)
    labels = STG50F.kmeans_cluster_points(xyz, cluster_point=500, seed=42)
    xyz_after = STG50F.embed_watermark_pseudoplane(
        xyz, labels, watermark_bits,
        beta=beta, graph_mode='knn', k=6, radius=0.03,
        flatness_weighting=0, k_neighbors=20,
        min_spectre=min_sp, max_spectre=max_sp
    )
    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
    mse, psnr = evaluate_vsnr(pcd_before, pcd_after)
    return xyz_after, labels, mse, psnr

def find_matching_beta(pcd_before, watermark_bits, min_sp, max_sp, target_mse, initial_beta):
    """二分探索でtarget_mseに最も近いbetaを自動探索する"""
    low_beta = 0.0
    high_beta = initial_beta * 5.0
    best_beta = initial_beta
    best_xyz_after = None
    best_labels = None
    best_psnr = 0
    
    print(f"\n[{min_sp}-{max_sp}] ターゲットMSE ({target_mse:.8e}) に合わせた beta を探索中...")
    for _ in range(5):
        mid_beta = (low_beta + high_beta) / 2
        xyz_after, labels, mse, psnr = run_embedding_only(pcd_before, watermark_bits, min_sp, max_sp, mid_beta)
        if mse < target_mse:
            low_beta = mid_beta
        else:
            high_beta = mid_beta
        best_beta = mid_beta
        best_xyz_after = xyz_after
        best_labels = labels
        best_psnr = psnr
        if abs(mse - target_mse) / target_mse < 0.05: # 誤差5%以内で終了
            break
    
    print(f"探索完了: optimal beta = {best_beta:.6e}, 実測PSNR = {best_psnr:.2f}dB")
    return best_xyz_after, best_labels, best_beta

def attack_and_extract(xyz_orig, xyz_after, labels, watermark_bits, min_sp, max_sp, attack_type, attack_param):
    """指定した攻撃を加えたあと、抽出精度を返す"""
    if attack_type == "noise":
        rng = np.random.RandomState(42)
        noise = rng.normal(0, attack_param * np.mean(np.linalg.norm(xyz_after, axis=1)), xyz_after.shape)
        xyz_att = xyz_after + noise
    elif attack_type == "smoothing":
        try:
            xyz_att = STG50F.smoothing_attack(xyz_after, lambda_val=attack_param, iterations=100, k=6) # iterを100など過酷に
        except AttributeError:
            xyz_att = xyz_after.copy()
    else:
        xyz_att = xyz_after.copy()

    extracted_bits = STG50F.extract_watermark_pseudoplane(
        xyz_att, xyz_orig, labels, len(watermark_bits),
        graph_mode='knn', k=6, radius=0.03,
        min_spectre=min_sp, max_spectre=max_sp
    )
    return eval_robustness(watermark_bits, extracted_bits)

def main():
    # シードの固定（再現性のため）
    np.random.seed(42)
    
    input_file = "C:/dragon_vrip_res2.ply"
    if os.path.exists(input_file):
        pcd = o3d.io.read_point_cloud(input_file)
    else:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
    pcd = STG50F.normalize_point_cloud(pcd)
    xyz_orig = np.asarray(pcd.points)
    
    # ペイロード（ビット数）の設定
    payload_size = 512  
    watermark_bits = np.random.randint(0, 2, payload_size).tolist()

    # --- 1. フル帯域を基準として実行 ---
    print("\n=== フル帯域 (0.0 - 1.0) ===")
    beta_full = 0.5e-3
    xyz_after_full, labels_full, mse_full, psnr_full = run_embedding_only(pcd, watermark_bits, 0.0, 1.0, beta_full)
    print(f"基準PSNR: {psnr_full:.2f}dB (MSE: {mse_full:.8e})")
    
    # ノイズ5%とスムージング(lambda 0.05, 100 iter)
    acc_full_noise = attack_and_extract(xyz_orig, xyz_after_full, labels_full, watermark_bits, 0.0, 1.0, "noise", 0.05)
    acc_full_smooth = attack_and_extract(xyz_orig, xyz_after_full, labels_full, watermark_bits, 0.0, 1.0, "smoothing", 0.05)
    print(f"Acc(Noise 5%): {acc_full_noise:.3f}, Acc(Smooth): {acc_full_smooth:.3f}")

    # --- 2. 複数の帯域で比較する ---
    bands = [
        (0.0, 0.2), # 極低周波（直流成分などを含む）
        (0.4, 0.6), # 中周波
        (0.8, 1.0)  # 極高周波
    ]

    for start_sp, end_sp in bands:
        print(f"\n=== 帯域 ({start_sp:.1f} - {end_sp:.1f}) ===")
        # 各帯域の係数数は約20%なので、初期betaは約sqrt(5)=2.23倍からスタート
        init_beta = beta_full * 2.23
        xyz_after_band, labels_band, best_beta = find_matching_beta(pcd, watermark_bits, start_sp, end_sp, mse_full, init_beta)
        
        acc_band_noise = attack_and_extract(xyz_orig, xyz_after_band, labels_band, watermark_bits, start_sp, end_sp, "noise", 0.05)
        acc_band_smooth = attack_and_extract(xyz_orig, xyz_after_band, labels_band, watermark_bits, start_sp, end_sp, "smoothing", 0.05)
        print(f"[{start_sp:.1f}-{end_sp:.1f}] Acc(Noise 5%): {acc_band_noise:.3f}, Acc(Smooth): {acc_band_smooth:.3f}")

if __name__ == "__main__":
    main()
