import numpy as np
import open3d as o3d
import DW1_func as DW1F
import DW1_Base_func as DW1B
import os

# ==========================================
# グローバルパラメータ設定
# ==========================================
# --- 共通＆入出力設定 ---
TARGET_PSNR = 60.0                           # 目標とするPSNR (dB) 【※埋め込み直後・攻撃前の視覚品質】
INPUT_FILE = "C:/bun_zipper.ply"             # 対象の点群データ
# INPUT_FILE = "C:/dragon_vrip_res2.ply"
# INPUT_FILE = "C:/Armadillo.ply"
# INPUT_FILE = "C:/longdress_vox12.ply"
# INPUT_FILE = "C:/soldier_vox12.ply"
IMAGE_PATH = "watermark16.bmp"                # 埋め込む透かし画像
WATERMARK_SIZE = 16                           # 埋め込む画像のサイズ (nxn)

# --- 攻撃に関する設定 ---
ATTACKS = [
    # ("noise", [0.2, 0.4, 0.6, 0.8, 1.0]),
    # ("smoothing", [5, 10, 20, 30]),
    ("cropping", [0.9, 0.7, 0.5, 0.3])
]
NUM_TRIALS = 5                               # 各攻撃条件ごとのテスト試行回数（平均BERを算出するため）

NOISE_MODE = 'gaussian'                      # ノイズの分布 ("gaussian" または "uniform")
SMOOTHING_LAMBDA = 0.2                       # スムージングの強さ (lambda)
SMOOTHING_K = 6                              # スムージングの近傍点数
CROPPING_MODE = 'axis'                       # 切り取りのモード ("axis" など)
CROPPING_AXIS = 0                            # 切り取り軸 (0:x, 1:y, 2:z)

# --- 提案手法(GFT)の詳細パラメータ ---
GRAPH_MODE = 'knn'                           # グラフ構築モード ("knn", "radius", "hybrid")
KNN_K = 6                                    # k-NNの近傍点数
GRAPH_RADIUS = 0.03                          # 半径グラフの半径値
FLATNESS_WEIGHTING = 0                       # 平面重み (0:なし, 1:平面部重み, 2:曲面部重み)
K_NEIGHBORS = 20                             # 局所曲率推定の近傍点数
CLUSTER_POINTS_PROPOSED = [2000]  # 検証する提案手法の1クラスタあたりの点数パターンのリスト

# 比較する周波数帯域 (帯域名, 最小周波数, 最大周波数, 初期betaの目安)
BANDS = [
    ("Full", 0.0, 1.0, 1.6e-3),
    ("Low ", 0.0, 0.2, 3.6e-3),
    # ("Mid ", 0.4, 0.6, 3.6e-3),
    ("High", 0.8, 1.0, 3.6e-3)
]

# --- ベースライン手法(FCM)の詳細パラメータ ---
BASELINE_KNN_K = 6                           # FCM処理におけるk-NNの近傍点数
# ==========================================

def calculate_target_mse(pcd_orig, target_psnr):
    xyz_orig = np.asarray(pcd_orig.points)
    p_max = np.max(xyz_orig, axis=0)
    p_min = np.min(xyz_orig, axis=0)
    peak = np.linalg.norm(p_max - p_min)
    target_mse = (peak ** 2) / (10 ** (target_psnr / 10))
    return target_mse

def apply_attack(xyz_after, attack_type, attack_param, seed):
    if attack_type == "noise":
        return DW1F.noise_addition_attack(xyz_after, noise_percent=attack_param, mode=NOISE_MODE, seed=seed)
    elif attack_type == "smoothing":
        try:
            return DW1F.smoothing_attack(xyz_after, lambda_val=SMOOTHING_LAMBDA, iterations=int(attack_param), k=SMOOTHING_K)
        except AttributeError:
            return xyz_after.copy()
    elif attack_type == "cropping":
        try:
            return DW1F.cropping_attack(xyz_after, keep_ratio=attack_param, mode=CROPPING_MODE, axis=CROPPING_AXIS)
        except AttributeError:
            return xyz_after.copy()
    else:
        return xyz_after.copy()

# ================= ベースライン手法の関数 =================
def run_embedding_baseline(pcd_before, xyz_orig, medium_indices, watermark_bits, a):
    # 二分探索を高速化するため、ベースラインの数式(加算/減算)を直接実行（再クラスタリング回避）
    xyz_after = xyz_orig.copy()
    n_points = len(watermark_bits)
    
    if len(medium_indices) < n_points:
        raise ValueError(f"Target points ({n_points}) exceeds available points in medium cluster ({len(medium_indices)}).")
        
    groups = np.array_split(medium_indices, n_points)
    
    for i in range(n_points):
        bit = watermark_bits[i]
        group_indices = groups[i]
        for idx in group_indices:
            v = xyz_after[idx]
            if bit == 1:
                xyz_after[idx] = v + a
            elif bit == 0:
                xyz_after[idx] = v - a

    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
    metrics = DW1F.evaluate_imperceptibility(pcd_before, pcd_after, verbose=False)
    return xyz_after, metrics['mse'], metrics['psnr']

def find_matching_alpha(pcd_before, xyz_orig, medium_indices, watermark_bits, target_mse, initial_a):
    if target_mse <= 0: return xyz_orig.copy()
    print(f"\n[Baseline] ターゲットMSE ({target_mse:.8e}) に合わせた 'a' を自動探索中...")
    
    # 手順1: ブランケット探索による上限の決定（TARGET_PSNRがどんな値でも対応できるようにするため）
    high_a = initial_a
    for _ in range(15):
        _, mse, _ = run_embedding_baseline(pcd_before, xyz_orig, medium_indices, watermark_bits, high_a)
        if mse >= target_mse:
            break
        high_a *= 2.0
    
    # 手順2: 二分探索による精密な決定
    low_a = 0.0
    best_a = high_a
    best_xyz_after = None
    best_psnr = 0
    
    for _ in range(15):
        mid_a = (low_a + high_a) / 2
        try:
            xyz_after, mse, psnr = run_embedding_baseline(pcd_before, xyz_orig, medium_indices, watermark_bits, mid_a)
        except Exception as e:
            print(f"探索エラー: {e}")
            break
            
        if mse < target_mse:
            low_a = mid_a
        else:
            high_a = mid_a
            
        best_a = mid_a
        best_xyz_after = xyz_after
        best_psnr = psnr
        
        # MSE誤差1%未満で最適とみなす
        if abs(mse - target_mse) / target_mse < 0.01:
            break
            
    print(f"--> [Baseline] 最適化完了: optimal a = {best_a:.6e}, 実測PSNR = {best_psnr:.2f}dB")
    return best_xyz_after

def attack_and_extract_baseline(xyz_orig, xyz_after, watermark_bits, attack_type, attack_param, seed):
    xyz_att = apply_attack(xyz_after, attack_type, attack_param, seed)
    try:
        extracted_bits = DW1B.extract_watermark_baseline(xyz_att, xyz_orig, n_points=len(watermark_bits), k=BASELINE_KNN_K, verbose=False)
    except Exception as e:
        print(f"Baseline抽出エラー: {e}")
        extracted_bits = []
    _, ber = DW1F.evaluate_robustness(watermark_bits, extracted_bits, verbose=False)
    return ber

# ================= 提案手法の関数 =================
def run_embedding_proposed(pcd_before, xyz_orig, labels, watermark_bits, min_sp, max_sp, beta):
    xyz_after = DW1F.embed_watermark_pseudoplane(
        xyz_orig, labels, watermark_bits,
        beta=beta, graph_mode=GRAPH_MODE, k=KNN_K, radius=GRAPH_RADIUS,
        flatness_weighting=FLATNESS_WEIGHTING, k_neighbors=K_NEIGHBORS,
        min_spectre=min_sp, max_spectre=max_sp
    )
    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
    metrics = DW1F.evaluate_imperceptibility(pcd_before, pcd_after, verbose=False)
    mse, psnr = metrics['mse'], metrics['psnr']
    return xyz_after, mse, psnr

def find_matching_beta(pcd_before, xyz_orig, labels, watermark_bits, min_sp, max_sp, target_mse, initial_beta, band_name):
    if target_mse <= 0: return xyz_orig.copy()
    print(f"\n[Proposed - {band_name}] ターゲットMSE に合わせた 'beta' を自動探索中...")
    
    # 手順1: ブランケット探索による上限の決定
    high_beta = initial_beta
    for _ in range(15):
        _, mse, _ = run_embedding_proposed(pcd_before, xyz_orig, labels, watermark_bits, min_sp, max_sp, high_beta)
        if mse >= target_mse:
            break
        high_beta *= 2.0
    
    # 手順2: 二分探索による精密な決定
    low_beta = 0.0
    best_beta = high_beta
    best_xyz_after = None
    best_psnr = 0
    
    for _ in range(15):
        mid_beta = (low_beta + high_beta) / 2
        xyz_after, mse, psnr = run_embedding_proposed(pcd_before, xyz_orig, labels, watermark_bits, min_sp, max_sp, mid_beta)
        
        if mse < target_mse:
            low_beta = mid_beta
        else:
            high_beta = mid_beta
            
        best_beta = mid_beta
        best_xyz_after = xyz_after
        best_psnr = psnr
        
        # MSE誤差1%未満で最適とみなす
        if abs(mse - target_mse) / target_mse < 0.01:
            break
            
    print(f"--> [{band_name}] 最適化完了: optimal beta = {best_beta:.6e}, 実測PSNR = {best_psnr:.2f}dB")
    return best_xyz_after

def attack_and_extract_proposed(xyz_orig, xyz_after, labels, watermark_bits, min_sp, max_sp, attack_type, attack_param, seed):
    xyz_att = apply_attack(xyz_after, attack_type, attack_param, seed)
    extracted_bits = DW1F.extract_watermark_pseudoplane(
        xyz_att, xyz_orig, labels, len(watermark_bits),
        graph_mode=GRAPH_MODE, k=KNN_K, radius=GRAPH_RADIUS,
        min_spectre=min_sp, max_spectre=max_sp
    )
    _, ber = DW1F.evaluate_robustness(watermark_bits, extracted_bits, verbose=False)
    return ber

# ================= メイン実行 =================
def main():
    np.random.seed(42)
    
    print("=== データとパラメータの準備 ===")
    if os.path.exists(INPUT_FILE):
        pcd = o3d.io.read_point_cloud(INPUT_FILE)
    else:
        print(f"入力ファイル {INPUT_FILE} が見つからないため、球体モデルで代替します。")
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        pcd = mesh.sample_points_poisson_disk(number_of_points=15000)
        
    pcd = DW1F.normalize_point_cloud(pcd)
    xyz_orig = np.asarray(pcd.points)
    
    try:
        watermark_bits = DW1F.image_to_bitarray(IMAGE_PATH, n=WATERMARK_SIZE)
    except Exception as e:
        print(f"画像読み込みエラー ({e})。ランダムなビット列を生成します。")
        watermark_bits = np.random.randint(0, 2, WATERMARK_SIZE * WATERMARK_SIZE).tolist()
    
    target_mse = calculate_target_mse(pcd, TARGET_PSNR)
    print(f"目標 PSNR: {TARGET_PSNR} dB -> 許容 MSE: {target_mse:.8e}")

    # ===============================================
    # 1. ベースライン手法の前計算（FCMクラスタリング）
    print("\nベースライン手法(FCM)のクラスタリング実行中...")
    medium_indices = DW1B.local_feature_clustering(xyz_orig, k=BASELINE_KNN_K, verbose=False)
    init_a = 0.001
    xyz_after_base = find_matching_alpha(pcd, xyz_orig, medium_indices, watermark_bits, target_mse, init_a)

    # ===============================================
    # 2. 提案手法の前計算（クラスタサイズごとに実行）
    proposed_configs = {} # key: display_name, value: (xyz_after, labels, min_sp, max_sp)
    
    for cp in CLUSTER_POINTS_PROPOSED:
        print(f"\n[Proposed] 1クラスタ {cp} 点の KMeansクラスタリング実行中...")
        labels_prop = DW1F.kmeans_cluster_points(xyz_orig, cluster_point=cp, seed=42)
        min_cluster_size = np.min([np.sum(labels_prop == c) for c in np.unique(labels_prop)])
        if min_cluster_size < len(watermark_bits):
            print(f"警告: 最小クラスタの点数({min_cluster_size})が埋め込みビット数({len(watermark_bits)})を満たしていません！1つのクラスタ内にすべてのビット列が格納できず、一部のビットが情報落ちする可能性があります。")
            
        for band_name, min_sp, max_sp, init_beta in BANDS:
            display_name = f"P-{band_name.strip()}({cp})"
            xyz_after = find_matching_beta(pcd, xyz_orig, labels_prop, watermark_bits, min_sp, max_sp, target_mse, init_beta, display_name)
            proposed_configs[display_name] = (xyz_after, labels_prop, min_sp, max_sp)

    # ===============================================
    # 3. 攻撃耐性の比較ループ
    results_all = {}
    
    for attack_type, params in ATTACKS:
        print(f"\n=== 攻撃テスト開始 (攻撃種類: {attack_type}) ===")
        results = {"Baseline": []}
        for name in proposed_configs.keys():
            results[name] = []
            
        for param in params:
            print(f"  -> テスト中: {attack_type} = {param} ({NUM_TRIALS}回平均)...")
            
            acc_base_sum = 0.0
            acc_prop_sums = {name: 0.0 for name in proposed_configs.keys()}
            
            for trial in range(NUM_TRIALS):
                seed = 42 + trial
                
                # ベースライン
                ber_base = attack_and_extract_baseline(xyz_orig, xyz_after_base, watermark_bits, attack_type, param, seed)
                acc_base_sum += (1.0 - ber_base)
                
                # 提案手法(全構成)
                for name, (xyz_emb, labels_prop, min_sp, max_sp) in proposed_configs.items():
                    ber_prop = attack_and_extract_proposed(xyz_orig, xyz_emb, labels_prop, watermark_bits, min_sp, max_sp, attack_type, param, seed)
                    acc_prop_sums[name] += (1.0 - ber_prop)
                    
            # 正答率(Accuracy)として保存
            results["Baseline"].append(acc_base_sum / NUM_TRIALS)
            for name in proposed_configs.keys():
                results[name].append(acc_prop_sums[name] / NUM_TRIALS)
                
        results_all[attack_type] = (params, results)

    # ===============================================
    # 4. 比較結果の表出力
    for attack_type, (params, results) in results_all.items():
        print(f"\n{'='*95}")
        print(f"   Robustness Comparison Table (Target PSNR: {TARGET_PSNR} dB | Model: {os.path.basename(INPUT_FILE)})")
        print(f"   Metric: Accuracy (1-BER) | Attack Type: {attack_type} (Trials: {NUM_TRIALS}) | Bits: {len(watermark_bits)}")
        print(f"{'='*95}")
        
        headers = ["Att Param", "Baseline"] + list(proposed_configs.keys())
        header_str = " | ".join([f"{h:<14}" for h in headers])
        print(header_str)
        print("-" * 95)
        
        for i, param in enumerate(params):
            row_vals = [f"{param:<14.4f}", f"{results['Baseline'][i]:<14.4f}"]
            for name in proposed_configs.keys():
                row_vals.append(f"{results[name][i]:<14.4f}")
            print(" | ".join(row_vals))
        print(f"{'='*95}\n")

if __name__ == "__main__":
    main()
