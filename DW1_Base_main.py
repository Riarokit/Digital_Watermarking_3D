import numpy as np
import open3d as o3d
import DW1_func as DW1F
import DW1_Base_func as DW1B
import time

if __name__ == "__main__":
    """
    main関数概要:
    El Zeinらが提案したベースラインとなる電子透かし手法。
    点群の局所特徴量(法線のなす角)にFCMクラスタリングを適用し、中程度の粗さ(Medium roughness)の
    クラスタを特定。取得した頂点座標に対して直接加減算操作を行うことでビットを埋め込む。
    抽出は元の点群を用いる完全ノンブラインド型で行われる。

    パラメータ:
    n : 画像サイズ (nxn の透かし画像)
    a : 透かしの埋め込み強度 (加減算される値)
    k : 局所特徴量推定のための近傍点数
    """
    
    # === パラメータ設定 ===
    # 埋め込みに用いる画像サイズ n×n
    n = 16
    n_points = n * n  # 埋め込む頂点数（1頂点につき1ビット）
    
    # 透かしの埋め込み強度
    a = 2.66e-3
    
    # 近傍点数k
    k = 6
    # ======================

    # 1. データ取得
    image_path = "watermark16.bmp"  # 埋め込みたい画像ファイル
    input_file = "C:/bun_zipper.ply"
    # input_file = "C:/dragon_vrip_res2.ply"
    # input_file = "C:/Armadillo.ply"
    # input_file = "C:/longdress_vox12.ply"
    # input_file = "C:/soldier_vox12.ply"
    
    print("--- データの読み込みと前処理 ---")
    pcd_before = o3d.io.read_point_cloud(input_file)

    # 前処理（正規化、色情報の追加）
    pcd_before = DW1F.normalize_point_cloud(pcd_before)
    pcd_before = DW1F.add_colors(pcd_before, color="grad")
    
    xyz = np.asarray(pcd_before.points)
    colors = np.asarray(pcd_before.colors)

    # 2. 埋め込みビット生成
    watermark_bits = DW1F.image_to_bitarray(image_path, n=n)
    watermark_bits_length = len(watermark_bits)
    print(f"[Debug] 埋込ビット数：{watermark_bits_length} (画像: {n}x{n})")
    
    # 埋め込み後の点群保存用
    pcd_after = o3d.geometry.PointCloud()

    # 3. 埋め込み処理
    print("\n--- 埋め込み処理開始 ---")
    start_embed = time.time()
    try:
        xyz_after = DW1B.embed_watermark_baseline(
            xyz, watermark_bits, 
            n_points=n_points, a=a, k=k
        )
    except ValueError as e:
        print(f"[Error] {e}")
        exit()
    embed_time = time.time() - start_embed
    
    diffs = np.linalg.norm(xyz_after - xyz, axis=1)
    max_embed_shift = np.max(diffs)
    print(f"[Debug] 最大埋め込み誤差: {max_embed_shift}")

    # OP. ノイズ攻撃
    # xyz_after = DW1F.noise_addition_attack(xyz_after, noise_percent=0.1, mode='gaussian', seed=42)

    # OP. スムージング攻撃
    # xyz_after = DW1F.smoothing_attack(xyz_after, lambda_val=0.1, iterations=30, k=6)

    # OP. 切り取り攻撃（不可視性評価はコメントアウト）
    # xyz_after = DW1F.cropping_attack(xyz_after, keep_ratio=0.9, mode='axis', axis=0)

    # 4. 抽出処理
    print("\n--- 抽出処理開始 ---")
    start_extract = time.time()
    extracted_bits = DW1B.extract_watermark_baseline(
        xyz_after, xyz, 
        n_points=n_points, k=k
    )
    extract_time = time.time() - start_extract

    # 5. 評価結果出力
    print("\n--- 評価結果 ---")
    pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
    pcd_after.colors = o3d.utility.Vector3dVector(colors)
    print(pcd_after)
    
    DW1F.evaluate_imperceptibility(pcd_before, pcd_after, by_index=True)
    DW1F.evaluate_ssim(pcd_before, pcd_after, save_dir="ssim_out")
    print(f"埋込ビット長：{len(watermark_bits)}")
    print(f"抽出ビット長：{len(extracted_bits)}")
    
    DW1F.evaluate_robustness(watermark_bits, extracted_bits)
    DW1F.bitarray_to_image(extracted_bits, n=n, save_path="baseline_recovered.bmp")
    print(f"埋込時間: {embed_time:.2f}秒")
    print(f"抽出時間: {extract_time:.2f}秒\n")
    
    # 6. 点群の可視化 (確認用)
    o3d.visualization.draw_geometries([pcd_after])
