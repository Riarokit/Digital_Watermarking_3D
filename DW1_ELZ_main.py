import numpy as np
import open3d as o3d
import DW2_func as DW2F
import DW1_ELZ_func as DW1ELZ
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
    a = 2.66e-3 #Bunny用
    # a = 3.56e-3 #Dragon用
    
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
    pcd_before = o3d.io.read_point_cloud(input_file)

    # 2. 前処理（正規化、色情報の追加）
    pcd_before = DW2F.normalize_point_cloud(pcd_before)
    pcd_before = DW2F.add_colors(pcd_before, color="grad")
    xyz = np.asarray(pcd_before.points)
    colors = np.asarray(pcd_before.colors)

    # 3. 埋め込みビット生成
    watermark_bits = DW2F.image_to_bitarray(image_path, n=n)
    watermark_bits_length = len(watermark_bits)
    print(f"[Debug] 埋込ビット数：{watermark_bits_length} (画像: {n}x{n})")

    # 4. 埋め込み処理
    start_embed = time.time()
    try:
        xyz_after = DW1ELZ.embed_watermark_elzein(
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
    # xyz_after = DW2F.noise_addition_attack(xyz_after, noise_percent=0.1, mode='gaussian', seed=42)

    # OP. スムージング攻撃
    # xyz_after = DW2F.smoothing_attack(xyz_after, lambda_val=0.1, iterations=30, k=6)

    # OP. 切り取り攻撃（不可視性評価はコメントアウト）
    # xyz_after = DW2F.cropping_attack(xyz_after, keep_ratio=0.9, mode='axis', axis=0)

    # OP. ダウンサンプリング攻撃 (不可視性評価はコメントアウト)
    xyz_after = DW2F.downsampling_attack(xyz_after, keep_ratio=0.5, mode='voxel', voxel_size_percent=1.0, seed=42)

    # 5. 抽出処理
    start_extract = time.time()
    extracted_bits = DW1ELZ.extract_watermark_elzein(
        xyz_after, xyz, 
        n_points=n_points, k=k
    )
    extract_time = time.time() - start_extract

    # 6. 視覚品質評価
    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
    pcd_after.colors = o3d.utility.Vector3dVector(colors)
    print(pcd_after)
    DW2F.evaluate_psnr(pcd_before, pcd_after, by_index=True)
    DW2F.evaluate_pc_msdm(pcd_before, pcd_after)
    DW2F.evaluate_point_ssim(pcd_before, pcd_after)
    DW2F.visualize_embedded_points(xyz, xyz_after)
    o3d.visualization.draw_geometries([pcd_after])

    # 7. ロバスト性評価
    print(f"埋込ビット長：{len(watermark_bits)}")
    print(f"抽出ビット長：{len(extracted_bits)}")
    DW2F.evaluate_robustness(watermark_bits, extracted_bits)
    DW2F.bitarray_to_image(extracted_bits, n=n, save_path="elzein_recovered.bmp")
    print(f"埋込時間: {embed_time:.2f}秒")
    print(f"抽出時間: {extract_time:.2f}秒\n")
