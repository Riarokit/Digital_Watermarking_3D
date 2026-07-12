import numpy as np
import open3d as o3d
import DW2_func as DW2F  # 共通ユーティリティ（正規化、評価等）
import DW1_HU_func as DW1HU  # 先ほど作成したHuらの関数ファイル
import time

if __name__ == "__main__":
    """
    main関数概要:
    Huらが提案した表面上のEMDに基づく高精度電子透かし手法の実行プログラム。
    1. 点群をメッシュ化し、表面上でEMDを実行。
    2. 第一IMFの安定した極値点（山と谷）を特定。
    3. 目標PSNR(FideP)に基づき強度alphaを最適化し、円状にT回繰り返し埋め込む。
    4. 抽出時は多数決投票(Voting)により堅牢にビットを復元する。
    """
    
    # === パラメータ設定 ===
    # 埋め込みに用いる画像サイズ n×n (論文では32x32 [Source 9: Sect 5])
    n = 16 
    # フィデリティパラメータ (論文推奨: 115 [Source 9: Sect 5.1.2])
    fide_p = 115.0
    # 円状埋め込み回数 (論文推奨: 25 [Source 9: Sect 5.1.3])
    T_rep = 25
    
    # ======================
    # 1. データ取得
    image_path = "watermark16.bmp"
    input_file = "C:/bun_zipper.ply"
    # input_file = "C:/dragon_vrip_res2.ply"
    # input_file = "C:/Armadillo.ply"
    # input_file = "C:/longdress_vox12.ply"
    # input_file = "C:/soldier_vox12.ply"
    pcd_before = o3d.io.read_point_cloud(input_file)
    
    # 2. 前処理
    # ElZeinと同様に正規化を実行
    pcd_before = DW2F.normalize_point_cloud(pcd_before)
    pcd_before = DW2F.add_colors(pcd_before, color="grad")
    xyz = np.asarray(pcd_before.points)
    colors = np.asarray(pcd_before.colors)

    # 3. 埋め込みビット生成
    watermark_bits = DW2F.image_to_bitarray(image_path, n=n)
    watermark_bits_length = len(watermark_bits)
    print(f"[Debug] 埋込ビット数：{len(watermark_bits)} ({n}x{n}画像)")

    # 4. 埋め込み処理
    start_embed = time.time()
    try:
        # メッシュベースの埋め込み関数を呼び出し
        xyz_after, key_info = DW1HU.embed_watermark_hu_mesh(
            xyz, watermark_bits, FideP=fide_p, T=T_rep
        )
    except Exception as e:
        print(f"[Error] 埋め込み失敗: {e}")
        exit()
    
    embed_time = time.time() - start_embed

    # OP. ノイズ攻撃 (論文では0.1%〜0.5%を検証 [Source 9: Sect 5.3.1])
    # xyz_after = DW2F.noise_addition_attack(xyz_after, noise_percent=0.1, mode='gaussian')
    
    # OP. スムージング攻撃 (論文ではLaplacian smoothingを検証 [Source 9: Sect 5.3.2])
    # xyz_after = DW2F.smoothing_attack(xyz_after, lambda_val=0.1, iterations=10)

    # OP. 切り取り攻撃（不可視性評価はコメントアウト）
    # xyz_after = DW2F.cropping_attack(xyz_after, keep_ratio=0.9, mode='axis', axis=0)

    # OP. ダウンサンプリング攻撃 (不可視性評価はコメントアウト)
    # xyz_after = DW2F.downsampling_attack(xyz_after, mode='voxel', voxel_size_percent=1.0, seed=42)

    # 5. 抽出処理
    start_extract = time.time()
    
    extracted_bits = DW1HU.extract_watermark_hu_mesh(
        xyz_after, key_info, wm_len=len(watermark_bits)
    )
    
    extract_time = time.time() - start_extract
    
    # 6. 視覚品質評価
    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
    pcd_after.colors = o3d.utility.Vector3dVector(colors)
    print(pcd_after)
    # DW2F.evaluate_psnr(pcd_before, pcd_after, by_index=True)
    # DW2F.evaluate_pc_msdm(pcd_before, pcd_after)
    # DW2F.evaluate_point_ssim(pcd_before, pcd_after)
    # DW2F.visualize_embedded_points(xyz, xyz_after)
    # o3d.visualization.draw_geometries([pcd_after])

    # 7. ロバスト性評価
    print(f"埋込ビット長：{len(watermark_bits)}")
    print(f"抽出ビット長：{len(extracted_bits)}")
    DW2F.evaluate_robustness(watermark_bits, extracted_bits)
    DW2F.bitarray_to_image(extracted_bits, n=n, save_path="hu_recovered.bmp")
    print(f"使用した極値点数: {len(key_info)} (Max) / {len(key_info[1])} (Min)")
    print(f"埋込時間: {embed_time:.2f}秒")
    print(f"抽出時間: {extract_time:.2f}秒\n")