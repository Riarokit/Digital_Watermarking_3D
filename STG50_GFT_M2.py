import numpy as np
import open3d as o3d
import STG50_GFT_func as STG50F
import time

if __name__ == "__main__":
    """
    main関数概要:
    点群をクラスタリングし、各クラスタの座標を信号としてGFTを行う。
    各クラスタのGFT係数にビット列を埋め込み、1クラスタから復元した冗長ビットのみでまず多数決。
    その後、そのビット列の検査符号をチェックし、通過したビット列のみ同ビットで再び多数決。

    使用変数説明:
    message_length         = 埋め込む文字列の長さ
    num_clusters           = クラスタ数
    beta                   = 調整係数
    split_mode             = 0:チャネル間に同一の埋め込み, 1:チャネル間に異なる埋め込み
    flatness_weighting     = 0:なし, 1:平面部重み, 2:曲面部重み
    error_correction       = "none", "parity", "hamming" のいずれか

    オプション設定:
    OP: 切り取りやノイズ付加などのオプション手順
    """
    # 基礎
    n = 16  # 画像サイズn×n
    beta = 1e-3
    # 平面曲面アプローチ
    flatness_weighting = 1
    # 埋め込み容量アプローチ
    split_mode = 1
    # 誤り訂正符号アプローチ
    error_correction = "none"

    # 1. 点群取得
    image_path = "watermark16.bmp"  # 埋め込みたい画像ファイル
    input_file = "C:/bun_zipper.ply"
    # input_file = "C:/Armadillo.ply"
    # input_file = "C:/longdress_vox12.ply"
    # input_file = "C:/soldier_vox12.ply"
    pcd_before = o3d.io.read_point_cloud(input_file)

    # 2. 前処理（色情報追加・理想クラスタ数計算）
    pcd_before = STG50F.add_colors(pcd_before, color="grad")
    # o3d.visualization.draw_geometries([pcd_before])
    xyz = np.asarray(pcd_before.points)
    colors = np.asarray(pcd_before.colors)

    # 3. 埋め込みビット生成
    watermark_bits = STG50F.image_to_bitarray(image_path, n=n)
    watermark_bits_length = len(watermark_bits)
    print(f"埋込ビット数：{watermark_bits_length} (画像: {n}x{n})")
    pcd_after = o3d.geometry.PointCloud() # 埋め込み後の点群基盤

    # 4. クラスタリング
    start = time.time()
    labels = STG50F.kmeans_cluster_points(xyz)
    # labels = STG50F.region_growing_cluster_points(xyz)
    # labels = STG50F.ransac_cluster_points(xyz)
    # labels = STG50F.split_large_clusters(xyz, labels, limit_points=3000)

    # 5. 2段階多数決方式の埋め込み
    xyz_after, checked_bit_length = STG50F.embed_watermark_xyz_check(
        xyz, labels, watermark_bits, beta=beta, split_mode=split_mode,
        flatness_weighting=flatness_weighting, k_neighbors=20, 
        error_correction=error_correction
    )

    diffs = np.linalg.norm(xyz_after - xyz, axis=1)
    max_embed_shift = np.max(diffs)
    print("最大埋め込み誤差:", max_embed_shift)

    # OP. ノイズ攻撃
    xyz_after = STG50F.add_noise(xyz_after, noise_percent=0.005, mode='uniform', seed=42)

    # OP. 切り取り攻撃
    # xyz_after = STG50F.crop_point_cloud_xyz(xyz_after, crop_ratio=0.9, mode='center')
    # xyz_after = STG50F.reconstruct_point_cloud(xyz_after, xyz, threshold=max_embed_shift*2)
    # print(len(xyz_after))
    # xyz_after = STG50F.reorder_point_cloud(xyz_after, xyz)
    # print(len(xyz_after))

    # 6. 2段階多数決方式の抽出
    extracted_bits = STG50F.extract_watermark_xyz_check(
        xyz_after, xyz, labels, watermark_bits_length, checked_bit_length,
        split_mode=split_mode,  error_correction=error_correction
    )

    # 7. 評価
    pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
    pcd_after.colors = o3d.utility.Vector3dVector(colors)
    print(pcd_after)
    STG50F.calc_psnr_xyz(pcd_before, pcd_after)

    # 8. 確認用
    o3d.visualization.draw_geometries([pcd_after])
    print(f"埋込ビット：{len(watermark_bits)}")
    print(f"抽出ビット：{len(extracted_bits)}")
    STG50F.evaluate_watermark(watermark_bits, extracted_bits)
    STG50F.bitarray_to_image(extracted_bits, n=n, save_path="recovered.bmp")
    all_time = time.time() - start
    print(f"実行時間: {all_time:.2f}秒\n")