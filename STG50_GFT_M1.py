import numpy as np
import open3d as o3d
import STG50_GFT_func as STG50F
import time

if __name__ == "__main__":
    """
    main関数概要:
    点群をクラスタリングし、各クラスタの座標を信号としてGFTを行う。
    各クラスタのGFT係数にビット列を埋め込み、各クラスタから復元した複数の同ビットを最後に一度のみ多数決して決定。

    使用変数説明:
    message_length         = 埋め込む文字列の長さ
    num_clusters           = クラスタ数
    beta                   = 調整係数
    split_mode             = 0:チャネル間に同一の埋め込み, 1:チャネル間に異なる埋め込み
    flatness_weighting     = 0:なし, 1:平面部重み, 2:曲面部重み
    min_weight, max_weight = 最小・最大重み係数 (平均1になるようにする)

    オプション設定:
    OP: 切り取りやノイズ付加などのオプション手順
    """
    # 基礎
    message_length = 100
    beta = 1e-3
    # 平面曲面アプローチ
    flatness_weighting = 1
    min_weight = 0
    max_weight = 2.0
    # 埋め込み容量アプローチ
    split_mode = 1
    # 周波数帯域アプローチ
    min_spectre = 0.5
    max_spectre = 1.0

    # 1. 点群取得
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
    upper_cluster_num = int(len(pcd_before.points) / (message_length*8*2/3))
    print(f"望ましいクラスタ数: 8 - {upper_cluster_num}")

    # 3. 埋め込みビット生成
    watermark_message = STG50F.generate_random_string(message_length)
    watermark_bits = STG50F.string_to_binary(watermark_message)
    watermark_bits_length = len(watermark_bits)
    pcd_after = o3d.geometry.PointCloud() # 埋め込み後の点群基盤

    # 4. クラスタリング
    start = time.time()
    labels = STG50F.kmeans_cluster_points(xyz)
    # labels = STG50F.region_growing_cluster_points(xyz)
    # labels = STG50F.ransac_cluster_points(xyz)
    # labels = STG50F.split_large_clusters(xyz, labels, limit_points=3000)

    # 5. 単多数決方式の埋め込み
    xyz_after = STG50F.embed_watermark_xyz(
        xyz, labels, watermark_bits, beta=beta, split_mode=split_mode,
        flatness_weighting=flatness_weighting, k_neighbors=20, 
        min_weight=min_weight, max_weight=max_weight,
        min_spectre=min_spectre, max_spectre=max_spectre
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

    # 6. 単多数決方式の抽出
    extracted_bits = STG50F.extract_watermark_xyz(
        xyz_after, xyz, labels, watermark_bits_length, split_mode=split_mode,
        min_spectre=min_spectre, max_spectre=max_spectre
    )

    # 7. 評価
    pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
    pcd_after.colors = o3d.utility.Vector3dVector(colors)
    print(pcd_after)
    psnr = STG50F.calc_psnr_xyz(pcd_before, pcd_after)

    # 8. 確認用
    o3d.visualization.draw_geometries([pcd_after])
    ber = 1.0000-np.mean(np.array(watermark_bits) == np.array(extracted_bits))
    extracted_message = STG50F.binary_to_string(extracted_bits)
    all_time = time.time() - start
    print(f"埋込文字列：{watermark_message}")
    print(f"抽出文字列：{extracted_message}")
    print(f"埋込ビット：{len(watermark_bits)}")
    print(f"抽出ビット：{len(extracted_bits)}")
    print(f"BER: {ber:.4f}")
    print(f"実行時間: {all_time:.2f}秒\n")