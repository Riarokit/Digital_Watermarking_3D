import numpy as np
import open3d as o3d
import DW2_func as DW2F
import DW1_M4_func as DW1M4
import time

if __name__ == "__main__":
    """
    main関数概要:
    点群をクラスタリングし、各クラスタの疑似平面からの高さを信号値としてGFTを行う。
    各クラスタのGFT係数にビット列を埋め込み、各クラスタから復元した複数の同ビットを最後に一度のみ多数決して決定。

    使用変数説明:
    n                        = 画像サイズn×n
    beta                     = 埋め込み強度の調整係数
    cluster_point            = 1クラスタあたりの点数目安
    graph_mode               = グラフ構築モード: 'knn' or 'radius' or 'hybrid'
    k                        = k-NNグラフのk値
    radius                   = 半径グラフの半径値
    flatness_weighting       = 0:なし, 1:平面部重み, 2:曲面部重み
    min_spectre, max_spectre = 最小・最大周波数帯域

    オプション設定:
    OP: 切り取りやノイズ付加などのオプション手順
    """
    # 画像サイズn×n
    n = 16
    # 1クラスタあたりの点数目安(k-means用)
    cluster_point = 2000
    # グラフ構築モード
    graph_mode = 'knn'
    k = 6
    radius = 0.03
    # 平面曲面アプローチ
    flatness_weighting = 0

    # Bunny全周波用
    # beta = 1.62e-3; min_spectre = 0.0; max_spectre = 1.0; input_file = "C:/bun_zipper.ply"
    # Bunny低周波用
    beta = 3.61e-3; min_spectre = 0.0; max_spectre = 0.2; input_file = "C:/bun_zipper.ply"
    # Bunny高周波用
    # beta = 3.70e-3; min_spectre = 0.8; max_spectre = 1.0; input_file = "C:/bun_zipper.ply"
    # Dragon全周波用
    # beta = 1.32e-3; min_spectre = 0.0; max_spectre = 1.0; input_file = "C:/dragon_vrip_res2.ply"
    # Dragon低周波用
    # beta = 2.96e-3; min_spectre = 0.0; max_spectre = 0.2; input_file = "C:/dragon_vrip_res2.ply"
    # Dragon高周波用
    # beta = 3.15e-3; min_spectre = 0.8; max_spectre = 1.0; input_file = "C:/dragon_vrip_res2.ply"
    # Armadillo全周波用
    # beta = 1.68e-3; min_spectre = 0.0; max_spectre = 1.0; input_file = "C:/Armadillo.ply"
    # Armadillo低周波用
    # beta = 3.74e-3; min_spectre = 0.0; max_spectre = 0.2; input_file = "C:/Armadillo.ply"
    # Armadillo高周波用
    # beta = 4.16e-3; min_spectre = 0.8; max_spectre = 1.0; input_file = "C:/Armadillo.ply"

    # 1. データ取得
    image_path = "watermark16.bmp"  # 埋め込みたい画像ファイル
    pcd_before = o3d.io.read_point_cloud(input_file)

    # 2. 前処理（色情報追加・理想クラスタ数計算）
    pcd_before = DW2F.normalize_point_cloud(pcd_before)
    pcd_before = DW2F.add_colors(pcd_before, color="grad")
    xyz = np.asarray(pcd_before.points)
    colors = np.asarray(pcd_before.colors)

    # 3. 埋め込みビット生成
    watermark_bits = DW2F.image_to_bitarray(image_path, n=n)
    watermark_bits_length = len(watermark_bits)
    print(f"[Debug] 埋込ビット数：{watermark_bits_length} (画像: {n}x{n})")

    # 4. クラスタリング
    start = time.time()
    labels = DW1M4.kmeans_cluster_points(xyz, cluster_point=cluster_point)
    # labels = DW1M4.region_growing_cluster_points(xyz)
    # labels = DW1M4.ransac_cluster_points(xyz)
    # labels = DW1M4.split_large_clusters(xyz, labels, limit_points=3000)

    # 5. 単多数決方式の埋め込み
    xyz_after = DW1M4.embed_watermark_m4(
        xyz, labels, watermark_bits,
        beta=beta,
        graph_mode=graph_mode, k=k, radius=radius,
        flatness_weighting=flatness_weighting, k_neighbors=20,
        min_spectre=min_spectre, max_spectre=max_spectre,
    )

    embed_time = time.time() - start
    diffs = np.linalg.norm(xyz_after - xyz, axis=1)
    max_embed_shift = np.max(diffs)
    print(f"[Debug] 最大埋め込み誤差: {max_embed_shift}")

    # OP. ノイズ攻撃
    # xyz_after = DW2F.noise_addition_attack(xyz_after, noise_percent=0.75, mode='gaussian', seed=42)

    # OP. スムージング攻撃
    # xyz_after = DW2F.smoothing_attack(xyz_after, lambda_val=0.2, iterations=30, k=6)

    # OP. 切り取り攻撃 (不可視性評価はコメントアウト)
    # xyz_after = DW2F.cropping_attack(xyz_after, keep_ratio=0.5, mode='axis', axis=0)

    # OP. ダウンサンプリング攻撃 (不可視性評価はコメントアウト)
    # xyz_after = DW2F.downsampling_attack(xyz_after, mode='voxel', voxel_size_percent=2.0, seed=42)

    # 6. 単多数決方式の抽出
    start = time.time()
    extracted_bits = DW1M4.extract_watermark_m4(
        xyz_after, xyz, labels, watermark_bits_length,
        graph_mode=graph_mode, k=k, radius=radius,
        min_spectre=min_spectre, max_spectre=max_spectre
    )
    extract_time = time.time() - start

    # 7. 視覚品質評価
    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
    pcd_after.colors = o3d.utility.Vector3dVector(colors)
    print(pcd_after)
    DW2F.evaluate_psnr(pcd_before, pcd_after)
    DW2F.evaluate_pc_msdm(pcd_before, pcd_after)
    DW2F.evaluate_angular_similarity(pcd_before, pcd_after)
    DW2F.evaluate_p2d(pcd_before, pcd_after)
    DW2F.evaluate_point_ssim(pcd_before, pcd_after)
    # DW2F.visualize_embedded_points(xyz, xyz_after)
    o3d.visualization.draw_geometries([pcd_after])

    # 8. ロバスト性評価
    print(f"埋込ビット：{len(watermark_bits)}")
    print(f"抽出ビット：{len(extracted_bits)}")
    DW2F.evaluate_robustness(watermark_bits, extracted_bits)
    DW2F.bitarray_to_image(extracted_bits, n=n, save_path="recovered.bmp")
    print(f"埋込時間: {embed_time:.2f}秒")
    print(f"抽出時間: {extract_time:.2f}秒\n")