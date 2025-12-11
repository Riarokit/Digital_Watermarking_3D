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
    n                        = 画像サイズn×n
    beta                     = 埋め込み強度の調整係数
    cluster_point            = 1クラスタあたりの点数目安
    graph_mode               = グラフ構築モード: 'knn' or 'radius' or 'hybrid'
    k                        = k-NNグラフのk値
    radius                   = 半径グラフの半径値
    split_mode               = 0:チャネル間に同一の埋め込み, 1:チャネル間に異なる埋め込み
    flatness_weighting       = 0:なし, 1:平面部重み, 2:曲面部重み
    min_spectre, max_spectre = 最小・最大周波数帯域

    オプション設定:
    OP: 切り取りやノイズ付加などのオプション手順
    """
    # 画像サイズn×n
    n = 16
    # 埋め込み強度
    beta = 2e-3
    # 1クラスタあたりの点数目安(k-means用)
    cluster_point = 500
    # グラフ構築モード
    graph_mode = 'knn'
    k = 6
    radius = 0.03
    # 平面曲面アプローチ
    flatness_weighting = 0
    # 埋め込み容量アプローチ
    split_mode = 1
    # 周波数帯域アプローチ
    min_spectre = 0.0
    max_spectre = 0.2

    # 1. データ取得
    image_path = "watermark16.bmp"  # 埋め込みたい画像ファイル
    input_file = "C:/bun_zipper.ply"
    # input_file = "C:/Armadillo.ply"
    # input_file = "C:/longdress_vox12.ply"
    # input_file = "C:/soldier_vox12.ply"
    pcd_before = o3d.io.read_point_cloud(input_file)

    # 2. 前処理（色情報追加・理想クラスタ数計算）
    pcd_before = STG50F.normalize_point_cloud(pcd_before)
    pcd_before = STG50F.add_colors(pcd_before, color="grad")
    # o3d.visualization.draw_geometries([pcd_before])
    xyz = np.asarray(pcd_before.points)
    colors = np.asarray(pcd_before.colors)

    # 3. 埋め込みビット生成
    watermark_bits = STG50F.image_to_bitarray(image_path, n=n)
    watermark_bits_length = len(watermark_bits)
    print(f"[Debug] 埋込ビット数：{watermark_bits_length} (画像: {n}x{n})")
    pcd_after = o3d.geometry.PointCloud() # 埋め込み後の点群基盤

    # 4. クラスタリング
    start = time.time()
    labels = STG50F.kmeans_cluster_points(xyz, cluster_point=cluster_point)
    # labels = STG50F.region_growing_cluster_points(xyz)
    # labels = STG50F.ransac_cluster_points(xyz)
    # labels = STG50F.split_large_clusters(xyz, labels, limit_points=3000)

    # 5. 単多数決方式の埋め込み
    xyz_after = STG50F.embed_watermark_xyz(
        xyz, labels, watermark_bits, beta=beta, 
        graph_mode=graph_mode, k=k, radius=radius,
        split_mode=split_mode, flatness_weighting=flatness_weighting, k_neighbors=20, 
        min_spectre=min_spectre, max_spectre=max_spectre
    )
    embed_time = time.time() - start
    diffs = np.linalg.norm(xyz_after - xyz, axis=1)
    max_embed_shift = np.max(diffs)
    print(f"[Debug] 最大埋め込み誤差: {max_embed_shift}")

    # OP. ノイズ攻撃
    # xyz_after = STG50F.noise_addition_attack(xyz_after, noise_percent=3.0, mode='gaussian', seed=42)

    # OP. 切り取り攻撃
    # xyz_after = STG50F.cropping_attack(xyz_after, keep_ratio=0.3, mode='axis', axis=0)
    # xyz_after = STG50F.reconstruct_point_cloud(xyz_after, xyz, threshold=max_embed_shift*2)
    # xyz_after = STG50F.reorder_point_cloud(xyz_after, xyz)
    # print(len(xyz_after))

    # OP. スムージング攻撃
    # xyz_after = STG50F.smoothing_attack(xyz_after, lambda_val=0.1, iterations=5, k=6)

    # 6. 単多数決方式の抽出
    start = time.time()
    extracted_bits = STG50F.extract_watermark_xyz(
        xyz_after, xyz, labels, watermark_bits_length,
        graph_mode=graph_mode, k=k, radius=radius,
        split_mode=split_mode, min_spectre=min_spectre, max_spectre=max_spectre
    )
    extract_time = time.time() - start

    # 7. 評価
    pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
    pcd_after.colors = o3d.utility.Vector3dVector(colors)
    print(pcd_after)
    STG50F.evaluate_imperceptibility(pcd_before, pcd_after, by_index=True)

    # 8. 確認用
    o3d.visualization.draw_geometries([pcd_after])
    print(f"埋込ビット：{len(watermark_bits)}")
    print(f"抽出ビット：{len(extracted_bits)}")
    STG50F.evaluate_robustness(watermark_bits, extracted_bits)
    STG50F.bitarray_to_image(extracted_bits, n=n, save_path="recovered.bmp")
    print(f"埋込時間: {embed_time:.2f}秒")
    print(f"抽出時間: {extract_time:.2f}秒\n")