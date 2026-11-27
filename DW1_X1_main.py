import numpy as np
import open3d as o3d
import DW2_func as DW2F
import DW1_X1_func as DW1X1
import time

if __name__ == "__main__":
    """
    main関数概要:
    点群をクラスタリングし、各クラスタの疑似平面からの高さを信号値としてGFTを行う。
    各クラスタのGFT係数にビット列を埋め込み、各クラスタから復元した複数の同ビットを最後に一度のみ多数決して決定。

    パラメータ詳細:
      beta / cluster_point
          埋込強度 / GFT を行う1クラスタの目標頂点数。
      graph_mode
          "knn", "radius", "hybrid"。k は knn の近傍数、radius は radius/hybrid の半径。
      flatness_weighting
          0=なし、1=平面部重視、2=曲面部重視, 3=平面部のみ, 4=曲面部のみ。
      min_spectre / max_spectre
          埋込対象にする GFT スペクトル帯域の下限・上限。

      correspondence_mode
          "auto", "index", "coordinate", "hk" (HK+厳密割当),
          "auction" (HK+auction割当), "bidir" (双方向候補+厳密割当)。
      index_distance_factor / max_index_inconsistency_ratio
          index/auto 用の距離しきい値倍率 / 許容するインデックス不整合率。
      coordinate_distance_factor
          coordinate/hk/auction/bidir 用の距離しきい値倍率。
      one_to_one_candidates / one_to_one_workers
          hk/auction の候補数列 / 0=自動（大規模時は最大8）、正数=ワーカ数。

      attack_type / attack_parameter
          "none"（未使用）、"noise"（率%）、"smoothing"（反復回数）、
          "cropping"（保持率 0–1）、"downsampling"（voxelサイズ%）、
          "reordering"（並べ替え率 0–1）。attack_seed は乱数シード。
    """
    # === 埋込設定 ===
    image_path = "bmp/original/watermark16.bmp"
    n = 16
    cluster_point = 2000
    graph_mode = 'knn'
    k = 24
    radius = 0.03
    flatness_weighting = 0
    # === 対応付け設定 ===
    index_distance_factor = 8.0
    coordinate_distance_factor = 8.0
    max_index_inconsistency_ratio = 0.01
    correspondence_mode = "bidir"
    one_to_one_candidates = DW1X1.ONE_TO_ONE_CANDIDATES
    one_to_one_workers = 0
    # === 可視化設定 ===
    show = True
    # === 攻撃設定 ===
    attack_type = "smoothing"
    attack_parameter = 50
    attack_seed = 42
    noise_mode = "gaussian"
    smoothing_lambda = 0.1
    cropping_mode = "axis"
    cropping_axis = 0
    downsampling_mode = "voxel"

    # Bunny全周波用
    # === メッシュ別埋込プリセット ===
    # beta = 1.62e-3; min_spectre = 0.0; max_spectre = 1.0; input_file = "C:/bun_zipper.ply"
    # Bunny低周波用
    # beta = 3.596e-3; min_spectre = 0.0; max_spectre = 0.2; input_file = "C:/bun_zipper.ply"
    # Bunny高周波用
    # beta = 3.59e-3; min_spectre = 0.8; max_spectre = 1.0; input_file = "C:/bun_zipper.ply"
    # Dragon全周波用
    # beta = 1.32e-3; min_spectre = 0.0; max_spectre = 1.0; input_file = "C:/dragon.ply"
    # Dragon低周波用
    # beta = 2.9e-3; min_spectre = 0.0; max_spectre = 0.2; input_file = "C:/dragon.ply"
    # Dragon高周波用
    # beta = 2.9e-3; min_spectre = 0.8; max_spectre = 1.0; input_file = "C:/dragon.ply"
    # Armadillo全周波用
    # beta = 1.68e-3; min_spectre = 0.0; max_spectre = 1.0; input_file = "C:/Armadillo.ply"
    # Armadillo低周波用
    beta = 3.2e-3; min_spectre = 0.0; max_spectre = 0.2; input_file = "C:/Armadillo.ply"
    # Armadillo高周波用
    # beta = 4.16e-3; min_spectre = 0.8; max_spectre = 1.0; input_file = "C:/Armadillo.ply"

    # 1. データ取得
    mesh_before = o3d.io.read_triangle_mesh(input_file)
    if len(mesh_before.vertices) == 0 or len(mesh_before.triangles) == 0:
        raise ValueError("Proposed method robustness tests require a triangle mesh.")

    # 2. 前処理（色情報追加・理想クラスタ数計算）
    raw_vertices = np.asarray(mesh_before.vertices)
    raw_triangles = np.asarray(mesh_before.triangles)
    xyz, triangles, _ = DW2F.remove_unreferenced_vertices(
        raw_vertices, raw_triangles
    )
    pcd_before = o3d.geometry.PointCloud()
    pcd_before.points = o3d.utility.Vector3dVector(xyz)
    pcd_before = DW2F.normalize_point_cloud(pcd_before)
    pcd_before = DW2F.add_colors(pcd_before, color="grad")
    xyz = np.asarray(pcd_before.points).copy()
    colors = np.asarray(pcd_before.colors).copy()
    if show:
        o3d.visualization.draw_geometries([pcd_before])

    # 3. 埋め込みビット生成
    watermark_bits = DW2F.image_to_bitarray(image_path, n=n)
    watermark_bits_length = len(watermark_bits)
    print(f"[Debug] 埋込ビット数：{watermark_bits_length} (画像: {n}x{n})")

    # 4. クラスタリング
    start = time.time()
    labels = DW1X1.kmeans_cluster_points(xyz, cluster_point=cluster_point)
    # labels = DW1X1.region_growing_cluster_points(xyz)
    # labels = DW1X1.ransac_cluster_points(xyz)
    # labels = DW1X1.split_large_clusters(xyz, labels, limit_points=3000)
    if show:
        cluster_colors = DW1X1.visualize_clusters(xyz, labels)
        DW1X1.visualize_clusters_with_pseudoplanes(
            xyz,
            labels,
            cluster_colors,
            scale=1.2,
            max_planes=None,
            show_frame=False,
        )

    # 5. 単多数決方式の埋め込み
    xyz_after = DW1X1.embed_watermark_x1(
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

    # Keep the pre-attack result: yellow visualization must show watermark
    # embedding changes, not changes introduced by the subsequent attack.
    xyz_embedded = xyz_after.copy()
    triangles_after = triangles.copy()
    xyz_after, triangles_after = DW2F.apply_mesh_attack(
        xyz_after,
        triangles_after,
        attack_type,
        attack_parameter,
        attack_seed,
        noise_mode=noise_mode,
        smoothing_lambda=smoothing_lambda,
        cropping_mode=cropping_mode,
        cropping_axis=cropping_axis,
        downsampling_mode=downsampling_mode,
    )

    # 6. 単多数決方式の抽出
    start = time.time()
    extracted_bits = DW1X1.extract_watermark_x1(
        xyz_after, xyz, labels, watermark_bits_length,
        graph_mode=graph_mode, k=k, radius=radius,
        min_spectre=min_spectre, max_spectre=max_spectre,
        index_distance_factor=index_distance_factor,
        coordinate_distance_factor=coordinate_distance_factor,
        max_index_inconsistency_ratio=max_index_inconsistency_ratio,
        correspondence_mode=correspondence_mode,
        one_to_one_candidates=one_to_one_candidates,
        one_to_one_workers=one_to_one_workers,
    )
    extract_time = time.time() - start

    # 7. 視覚品質評価
    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
    if len(xyz_after) == len(colors):
        pcd_after.colors = o3d.utility.Vector3dVector(colors)
    print(pcd_after)
    if len(xyz_after) == len(xyz):
        DW2F.evaluate_psnr(pcd_before, pcd_after, by_index=True)
        # DW2F.evaluate_pc_msdm(pcd_before, pcd_after)
        DW2F.evaluate_p2d(pcd_before, pcd_after)
        DW2F.evaluate_point_ssim(pcd_before, pcd_after)
        if not show:
            DW2F.evaluate_angular_similarity(pcd_before, pcd_after)
    if show:
        DW2F.visualize_embedded_points(xyz, xyz_embedded)
        DW2F.visualize_triangle_mesh(xyz_after, triangles_after)
        o3d.visualization.draw_geometries([pcd_after])

    # 8. ロバスト性評価
    print(f"埋込ビット：{len(watermark_bits)}")
    print(f"抽出ビット：{len(extracted_bits)}")
    DW2F.evaluate_robustness(watermark_bits, extracted_bits)
    if show:
        DW1X1.visualize_cluster_recovery(
            xyz_after,
            xyz,
            labels,
            watermark_bits,
            graph_mode=graph_mode,
            k=k,
            radius=radius,
            min_spectre=min_spectre,
            max_spectre=max_spectre,
            index_distance_factor=index_distance_factor,
            coordinate_distance_factor=coordinate_distance_factor,
            max_index_inconsistency_ratio=max_index_inconsistency_ratio,
            correspondence_mode=correspondence_mode,
            one_to_one_candidates=one_to_one_candidates,
            one_to_one_workers=one_to_one_workers,
        )
    unknown_count = int(np.count_nonzero(np.asarray(extracted_bits) < 0))
    if unknown_count:
        print(f"[Proposed] undecodable watermark bits: {unknown_count}")
    display_bits = np.where(np.asarray(extracted_bits) < 0, 0, extracted_bits)
    DW2F.bitarray_to_image(
        display_bits, n=n, save_path="bmp/recovered/proposed_recovered.bmp"
    )

    # 9. 固有評価
    print(f"埋込時間: {embed_time:.2f}秒")
    print(f"抽出時間: {extract_time:.2f}秒\n")
