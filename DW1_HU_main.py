import numpy as np
import open3d as o3d
import DW2_func as DW2F  # 共通ユーティリティ（正規化、評価等）
import DW1_HU_func as DW1HU  # 先ほど作成したHuらの関数ファイル
import time

if __name__ == "__main__":
    """
    main関数概要:
    Huらが提案した表面上のEMDに基づく高精度電子透かし手法の実行プログラム。
    点群をメッシュ化して表面 EMD を実行し、第一 IMF の安定極値に透かしを
    T_rep 回繰り返し埋め込む。抽出時は多数決投票でビットを復元する。

    パラメータ詳細:
      fide_p / T_rep
          Hu 法の目標 FideP と各ビットの円状埋込回数。fide_p を小さくすると
          埋込強度 alpha は大きくなる。
      correspondence_mode
          "auto", "index", "coordinate"。coordinate は保存した埋込極値と
          攻撃後に再検出した極値を座標距離で対応付ける。
      hu_coordinate_distance_factor
          coordinate 対応付けに用いる距離しきい値の倍率。

      attack_type / attack_parameter
          "none"（未使用）、"noise"（率%）、"smoothing"（反復回数）、
          "cropping"（保持率 0–1）、"downsampling"（voxelサイズ%）、
          "reordering"（並べ替え率 0–1）。attack_seed は乱数シード。
    """
    
    # === 埋込設定 ===
    image_path = "watermark16.bmp"
    n = 16
    # === 対応付け設定 ===
    correspondence_mode = "coordinate"
    hu_coordinate_distance_factor = 30.0
    # === 可視化設定 ===
    show = True
    # === 攻撃設定 ===
    attack_type = "downsampling"
    attack_parameter = 1.0
    attack_seed = 42
    noise_mode = "gaussian"
    smoothing_lambda = 0.1
    cropping_mode = "axis"
    cropping_axis = 0
    downsampling_mode = "voxel"

    # === Hu 固有の埋込設定 ===
    fide_p = 52.5
    T_rep = 25
    
    # ======================
    # 1. データ取得
    # input_file = "C:/bun_zipper.ply"
    input_file = "C:/dragon.ply"
    # input_file = "C:/Armadillo.ply"
    mesh_before = o3d.io.read_triangle_mesh(input_file)
    if len(mesh_before.vertices) == 0 or len(mesh_before.triangles) == 0:
        raise ValueError("Hu 法は三角形メッシュ M={V,F} を必要とします。面を含む PLY/OBJ を指定してください。")
    
    # 2. 前処理
    # 面は青、面に属さない頂点は赤で表示する。wireframe は面接続を確認するためのもの。
    raw_vertices = np.asarray(mesh_before.vertices)
    raw_triangles = np.asarray(mesh_before.triangles)
    isolated_indices = DW2F.find_unreferenced_vertex_indices(raw_vertices, raw_triangles)
    if show:
        DW2F.visualize_mesh_with_highlighted_vertices(mesh_before, highlighted_indices=isolated_indices)
    xyz, triangles, _ = DW2F.remove_unreferenced_vertices(
        raw_vertices, raw_triangles
    )
    pcd_before = o3d.geometry.PointCloud()
    pcd_before.points = o3d.utility.Vector3dVector(xyz)
    pcd_before = DW2F.normalize_point_cloud(pcd_before)
    pcd_before = DW2F.add_colors(pcd_before, color="grad")
    xyz = np.asarray(pcd_before.points).copy()
    colors = np.asarray(pcd_before.colors).copy()
    print(f"[Hu] 面に未参照の孤立頂点を {len(isolated_indices)} 個除去しました。")

    # 3. 埋め込みビット生成
    watermark_bits = DW2F.image_to_bitarray(image_path, n=n)
    watermark_bits_length = len(watermark_bits)
    print(f"[Debug] 埋込ビット数：{len(watermark_bits)} ({n}x{n}画像)")

    # 4. 埋め込み処理
    start_embed = time.time()
    try:
        # メッシュベースの埋め込み関数を呼び出し
        xyz_after, key_info, alpha = DW1HU.embed_watermark_hu_mesh(
            xyz, triangles, watermark_bits, FideP=fide_p, T=T_rep,
            watermark_size=n,
        )
    except Exception as e:
        print(f"[Error] 埋め込み失敗: {e}")
        exit()
    embed_time = time.time() - start_embed
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

    # 5. 抽出処理
    start_extract = time.time()
    extracted_bits = DW1HU.extract_watermark_hu_mesh(
        xyz_after,
        triangles_after,
        key_info,
        correspondence_mode=correspondence_mode,
        coordinate_distance_factor=hu_coordinate_distance_factor,
        verbose=True,
    )
    extract_time = time.time() - start_extract
    
    # 6. 視覚品質評価
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

    # 7. ロバスト性評価
    print(f"埋込ビット長：{len(watermark_bits)}")
    print(f"抽出ビット長：{len(extracted_bits)}")
    DW2F.evaluate_robustness(watermark_bits, extracted_bits)
    unknown_count = int(np.count_nonzero(np.asarray(extracted_bits) < 0))
    if unknown_count:
        print(f"[Hu] undecodable watermark bits: {unknown_count}")
    display_bits = np.where(np.asarray(extracted_bits) < 0, 0, extracted_bits)
    DW2F.bitarray_to_image(display_bits, n=n, save_path="hu_recovered.bmp")
    
    # 8. 固有評価
    print(f"[Hu] 埋込強度 alpha: {alpha:.6e}")
    print(f"[Hu] 埋込位置数: {len(key_info.embedding_indices)} "
          f"({T_rep} repetitions x {len(watermark_bits)} bits)")
    print(f"埋込時間: {embed_time:.2f}秒")
    print(f"抽出時間: {extract_time:.2f}秒\n")
