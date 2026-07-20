import time

import numpy as np
import open3d as o3d

import DW1_VER_func as DW1VER
import DW2_func as DW2F


if __name__ == "__main__":
    """Vermaらの非ブラインド3Dメッシュデータ隠蔽法を実行・評価する。"""

    # === パラメータ設定 ===
    n = 16
    show_input_mesh = False
    
    # 1. データ取得
    image_path = "watermark16.bmp"
    input_file = "C:/bun_zipper.ply"
    # input_file = "C:/dragon.ply"
    # input_file = "C:/Armadillo.ply"
    mesh_before = o3d.io.read_triangle_mesh(input_file)
    if len(mesh_before.vertices) == 0 or len(mesh_before.triangles) == 0:
        raise ValueError("Verma法には三角形メッシュ M={V,F} が必要です。面を含む PLY/OFF/OBJ を指定してください。")

    # 2. 前処理
    raw_vertices = np.asarray(mesh_before.vertices)
    raw_triangles = np.asarray(mesh_before.triangles)
    isolated_indices = DW2F.find_unreferenced_vertex_indices(raw_vertices, raw_triangles)
    if show_input_mesh:
        DW2F.visualize_mesh_with_highlighted_vertices(mesh_before, highlighted_indices=isolated_indices)
    pcd_before = o3d.geometry.PointCloud()
    pcd_before.points = mesh_before.vertices
    pcd_before = DW2F.add_colors(pcd_before, color="grad")
    xyz = np.asarray(pcd_before.points).copy()
    triangles = raw_triangles.copy()
    colors = np.asarray(pcd_before.colors).copy()
    xyz, triangles, retained_indices = DW2F.remove_unreferenced_vertices(xyz, triangles)
    colors = colors[retained_indices]
    pcd_before.points = o3d.utility.Vector3dVector(xyz)
    pcd_before.colors = o3d.utility.Vector3dVector(colors)
    print(f"[Verma] 面に未参照の孤立頂点を {len(isolated_indices)} 個除去しました。")

    # 3. 埋め込みビット生成
    watermark_bits = DW2F.image_to_bitarray(image_path, n=n)
    watermark_bits_length = len(watermark_bits)
    capacity = DW1VER.compute_embedding_capacity_verma(xyz)
    print(
        f"[Debug] 埋込ビット数：{watermark_bits_length} "
        f"({n}x{n}画像), 最大容量：{capacity['capacity_bits']} bit"
    )

    # 4. 埋め込み処理
    start_embed = time.time()
    try:
        xyz_after, key_info, embed_details = DW1VER.embed_watermark_verma_mesh(
            xyz, triangles, watermark_bits
        )
    except ValueError as error:
        raise RuntimeError(f"Verma法の埋め込みに失敗しました: {error}") from error
    embed_time = time.time() - start_embed

    # OP. ノイズ攻撃
    xyz_after = DW2F.noise_addition_attack(xyz_after, noise_percent=0.1, mode="gaussian", seed=42)

    # OP. スムージング攻撃
    # xyz_after = DW2F.smoothing_attack(xyz_after, lambda_val=0.1, iterations=10)

    # OP. 切り取り攻撃（不可視性評価はコメントアウト）
    # xyz_after = DW2F.cropping_attack(xyz_after, keep_ratio=0.9, mode='axis', axis=0)

    # OP. ダウンサンプリング攻撃 (不可視性評価はコメントアウト)
    # xyz_after = DW2F.downsampling_attack(xyz_after, mode='voxel', voxel_size_percent=1.0, seed=42)

    # 5. 抽出処理
    start_extract = time.time()
    extracted_bits = DW1VER.extract_watermark_verma_mesh(
        xyz, xyz_after, triangles, key_info=key_info
    )
    extract_time = time.time() - start_extract

    # 6. 視覚品質評価
    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
    pcd_after.colors = o3d.utility.Vector3dVector(colors)
    print(pcd_after)
    DW2F.evaluate_psnr(pcd_before, pcd_after, by_index=True)
    DW2F.evaluate_pc_msdm(pcd_before, pcd_after)
    DW2F.evaluate_angular_similarity(pcd_before, pcd_after)
    DW2F.evaluate_p2d(pcd_before, pcd_after)
    DW2F.evaluate_point_ssim(pcd_before, pcd_after)
    DW2F.visualize_embedded_points(xyz, xyz_after)
    o3d.visualization.draw_geometries([pcd_after])

    # 7. ロバスト性評価
    print(f"埋込ビット長：{len(watermark_bits)}")
    print(f"抽出ビット長：{len(extracted_bits)}")
    DW2F.evaluate_robustness(watermark_bits, extracted_bits)
    DW2F.bitarray_to_image(extracted_bits, n=n, save_path="verma_recovered.bmp")

    # 8. 固有評価
    print(f"[Verma] peak bin: {key_info.peak_bin:02d}")
    print(f"[Verma] peak位置数: {len(embed_details['all_embedding_positions'])}")
    print(f"[Verma] 使用位置数: {key_info.used_position_count}")
    used_columns = np.bincount(
        embed_details["used_embedding_positions"][:, 1], minlength=9
    )
    print(f"[Verma] 使用位置の列別内訳: {used_columns.tolist()}")
    print(f"[Verma] 平均反復回数: {embed_details['average_repetitions']:.2f} 回/bit")
    print(f"[Verma] 最大埋め込み率: {embed_details['maximum_embedding_rate_bpv']:.4f} bpv")
    print(f"埋込時間: {embed_time:.2f}秒")
    print(f"抽出時間: {extract_time:.2f}秒\n")
