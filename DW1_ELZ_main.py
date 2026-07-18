import time

import numpy as np
import open3d as o3d

import DW1_ELZ_func as DW1ELZ
import DW2_func as DW2F


if __name__ == "__main__":
    """El Zein Portion 2ベースの冗長埋め込み方式を実行・評価する。"""

    # === パラメータ設定 ===
    n = 16
    a = 2.53e-3  # Bunny用
    # a = 3.56e-3  # Dragon用
    # a = 3.21e-3  # Armadillo用
    show_input_mesh = False

    # 1. データ取得
    image_path = "watermark16.bmp"
    input_file = "C:/bun_zipper.ply"
    # input_file = "C:/dragon_vrip_res2.ply"
    # input_file = "C:/Armadillo.ply"
    mesh_before = o3d.io.read_triangle_mesh(input_file)
    if len(mesh_before.vertices) == 0 or len(mesh_before.triangles) == 0:
        raise ValueError(
            "El Zein法には三角形メッシュ M={V,F} が必要です。"
            "面を含むPLY/OFF/OBJを指定してください。"
        )

    # 2. 前処理
    raw_vertices = np.asarray(mesh_before.vertices)
    raw_triangles = np.asarray(mesh_before.triangles)
    isolated_indices = DW2F.find_unreferenced_vertex_indices(
        raw_vertices, raw_triangles
    )
    if show_input_mesh:
        DW2F.visualize_mesh_with_highlighted_vertices(
            mesh_before, highlighted_indices=isolated_indices
        )
    pcd_before = o3d.geometry.PointCloud()
    pcd_before.points = mesh_before.vertices
    pcd_before = DW2F.normalize_point_cloud(pcd_before)
    pcd_before = DW2F.add_colors(pcd_before, color="grad")
    xyz = np.asarray(pcd_before.points).copy()
    triangles = raw_triangles.copy()
    colors = np.asarray(pcd_before.colors).copy()
    xyz, triangles, retained_indices = DW2F.remove_unreferenced_vertices(
        xyz, triangles
    )
    colors = colors[retained_indices]
    pcd_before.points = o3d.utility.Vector3dVector(xyz)
    pcd_before.colors = o3d.utility.Vector3dVector(colors)
    print(f"[ElZein] 面に未参照の孤立頂点を {len(isolated_indices)} 個除去しました。")

    # 3. 埋め込みビット生成
    watermark_bits = DW2F.image_to_bitarray(image_path, n=n)
    print(f"[Debug] 埋込ビット数：{len(watermark_bits)} ({n}x{n}画像)")

    # 4. 埋め込み処理
    start_embed = time.time()
    try:
        xyz_after, key_info = DW1ELZ.embed_watermark_elzein_mesh(
            xyz,
            triangles,
            watermark_bits,
            a=a,
            verbose=True,
        )
    except ValueError as error:
        raise RuntimeError(f"ElZeinの埋め込みに失敗しました: {error}") from error
    embed_time = time.time() - start_embed

    # OP. ノイズ攻撃
    xyz_after = DW2F.noise_addition_attack(xyz_after, noise_percent=1.5, mode="gaussian", seed=42)

    # OP. スムージング攻撃
    # xyz_after = DW2F.smoothing_attack(xyz_after, lambda_val=0.2, iterations=30, k=6)

    # OP. 切り取り攻撃
    # xyz_after = DW2F.cropping_attack(xyz_after, keep_ratio=0.5, mode="axis", axis=0)

    # OP. ダウンサンプリング攻撃
    # xyz_after = DW2F.downsampling_attack(xyz_after, mode="voxel", voxel_size_percent=1.5, seed=42)

    # 5. 抽出処理
    start_extract = time.time()
    extracted_bits = DW1ELZ.extract_watermark_elzein_mesh(
        xyz_after, xyz, triangles, key_info=key_info, verbose=True
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
        DW2F.evaluate_pc_msdm(pcd_before, pcd_after)
        DW2F.evaluate_angular_similarity(pcd_before, pcd_after)
        DW2F.evaluate_p2d(pcd_before, pcd_after)
        DW2F.evaluate_point_ssim(pcd_before, pcd_after)
        # DW2F.visualize_embedded_points(xyz, xyz_after)
    o3d.visualization.draw_geometries([pcd_after])

    # 7. ロバスト性評価
    print(f"埋込ビット長：{len(watermark_bits)}")
    print(f"抽出ビット長：{len(extracted_bits)}")
    DW2F.evaluate_robustness(watermark_bits, extracted_bits)
    unknown_count = int(np.count_nonzero(np.asarray(extracted_bits) < 0))
    if unknown_count:
        print(f"[ElZein] undecodable watermark bits: {unknown_count}")
    display_bits = np.where(np.asarray(extracted_bits) < 0, 0, extracted_bits)
    DW2F.bitarray_to_image(
        display_bits, n=n, save_path="elzein_recovered.bmp"
    )

    # 8. 固有評価
    print("[ElZein] Portion 2-only / grouped redundant majority voting")
    print(f"[ElZein] scaling factor a: {a}")
    print(f"[ElZein] carrier vertices: {len(key_info.carrier_indices)}")
    print(f"埋込時間: {embed_time:.2f}秒")
    print(f"抽出時間: {extract_time:.2f}秒\n")
