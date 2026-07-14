import time

import numpy as np
import open3d as o3d

import DW1_ELZ_func as DW1ELZ
import DW2_func as DW2F


if __name__ == "__main__":
    """El Zein Method I を三角形メッシュに対して実行・評価する。"""
    n = 16
    n_points = n * n
    a = 2.66e-3  #Bunny用
    # a = 3.56e-3 #Dragon用
    # a = 3.21e-3 #Armadillo用

    # 1. データ取得
    image_path = "watermark16.bmp"
    input_file = "C:/bun_zipper.ply"
    # input_file = "C:/dragon_vrip_res2.ply"
    # input_file = "C:/Armadillo.ply"
    mesh_before = o3d.io.read_triangle_mesh(input_file)
    if len(mesh_before.vertices) == 0 or len(mesh_before.triangles) == 0:
        raise ValueError("El Zein method requires a triangle mesh (PLY/OFF/OBJ).")

    # 2. 前処理
    xyz = np.asarray(mesh_before.vertices).copy()
    triangles = np.asarray(mesh_before.triangles).copy()
    pcd_before = o3d.geometry.PointCloud()
    pcd_before.points = o3d.utility.Vector3dVector(xyz)
    pcd_before = DW2F.normalize_point_cloud(pcd_before)
    pcd_before = DW2F.add_colors(pcd_before, color="grad")
    xyz = np.asarray(pcd_before.points).copy()
    colors = np.asarray(pcd_before.colors).copy()

    # 3. 埋め込みビット生成
    watermark_bits = DW2F.image_to_bitarray(image_path, n=n)
    watermark_bits_length = len(watermark_bits)
    print(f"[Debug] 埋込ビット数：{watermark_bits_length} (画像: {n}x{n})")

    # 4. 埋め込み処理
    start_embed = time.time()
    xyz_after = DW1ELZ.embed_watermark_elzein_mesh(
        xyz, triangles, watermark_bits, n_points=n_points, a=a, verbose=True
    )
    embed_time = time.time() - start_embed

    # OP. ノイズ攻撃
    # xyz_after = DW2F.noise_addition_attack(xyz_after, noise_percent=0.75, mode='gaussian', seed=42)

    # OP. スムージング攻撃
    # xyz_after = DW2F.smoothing_attack(xyz_after, lambda_val=0.2, iterations=30, k=6)

    # OP. 切り取り攻撃 (不可視性評価はコメントアウト)
    # xyz_after = DW2F.cropping_attack(xyz_after, keep_ratio=0.5, mode='axis', axis=0)

    # OP. ダウンサンプリング攻撃 (不可視性評価はコメントアウト)
    # xyz_after = DW2F.downsampling_attack(xyz_after, mode='voxel', voxel_size_percent=2.0, seed=42)

    # 5. 抽出処理
    start_extract = time.time()
    extracted_bits = DW1ELZ.extract_watermark_elzein_mesh(
        xyz_after, xyz, triangles, n_points=n_points
    )
    extract_time = time.time() - start_extract

    # 6. 視覚品質評価
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

    # 9. 固有評価
    print(f"埋込時間: {embed_time:.2f}秒")
    print(f"抽出時間: {extract_time:.2f}秒\n")
