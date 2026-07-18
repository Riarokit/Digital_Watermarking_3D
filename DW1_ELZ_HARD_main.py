import time

import numpy as np
import open3d as o3d

import DW1_ELZ_HARD_func as DW1ELZ
import DW2_func as DW2F


if __name__ == "__main__":
    """El Zein Method II の論文準拠 HARD 版を実行・評価する。"""

    # === パラメータ設定 ===
    n = 16
    a = 0.01  # 論文の式(6)で指定された固定値
    fcm_seed = 42
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
            "面を含む PLY/OFF/OBJ を指定してください。"
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
    print(f"[ElZein-HARD] 面に未参照の孤立頂点を {len(isolated_indices)} 個除去しました。")

    # 3. 埋め込みビット生成
    watermark_bits = DW2F.image_to_bitarray(image_path, n=n)
    portion1_length, portion2_length = DW1ELZ.split_watermark_5_to_3(
        len(watermark_bits)
    )
    print(
        f"[Debug] 埋込ビット数：{len(watermark_bits)} ({n}x{n}画像), "
        f"Portion 1/2：{portion1_length}/{portion2_length}"
    )

    # 4. 埋め込み処理
    start_embed = time.time()
    try:
        xyz_after, key_info = DW1ELZ.embed_watermark_elzein_hard_mesh(
            xyz,
            triangles,
            watermark_bits,
            a=a,
            verbose=True,
            seed=fcm_seed,
        )
    except ValueError as error:
        raise RuntimeError(f"ElZein-HARDの埋め込みに失敗しました: {error}") from error
    embed_time = time.time() - start_embed

    # OP. ノイズ攻撃（頂点数と順序を維持する攻撃のみHARD抽出可能）
    # xyz_after = DW2F.noise_addition_attack(
    #     xyz_after, noise_percent=0.1, mode="gaussian", seed=42
    # )

    # OP. スムージング攻撃（頂点数と順序を維持する攻撃のみHARD抽出可能）
    # xyz_after = DW2F.smoothing_attack(
    #     xyz_after, lambda_val=0.1, iterations=10, k=6
    # )

    # 切り取り・ダウンサンプリングには同期を使わないため対応しない。

    # 5. 抽出処理
    start_extract = time.time()
    extracted_bits = DW1ELZ.extract_watermark_elzein_hard_mesh(
        xyz_after, xyz, triangles, key_info=key_info, verbose=False
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
    # DW2F.visualize_embedded_points(xyz, xyz_after)
    o3d.visualization.draw_geometries([pcd_after])

    # 7. ロバスト性評価
    print(f"埋込ビット長：{len(watermark_bits)}")
    print(f"抽出ビット長：{len(extracted_bits)}")
    DW2F.evaluate_robustness(watermark_bits, extracted_bits)
    DW2F.bitarray_to_image(
        extracted_bits, n=n, save_path="elzein_hard_recovered.bmp"
    )

    # 8. 固有評価
    print(
        f"[ElZein-HARD] Portion 1/2: "
        f"{key_info.portion1_length}/{key_info.portion2_length}"
    )
    print(f"[ElZein-HARD] Portion 2 scaling factor a: {key_info.scaling_factor}")
    print(f"埋込時間: {embed_time:.2f}秒")
    print(f"抽出時間: {extract_time:.2f}秒\n")
