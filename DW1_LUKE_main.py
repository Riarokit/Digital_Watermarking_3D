import time

import numpy as np
import open3d as o3d

import DW1_LUKE_func as DW1LUKE
import DW2_func as DW2F


if __name__ == "__main__":
    """LukeらのFlux法を画像ビット透かしへ拡張した実行プログラム。

    1. 三角形メッシュを正規化し、面に未参照の孤立頂点を除去する。
    2. 画像ビットを-1/+1へ変換し、互いに干渉しない三角形面のFluxへ埋め込む。
    3. 勾配法で頂点を移動し、目標Fluxを実現する。
    4. 鍵に保存した元FluxとFlux順位を使って実数透かし列を抽出する。

    0/1をそのまま埋め込むと0だけFluxが変化しないため、0を-1、1を+1へ変換する。
    抽出した実数値は0を閾値として0/1へ戻し、BERを評価する。
    """

    # === パラメータ設定 ===
    # 他手法と同じ16x16ビットマップ画像を使用する。
    n = 16

    # 論文Table 2のModel 1を参考にした初期値。モデルごとの調整が必要。
    seed = 42
    series_iterations = 8  # 論文中のランダムベクトル場の級数項数N
    alpha = 0.28
    learning_rate = 0.004
    training_iterations = 100
    # 論文の歪み閾値Dは入力モデルの元座標系で定義する。
    distortion_threshold_original = 0.0015
    # 解析勾配の一時メモリを抑える面バッチ数。
    gradient_batch_size = 256
    show_input_mesh = False
    verbose = False
    # ======================

    # 1. データ取得
    image_path = "watermark16.bmp"
    input_file = "C:/bun_zipper.ply"
    # input_file = "C:/dragon.ply"
    # input_file = "C:/Armadillo.ply"
    mesh_before = o3d.io.read_triangle_mesh(input_file)
    if len(mesh_before.vertices) == 0 or len(mesh_before.triangles) == 0:
        raise ValueError("Luke法は三角形メッシュを必要とします。面を含むPLY/OBJ/STLファイルを指定してください。")

    # 2. 前処理（正規化、色情報追加、孤立頂点除去）
    raw_vertices = np.asarray(mesh_before.vertices)
    raw_triangles = np.asarray(mesh_before.triangles)
    raw_centroid = np.mean(raw_vertices, axis=0)
    normalization_radius = float(
        np.max(np.linalg.norm(raw_vertices - raw_centroid, axis=1))
    )
    if normalization_radius <= 0.0:
        raise ValueError("モデルの正規化半径が0です。頂点座標を確認してください。")
    # normalize_point_cloud() が座標を normalization_radius で割るため、
    # 歪み閾値にも同じ尺度変換を適用する。
    distortion_threshold = (
        distortion_threshold_original / normalization_radius
    )
    isolated_indices = DW2F.find_unreferenced_vertex_indices(raw_vertices, raw_triangles)
    if show_input_mesh:
        DW2F.visualize_mesh_with_highlighted_vertices(mesh_before, highlighted_indices=isolated_indices)
    pcd_before = o3d.geometry.PointCloud()
    pcd_before.points = mesh_before.vertices
    pcd_before = DW2F.normalize_point_cloud(pcd_before)
    pcd_before = DW2F.add_colors(pcd_before, color="grad")
    xyz = np.asarray(pcd_before.points).copy()
    triangles = raw_triangles.copy()
    colors = np.asarray(pcd_before.colors).copy()
    xyz, triangles, retained_indices = DW2F.remove_unreferenced_vertices(xyz, triangles)
    colors = colors[retained_indices]
    pcd_before.points = o3d.utility.Vector3dVector(xyz)
    pcd_before.colors = o3d.utility.Vector3dVector(colors)
    print(
        "[Luke] 歪み閾値: "
        f"元座標={distortion_threshold_original:.6e}, "
        f"正規化後={distortion_threshold:.6e} "
        f"(正規化半径={normalization_radius:.6e})"
    )
    print(f"[Luke] 面に未参照の孤立頂点を {len(isolated_indices)} 個除去しました。")
    
    # 3. 埋め込みビット生成
    watermark_bits = DW2F.image_to_bitarray(image_path, n=n)
    watermark_bits_length = len(watermark_bits)
    print(f"[Debug] 埋込ビット数：{len(watermark_bits)} ({n}x{n}画像)")

    # 4. 埋め込み処理
    start_embed = time.time()
    try:
        xyz_after, key_info, embed_details = (
            DW1LUKE.embed_watermark_luke_mesh(
                xyz, triangles, watermark_size=watermark_bits_length, seed=seed,
                N=series_iterations, alpha=alpha,
                learning_rate=learning_rate, iterations=training_iterations,
                distortion_threshold=distortion_threshold,
                watermark=watermark_bits,
                center_before_embedding=True,
                gradient_batch_size=gradient_batch_size, verbose=verbose,
            )
        )
    except Exception as exc:
        print(f"[Error] Luke法の埋め込みに失敗しました: {exc}")
        raise
    embed_time = time.time() - start_embed

    print(f"[Luke] 歪み判定前の選択面数: {embed_details['selected_size_before_refinement']}")
    print(f"[Luke] 歪み閾値による除外面数: {embed_details['rejected_by_distortion']}")
    displacement = np.asarray(embed_details["pre_refinement_max_displacement"])
    if len(displacement):
        print(
            "[Luke] 閾値判定前の面別最大頂点変位: "
            f"min={np.min(displacement):.6e}, "
            f"median={np.median(displacement):.6e}, "
            f"p90={np.percentile(displacement, 90):.6e}, "
            f"max={np.max(displacement):.6e}"
        )
    gradient_errors = np.asarray(embed_details["gradient_descent_error"])
    if len(gradient_errors):
        print(f"[Luke] 勾配誤差: 初回={gradient_errors[0]:.6e}, 最終={gradient_errors[-1]:.6e}, 反復数={len(gradient_errors)}")

    # 注意: cropping/downsamplingは頂点数とtrianglesの対応を壊すため、
    # メッシュ接続を同時更新する攻撃関数を用意してから適用する。

    # OP. ノイズ攻撃
    # xyz_after = DW2F.noise_addition_attack(xyz_after, noise_percent=1.0, mode='gaussian', seed=42)
    
    # OP. スムージング攻撃
    # xyz_after = DW2F.smoothing_attack(xyz_after, lambda_val=0.1, iterations=10)

    # OP. 切り取り攻撃（不可視性評価はコメントアウト）
    # xyz_after = DW2F.cropping_attack(xyz_after, keep_ratio=0.9, mode='axis', axis=0)

    # OP. ダウンサンプリング攻撃 (不可視性評価はコメントアウト)
    # xyz_after = DW2F.downsampling_attack(xyz_after, mode='voxel', voxel_size_percent=1.0, seed=42)

    # 5. 抽出処理
    start_extract = time.time()
    extracted_bits, extract_details = DW1LUKE.extract_watermark_luke_mesh(
        xyz_after, triangles, key_info,
        translation_align=True, rotation_align=True, return_details=True,
    )
    extract_time = time.time() - start_extract

    # 歪み閾値で残ったビットを元画像上の位置へ戻す
    embedded_positions = key_info.watermark_element_indices
    extracted_bits = np.asarray(extracted_bits, dtype=np.uint8)
    missing_bits = watermark_bits_length - len(embedded_positions)
    if missing_bits != 0:
        raise RuntimeError(
            "固定長透かしを全て埋め込めなかったため、BER評価を中止します: "
            f"embedded={len(embedded_positions)}, requested={watermark_bits_length}, "
            f"missing={missing_bits}. alpha、learning_rate、iterations、"
            "distortion_thresholdを調整してください。"
        )

    recovered_bits = np.empty(watermark_bits_length, dtype=np.uint8)
    recovered_bits[embedded_positions] = extracted_bits

    # 6. 視覚品質評価
    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
    pcd_after.colors = o3d.utility.Vector3dVector(colors)
    print(pcd_after)
    DW2F.evaluate_psnr(pcd_before, pcd_after, by_index=True)
    DW2F.evaluate_pc_msdm(pcd_before, pcd_after)
    DW2F.evaluate_point_ssim(pcd_before, pcd_after)
    DW2F.visualize_embedded_points(xyz, xyz_after)
    o3d.visualization.draw_geometries([pcd_after])

    # 7. ロバスト性評価
    print(f"埋込ビット長：{len(watermark_bits)}")
    print(f"抽出ビット長：{len(recovered_bits)}")
    DW2F.evaluate_robustness(watermark_bits, recovered_bits)
    DW2F.bitarray_to_image(recovered_bits, n=n, save_path="luke_recovered.bmp")
    
    # 8. 固有評価
    print(f"[Luke] 埋め込み要求要素数: {key_info.requested_watermark_size}")
    print(f"[Luke] 実際の埋め込み要素数: {key_info.embedded_watermark_size}")
    print(f"[Luke] Flux順位参照: {extract_details['valid_rank_lookup']}")
    print(f"埋込時間: {embed_time:.2f}秒")
    print(f"抽出時間: {extract_time:.2f}秒\n")
