import numpy as np
import open3d as o3d
import STG50_GFT_func as STG50F
import time

if __name__ == "__main__":
    # 0. 各パラメータ設定
    """
    各パラメータ設定

    message_length = 埋め込む文字列の長さ
    num_clusters = クラスタ数
    beta = 調整係数
    split_mode = 0:チャネル間に同一の埋め込み, 1:チャネル間に異なる埋め込み
    flatness_weighting = 0:なし, 1:平面部重み, 2:曲面部重み
    min_weight, max_weight = 最小・最大重み係数 (平均1になるようにする)
    embed_spectre = (0,1] 低周波成分側からembed_spectre*100%まで埋め込む
    """
    # 基礎
    message_length = 100
    num_clusters = 50
    beta = 1e-3
    # 埋め込み容量アプローチ
    split_mode = 0
    # 平面曲面アプローチ
    flatness_weighting = 1
    min_weight = 0
    max_weight = 2.0
    # 周波数アプローチ
    embed_spectre = 1.0

    # 1. 点群読み込み・色情報の追加と表示
    input_file = "C:/bun_zipper.ply"
    pcd_before = o3d.io.read_point_cloud(input_file)
    pcd_before = STG50F.add_colors(pcd_before, color="grad")
    # o3d.visualization.draw_geometries([pcd_before])
    print("points: ", pcd_before)

    # 2. numpy配列として点群を取得
    xyz = np.asarray(pcd_before.points)
    colors = np.asarray(pcd_before.colors)

    # 3. 埋め込みビット生成
    embed_message = STG50F.generate_random_string(message_length)
    embed_bits = STG50F.string_to_binary(embed_message)
    embed_bits_length = len(embed_bits)
    pcd_after = o3d.geometry.PointCloud() # 埋め込み後の点群基盤

    # 4. クラスタリング
    start = time.time()
    labels = STG50F.kmeans_cluster_points(xyz, num_clusters=num_clusters, seed=42)

    ########################################## OP. 色情報埋め込み #############################################

    # 5. 埋め込み（channel=0: R, 1: G, 2: B）
    # channel = 0
    # beta = 0.01
    # colors_after = STG50F.embed_watermark_rgb(colors, xyz, labels, embed_bits, channel=channel, beta=beta)

    # 6. 抽出
    # extracted_bits = STG50F.extract_watermark_rgb(colors_after, colors, xyz, labels, embed_bits_length=embed_bits_length, channel=channel)
    # pcd_after.points = o3d.utility.Vector3dVector(xyz)
    # pcd_after.colors = o3d.utility.Vector3dVector(colors_after)

    # 7. 評価

    ###########################################################################################################

    ########################################## OP. 座標情報埋め込み #############################################

    # 5. 埋め込み
    xyz_after = STG50F.embed_watermark_xyz(xyz, labels, embed_bits, beta=beta,
                                        split_mode=split_mode , flatness_weighting=flatness_weighting, k_neighbors=20, 
                                        min_weight=min_weight, max_weight=max_weight, embed_spectre=embed_spectre)
    diffs = np.linalg.norm(xyz_after - xyz, axis=1)
    max_embed_shift = np.max(diffs)
    print("最大埋め込み誤差:", max_embed_shift)

    # OP. ノイズ攻撃
    # xyz_after = STG50F.add_noise(xyz_after, noise_percent=0.05, mode='uniform', seed=42)

    # OP. 切り取り攻撃
    xyz_after = STG50F.crop_point_cloud_xyz(xyz_after, crop_ratio=0.9, mode='center')
    xyz_after = STG50F.reconstruct_point_cloud(xyz_after, xyz, threshold=max_embed_shift*2)
    print(len(xyz_after))
    xyz_after = STG50F.reorder_point_cloud(xyz_after, xyz)
    print(len(xyz_after))

    # 6. 抽出
    extracted_bits = STG50F.extract_watermark_xyz(xyz_after, xyz, labels, embed_bits_length=embed_bits_length,
                                                  split_mode=split_mode, embed_spectre=embed_spectre)
    pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
    pcd_after.colors = o3d.utility.Vector3dVector(colors)

    # 7. 評価
    print(pcd_after)
    psnr = STG50F.calc_psnr_xyz(pcd_before, pcd_after)

    ############################################################################################################

    # 8. 確認用
    o3d.visualization.draw_geometries([pcd_after])
    accuracy = np.mean(np.array(embed_bits) == np.array(extracted_bits))
    extracted_message = STG50F.binary_to_string(extracted_bits)
    all_time = time.time() - start
    print(f"埋込文字列：{embed_message}")
    print(f"抽出文字列：{extracted_message}")
    print(f"埋込ビット：{len(embed_bits)}")
    print(f"抽出ビット：{len(extracted_bits)}")
    print(f"ビット一致率: {accuracy:.3f}")
    print(f"全体実行時間: {all_time:.2f}秒")