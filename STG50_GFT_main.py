import numpy as np
import open3d as o3d
import STG50_GFT_func as STG50F
import time

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
message_length = 10  # 埋め込む文字列の長さ
embed_message = STG50F.generate_random_string(message_length)
embed_bits = STG50F.string_to_binary(embed_message)
embed_bits_length = len(embed_bits)
pcd_after = o3d.geometry.PointCloud() # 埋め込み後の点群基盤

# 4. クラスタリング
start = time.time()
labels = STG50F.kmeans_cluster_points(xyz, num_clusters=8, seed=42)

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

# 5. 埋め込み（channel=0: x, 1: y, 2: z）
channel = 0
beta = 1e-3
xyz_after = STG50F.embed_watermark_xyz(xyz, labels, embed_bits, channel=channel, beta=beta)

# OP. 攻撃
xyz_after = STG50F.add_noise(xyz_after, noise_percent=0.01, mode='uniform', seed=42)

# 6. 抽出
extracted_bits = STG50F.extract_watermark_xyz(xyz_after, xyz, labels, embed_bits_length=embed_bits_length, channel=channel)
pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
pcd_after.colors = o3d.utility.Vector3dVector(colors)

# 7. 評価
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