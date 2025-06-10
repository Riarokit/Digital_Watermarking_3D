import numpy as np
import open3d as o3d
import STG50_GFT_func as STG50F

# 1. 点群読み込み・色情報の追加と表示
input_file = "C:/bun_zipper.ply"
pcd_before = o3d.io.read_point_cloud(input_file)
pcd_before = STG50F.add_colors(pcd_before, color="grad")
print("元の点群データ")
o3d.visualization.draw_geometries([pcd_before])

# 2. numpy配列として点群座標を取得
xyz = np.asarray(pcd_before.points)

# 3. ランダムな文字列を生成→ビット列に変換
message_length = 10  # 埋め込む文字列の長さ（例：10文字で80ビット）
message = STG50F.generate_random_string(message_length)
watermark_bits = STG50F.string_to_binary(message)
print("埋め込むメッセージ：", message)
print("ビット列長：", len(watermark_bits))

# 4. クラスタリング（ビット長に合わせてクラスタ数を調整）
clusters_needed = len(watermark_bits)
labels = STG50F.cluster_points(
    xyz, num_clusters=clusters_needed, 
    points_per_cluster=max(50, len(xyz)//clusters_needed), seed=123
)

# 5. GFT係数を変調して埋め込み
xyz_after = STG50F.embed_watermark(
    xyz, labels, watermark_bits, coeff_idx=0, alpha=0.001, seed=123
)

# 6. 埋め込み後点群の作成・表示
pcd_after = o3d.geometry.PointCloud()
pcd_after.points = o3d.utility.Vector3dVector(xyz_after)
if pcd_before.has_colors():
    pcd_after.colors = pcd_before.colors
print("埋め込み後の点群データ")
o3d.visualization.draw_geometries([pcd_after])

# 7. 抽出
extracted_bits = STG50F.extract_watermark(xyz_after, xyz, labels, coeff_idx=0)

# 8. 抽出ビット列からメッセージ復元
recovered_message = STG50F.binary_to_string(extracted_bits)
print("抽出ビット列長:", len(extracted_bits))
print("抽出されたメッセージ：", recovered_message)
print("一致率:", np.mean(np.array(watermark_bits) == np.array(extracted_bits)))