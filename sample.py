import open3d as o3d
import numpy as np
import random

# 点群ファイルのパス
point_cloud_path = "C:/bun_zipper.ply"

# 点群を読み込む
pcd = o3d.io.read_point_cloud(point_cloud_path)
points = np.asarray(pcd.points)

# 削除する点の数
num_points_to_remove = 500

# 点の数を確認
num_points = points.shape[0]
if num_points_to_remove > num_points:
    raise ValueError("削除する点の数が点群の総数を超えています。")

# 削除する点のインデックスをランダムに選択
indices_to_remove = random.sample(range(num_points), num_points_to_remove)

# 残す点のインデックスを選択
indices_to_keep = list(set(range(num_points)) - set(indices_to_remove))

# 残す点の点群を作成
remaining_points = points[indices_to_keep]
pcd.points = o3d.utility.Vector3dVector(remaining_points)

# 結果を可視化
o3d.visualization.draw_geometries([pcd])
