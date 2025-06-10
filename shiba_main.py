import open3d as o3d
import numpy as np
from modules import fileread as fr
from modules import preprocess as pp
import shiba_func as sf

pcdpath = "C:/bun_zipper.ply"
output_path_location = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/octree/shiba_point.txt"
output_path_color = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/octree/shiba_color.txt"
pcd = o3d.io.read_point_cloud(pcdpath)
# o3d.visualization.draw(pcd)

# backgroundpath = "C:/Users/Public/Pythoncode/LiDAR/Python/data/pcddata/20240427/background/frame_0.pcd"
# pcdpath = "C:/Users/Public/Pythoncode/LiDAR/Python/data/pcddata/20240427/readbook2/frame_0.pcd"
# background = fr.ReadPCD(backgroundpath)
# pcd = fr.ReadPCD(pcdpath)

# removed_bg_bef = pp.RemoveBackground(background,pcd,thresh=0.1)
# removed_bg = SPCDV2.SelectPCD(removed_bg_bef,[-100,100],[-5,2],[-100,100])
#############################################点群に色付ける################################################################
# 点の座標を取得
points = np.asarray(pcd.points)

# 各座標軸に基づいて色を生成
x_values = points[:, 0]
y_values = points[:, 1]
z_values = points[:, 2]

# 各軸の最小値と最大値を取得
x_min, x_max = x_values.min(), x_values.max()
y_min, y_max = y_values.min(), y_values.max()
z_min, z_max = z_values.min(), z_values.max()

# 色のグラデーションを計算
colors = np.zeros_like(points)
colors[:, 0] = (x_values - x_min) / (x_max - x_min)  # 赤色のグラデーション
colors[:, 1] = (y_values - y_min) / (y_max - y_min)  # 緑色のグラデーション
colors[:, 2] = (z_values - z_min) / (z_max - z_min)  # 青色のグラデーション

# 点群に色を追加
pcd.colors = o3d.utility.Vector3dVector(colors)

# 点群を表示
o3d.visualization.draw_geometries([pcd])
##########################################################################################################################
size_expand=0.01
octree = o3d.geometry.Octree(max_depth=20)
octree.convert_from_point_cloud(pcd,size_expand)
# print([isinstance(octree.root_node.children[0].children[i],o3d.geometry.OctreeInternalPointNode) for i in range(8)])
# print(isinstance(octree.root_node.children[0].children,o3d.geometry.OctreeInternalPointNode))
# print(octree.root_node.children[0].children[4])
print(octree.root_node)
o3d.visualization.draw_geometries([octree])
sf.encode_octree(octree.root_node,
                output_path_location = output_path_location,
                output_path_color = output_path_color)
print("check A")

points = np.asarray(pcd.points)
max_values = np.max(points,axis=0)
min_values = np.min(points,axis=0)
range_x=(max_values[0]-min_values[0])
range_y=(max_values[1]-min_values[1])
range_z=(max_values[2]-min_values[2])
cube_size = max(range_x,range_y,range_z)
octree_size = (cube_size * (1 + size_expand))
points = sf.decode_octree(input_path_location = output_path_location,
                          input_path_color = output_path_color,
                          max_size = octree_size)
print("check B")

o3d.visualization.draw_geometries([points])
sf.psnr_color(pcd,points)
