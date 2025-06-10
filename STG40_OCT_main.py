import open3d as o3d
import numpy as np
import STG40_OCT_func as STG40F

# 入力ファイルと出力ファイルのパス
input_file = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240513/extract_people/frame_0.pcd"
output_file = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/octree/STG33.txt"

# 点群データを読み込む
pcd = o3d.io.read_point_cloud(input_file)

# 1. 元の点群データを表示
print("元の点群データを表示します")
o3d.visualization.draw_geometries([pcd])

size_expand=0.01
octree = o3d.geometry.Octree(max_depth=20)
octree.convert_from_point_cloud(pcd,size_expand)
# print([isinstance(octree.root_node.children[0].children[i],o3d.geometry.OctreeInternalPointNode) for i in range(8)])
# print(isinstance(octree.root_node.children[0].children,o3d.geometry.OctreeInternalPointNode))
# print(octree.root_node.children[0].children[4])
print(octree.root_node)
# o3d.visualization.draw_geometries([octree])
STG40F.encode_octree(octree.root_node, output_path=output_file)

points = np.asarray(pcd.points)
max_values = np.max(points,axis=0)
min_values = np.min(points,axis=0)
range_x=(max_values[0]-min_values[0])
range_y=(max_values[1]-min_values[1])
range_z=(max_values[2]-min_values[2])
cube_size = max(range_x,range_y,range_z)
octree_size = (cube_size * (1 + size_expand))
points = STG40F.decode_octree(output_file, octree_size)
o3d.visualization.draw_geometries([points])
# snr = STG40F.calculate_snr(pcd,points)
# print("SNR:",snr)
