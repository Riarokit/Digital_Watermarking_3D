import open3d as o3d
import numpy as np

# 点群データの読み込み
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

# 点群データの保存
def save_point_cloud(pcd, file_path):
    o3d.io.write_point_cloud(file_path, pcd)

# Octreeを使った点群データの圧縮
def compress_point_cloud(point_cloud, max_depth):
    octree = o3d.geometry.Octree(max_depth)
    octree.convert_from_point_cloud(point_cloud, size_expand=0.01)
    return octree

# 圧縮されたOctreeから点群データを復元
def decompress_point_cloud(octree):
    points = []  # 復元された点の座標を格納するリストを初期化

    # Octreeの各ノードから点の座標を抽出する関数
    def extract_points_from_node(node, node_info):
        # ノードが葉ノード (OctreeLeafNode) の場合、そのノードの中心座標を計算し、リストに追加
        if isinstance(node, o3d.geometry.OctreeLeafNode):
            # node_info.originはノードの原点（始点）で、node_info.sizeはそのノードの大きさ
            # ノードの中心点を計算して、リストに追加
            points.append(node_info.origin + node_info.size * 0.5)

    # Octreeを走査し、各ノードについてextract_points_from_nodeを実行
    octree.traverse(extract_points_from_node)

    # 抽出された点のリストを使ってPointCloudオブジェクトを作成
    pcd = o3d.geometry.PointCloud()
    
    # PointCloudオブジェクトのpointsプロパティに抽出された点の座標をセット
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    # 復元されたPointCloudオブジェクトを返す
    return pcd


# 点群データを表示
def display_point_cloud(pcd, title="Point Cloud"):
    print(f"Displaying: {title}")
    o3d.visualization.draw_geometries([pcd])

# Octreeを可視化するための関数
def display_octree(octree, point_cloud, max_depth=5):
    o3d.visualization.draw_geometries([octree])
