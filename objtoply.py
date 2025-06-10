import trimesh
import open3d as o3d
import os

# OBJファイルを読み込み、点群データに変換する関数
def convert_obj_to_point_cloud(obj_file, ply_file):
    mesh = trimesh.load(obj_file)
    points = mesh.sample(10000)  # サンプリングする点の数を指定
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(ply_file, pcd)

# ディレクトリ内のすべてのOBJファイルを変換する関数
def convert_all_objs_in_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.obj'):
            obj_file = os.path.join(input_dir, file_name)
            ply_file = os.path.join(output_dir, file_name.replace('.obj', '.ply'))
            convert_obj_to_point_cloud(obj_file, ply_file)

# 使用例
input_directory = 'path_to_obj_files'  # OBJファイルが格納されているディレクトリ
output_directory = 'path_to_output_ply_files'  # 出力するPLYファイルを保存するディレクトリ
convert_all_objs_in_directory(input_directory, output_directory)