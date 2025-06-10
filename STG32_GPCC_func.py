import open3d as o3d
import numpy as np
import random

# 文字列をバイナリに変換
def string_to_binary(input_string):
    binary_string = ''.join(format(ord(char), '08b') for char in input_string)
    return binary_string

# バイナリを文字列に変換
def binary_to_string(binary_string):
    ascii_string = ''.join(chr(int(binary_string[i:i+8], 2)) for i in range(0, len(binary_string), 8))
    return ascii_string

# 点群データの読み込み
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

# 点群データをオクツリーに変換
def compress_point_cloud(point_cloud, max_depth):
    octree = o3d.geometry.Octree(max_depth)
    octree.convert_from_point_cloud(point_cloud, size_expand=0.01)
    return octree

# ランダムに葉ノードを選び、ビット情報を埋め込む
def embed_data_in_octree(octree, watermark_binary):
    binary_index = 0
    embedded_voxel_positions = []  # ビット情報を埋め込んだボクセルの位置を記録

    # 葉ノードをすべて集めるためのリスト
    leaf_nodes = []

    def collect_leaf_nodes(node, node_info):
        if isinstance(node, o3d.geometry.OctreeLeafNode):
            leaf_nodes.append(node_info)

    # 葉ノードを収集
    octree.traverse(collect_leaf_nodes)

    # ランダムに葉ノードを選び、ビット情報に基づいて識別
    selected_nodes = random.sample(leaf_nodes, len(watermark_binary))  # バイナリビット分ランダムに選択
    for i, node_info in enumerate(selected_nodes):
        if watermark_binary[i] == '1':
            embedded_voxel_positions.append(node_info.origin)
            print(f"Embedding '1' at {node_info.origin}")  # 埋め込む座標を出力

    return embedded_voxel_positions

# 埋め込まれたボクセルの位置からデータを復号
def extract_data_from_voxels(octree, embedded_voxel_positions, num_bits):
    extracted_binary = []
    bit_count = 0

    def check_embedded_voxel(node, node_info):
        nonlocal bit_count
        if isinstance(node, o3d.geometry.OctreeLeafNode) and bit_count < num_bits:
            # 許容誤差 (atol) を 1e-3 から 1e-2 に広げることで微細な位置ズレを吸収
            if any(np.allclose(node_info.origin, embedded_voxel, atol=1e-2) for embedded_voxel in embedded_voxel_positions):
                extracted_binary.append('1')
                print(f"Detected '1' at {node_info.origin}")  # 復号された座標を出力
            else:
                extracted_binary.append('0')
            bit_count += 1

    # オクツリーを走査して埋め込まれたボクセルを検出
    octree.traverse(check_embedded_voxel)

    return ''.join(extracted_binary)

# バイナリビットの比較を2行で表示
def display_bit_comparison(original_binary, extracted_binary):
    print("\nOriginal Binary:    ", original_binary)
    print("Extracted Binary:   ", extracted_binary)

# 点群データを表示
def display_point_cloud(pcd, title="Point Cloud"):
    print(f"Displaying: {title}")
    o3d.visualization.draw_geometries([pcd])

# Octreeを可視化するための関数
def display_octree(octree):
    print("Displaying Octree")
    o3d.visualization.draw_geometries([octree])

# バイナリの一致率を計算する
def calculate_bit_error_rate(original_binary, extracted_binary):
    total_bits = len(original_binary)
    error_bits = sum(1 for o, e in zip(original_binary, extracted_binary) if o != e)
    error_rate = (error_bits / total_bits) * 100
    return error_rate
