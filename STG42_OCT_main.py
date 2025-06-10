import open3d as o3d
import numpy as np
import STG42_OCT_func as STG42F
import random

"""
リーフノードを領域で分割して、埋め込む情報が"0"の領域では領域内のいずれかの"1"を1つ選択して"0"に、逆も同様。
共通鍵が不要な方法。元のデータと埋め込み後のデータを比較して情報を抽出する。
"""

# 入力ファイルと出力ファイルのパス
input_file = "C:/bun_zipper.ply"
original_output_file = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/octree/STG42_original_octree.txt"
modified_output_file = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/octree/STG42_modified_octree.txt"

# 0. 点群データを読み込む
pcd = o3d.io.read_point_cloud(input_file)

# 1. 元の点群データを表示
print("元の点群データを表示します")
o3d.visualization.draw(pcd)

# OP. 点群をオクツリー状態で表示
# max_depth = 10
# print("元の点群をオクツリー状態で表示します")
# STG42F.display_octree(pcd, max_depth)

# 2. 文字列をバイナリに変換
binary_string = STG42F.string_to_binary("Compareing")

# 3. 元の点群データのオクツリーを作成し、テキストファイルに出力
size_expand = 0.01
octree = o3d.geometry.Octree(max_depth=15)
octree.convert_from_point_cloud(pcd, size_expand)
STG42F.encode_octree(octree.root_node, output_path=original_output_file)

# 4. 情報を埋め込んだオクツリーを作成し、テキストファイルに出力
with open(original_output_file, 'r') as file:
    bit_stream = file.read()

level_bits_list, max_depth = STG42F.countlayer(bit_stream)
embedded_bit_stream = STG42F.embed_bits_in_octree_with_regions(original_output_file, binary_string, level_bits_list)

with open(modified_output_file, 'w') as file:
    file.write(embedded_bit_stream)

# 5. 点群の範囲を計算
octree_size = STG42F.sizing_octree(pcd, size_expand)

# 6. オクツリーをデコードし、点群を再構成して表示
points = STG42F.decode_octree(modified_output_file, octree_size)
print("埋め込み後の点群について表示します")
o3d.visualization.draw_geometries([points, pcd])

# OP. 点群をオクツリー状態で表示
# print("埋め込み後の点群をオクツリー状態で表示します")
# STG42F.display_octree(points, max_depth)

# OP. ノイズ除去
points = STG42F.Clustering(points, 0.02, 5)
octree = o3d.geometry.Octree(max_depth=15)
octree.convert_from_point_cloud(points, size_expand)
STG42F.encode_octree(octree.root_node, output_path=modified_output_file)

# 7. 情報を抽出してバイナリビットを復元（元のオクツリーと比較）
extracted_binary_string = STG42F.extract_bits_from_octree_with_comparison(original_output_file, modified_output_file, len(binary_string))

print("埋込んだバイナリビット:  ", binary_string)
print("抽出したバイナリビット:  ", extracted_binary_string)

# 8. 埋め込んだバイナリビットと抽出したバイナリビットの誤差率を計算
error_rate = STG42F.calculate_bit_error_rate(binary_string, extracted_binary_string)
print(f"ビット誤差率: {error_rate:.2f}%")

# 9. バイナリビット列を元の文字列に変換
extracted_string = STG42F.binary_to_string(extracted_binary_string)
print("抽出された文字列:", extracted_string)
