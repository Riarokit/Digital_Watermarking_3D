import open3d as o3d
import numpy as np
import STG41_OCT_func as STG41F
import random
import pandas as pd
import os

"""
点間の距離が一定の範囲内の部分にランダムで情報を埋め込む。
リストで埋め込み場所を共有（共通鍵）、リストを使って復号。情報の欠損には対応できない。
埋め込みOP: 文字列を埋め込まない場合にはコメントアウトしてよい手順
OP: 攻撃や色付けなどのオプション手順
"""

# 入力ファイルと出力ファイルのパス
input_file = "C:/bun_zipper.ply"
output_path_location = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/txtdata/STG41_point.txt"
output_path_color = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/txtdata/STG41_color.txt"
embedded_output_path_location = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/txtdata/STG41_point.txt"
embedded_output_path_color = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/txtdata/STG41_color.txt"

#埋め込み可能な点のcsvファイル。まだcsvファイルがないときは、[作る場所のパス/ファイル名.csv]を入れる
candidates_path= "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20241114/embedding_candidates10.csv"

# 0. 点群データを読み込む
pcd_before = o3d.io.read_point_cloud(input_file)
num_points_before = len(pcd_before.points)

# OP. 点群に色情報を追加 (color: "grad" = グラデーション、"black" = 全部黒)
# pcd_before = STG41F.add_colors(pcd_before, color="grad")

# 1. 元の点群データを表示
print("元の点群データについて表示します")
o3d.visualization.draw_geometries([pcd_before])

# OP. 点群をオクツリー状態で表示
# max_depth = 10
# print("元の点群をオクツリー状態で表示します")
# STG41F.display_octree(pcd_before, max_depth)

# 2. 文字列をバイナリに変換し、検査符号を追加 (埋め込みOP)
binary_string = STG41F.string_to_binary("HelloWorld")
binary_string_check = STG41F.add_crc(binary_string)

# 3. 元の点群データのオクツリーを作成し、テキストファイルに出力
size_expand = 0.01
max_depth = 10
octree = o3d.geometry.Octree(max_depth=max_depth)
octree.convert_from_point_cloud(pcd_before, size_expand)
octree_size = octree.size
STG41F.encode_octree(octree.root_node, 
                     output_path_location = output_path_location,
                     output_path_color = output_path_color)

# 4. 情報を埋め込んだオクツリーを作成し、テキストファイルに出力 (埋め込みOP)
"""
1.点群をデコードするために点群の範囲を計算
2.点群をデコードして、埋め込み可能な点の座標を計算*
3.2の中から文字列のバイナリビット分をランダムに選んでリストに保存
4.3で決まった位置で、埋め込む情報が"0"ならそのまま、"1"なら点群に点を追加
5.もう一度点群を符号化して透かし入りデータを得る

* 処理がすごい長いので1度処理結果計算したらcsvファイルにしまって、3以降はそのcsvファイルを参照することにする。
2回目からは2のステップを飛ばしている。
"""
points = np.asarray(pcd_before.points)  # 点の座標を取得

if not candidates_path:
    print("埋め込み可能な点のcsvファイルのパスか、埋め込み可能な点のcsvファイルを作りたいパスを指定してください")
elif not os.path.isdir(os.path.dirname(candidates_path)):
    print("指定されたパスが無効です。正しいディレクトリを指定してください")
elif not os.path.exists(candidates_path):
    pointcloud_to_embed, embedding_candidates =STG41F.select_embeddable_voxels(input_path_location = output_path_location,
                                                                  input_path_color = output_path_color,
                                                                  max_size = octree_size)
    STG41F.save_embedding_candidates_to_csv(embedding_candidates,
                                            file_path = candidates_path)
else:
    pointcloud_to_embed = STG41F.decode_octree(input_path_location = output_path_location,
                                   input_path_color = output_path_color,
                                   max_size = octree_size)

# 埋め込み候補点からランダムに埋め込み点を決定して埋め込み
candidates_df = pd.read_csv(candidates_path)
embed_points, embed_positions = STG41F.choose_candidates_positions(candidates_df,binary_string_check)
print(f"embed_points: {embed_points}")
embedded_pcd = STG41F.embed_to_pointcloud(pointcloud_to_embed,embed_points,binary_string_check)

# 再び点群をオクツリーに変換してテキスト化
embedded_octree = o3d.geometry.Octree(max_depth=10)
embedded_octree.convert_from_point_cloud(embedded_pcd, size_expand)
STG41F.encode_octree(embedded_octree.root_node, 
                     output_path_location = embedded_output_path_location,
                     output_path_color = embedded_output_path_color)

#######################################################################

# 6. オクツリーをデコードし、点群を再構成・位置合わせして表示
pcd_after = STG41F.decode_octree(input_path_location = embedded_output_path_location,
                              input_path_color = embedded_output_path_color,
                              max_size = octree_size)
# pcd_after = STG41F.modify_locate(pcd_before, pcd_after)
num_points_after = len(pcd_after.points)
print("埋め込み後の点群を表示します")
o3d.visualization.draw_geometries([pcd_after])
count_of_ones = binary_string_check.count('1')
print(f"点群の点の数(前,後): {num_points_before},{num_points_after}")
print(f"埋め込むビット数, 増える点の数: {len(binary_string_check)},{count_of_ones}")

# OP. 点群をオクツリー状態で表示
# print("埋め込み後の点群をオクツリー状態で表示します")
# STG42F.display_octree(pcd_after, max_depth)

# OP. ノイズ除去
# pcd_after = STG41F.Clustering(pcd_after, 0.02, 10)
# octree = o3d.geometry.Octree(max_depth=15)
# octree.convert_from_point_cloud(pcd_after, size_expand)
# STG41F.encode_octree(octree.root_node, 
#                      output_path_location = output_path_location,
#                      output_path_color = output_path_color)

# 7. 情報を抽出してバイナリビットを復元 (埋め込みOP)
"""
4で作成したリストより、埋め込み位置を特定して情報を抽出
"""
extracted_binary_string_check = STG41F.extract_bits_from_candidates(pcd_after, embed_points)
print("埋込んだバイナリビット:  ", binary_string_check)
# print("抽出したバイナリビット:  ", extracted_binary_string_check)

# 8. 埋め込んだバイナリビットと抽出したバイナリビットの誤差率を計算 (埋め込みOP)
error_rate = STG41F.calculate_bit_error_rate(binary_string, extracted_binary_string_check)
print(f"ビット誤差率: {error_rate:.2f}%")

# 9. 検査符号を検査し、データ部に分割 (埋め込みOP)
extracted_binary_string = STG41F.check_crc(extracted_binary_string_check)

# 10. バイナリビット列を元の文字列に変換 (埋め込みOP)
extracted_string = STG41F.binary_to_string(extracted_binary_string)
print("抽出された文字列:", extracted_string)

# OP. 性能評価
# threshold = octree_size / 2**(max_depth-1)  # 分岐OP. 違う点とみなす点の距離を、最下層ボクセルの大きさの半分に設定
# STG41F.point_to_point(pcd_before, pcd_after, threshold=threshold)
# STG41F.point_to_plane(pcd_before, pcd_after, threshold=threshold)
# mode= "compare"なら埋め込み前後点群比較して追加点抽出、"search"なら埋め込み点座標用いて追加点抽出(embed_points必要)
STG41F.evaluate_added_points(pcd_before, pcd_after, mode="search", embed_points=embed_points)
