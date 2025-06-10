import open3d as o3d
import numpy as np
import STG45_func as STG45F
import random
import pandas as pd
import os
import modules.functions as fct

"""
main関数概要:
点間の距離が一定の範囲内の部分にランダムで情報を埋め込む。
リストで埋め込み場所を共有（共通鍵）、リストを使って復号。

使用変数説明:
pcd_before:    [量子化前・埋め込みなし]
pcd_quantized: [量子化後・埋め込みなし]
pcd_embedded:  [量子化後・埋め込みあり] (pcd_afterと同じだが、エンコード前の点群という意味で分別)
pcd_after:     [量子化後・埋め込みあり] (pcd_embeddedと同じだが、デコード後の点群という意味で分別)

オプション設定:
埋め込みOP: 文字列を埋め込まない場合にはコメントアウトしてよい手順
OP: 攻撃や色付けなどのオプション手順
"""

################################# 入力ファイルと出力ファイルのパス ###################################

# quantized = [埋め込みなし], embedded = [埋め込みあり]でエンコード。ファイル名は同じでも別でもよい。
input_file = "C:/bun_zipper.ply"
quantized_output_path_location = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/txtdata/STG45_point.txt"
quantized_output_path_color = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/txtdata/STG45_color.txt"
embedded_output_path_location = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/txtdata/STG45_point.txt"
embedded_output_path_color = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/txtdata/STG45_color.txt"

# 埋め込み可能な点のcsvファイル。まだcsvファイルがないときは、[作る場所のパス/ファイル名.csv]を入れる
candidates_path= "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20241214/bunny_depth10_dist5to8_no1.csv"

##################################### 0. 点群データを読み込む #######################################

pcd_before = o3d.io.read_point_cloud(input_file)

############### OP. 点群に色情報を追加 (color: "grad" = グラデーション、"black" = 全部黒) #############

pcd_before = STG45F.add_colors(pcd_before, color="black")

##################################### 1. 元の点群データを表示 #######################################

print("元の点群データについて表示します")
o3d.visualization.draw_geometries([pcd_before])

################################## OP. 点群をオクツリー状態で表示 ####################################

# max_depth = 10
# print("元の点群をオクツリー状態で表示します")
# STG45F.display_octree(pcd_before, max_depth)

####################### 2. 文字列をバイナリに変換し、検査符号を追加 (埋め込みOP) ########################

# 引数に指定した長さの英数字の文字列を生成
embed_string = STG45F.generate_random_string(396)
embed_binary_string = STG45F.string_to_binary(embed_string)
embed_binary_string_check = STG45F.add_crc(embed_binary_string)

##################### 3. 元の点群データのオクツリーを作成し、テキストファイルに出力 ######################

size_expand = 0.01
max_depth = 10
octree = o3d.geometry.Octree(max_depth=max_depth)
octree.convert_from_point_cloud(pcd_before, size_expand)
max_voxelsize = octree.size
STG45F.encode_octree(octree.root_node, 
                     output_path_location = quantized_output_path_location,
                     output_path_color = quantized_output_path_color)

############### 4. 情報を埋め込んだオクツリーを作成し、テキストファイルに出力 (埋め込みOP) ################
"""
1.点群をデコードするために点群の範囲を計算
2.点群をデコードして、埋め込み可能な点の座標を計算*
3.2の中から文字列のバイナリビット分をランダムに選んでリストに保存
4.3で決まった位置で、埋め込む情報が"0"ならそのまま、"1"なら点群に点を追加
5.もう一度点群を符号化して透かし入りデータを得る

* 処理がすごい長いので1度処理結果計算したらcsvファイルにしまって、3以降はそのcsvファイルを参照することにする。
2回目からは2のステップを飛ばしている。
"""

pcd_quantized = STG45F.decode_octree(input_path_location = quantized_output_path_location,
                              input_path_color = quantized_output_path_color,
                              max_voxelsize = max_voxelsize)
if not candidates_path:
    print("埋め込み可能な点のcsvファイルのパスか、埋め込み可能な点のcsvファイルを作りたいパスを指定してください")
elif not os.path.isdir(os.path.dirname(candidates_path)):
    print("指定されたパスが無効です。正しいディレクトリを指定してください")
elif not os.path.exists(candidates_path):
    embedding_candidates = STG45F.select_embeddable_between(input_path_location = quantized_output_path_location,
                                                            input_path_color = quantized_output_path_color,
                                                            max_voxelsize = max_voxelsize)
    STG45F.save_embedding_candidates_to_csv(embedding_candidates,
                                            file_path = candidates_path)

# 埋め込み候補点からランダムに埋め込み点を決定して埋め込み
candidates_df = pd.read_csv(candidates_path)
embed_points, embed_positions = STG45F.choose_embedding_between(candidates_df,embed_binary_string_check)
pcd_embedded = STG45F.embed_to_pointcloud(pcd_quantized,embed_points,embed_binary_string_check)
# o3d.visualization.draw_geometries([pcd_embedded])

# 再び点群をオクツリーに変換してテキスト化
embedded_octree = o3d.geometry.Octree(max_depth=max_depth)
embedded_octree.convert_from_point_cloud(pcd_embedded, size_expand)
STG45F.encode_octree(embedded_octree.root_node, 
                     output_path_location = embedded_output_path_location,
                     output_path_color = embedded_output_path_color)

################################ 5. オクツリーをデコードして表示 ##################################

pcd_after = STG45F.decode_octree(input_path_location = embedded_output_path_location,
                              input_path_color = embedded_output_path_color,
                              max_voxelsize = max_voxelsize)
print("埋め込み後の点群を表示します")
o3d.visualization.draw_geometries([pcd_after])

################################### 6. 正確性確認 (埋め込みOP) ####################################

count_of_ones = embed_binary_string_check.count('1')
num_points_before = len(pcd_before.points)
num_points_after = len(pcd_after.points)
min_voxelsize = max_voxelsize / 2**max_depth

print("-------------------------正確性確認用----------------------------")
print(f"点群の点の数(前,後): {num_points_before},{num_points_after}")
print(f"埋め込むビット数, 追加されるべき点数: {len(embed_binary_string_check)},{count_of_ones}")
print(f"(最大,最小)ボクセルの大きさ: {max_voxelsize:.6f},{min_voxelsize:.6f}")

################################ OP. 点群をオクツリー状態で表示 ####################################

# print("埋め込み後の点群をオクツリー状態で表示します")
# STG45F.display_octree(pcd_after, max_depth)

######################################## OP. 切り取り攻撃 #########################################

# cropped_range = STG45F.crop_axis_range(pcd_after, axis="z", cropping_rate=0.7)
# pcd_after = fct.SelectPCD(pcd_after, xlim=[], ylim=[], zlim=cropped_range)
# print("切り取り攻撃後の点群を表示します")
# o3d.visualization.draw_geometries([pcd_after])

####################################### OP. ノイズ除去 ###########################################

# pcd_after = STG45F.Clustering(pcd_after, 0.02, 10)
# octree = o3d.geometry.Octree(max_depth=15)
# octree.convert_from_point_cloud(pcd_after, size_expand)
# STG45F.encode_octree(octree.root_node, 
#                      output_path_location = embedded_output_path_location,
#                      output_path_color = embedded_output_path_color)

###################### 7. 情報を抽出して文字列を復元・誤差率計算 (埋め込みOP) ########################
"""
4で作成したリストより、埋め込み位置を特定して情報を抽出
"""
print("--------------------------文字列復号----------------------------")
extracted_binary_string_check = STG45F.extract_bits_from_candidates(pcd_after, embed_points, min_voxelsize)
# print("埋込んだバイナリビット:  ", embed_binary_string_check)
# print("抽出したバイナリビット:  ", extracted_binary_string_check)

# 埋め込んだバイナリビットと抽出したバイナリビットの誤差率を計算
STG45F.calculate_bit_error_rate(embed_binary_string_check, extracted_binary_string_check)

# 検査符号を検査し、データ部に分割
extracted_binary_string = STG45F.check_crc(extracted_binary_string_check)

# バイナリビット列を元の文字列に変換
STG45F.binary_to_string(extracted_binary_string)

######################################### OP. 性能評価 ###########################################
"""
CV = "変動係数で評価"
→ 引数use_radius: False...分散・MAD・CVの近傍点探索を点数で行う。(num_neighborsに探索点数を入力)
→ 引数use_radius: True...分散・MAD・CVの近傍点探索を半径で行う。（radiusに探索半径を入力）

PSNR = "PSNRで評価"
→ modify_locateで、pcd_beforeをpcd_afterの位置に合わせる。戻り値に注意。この場合はbeforeを戻り値に戻す。
→ embed_pointsは一度Octreeに変換した後の座標でとっているため、before側に合わせるとembed_pointsが機能しなくなる
→ pcd_beforeにnon_added_pointsを入れることで、追加点を除外した埋め込み後の点群と比較できる
"""
print("-------------------追加点の近傍点との距離のCV--------------------")
# 近傍点探索に半径を用いる場合、最下層ボクセルサイズのx倍に設定
x = 10
radius = x * max_voxelsize / (2**max_depth)
STG45F.evaluate_CV(pcd_after, embed_points, max_voxelsize, min_voxelsize, use_radius=False, num_neighbors=4) # (埋め込みOP)
print("------------------Octree量子化前の点群とのPSNR-------------------")
# 評価用に位置合わせ
pcd_before = STG45F.modify_locate(pcd_after, pcd_before)
STG45F.evaluate_PSNR(pcd_before, pcd_after, max_voxelsize, min_voxelsize)
print("------------------Octree量子化後の点群とのPSNR-------------------")
STG45F.evaluate_PSNR(pcd_quantized, pcd_after, max_voxelsize, min_voxelsize)
print("---------------------------------------------------------------")