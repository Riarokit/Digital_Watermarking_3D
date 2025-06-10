import open3d as o3d
import numpy as np
import copy
import random
# from modules import fileread as fr
# from modules import preprocess as pp
# from modules import tools as t
# from selfmade import dctV2
# from selfmade import dct
# from selfmade import SelectPCD_Ver2 as SPCDV2
# from selfmade import comp
import time
import csv
import pandas as pd
import binascii
from scipy.spatial import KDTree

def string_to_binary(input_string):
    """
    文字列をバイナリビット列に変換する関数。
    
    Parameters:
    input_string (str): 変換する文字列。
    
    Returns:
    str: 文字列をバイナリに変換したビット列。
    """
    return ''.join(format(ord(char), '08b') for char in input_string)

def binary_to_string(extracted_binary_string):
    """
    バイナリビット列を文字列に変換する関数。
    
    Parameters:
    extracted_binary_string (str): 変換するバイナリビット列。
    
    Returns:
    str: バイナリを文字列に変換した文字列。
    """
    extracted_string = ''.join([chr(int(extracted_binary_string[i:i+8], 2)) for i in range(0, len(extracted_binary_string), 8)])
    return extracted_string

def display_octree(point_cloud, max_depth=8):
    """
    Octreeを表示する関数。

    Parameters:
    point_cloud (pcd): 点群
    max_depth (int): Octreeの深さ
    """
    octree = o3d.geometry.Octree(max_depth)
    octree.convert_from_point_cloud(point_cloud, size_expand=0.01)
    o3d.visualization.draw_geometries([octree])

def embed_bits_in_octree(input_path_location, input_path_color, bit_positions, binary_string):
    """
    オクツリーの最下層に、指定したバイナリビットを埋め込む関数。
    
    Parameters:
    input_path_location (str): テキストファイルのパス。
    bit_positions (list): ビットを埋め込む最下層の「0」のビット位置リスト。
    binary_string (str): 埋め込むバイナリビット列。
    """
    with open(input_path_location, 'r') as file:
        bit_stream = list(file.read())  # ビット列を文字のリストとして取得

    with open(input_path_color, 'r', encoding='utf-8') as file:
        color_lines = file.readlines()

    # 埋め込むビットを順番に、指定した位置に埋め込む
    for i, bit in enumerate(binary_string):
        if bit == '1':
            bit_stream[bit_positions[i]] = '1'  # ビットを1に変更
            # 色補正OP
            # rgb_value, count_of_ones = color_correction(bit_stream, color_lines, bit_positions, i)
            # color_lines.insert(count_of_ones, rgb_value)
    
    # 埋め込み後のビット列をテキストファイルに書き戻す
    embedded_bit_stream = ''.join(bit_stream) # 埋め込み後のビット列を文字列に戻す
    with open(input_path_location, 'w') as file:
        file.write(embedded_bit_stream)

    with open(input_path_color, 'w', encoding='utf-8') as file:
        file.writelines(color_lines)
    
    return None

def color_correction(bit_stream, color_lines, bit_positions, i):
    """
    埋め込み時に追加された点の色情報を補正するための関数(Octree由来のため正確性に欠ける)。

    Parameters:
    bit_stream (str): Octreeのバイナリビット列
    color_lines (str): Octreeの色情報
    bit_positions (list): 埋め込み位置のリスト
    i (int): バイナリビット列で何文字目の処理かを示すカウント変数

    Returns:
    rgb_value (str): 追加した点の色情報を書き出し用に文字列に変換した状態
    """
    # 各階層のビット数を計算
    level_bits_list, max_depth = countlayer(bit_stream)
    
    # 最下層の情報を取得
    deepest_layer_bits, start_position = level_bits_list[-1]

    if bit_positions[i] < start_position or bit_positions[i] >= len(bit_stream):
        return print("Bit position out of range\n")  # 指定された位置が範囲外の場合

    if bit_stream[bit_positions[i]] != '1':
        return print("There is not 1 at the point\n")  # 指定された位置に"1"がない場合

    # 開始位置から指定位置までに含まれる"1"の数を数える
    count_of_ones = bit_stream[start_position:bit_positions[i] + 1].count('1')

    # n-2, n-1, n, n+1行を取り出して平均を求め、追加された点の色とする。
    color_surrender = []  # 結果を格納するリスト
    for j in range(count_of_ones - 3, count_of_ones + 1):
        if 0 <= j < len(color_lines):
            # 各行の文字列をカンマで分割してリスト化し、結果に追加
            color_surrender.append([float(num) for num in color_lines[j].strip().split(',')])

    # 平均計算用
    column_sums = [0] * len(color_surrender[0])
    column_counts = [0] * len(color_surrender[0])
    for row in color_surrender:
        for j, value in enumerate(row):
            column_sums[j] += value
            column_counts[j] += 1
    column_averages = [column_sums[j] / column_counts[j] for j in range(len(column_sums))]
    # column_averages = [1, 0, 0] # OP. 視覚用(追加点確認用)
    
    rgb_value = f"{column_averages[0]},{column_averages[1]},{column_averages[2]}\n"

    return rgb_value, count_of_ones

def extract_bits_from_octree(input_path, bit_positions):
    """
    埋め込んだ場所からビットを抽出する関数。
    
    Parameters:
    input_path (str): テキストファイルのパス。
    bit_positions (list): ビットを抽出する最下層のビット位置リスト。
    
    Returns:
    extracted_bits (str): 抽出したバイナリビット列。
    """
    with open(input_path, 'r') as file:
        bit_stream = file.read()  # ビット列を取得

    extracted_bits = ''.join(bit_stream[pos] for pos in bit_positions)
    
    return extracted_bits


def encode_octree(node, output_path_location=None, output_path_color=None, depth=0, bit_dict=None, color_list=None):
    """
    Octreeのノードをビット列に符号化する関数

    Parameters:
    node: root_node
    depth (int): Octreeの深さ
    output_path_location (str): 座標情報ファイルのパス
    output_path_color (str): 色情報ファイルのパス
    bit_dict (?): 各層の占有マップを格納する辞書。再帰処理に使うだけだからこの関数使う人は気にしないでいい。
    color_list (?): 点の色情報を保存しとくリスト。この関数使う人は気にしないでいい。
    """
    if output_path_location is None:
        print("Specify the path to the location information file")
        return None
    
    if output_path_location is None:
        print("Specify the path to the color information file")
        return None
    
    if bit_dict is None:
        bit_dict = {}

    if color_list is None:
        color_list = []
    
    # 内部ノードのチェック（個々の子ノードを判定）
    if isinstance(node, o3d.geometry.OctreeInternalNode):
        # 子ノードが存在するかをビット列で表現
        children_bits = "".join([str(int(child is not None)) for child in node.children])
        
        if depth not in bit_dict:
            bit_dict[depth] = []
        bit_dict[depth].append(children_bits)

        # 各子ノードを再帰的に処理
        for child in node.children:
            if child is not None:
                encode_octree(child, output_path_location, output_path_color, depth + 1, bit_dict, color_list)

    elif isinstance(node,o3d.geometry.OctreePointColorLeafNode):
        color_list.append(node.color)

    if depth == 0:
        with open(output_path_location,'w') as file:
            for depth in sorted(bit_dict.keys()):
                for bits in bit_dict[depth]:
                    file.write(bits)
        with open(output_path_color,'w') as file:
            for color in color_list:
                file.write(f"{color[0]},{color[1]},{color[2]}\n")

    return None

def decode_octree(input_path_location,input_path_color,max_size=1):
    """
    テキストファイルのOctreeから点群を再構成する関数

    Parameters:
    input_path_location (str): 座標情報ファイルのパス
    input_path_color (str): 色情報ファイルのパス
    max_size (double): octreeの最初のボックス(ルートノード)の大きさ

    Returns:
    pcd (pcd): 点群データ
    """
    with open(input_path_location, 'r') as file:
        bit_stream = file.read()  # 0と1のみを取り出す

    color_list = []

    with open(input_path_color,'r') as file:
        for line in file:
            values = line.strip().split(',')
            if len (values)== 3:
                try:
                    color = list(map(float, values))
                    color_list.append(color)
                except ValueError:
                    print(f"Invalid color data: {line}")
                    continue

    level_ptrs = [0] # 現在地

    level_bits_list,max_depth = countlayer(bit_stream)
    print("level_bits_list:",level_bits_list)
    max_depth_check = len(level_bits_list)
    if max_depth != max_depth_check:
        print("max_depth calculate error:max_depth=",max_depth,"max_depth_check=",max_depth_check)
        return None
    print("Calculated max_depth:",max_depth," check:",max_depth_check)

    reconstruct_count = [0] * max_depth
    points = []

    # voxel_size = 1.0 / (2 ** max_depth)
    # max_size = voxel_size * (2 ** max_depth)
    reconstruct_pointcloud(bit_stream,level_ptrs,1,max_depth,level_bits_list,reconstruct_count,points,max_size)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(color_list))

    return pcd

def countlayer(bit_stream):
    """
    Octreeの各層に対するノードの数を数える関数

    Parameters:
    bit_stream (str): Octreeのバイナリビット列

    Returns:
    level_bits_list (tuple): (各層のノード数, ビット列でのその層のノードの開始地点)
    depth (int): Octreeの深さ
    """
    level_bits_list = []
    depth = 1
    bit_ptr = 0
    nodes_in_current_level = 8

    while bit_ptr < len(bit_stream):
        # この階層のビット数を保持
        level_bits_list.append((nodes_in_current_level,bit_ptr))

        # 次の階層のビット数を計算
        children_count = sum(1 for i in range(nodes_in_current_level) if bit_ptr + i < len(bit_stream) and bit_stream[bit_ptr + i] == '1')

        # 次の階層に子ノードがない場合、終了
        if children_count == 0:
            break

        # 次の階層に移動
        depth += 1
        bit_ptr += nodes_in_current_level
        nodes_in_current_level = children_count * 8
    
    return level_bits_list,depth - 1

def reconstruct_pointcloud(bit_stream,level_ptrs,current_depth,max_depth,level_bits_list,reconstruct_count,points,size,origin=np.array([0,0,0]),num = 0):
    """
    点群を再構成するための関数

    Parameters:
    bit_stream (str): Octreeのバイナリビット列
    level_ptrs (list): 層における点の数計算用(層内の点全探査終了で関数を終えるため)
    current_depth (int): Octreeの深さの現在地
    max_depth (int): Octreeの深さ
    level_bits_list (tuple): (各層のノード数, ビット列でのその層のノードの開始地点)
    reconstruct_count (int): 再帰した回数
    size (int): ボクセルのオフセット計算用
    origin (3次元np.array): 点を追加するときの原点（これにオフセットプラスして追加位置を特定)
    num (int?): 不要説

    Returns:
    min_size: 最下層ボクセルの大きさ（ユークリッド距離）
    """
    if current_depth > max_depth:
        return
    
    # 今の階層のノードの数と読み込み地点を取得
    nodes_in_current_level, start_ptr = level_bits_list[current_depth -1]
    
    # 1階層目以外は現在の階層のlevel_ptrsを読み取り位置にセット（start_ptrは各階層の最初の位置、countはすでに読み込んだビット数）
    if len(level_ptrs) < current_depth:
        # reconstruct_countは再帰した回数＝8ビットずつ読み込んだ回数
        count = reconstruct_count[current_depth - 2] - 1 if current_depth > 1 else 0
        level_ptrs.append(start_ptr + (count * 8))
    
    # 8ビットずつ1を走査。最深階層以外は再帰処理、最深階層は点を生成。
    for i in range(8):
        if level_ptrs[current_depth - 1] >= start_ptr + nodes_in_current_level:
            return

        # ポインタで現在地のビットを取り出し
        bit = bit_stream[level_ptrs[current_depth - 1]]
        
        # ポインタを１進める
        level_ptrs[current_depth - 1] += 1

        # 現在地が1なら処理開始
        if bit == '1':
            # ビットに対応するボクセルのオフセットを計算
            # i は 0 から 7 の値を取り、それに応じたオフセットを返す
            voxel_offset = np.array([i & 1, (i >> 1) & 1, (i >> 2) & 1],dtype=np.float32) * (size/2.0)

            #現在最深階層なら点を追加
            if current_depth == max_depth:
                min_size = size/2
                point = origin + voxel_offset
                points.append(point)
            # 現在最深階層以外なら再帰処理で階層を進む
            else:
                reconstruct_count[current_depth - 1] += 1
                next_origin = origin + voxel_offset
                min_size = reconstruct_pointcloud(bit_stream, level_ptrs,current_depth + 1,max_depth,level_bits_list,reconstruct_count,points,size/2.0,next_origin,num)
    return min_size

def find_zero_bits_in_deepest_layer(input_path):
    """
    テキストファイルから、オクツリーの最下層にある「0」のビットの位置を探しリストに格納する関数。
    
    Parameters:
    input_path (str): テキストファイルのパス。
    
    Returns:
    list: 最下層にある「0」のビットの位置を格納したリスト。
    """
    # テキストファイルを読み込む
    with open(input_path, 'r') as file:
        bit_stream = file.read()  # 0と1のみを取得

    # 各階層のビット数を計算
    level_bits_list, max_depth = countlayer(bit_stream)
    
    # 最下層の情報を取得
    deepest_layer_bits, start_ptr = level_bits_list[-1]

    # 最下層における「0」のビット位置を格納するリスト
    zero_bit_positions = []

    # 最下層のビット列をスキャンして「0」の位置を記録
    for i in range(deepest_layer_bits):
        bit_position = start_ptr + i  # ビット位置
        if bit_stream[bit_position] == '0':
            zero_bit_positions.append(bit_position)

    return zero_bit_positions

def find_zero_bits_in_deepx_layer(input_path):
    """
    テキストファイルから、オクツリーの最下層からx層分にある「0」のビットの位置を探しリストに格納する関数。
    
    Parameters:
    input_path (str): テキストファイルのパス。
    
    Returns:
    list: 最下層にある「0」のビットの位置を格納したリスト。
    """
    # テキストファイルを読み込む
    with open(input_path, 'r') as file:
        bit_stream = file.read()  # 0と1のみを取得

    # 各階層のビット数を計算
    level_bits_list, max_depth = countlayer(bit_stream)
    deepall_layer_bits = 0
    
    # 最下x層の情報を取得
    for i in range(-1, -3, -1):
        deepx_layer_bits, start_ptr = level_bits_list[i]
        deepall_layer_bits += deepx_layer_bits

    # 最下x層における「0」のビット位置を格納するリスト
    zero_bit_positions = []

    # 最下x層のビット列をスキャンして「0」の位置を記録
    for i in range(deepall_layer_bits):
        bit_position = start_ptr + i  # ビット位置
        if bit_stream[bit_position] == '0':
            zero_bit_positions.append(bit_position)

    return zero_bit_positions

def choose_positions(zero_bit_positions, binary_string_check):
    """
    find_zero_bits_in_deepest_layer関数で見つけたOctree最下層"0"より、ランダムに埋め込む位置を決定する関数。

    Parameters:
    zero_bit_positions (list): find_zero_bits_in_deepest_layer関数で見つけたOctree最下層"0"の位置
    binary_string_check (str): 検査符号付きのバイナリビット列

    Returns:
    embed_positions (list): ソート済みの埋め込み位置
    """
    embed_positions = random.sample(zero_bit_positions, len(binary_string_check))
    embed_positions.sort()
    return embed_positions

def calculate_bit_error_rate(embedded_bits, extracted_bits):
    """
    埋め込んだバイナリビットと抽出したバイナリビットの誤差率を計算する関数。
    
    Parameters:
    embedded_bits (str): 埋め込んだバイナリビット列。
    extracted_bits (str): 抽出したバイナリビット列。
    
    Returns:
    float: 誤差率（%）。
    """
    total_bits = len(embedded_bits)
    error_bits = sum(1 for emb, ext in zip(embedded_bits, extracted_bits) if emb != ext)
    
    # 誤差率の計算
    error_rate = (error_bits / total_bits) * 100
    return error_rate


def add_crc(binary_string):
    """
    OP. バイナリビット列にCRC-32検査符号を付加する関数。
    
    Parameters:
    binary_string (str): 付加するバイナリビット列。
    
    Returns:
    str: CRC-32検査符号を付加したバイナリビット列。
    """
    # バイナリ文字列をバイトに変換
    data_bytes = int(binary_string, 2).to_bytes((len(binary_string) + 7) // 8, byteorder='big')

    # CRC-32計算
    crc = binascii.crc32(data_bytes)

    # CRCをバイナリビット列に変換（32ビット）
    crc_binary = format(crc, '032b')

    # データ部とCRC-32を連結して返す
    return binary_string + crc_binary

def check_crc(extracted_binary_string):
    """
    OP. 抽出したバイナリビット列のデータ部とCRC-32検査符号部に分割し、CRCが正しいかをチェックする関数。
    
    Parameters:
    extracted_binary_string (str): 抽出したバイナリビット列。
    
    Returns:
    str: 正しければデータ部のバイナリビット列。誤っていればエラーメッセージを返す。
    """
    # データ部（最後の32ビットはCRC-32検査符号）
    data_part = extracted_binary_string[:-32]
    crc_part = extracted_binary_string[-32:]

    # バイナリ文字列をバイトに変換
    data_bytes = int(data_part, 2).to_bytes((len(data_part) + 7) // 8, byteorder='big')

    # CRC-32計算を行い、抽出されたCRCと比較
    crc_calculated = binascii.crc32(data_bytes)
    crc_calculated_binary = format(crc_calculated, '032b')

    if crc_calculated_binary == crc_part:
        print("CRC-32 check completed.")
        return data_part  # データ部を返す
    else:
        return "CRC-32 check failed, error in data."

import random

def attack(input_path, x_percent, mode='random', y=0):
    """
    OP. 攻撃想定用
    オクツリーの最下層のビットをランダムにx%変更するか、ランダムな開始位置から連続でyビット変更する関数。
    
    Parameters:
    input_path (str): オクツリーのビット列が格納されたテキストファイルのパス。
    x_percent (float): ランダムに変更するビットの割合（0～100%）。
    mode (str): 'random'または'continuous'。'random'はx%のビットをランダムに変更、'continuous'はyビットをランダムな開始位置から連続して変更。
    y (int): 'continuous'モードのときに変更する連続ビット数。
    
    Returns:
    str: 変更後のビット列。
    """
    with open(input_path, 'r') as file:
        bit_stream = list(file.read())  # ビット列を文字のリストとして取得

    # 最下層のビット範囲を取得
    level_bits_list, max_depth = countlayer(''.join(bit_stream))  # countlayer関数を使用
    nodes_in_deepest_layer, start_ptr = level_bits_list[-1]  # 最下層のノード数と開始地点を取得
    end_ptr = start_ptr + nodes_in_deepest_layer  # 最下層の終了位置

    # 最下層のビット列だけを対象にする
    bit_positions = list(range(start_ptr, end_ptr))

    if mode == 'random':
        # 最下層のx%のビットをランダムに変更
        num_bits_to_change = int(len(bit_positions) * (x_percent / 100))
        random_positions = random.sample(bit_positions, min(num_bits_to_change, len(bit_positions)))

        for pos in random_positions:
            bit_stream[pos] = '1' if bit_stream[pos] == '0' else '0'
    
    elif mode == 'continuous':
        # 開始位置をランダムに選択（yビット分の変更が可能な範囲内で選ぶ）
        max_start_position = len(bit_positions) - y  # yビット変更できる最大の開始位置
        if max_start_position < 0:
            raise ValueError(f"y ({y}) is larger than the number of available bits in the deepest layer.")
        
        start_position = random.randint(0, max_start_position)  # ランダムに開始位置を決定
        for i in range(y):
            bit_stream[bit_positions[start_position + i]] = '1' if bit_stream[bit_positions[start_position + i]] == '0' else '0'

    else:
        raise ValueError("mode should be either 'random' or 'continuous'.")

    # 変更後のビット列をテキストファイルに書き戻す
    modified_bit_stream = ''.join(bit_stream)
    with open(input_path, 'w') as file:
        file.write(modified_bit_stream)
    
    return modified_bit_stream


def Clustering(pcdData,epsm,points):
    """
    OP. 点群のノイズ除去耐性チェック用。

    Parameters:
    pcdData (pcd): ノイズ除去対象の点群データ。
    epsm (double): DBSCANの半径。
    points (int): 半径内に存在する点の数の閾値。

    Returns:
    pcd: ノイズ除去後のPCDデータ。
    """
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        clusterLabels = np.array(pcdData.cluster_dbscan(eps=epsm, min_points=points,print_progress=True))#クラスタごとにラベル付けしたものを配列で返す
        clusterLabels_noNoise = clusterLabels[clusterLabels>-1]#ノイズでないラベルを抽出
        noiseIndex = np.where(clusterLabels == -1)[0]#ノイズの点（インデックス）
        pcdData = pcdData.select_by_index(np.delete(np.arange(len(pcdData.points)),noiseIndex))#全点群の数分の等間隔のリストから、ノイズのインデックスに対応するものを削除->ノイズ出ない点（インデックス）の抽出
        return pcdData


def evaluate_added_points(pcd_before, pcd_after, mode, embed_points=None, atol=1e-5):
    # 点群の座標取得
    points_before = np.asarray(pcd_before.points)
    points_after = np.asarray(pcd_after.points)

    if mode == "compare":
        # 一致点の除外
        is_matched = np.any([np.all(np.isclose(points_after, point, atol=atol), axis=1) for point in points_before], axis=0)
        added_list = np.array([not x for x in is_matched])
    elif mode == "search":
        added_list = np.any([np.all(np.isclose(points_after, target_point), axis=1) for target_point in embed_points], axis=0)
    else:
        print("Mode error: compareかsearchを選択してください。")
        return

    added_points = points_after[added_list]
    print(f"追加点リスト型確認用: {added_list}")
    print(f"追加点の数: {len(added_points)}")

    # KDTree構築 (pcd_after から追加点を除いたもの)
    non_added_points = points_after[~added_list]

    # Open3Dの点群オブジェクトを作成
    non_added_cloud = o3d.geometry.PointCloud()
    non_added_cloud.points = o3d.utility.Vector3dVector(non_added_points)

    # KDTree構築
    kdtree = o3d.geometry.KDTreeFlann(non_added_cloud)

    # Point-to-Point と Point-to-Plane の計算
    point_to_point_distances = []
    point_to_plane_distances = []

    # 法線の計算
    non_added_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    if non_added_cloud.has_normals():
        normals_after = np.asarray(non_added_cloud.normals)
        print(f"法線の数: {len(normals_after)}")
        print(f"法線のサンプル: {normals_after[:5]}")
    else:
        print("法線が計算されませんでした。点群の密度やパラメータを確認してください。")
        normals_after = np.zeros_like(np.asarray(non_added_cloud.points))  # 法線がない場合の代替


    for added_point in added_points:
        _, idx, _ = kdtree.search_knn_vector_3d(added_point, 1)  # 最近傍点を探索
        nearest_point = non_added_points[idx[0]]  # 最近傍点（追加点以外）
        nearest_normal = normals_after[idx[0]]    # 最近傍点の法線

        # Point-to-Point
        p2p_distance = np.linalg.norm(added_point - nearest_point)
        point_to_point_distances.append(p2p_distance)

        # Point-to-Plane
        vector = added_point - nearest_point
        p2plane_distance = abs(np.dot(vector, nearest_normal))
        point_to_plane_distances.append(p2plane_distance)

    # 結果の表示: 科学技術表記 (指数表記)
    print("Point-to-Point")
    print(f"平均値: {np.mean(point_to_point_distances):.4e}, 最大値: {np.max(point_to_point_distances):.4e}")

    print("Point-to-Plane")
    print(f"平均値: {np.mean(point_to_plane_distances):.4e}, 最大値: {np.max(point_to_plane_distances):.4e}")


    return None
    
def modify_locate(before_points, after_points):
    """
    埋め込み前後の点群の位置調整用関数。

    Parameters:
    before_points (pcd): 埋め込み前点群
    after_points (pcd): 埋め込み後点群

    Returns:
    after_points (pcd): 位置を埋め込み前点群と合わせた状態の埋め込み後点群
    """
    before = np.array(before_points.points)
    after = np.array(after_points.points)
    min_x_before = np.min(before[:, 0])
    min_y_before = np.min(before[:, 1])
    min_z_before = np.min(before[:, 2])
    min_x_after = np.min(after[:, 0])
    min_y_after = np.min(after[:, 1])
    min_z_after = np.min(after[:, 2])
    dif_x = min_x_before-min_x_after
    dif_y = min_y_before-min_y_after
    dif_z = min_z_before-min_z_after
    # print(dif_x,dif_y,dif_z)
    transformation = np.array([[1, 0, 0, dif_x],
                                [0, 1, 0, dif_y],
                                [0, 0, 1, dif_z],
                                [0, 0, 0, 1]])
    after_points.transform(transformation)

    # ICPによる位置合わせ
    threshold = 0.02  # 対応点を探索する距離のしきい値
    trans_init = np.identity(4)  # 初期の変換行列（単位行列）

    # 点群同士のICP位置合わせ
    reg_p2p = o3d.pipelines.registration.registration_icp(
        before_points, after_points, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # 最適な変換行列を取得
    transformation = reg_p2p.transformation

    # 変換後の視覚化
    after_points.transform(transformation)

    return after_points

def point_to_point(A, B, threshold=0.1):
    """
    点群Aと点群BにおけるPoint-to-Point誤差を計算する関数。
    
    Parameters:
    A (pcd): 元の点群
    B (pcd): 追加点を含む点群
    threshold (double): 違う点とみなす閾値
    """
    A_kd_tree = o3d.geometry.KDTreeFlann(A)  # 点群AのKDTreeを構築し、Bの各点に対してAの最近傍点を効率的に探索
    point_errors = []  # 各点のPoint-to-Point誤差を格納するリストを初期化
    different_points_count = 0  # 位置が違う点の数をカウントする変数を初期化

    # 点群Bの各点についてループ処理
    for point in B.points:
        [_, idx, _] = A_kd_tree.search_knn_vector_3d(point, 1)  # 点群A内で、点Bの現在の点に最も近い点を探索（距離が最小の1点を見つける）
        
        nearest_point = np.asarray(A.points)[idx[0]]  # 最近傍点の座標を取得
        error = np.linalg.norm(np.asarray(point) - nearest_point)  # 点Bの点とAの最近傍点とのユークリッド距離を計算し、誤差として取得

        # 誤差が閾値以上であれば、位置が違う点としてカウント
        if error > threshold:
            different_points_count += 1
        # 誤差が閾値以下であれば、位置が同じ点とみなす
        if error <= threshold:
            error = 0

        point_errors.append(error)  # 計算した誤差をリストに追加

    average_error = np.mean(point_errors)  # Point-to-Point誤差の平均値を計算
    max_error = np.max(point_errors)  # Point-to-Point誤差の最大値を計算

    print(f"Point-to-Point 検知数: {different_points_count}")
    print(f"Point-to-Point 誤差値: 平均 = {average_error}, 最大 = {max_error}")
    
    return None

def point_to_plane(A, B, threshold=0.1):
    """
    点群Aと点群BにおけるPoint-to-Plane誤差を計算し、平均誤差と最大誤差を返す関数。

    Parameters:
    A (pcd): 元の点群
    B (pcd): 追加点を含む点群
    threshold (double): 違う点とみなす閾値
    """
    # 元の点群Aの法線を推定する（局所的な平面を求めるために必要）
    A.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # 点群AのKDTreeを構築して、Bの各点に対してAの最近傍点を効率的に探索
    A_kd_tree = o3d.geometry.KDTreeFlann(A)

    point_errors = []  # 誤差を保存するリストを初期化
    different_points_count = 0  # 位置が違う点のカウントを初期化

    # 点群Bの各点に対して処理を行う
    for point in B.points:
        # 点群A内で、点Bの現在の点に最も近い点を探索（距離が最小の1点を見つける）
        [_, idx, _] = A_kd_tree.search_knn_vector_3d(point, 1)
        
        nearest_point = np.asarray(A.points)[idx[0]]  # 最も近い点の座標を取得
        nearest_normal = np.asarray(A.normals)[idx[0]]  # 最近傍点の法線ベクトルを取得
        
        # 点Bの点と最近傍点との差分（ベクトル）を計算
        displacement = np.asarray(point) - nearest_point
        
        # 法線方向の誤差を計算（displacementと法線ベクトルの内積の絶対値）
        error = np.abs(np.dot(displacement, nearest_normal))

        # 誤差が閾値以上の場合、位置が違う点としてカウント
        if error > threshold:
            different_points_count += 1
        # 誤差が閾値以下であれば、位置が同じ点とみなす
        if error <= threshold:
            error = 0

        point_errors.append(error)  # 計算した誤差をリストに追加

    average_error = np.mean(point_errors)  # Point-to-Plane誤差の平均値を計算
    max_error = np.max(point_errors)  # Point-to-Plane誤差の最大値を計算

    print(f"Point-to-Point 検知数: {different_points_count}")
    print(f"Point-to-Plane 誤差値: 平均 = {average_error}, 最大 = {max_error}")
    
    return None

def add_colors(pcd_before, color="grad"):
    """
    色情報を追加する関数。

    Parameters:
    pcd_before (pcd): 埋め込み前点群
    color (str): "grad" = グラデーション、"black" = 全部黒(視認用)

    Returns:
    pcd_before (pcd): 色情報がついた埋め込み前点群
    """
    if color == "grad":
        # 分岐OP. 点群に色情報を追加
        points = np.asarray(pcd_before.points)  # 点の座標を取得
        x_values = points[:, 0]  # x軸に基づいて色を生成
        y_values = points[:, 1]  # y軸に基づいて色を生成
        z_values = points[:, 2]  # z軸に基づいて色を生成
        x_min, x_max = x_values.min(), x_values.max()  # x軸の最小値と最大値を取得
        y_min, y_max = y_values.min(), y_values.max()  # y軸の最小値と最大値を取得
        z_min, z_max = z_values.min(), z_values.max()  # z軸の最小値と最大値を取得
        colors = np.zeros_like(points)
        colors[:, 0] = (x_values - x_min) / (x_max - x_min)  # 赤色のグラデーション
        colors[:, 1] = (y_values - y_min) / (y_max - y_min)  # 緑色のグラデーション
        colors[:, 2] = (z_values - z_min) / (z_max - z_min)  # 青色のグラデーション

    if color == "black":
        # 分岐OP. 全ての色を黒に設定 (視認用)
        points = np.asarray(pcd_before.points)  # 点の座標を取得
        colors = np.zeros_like(points)  #全点を黒にする
    
    pcd_before.colors = o3d.utility.Vector3dVector(colors)
    return pcd_before
    
def sizing_octree(points, size_expand):
    """
    点群の範囲を計算する関数
    デコーダではこの範囲に従ってオフセットを計算して点群の縮尺を合わせてる。
    デコーダではこの大きさを初期値1にしてるので、なくても復号できるけど縮尺が全然違くなる。

    Parameters:
    points (np.asarray): 点群データの座標情報
    size_expand (double): 点群の範囲の拡張度合い

    Returns:
    octree_size (double): ルートノードの大きさ
    """
    max_values = np.max(points, axis=0)
    min_values = np.min(points, axis=0)
    range_x = (max_values[0] - min_values[0])
    range_y = (max_values[1] - min_values[1])
    range_z = (max_values[2] - min_values[2])
    cube_size = max(range_x, range_y, range_z)
    octree_size = (cube_size * (1 + size_expand))
    return octree_size


# --------------------------ここから点の間に電子透かし埋め込む関数---------------------------------
def select_embeddable_voxels(input_path_location,input_path_color=None,max_size=1):
    """

    """
    with open(input_path_location, 'r') as file:
        bit_stream = file.read()  # 0と1のみを取り出す

    color_list = []

    if input_path_color is not None:
        with open(input_path_color,'r') as file:
            for line in file:
                values = line.strip().split(',')
                if len (values)== 3:
                    try:
                        color = list(map(float, values))
                        color_list.append(color)
                    except ValueError:
                        print(f"Invalid color data: {line}")
                        continue

    level_ptrs = [0] # 現在地

    level_bits_list,max_depth = countlayer(bit_stream)
    print("level_bits_list:",level_bits_list)
    max_depth_check = len(level_bits_list)
    if max_depth != max_depth_check:
        print("max_depth calculate error:max_depth=",max_depth,"max_depth_check=",max_depth_check)
        return None
    print("Calculated max_depth:",max_depth," check:",max_depth_check)

    reconstruct_count = [0] * max_depth
    points = []
    voxel_info = []
    voxel_index = 0

    # voxel_size = 1.0 / (2 ** max_depth)
    # max_size = voxel_size * (2 ** max_depth)
    min_size,voxel_index = create_index(bit_stream,level_ptrs,1,max_depth,level_bits_list,reconstruct_count,points,voxel_info,max_size,voxel_index)
    min_size_check = max_size
    for i in range(max_depth):
        min_size_check = min_size_check/2
    if min_size != min_size_check:
        print("【error】min_sizeがなんかうまくいってません")
        return
    else:
        print("処理開始")
    
    embedding_candidates = find_embedding_candidates(voxel_info,min_size)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    if color_list:
        pcd.colors = o3d.utility.Vector3dVector(np.array(color_list))

    return pcd,embedding_candidates

def create_index(bit_stream,level_ptrs,current_depth,max_depth,level_bits_list,reconstruct_count,points,voxel_info,size,voxel_index,origin=np.array([0,0,0]),min_size=None):
    if current_depth > max_depth:
        return
    
    # 今の階層のノードの数と読み込み地点を取得
    nodes_in_current_level, start_ptr = level_bits_list[current_depth -1]
    
    # 1階層目以外は現在の階層のlevel_ptrsを読み取り位置にセット（start_ptrは各階層の最初の位置、countはすでに読み込んだビット数）
    if len(level_ptrs) < current_depth:
        # reconstruct_countは再帰した回数＝8ビットずつ読み込んだ回数
        count = reconstruct_count[current_depth - 2] - 1 if current_depth > 1 else 0
        level_ptrs.append(start_ptr + (count * 8))
    
    # 8ビットずつ1を走査。最深階層以外は再帰処理、最深階層は点を生成。
    for i in range(8):
        if level_ptrs[current_depth - 1] >= start_ptr + nodes_in_current_level:
            return

        # ポインタで現在地のビットを取り出し
        bit = bit_stream[level_ptrs[current_depth - 1]]
        
        # ポインタを１進める
        level_ptrs[current_depth - 1] += 1

        # 現在地が1なら処理開始
        if bit == '1':
            # オフセット計算
            voxel_offset = np.array([i & 1, (i >> 1) & 1, (i >> 2) & 1],dtype=np.float32) * (size/2.0)

            #現在最深階層なら点を追加
            if current_depth == max_depth:
                if min_size is None:
                    min_size = size/2
                point = origin + voxel_offset
                points.append(point)
                voxel_info.append({
                    'index': voxel_index,
                    'depth': current_depth,
                    'child_index': i,
                    'exist': 1,
                    'coordinate': point
                })
                voxel_index += 1
            # 現在最深階層以外なら再帰処理で階層を進む
            else:
                reconstruct_count[current_depth - 1] += 1
                next_origin = origin + voxel_offset
                min_size,voxel_index = create_index(bit_stream, level_ptrs,current_depth + 1,max_depth,level_bits_list,reconstruct_count,points,voxel_info,size/2.0,voxel_index,next_origin,min_size)
        else:
            if current_depth == max_depth:
                voxel_offset = np.array([i & 1, (i >> 1) & 1, (i >> 2) & 1],dtype=np.float32) * (size/2.0)
                point = origin + voxel_offset
                voxel_info.append({
                    'index': voxel_index,
                    'depth': current_depth,
                    'child_index': i,
                    'exist': 0,
                    'coordinate': point
                })
                voxel_index += 1
                
    return min_size,voxel_index

def find_embedding_candidates(voxel_info, voxelsize):
    embedding_candidates = []
    seen_candidates = []

    start_time = time.time()

    total_voxels = len(voxel_info)

    for i,target in enumerate(voxel_info):
        if target['exist'] == 1:
            target_coord = target['coordinate']
            for hit in voxel_info:
                if hit['exist'] == 1 and hit != target:
                    hit_coord = hit['coordinate']
                    # 各軸の距離を計算
                    distance_x = abs(target_coord[0] - hit_coord[0])
                    distance_y = abs(target_coord[1] - hit_coord[1])
                    distance_z = abs(target_coord[2] - hit_coord[2])
                    
                    # いずれかの軸がvoxelsize * 3以上離れている場合は除外
                    if distance_x >= voxelsize * 3 or distance_y >= voxelsize * 3 or distance_z >= voxelsize * 3:
                        continue

                    distance = np.linalg.norm(np.array(target_coord) - np.array(hit_coord))
                    if voxelsize * 2 <= distance < voxelsize * 3:
                        min_x, max_x = min(target_coord[0], hit_coord[0]), max(target_coord[0], hit_coord[0])
                        min_y, max_y = min(target_coord[1], hit_coord[1]), max(target_coord[1], hit_coord[1])
                        min_z, max_z = min(target_coord[2], hit_coord[2]), max(target_coord[2], hit_coord[2])
                        for candidate in voxel_info:
                            if candidate['exist'] == 0:
                                candidate_coord = candidate['coordinate']
                                if not any(np.array_equal(candidate_coord, seen) for seen in seen_candidates):
                                    if (min_x <= candidate_coord[0] <= max_x and
                                        min_y <= candidate_coord[1] <= max_y and
                                        min_z <= candidate_coord[2] <= max_z):
                                        embedding_candidates.append({
                                            'target': target_coord,
                                            'hit': hit_coord,
                                            'candidate': candidate_coord
                                        })
                                        seen_candidates.append(candidate_coord)
        # 進捗を表示
        if (i + 1) % 100 == 0 or (i + 1) == total_voxels:
            elapsed_time = time.time() - start_time
            print(f"Progress: {i + 1}/{total_voxels} ({(i + 1) / total_voxels * 100:.2f}%) - Elapsed time: {elapsed_time:.2f} seconds")

    
    end_time = time.time()
    elapsed_time=end_time - start_time
    print(f"埋め込み可能点の探索処理時間：{elapsed_time}秒")
    return embedding_candidates

def save_embedding_candidates_to_csv(embedding_candidates, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Target_X', 'Target_Y', 'Target_Z', 'Hit_X', 'Hit_Y', 'Hit_Z', 'Candidate_X', 'Candidate_Y', 'Candidate_Z'])  # ヘッダー行
        for candidate in embedding_candidates:
            target_x, target_y, target_z = candidate['target']
            hit_x, hit_y, hit_z = candidate['hit']
            candidate_x, candidate_y, candidate_z = candidate['candidate']
            writer.writerow([target_x, target_y, target_z, hit_x, hit_y, hit_z, candidate_x, candidate_y, candidate_z])


def choose_candidates_positions(candidates_df, binary_string_check):
    """
    find_zero_bits_in_deepest_layer関数で見つけたOctree最下層"0"より、ランダムに埋め込む位置を決定する関数。

    Parameters:
    candidates_df (pandas dataframe): find_zero_bits_in_deepest_layer関数で見つけたOctree最下層"0"の位置
    binary_string_check (str): 検査符号付きのバイナリビット列

    Returns:
    embed_points(list[array(3)]):ソート済みの埋め込み位置の点の座標['Candidate_X', 'Candidate_Y', 'Candidate_Z']
    embed_positions_dict (dict): ソート済みの埋め込み位置(['Target_X', 'Target_Y', 'Target_Z', 'Hit_X', 'Hit_Y', 'Hit_Z', 'Candidate_X', 'Candidate_Y', 'Candidate_Z'])
                                 多分後で色情報つけるのとかに使うんじゃないかと思ってる
    """
    num_sample = len(binary_string_check)

    # candidates_dfから埋め込むビット数分ランダムに選ぶ
    sampled_rows = candidates_df.sample(n = num_sample,random_state = 1)
    
    # x,y,zの優先度でソートする
    sorted_rows = sampled_rows.sort_values(by=['Candidate_X', 'Candidate_Y', 'Candidate_Z'])
    
    # ソートしたのを辞書型とnumpyで格納する
    embed_positions_dict = sorted_rows.to_dict(orient='records')
    embed_points = [row.to_numpy() for _, row in sorted_rows[['Candidate_X', 'Candidate_Y', 'Candidate_Z']].iterrows()]
    return embed_points,embed_positions_dict

def embed_to_pointcloud(pointcloud_to_embed,embed_points,binary_string_check):
    """
    点群に埋め込む点を追加する関数。
    
    Parameters:
    pointcloud_to_embed (o3d.pointcloud): 埋め込み前の点群
    embed_points (list[array(3)]):埋め込み位置の点の座標
    binary_string_check (str): 検査符号付きのバイナリビット列

    Returns:
    embedded_pointcloud (o3d.pointcloud): 透かし追加した点群
    """
    after_points = []
    after_colors = []
    for i, bit in enumerate(binary_string_check):
        if bit == '1':
            after_points.append(embed_points[i])  # 埋め込む点の座標をafter_pointsに入れる
            after_colors.append([1,0,0]) # 赤色で追加
            # # 色補正OP
            # rgb_value = color_correction(bit_stream, color_lines, bit_positions, i)
            # color_lines.insert(bit_positions[i] - 1 + i, rgb_value)
            
    # 埋め込み前の点群の座標と色を取得する
    bef_points = np.asarray(pointcloud_to_embed.points)
    bef_colors = np.zeros_like(bef_points) # 色を全部黒にする
    
    # 埋め込み前の点群と埋め込む点の座標と色を一つにまとめる
    all_points = np.vstack((bef_points,after_points))
    all_colors = np.vstack((bef_colors,after_colors))
    
    # まとめたやつで点群を作る
    embedded_pointcloud = o3d.geometry.PointCloud()
    embedded_pointcloud.points = o3d.utility.Vector3dVector(all_points)
    embedded_pointcloud.colors = o3d.utility.Vector3dVector(all_colors)
    return embedded_pointcloud

def extract_bits_from_candidates(pcd, embed_points):
    """
    埋め込んだ場所からビットを抽出する関数。
    
    Parameters:
    pcd (o3d.pointcloud): 復号した点群
    embed_points (list[array(3)]): 埋め込み位置の点の座標
    
    Returns:
    extracted_bits (str): 抽出したバイナリビット列。
    """

    point_array = np.asarray(pcd.points) #点群の位置情報をnumpyで格納
    
    # 透かしの埋め込み位置の中で復号した点群の中にほぼ同じ座標の点があれば1、なければ0でビット列を作る
    bit_array = [
        1 if np.any(np.all(np.isclose(point_array, target_point), axis=1)) else 0
        for target_point in embed_points
        ]

    extracted_bits = ''.join(map(str, bit_array)) # 一応文字列に変えとく
    count_of_ones = bit_array.count(1)
    print(f"追加点 の数: {count_of_ones}")
    return extracted_bits