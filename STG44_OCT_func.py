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
import string


"""
STG43との変更点
binary_to_string
check_crc
caluculate_bit_error_rate
caluculate_centroid
select_embeddable_random
evaluate_CV
create_index
find_choose_embedding_random
embed_to_pointcloud

変更すべき点
binary_to_string: リストに対応できてない
"""

def generate_random_string(length):
    """
    指定した長さのランダムな英数字文字列を生成する関数。

    Parameters:
    length (int): 生成する文字列の長さ

    Returns:
    str: 生成されたランダムな英数字文字列
    """
    random.seed(42)
    if length <= 0:
        raise ValueError("文字列の長さは正の整数で指定してください。")

    characters = string.ascii_letters + string.digits  # 英字（大小）と数字を含む
    random_string = ''.join(random.choices(characters, k=length))  # ランダムな文字列を生成
    return random_string

def string_to_binary(input_string):
    """
    文字列をUTF-8バイナリビット列に変換する関数。
    
    Parameters:
    input_string (str): 変換する文字列。
    
    Returns:
    str: UTF-8エンコードされた文字列をバイナリに変換したビット列。
    """
    # UTF-8エンコード後、各バイトを2進数に変換して結合
    return ''.join(format(byte, '08b') for byte in input_string.encode('utf-8'))

def binary_to_string(extracted_binary_strings):
    """
    バイナリビット列をUTF-8文字列に変換する関数。
    
    Parameters:
    extracted_binary_string (str): 変換するバイナリビット列。
    """
    for i, extracted_binary_string in enumerate(extracted_binary_strings):
        # 8ビットごとに区切り、整数に変換したリストをUTF-8デコード
        byte_arrays = []
        byte_array = bytearray(int(extracted_binary_string[i:i+8], 2) for i in range(0, len(extracted_binary_string), 8))
        byte_array.decode('utf-8')
        byte_arrays.append(byte_array)
        print(f"Area:{i} で抽出された文字列: {byte_arrays}")

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

def calculate_bit_error_rate(embedded_bits, extracted_bits_lists):
    """
    埋め込んだバイナリビットと抽出したバイナリビットの誤差率を計算する関数。
    
    Parameters:
    embedded_bits (str): 埋め込んだバイナリビット列。
    extracted_bits (str): 抽出したバイナリビット列。
    """
    for i, extracted_bits in enumerate(extracted_bits_lists):
        total_bits = len(embedded_bits)
        error_bits = sum(1 for emb, ext in zip(embedded_bits, extracted_bits) if emb != ext)
        
        # 誤差率の計算
        error_rate = (error_bits / total_bits) * 100
        print(f"Area:{i} ビット誤差率: {error_rate:.2f}%")


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


def check_crc(extracted_binary_strings):
    """
    抽出したバイナリビット列を8つのデータ部とCRC部に分割し、それぞれのCRC-32チェックを行う関数。

    Parameters:
    extracted_binary_string (["010100", "1010101", ..., "1000100"]): 抽出したバイナリビット列。

    Returns:
    list: 8つのデータ部を格納した文字列のリスト。
    """
    data_lists = []
    for i, extracted_binary_string in enumerate(extracted_binary_strings):
        data_list = []
        # データ部（最後の32ビットはCRC-32検査符号）
        data_part = extracted_binary_string[:-32]
        crc_part = extracted_binary_string[-32:]

        # バイナリ文字列をバイトに変換
        data_bytes = int(data_part, 2).to_bytes((len(data_part) + 7) // 8, byteorder='big')

        # CRC-32計算を行い、抽出されたCRCと比較
        crc_calculated = binascii.crc32(data_bytes)
        crc_calculated_binary = format(crc_calculated, '032b')

        if crc_calculated_binary == crc_part:
            print(f"CRC-32 Area: {i} check completed.")
            data_list = data_part
            data_lists.append(data_list)
        else:
            print(f"CRC-32 Area: {i} check failed, error in data.")
    return data_lists       # データ部を返す
    


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

def modify_locate(pcd_standard, pcd_movement):
    """
    埋め込み前後の点群の位置調整用関数。

    Parameters:
    pcd_standard (pcd): 位置の基準にする点群
    pcd_movement (pcd): 位置を変更する点群

    Returns:
    pcd_movement (pcd): 位置をpcd_standardに合わせたあとの点群pcd_movement
    """
    standard = np.array(pcd_standard.points)
    movement = np.array(pcd_movement.points)
    min_x_standard = np.min(standard[:, 0])
    min_y_standard = np.min(standard[:, 1])
    min_z_standard = np.min(standard[:, 2])
    min_x_movement = np.min(movement[:, 0])
    min_y_movement = np.min(movement[:, 1])
    min_z_movement = np.min(movement[:, 2])
    dif_x = min_x_standard-min_x_movement
    dif_y = min_y_standard-min_y_movement
    dif_z = min_z_standard-min_z_movement
    # print(dif_x,dif_y,dif_z)
    transformation = np.array([[1, 0, 0, dif_x],
                                [0, 1, 0, dif_y],
                                [0, 0, 1, dif_z],
                                [0, 0, 0, 1]])
    pcd_movement.transform(transformation)


    ############## ICPによる位置合わせ (したら、点全体の評価の意味は若干損なわれる) #############
    # threshold = 0.02  # 対応点を探索する距離のしきい値
    # trans_init = np.identity(4)  # 初期の変換行列（単位行列）

    # # 点群同士のICP位置合わせ
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     pcd_standard, pcd_movement, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # )

    # # 最適な変換行列を取得
    # transformation = reg_p2p.transformation

    # # 変換後の視覚化
    # pcd_movement.transform(transformation)

    return pcd_movement

def evaluate_CV(pcd_after, embed_points, octree_size, use_radius=False, num_neighbors=8, radius=0.005):
    """
    追加点の近傍点を"埋め込み後の点群"から探し、そのばらつきを分散・MAD・CVで評価する関数。

    Parameters:
    pcd_after (pcd): 埋め込み後の点群
    embed_points [[(array([a, b, c]),...,(array([a, b, c])],...,[(array([a, b, c]),...,(array([a, b, c])]]: 埋め込み位置の座標情報の3次元リスト
    octree_size (float): 点群の最大座標範囲（PSNR計算用）
    use_radius (bool): Trueなら半径で近傍点を探す。Falseなら固定点数で探す。
    radius (float): 半径指定（use_radius=True の場合に使用）
    """
    # 点群の座標取得
    points_after = np.asarray(pcd_after.points)

    #3次元リストから2次元リストに変換
    flattened_embed_points = [point for sublist in embed_points for point in sublist]
    added_list = np.any([np.all(np.isclose(points_after, target_point), axis=1) for target_point in flattened_embed_points], axis=0)

    added_points = points_after[added_list]
    # print(f"追加点リスト型確認用: {added_list}")

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
    non_added_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=80))

    if non_added_cloud.has_normals():
        normals_after = np.asarray(non_added_cloud.normals)
        # print(f"法線の数: {len(normals_after)}")
    else:
        print("法線が計算されませんでした。点群の密度やパラメータを確認してください。")
        normals_after = np.zeros_like(np.asarray(non_added_cloud.points))  # 法線がない場合の代替

    distance_variances = []
    distance_mads = []
    distance_cvs = []

    for added_point in added_points:
        if use_radius:
            # 半径で近傍点を探す
            _, indices, _ = kdtree.search_radius_vector_3d(added_point, radius)
            # 近傍点が4点未満なら十分に評価できないとして終了
            if len(indices) < 4:
                print(f"追加点 {added_point} の近傍点が4点未満でした。半径を大きくしてください。")
                return None
        else:
            # 固定点数で近傍点を探す
            _, indices, _ = kdtree.search_knn_vector_3d(added_point, num_neighbors)

        neighbors = non_added_points[indices]

        # Point-to-Point
        nearest_point = neighbors[0]  # 最も近い点
        p2p_distance = np.linalg.norm(added_point - nearest_point)
        point_to_point_distances.append(p2p_distance)

        # Point-to-Plane
        nearest_normal = normals_after[indices[0]]
        vector = added_point - nearest_point
        p2plane_distance = abs(np.dot(vector, nearest_normal))
        point_to_plane_distances.append(p2plane_distance)

        # 距離の分散とMAD
        distances_to_neighbors = [np.linalg.norm(added_point - neighbor) for neighbor in neighbors]
        variance = np.var(distances_to_neighbors)
        mad = np.mean(np.abs(distances_to_neighbors - np.mean(distances_to_neighbors)))

        # 距離の変動係数 (CV)
        mean_distance = np.mean(distances_to_neighbors)
        if mean_distance > 0:
            cv = (np.std(distances_to_neighbors) / mean_distance) * 100  # パーセント表示
        else:
            cv = 0  # 平均が0の場合、CVは計算できないので0とする

        distance_variances.append(variance)
        distance_mads.append(mad)
        distance_cvs.append(cv)

    # 平均二乗誤差 (MSE)
    point_to_point_mse = np.mean(np.array(point_to_point_distances) ** 2)
    point_to_plane_mse = np.mean(np.array(point_to_plane_distances) ** 2)

    # PSNRの計算
    point_to_point_psnr = 10 * np.log10(octree_size**2 / point_to_point_mse) if point_to_point_mse > 0 else float('inf')
    point_to_plane_psnr = 10 * np.log10(octree_size**2 / point_to_plane_mse) if point_to_plane_mse > 0 else float('inf')

    # 結果の表示: 科学技術表記 (指数表記)
    # print("Point-to-Point")
    # print(f"平均値: {np.mean(point_to_point_distances):.4e}, 最大値: {np.max(point_to_point_distances):.4e}")
    # print(f"MSE: {point_to_point_mse:.4e}, PSNR: {point_to_point_psnr:.2f} dB")
    # print("Point-to-Plane")
    # print(f"平均値: {np.mean(point_to_plane_distances):.4e}, 最大値: {np.max(point_to_plane_distances):.4e}")
    # print(f"MSE: {point_to_plane_mse:.4e}, PSNR: {point_to_plane_psnr:.2f} dB")
    if use_radius:
        print(f"近傍点との距離(半径: {radius})")
    else:
        print(f"近傍点との距離(点数: {num_neighbors})")
    print(f"分散 - 平均値: {np.mean(distance_variances):.4e}, 最大値: {np.max(distance_variances):.4e}")
    print(f"MAD - 平均値: {np.mean(distance_mads):.4e}, 最大値: {np.max(distance_mads):.4e}")
    print(f"CV  - 平均値: {np.mean(distance_cvs):.2f}%, 最大値: {np.max(distance_cvs):.2f}%")

    return None

def evaluate_PSNR(pcd_before, pcd_after, octree_size):
    """
    埋め込み後の点群に対して、埋め込み前の点群から最近傍点を探してp2p, p2lを求め、そのPSNRで評価する関数。
    pcd_beforeにpcd_non_addedを代入すると、Octree量子化後の点群から最近傍点を探せる。

    Parameters:
    pcd_before (pcd): 埋め込み前の点群
    pcd_after (pcd): 埋め込み後の点群
    octree_size (double): 点群の最大座標範囲（PSNR計算用）
    """
    import numpy as np
    import open3d as o3d
    from scipy.spatial import KDTree

    # 点群データをnumpy配列に変換
    points_before = np.asarray(pcd_before.points)
    points_after = np.asarray(pcd_after.points)

    # KDTreeを構築して、pcd_after内の各点に対するpcd_beforeの最近傍点を検索
    kdtree = KDTree(points_before)
    distances, indices = kdtree.query(points_after)  # 最近傍点のインデックスと距離を取得

    # 法線の推定（point-to-plane誤差の計算に必要）
    pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=80))
    normals_before = np.asarray(pcd_before.normals)

    # point-to-point誤差（最近傍点とのユークリッド距離）
    point_to_point_errors = distances

    # Point-to-Plane誤差の計算
    point_to_plane_errors = []
    for idx_after, idx_before in enumerate(indices):
        normal = normals_before[idx_before]
        diff_vector = points_after[idx_after] - points_before[idx_before]
        point_to_plane_errors.append(np.abs(np.dot(diff_vector, normal)))

    point_to_plane_errors = np.array(point_to_plane_errors)

    # 平均値と最大値を計算
    point_to_point_mean = np.mean(point_to_point_errors)
    point_to_point_max = np.max(point_to_point_errors)
    point_to_plane_mean = np.mean(point_to_plane_errors)
    point_to_plane_max = np.max(point_to_plane_errors)

    # 平均二乗誤差 (MSE)
    point_to_point_mse = np.mean(point_to_point_errors ** 2)
    point_to_plane_mse = np.mean(point_to_plane_errors ** 2)

    # PSNRの計算
    point_to_point_psnr = 10 * np.log10(octree_size**2 / point_to_point_mse) if point_to_point_mse > 0 else float('inf')
    point_to_plane_psnr = 10 * np.log10(octree_size**2 / point_to_plane_mse) if point_to_plane_mse > 0 else float('inf')

    # 点群の可視化
    # o3d.visualization.draw_geometries([pcd_before.paint_uniform_color([1, 0, 0]),  # 赤
    #                                   pcd_after.paint_uniform_color([0, 1, 0])])  # 緑

    # 結果をフォーマットして表示
    print("Point-to-Point")
    print(f"平均値: {point_to_point_mean:.4e}, 最大値: {point_to_point_max:.4e}")
    print(f"MSE: {point_to_point_mse:.4e}, PSNR: {point_to_point_psnr:.2f} dB")
    print("Point-to-Plane")
    print(f"平均値: {point_to_plane_mean:.4e}, 最大値: {point_to_plane_max:.4e}")
    print(f"MSE: {point_to_plane_mse:.4e}, PSNR: {point_to_plane_psnr:.2f} dB")

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

import numpy as np
import open3d as o3d

def calculate_centroid(pcd):
    """
    点群の重心点を計算して表示する関数

    Parameters:
    pcd (open3d.geometry.PointCloud): 点群データ

    Returns:
    np.ndarray: 重心点の座標 (x, y, z)
    """
    # 点群の座標を取得
    points = np.asarray(pcd.points)
    
    # 重心点を計算 (x, y, z の平均値)
    centroid = np.mean(points, axis=0)
    
    # 結果の表示
    print(f"重心点の座標: {centroid}")
    
    return centroid


### --------------------------埋め込み位置決定、透かし情報抽出に関する関数---------------------------
def select_embeddable_random(centroid,binary_string_check,input_path_location,input_path_color=None,max_size=1):
    """
    ベースラインとなる、ランダム埋め込み用の埋め込み位置を決定する総合関数。

    Parameters:
    binary_string_check (str): 埋め込む情報のバイナリビット列
    input_path_location (str): 埋め込み前点群の座標情報が格納されたテキストファイルのパス
    input_path_color (str): 埋め込み前点群の色情報が格納されたテキストファイルのパス
    max_size (double): Octreeのルートノードのボクセルサイズ
    centroid (list): 重心点座標

    Returns:
    pcd (pcd): 埋め込み前点群
    embed_points (array([a, b, c])): 埋め込み位置の座標情報のリスト
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
    min_size,voxel_index = create_index(bit_stream,level_ptrs,1,max_depth,level_bits_list,
                                        reconstruct_count,points,voxel_info,max_size,voxel_index,centroid)
    min_size_check = max_size
    for i in range(max_depth):
        min_size_check = min_size_check/2
    if min_size != min_size_check:
        print("【error】min_sizeがなんかうまくいってません")
        return
    else:
        print("処理開始")
    
    embed_points = find_choose_embedding_random(voxel_info,binary_string_check)

    return embed_points

def select_embeddable_between(centroid,input_path_location,input_path_color=None,max_size=1):
    """
    提案手法用の、点の間に埋め込み位置の"候補"を決定する関数。

    Parameters:
    input_path_location (str): 埋め込み前点群の座標情報が格納されたテキストファイルのパス
    input_path_color (str): 埋め込み前点群の色情報が格納されたテキストファイルのパス
    max_size (double): Octreeのルートノードのボクセルサイズ
    centroid (list): 重心点座標

    Returns:
    pcd (pcd): 埋め込み前点群
    embedding_candidate ([{"target", "hit", "coord"}]): 埋め込み候補位置の情報の辞書型リスト
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
    min_size,voxel_index = create_index(bit_stream,level_ptrs,1,max_depth,level_bits_list,
                                        reconstruct_count,points,voxel_info,max_size,voxel_index,centroid)
    min_size_check = max_size
    for i in range(max_depth):
        min_size_check = min_size_check/2
    if min_size != min_size_check:
        print("【error】min_sizeがなんかうまくいってません")
        return
    else:
        print("処理開始")
    
    embedding_candidates = find_embedding_between(voxel_info,min_size)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    if color_list:
        pcd.colors = o3d.utility.Vector3dVector(np.array(color_list))

    return pcd,embedding_candidates

def create_index(bit_stream,level_ptrs,current_depth,max_depth,level_bits_list,
                 reconstruct_count,points,voxel_info,size,voxel_index,centroid,origin=np.array([0,0,0]),min_size=None):
    """
    点群から、ボクセルのインデックスや"0","1",座標情報を格納できるOctreeを作成する関数。

    Parameters:
    bit_stream (str): 点群の座標情報のテキストファイルのバイナリビット列
    level_ptrs (list): 層における点の数計算用(層内の点全探査終了で関数を終えるため)
    current_depth (int): Octreeの深さの現在地
    max_depth (int): Octreeの深さ
    level_bits_list (tuple): (各層のノード数, ビット列でのその層のノードの開始地点)
    reconstruct_count (int): 再帰した回数
    points ():
    voxel_info ([{'index','depth','child_index','exist','coordinate','area'}]): ボクセルの情報を保持するためのリスト
    size (int): ボクセルのオフセット計算用
    voxel_index (int): ボクセルのナンバリング
    centroid (list): 重心点座標
    origin (3次元np.array): 点を追加するときの原点（これにオフセットプラスして追加位置を特定)

    Returns:
    min_size: 最下層ボクセルの大きさ（ユークリッド距離）
    voxel_index (int): ボクセルのナンバリング
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
            # オフセット計算
            voxel_offset = np.array([i & 1, (i >> 1) & 1, (i >> 2) & 1],dtype=np.float32) * (size/2.0)

            #現在最深階層なら点を追加
            if current_depth == max_depth:
                if min_size is None:
                    min_size = size/2
                point = origin + voxel_offset
                points.append(point)
                area_number = 0
                # 各軸に対して、点が重心より大きいか小さいかを確認
                if point[0] > centroid[0]: area_number |= 1  # x軸方向
                if point[1] > centroid[1]: area_number |= 2  # y軸方向
                if point[2] > centroid[2]: area_number |= 4  # z軸方向
                voxel_info.append({
                    'index': voxel_index,
                    'depth': current_depth,
                    'child_index': i,
                    'exist': 1,
                    'coordinate': point,
                    'area': area_number
                })
                voxel_index += 1
            # 現在最深階層以外なら再帰処理で階層を進む
            else:
                reconstruct_count[current_depth - 1] += 1
                next_origin = origin + voxel_offset
                min_size,voxel_index = create_index(bit_stream, level_ptrs,current_depth + 1,max_depth,level_bits_list,
                                                    reconstruct_count,points,voxel_info,size/2.0,voxel_index,centroid,next_origin,min_size)
        else:
            if current_depth == max_depth:
                voxel_offset = np.array([i & 1, (i >> 1) & 1, (i >> 2) & 1],dtype=np.float32) * (size/2.0)
                point = origin + voxel_offset
                area_number = 0
                # 各軸に対して、点が重心より大きいか小さいかを確認
                if point[0] > centroid[0]: area_number |= 1  # x軸方向
                if point[1] > centroid[1]: area_number |= 2  # y軸方向
                if point[2] > centroid[2]: area_number |= 4  # z軸方向
                voxel_info.append({
                    'index': voxel_index,
                    'depth': current_depth,
                    'child_index': i,
                    'exist': 0,
                    'coordinate': point,
                    'area': area_number
                })
                voxel_index += 1
                
    return min_size,voxel_index

def find_choose_embedding_random(voxel_info, binary_string_check):
    """
    ベースラインでの、埋め込み位置の検出と決定を行う内部動作関数。

    Parameters:
    voxel_info ([{'index','depth','child_index','exist','coordinate','area'}]): ボクセルの情報を保持するためのリスト
    binary_string_check (str): 埋め込む情報のバイナリビット列

    Returns:
    embed_points (list): 埋め込み位置の座標情報のリスト
    """
    # シード値を固定
    random.seed(42)

    # exist == 0 の要素をフィルタリングして埋め込み候補位置を全探査
    filtered_data = [item for item in voxel_info if item['exist'] == 0]

    # 各エリア（1から8）ごとに分ける
    area_dict = {i: [] for i in range(8)}  # area 0～7 に対応する空のリストを用意
    
    # "area" ごとに filtered_data を分割
    for item in filtered_data:
        area_dict[item['area']].append(item)

    # 埋め込む情報の数
    num = len(binary_string_check)  # 埋め込むビット列の長さを抽出する個数とする

    # 埋め込む座標情報のリスト
    embed_points = []
    empty_list = []

    # 各エリアごとにランダムに num 個の座標を選択して追加
    for area in range(8):
        selected_items = random.sample(area_dict[area], num) if len(area_dict[area]) >= num else empty_list
        
        # 座標情報をリストに追加
        area_points = [np.array(item['coordinate']) for item in selected_items]
        embed_points.append(area_points)

    return embed_points

def find_embedding_between(voxel_info, voxelsize):
    """
    提案手法での、埋め込み位置の検出を行うための内部動作関数。

    Parameters:
    voxel_info ([{'index','depth','child_index','exist','coordinate','area'}]): ボクセルの情報を保持するためのリスト
    voxelsize (double): 最下層ボクセルのサイズ

    Returns:
    embedding_candidate ([{"target", "hit", "coord"}]): 埋め込み候補位置の情報の辞書型リスト
    """
    # サーチ対象のボクセル間隔を0~between_thresholdに設定
    between_threshold = 4
    embedding_candidates = []
    # seen_candidates = []
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
                    
                    # いずれかの軸がvoxelsize * between_threshold以上離れている場合は除外
                    if distance_x >= voxelsize * between_threshold or distance_y >= voxelsize * between_threshold or distance_z >= voxelsize * between_threshold:
                        continue
                    distance = np.linalg.norm(np.array(target_coord) - np.array(hit_coord))
                    # 下限の設定をなくしてボクセルbetween_threshold個分に拡大(多分hit != targetあるからうまくいくと思う)
                    # そもそも5行上の処理でdistanceの計算いらないかも？
                    if distance < voxelsize * between_threshold:
                        min_x, max_x = min(target_coord[0], hit_coord[0]), max(target_coord[0], hit_coord[0])
                        min_y, max_y = min(target_coord[1], hit_coord[1]), max(target_coord[1], hit_coord[1])
                        min_z, max_z = min(target_coord[2], hit_coord[2]), max(target_coord[2], hit_coord[2])
                        for candidate in voxel_info:
                            if candidate['exist'] == 0:
                                candidate_coord = candidate['coordinate']
                                if (min_x <= candidate_coord[0] <= max_x and
                                    min_y <= candidate_coord[1] <= max_y and
                                    min_z <= candidate_coord[2] <= max_z):
                                    embedding_candidates.append({
                                        'target': target_coord,
                                        'hit': hit_coord,
                                        'candidate': candidate_coord
                                    })
                                    # seen_candidates.append(candidate_coord)
        # 進捗を表示
        if (i + 1) % 100 == 0 or (i + 1) == total_voxels:
            elapsed_time = time.time() - start_time
            print(f"Progress: {i + 1}/{total_voxels} ({(i + 1) / total_voxels * 100:.2f}%) - Elapsed time: {elapsed_time:.2f} seconds")
    print("埋め込み候補点の探索完了")
    print("重複を消去")
    unique_candidates = set()
    filtered_candidates = []
    for i,entry in enumerate(embedding_candidates):
        candidate_key = tuple(entry['candidate'])  # 'candidate'をタプル化してキーに
        if candidate_key not in unique_candidates:
            unique_candidates.add(candidate_key)  # 新しいキーを登録
            filtered_candidates.append(entry)     # 元の辞書形式をリストに追加
    embedding_candidates = filtered_candidates
    print("重複の消去完了")
    end_time = time.time()
    elapsed_time=end_time - start_time
    print(f"埋め込み可能点の探索処理時間：{elapsed_time}秒")
    return embedding_candidates

def save_embedding_candidates_to_csv(embedding_candidates, file_path):
    """
    提案手法での、埋め込み候補位置をcsvに出力するための関数。説明なし
    """
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Target_X', 'Target_Y', 'Target_Z', 'Hit_X', 'Hit_Y', 'Hit_Z', 'Candidate_X', 'Candidate_Y', 'Candidate_Z'])  # ヘッダー行
        for candidate in embedding_candidates:
            target_x, target_y, target_z = candidate['target']
            hit_x, hit_y, hit_z = candidate['hit']
            candidate_x, candidate_y, candidate_z = candidate['candidate']
            writer.writerow([target_x, target_y, target_z, hit_x, hit_y, hit_z, candidate_x, candidate_y, candidate_z])

def choose_between_positions(candidates_df, binary_string_check):
    """
    select_embeddable_between関数で決めた埋め込み位置の候補より、ランダムに埋め込む位置を決定する関数。

    Parameters:
    candidates_df (pandas dataframe): find_zero_bits_in_deepest_layer関数で見つけたOctree最下層"0"の位置
    binary_string_check (str): 検査符号付きのバイナリビット列

    Returns:
    embed_points(list[array(3)]):ソート済みの埋め込み位置の点の座標['Candidate_X', 'Candidate_Y', 'Candidate_Z']
    embed_positions_dict (dict): ソート済みの埋め込み位置(['Target_X', 'Target_Y', 'Target_Z', 'Hit_X', 'Hit_Y', 'Hit_Z', 'Candidate_X', 'Candidate_Y', 'Candidate_Z'])
                                 多分後で色情報つけるのとかに使うんじゃないかと思ってる
    """
    # シード値を固定
    random.seed(42)

    num_sample = len(binary_string_check)

    print(f"埋め込み候補位置の数: {len(candidates_df)}")

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
    embed_area = []
    for i in range(8):
        if embed_points[i] == []: continue
        embed_area.append(i)
        for j in range(len(binary_string_check)):
            if binary_string_check[j] == '1':
                after_points.append(embed_points[i][j])  # 埋め込む点の座標をafter_pointsに入れる
                after_colors.append([1,0,0]) # 赤色で追加
    
    print(f"embed_area: {embed_area}")
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

    extracted_bits_lists = []
    extracted_bits_list = []

    for target_points in embed_points:
        bit_array = []
        for target_point in target_points:
            if np.any(np.all(np.isclose(point_array, target_point), axis=1)):
                bit_array.append(1)
            else:
                bit_array.append(0)
        extracted_bits = ''.join(map(str, bit_array)) # 一応文字列に変えとく
        extracted_bits_list = extracted_bits
        extracted_bits_lists.append(extracted_bits_list)

    return extracted_bits_lists


### エラー集
"""
ValueError: invalid literal for int() with base 2: ''
>> 埋め込む文字列が空白
"""