from modules.sharemodule import o3d, np
import copy
import random
import sys

def string_to_binary(input_string):
    """文字列をバイナリビット列に変換する関数。"""
    return ''.join(format(ord(char), '08b') for char in input_string)

def binary_to_string(extracted_binary_string):
    extracted_string = ''.join([chr(int(extracted_binary_string[i:i+8], 2)) for i in range(0, len(extracted_binary_string), 8)])
    return extracted_string

# Octreeを可視化するための関数
def display_octree(point_cloud, max_depth=8):
    octree = o3d.geometry.Octree(max_depth)
    octree.convert_from_point_cloud(point_cloud, size_expand=0.01)
    o3d.visualization.draw_geometries([octree])

def embed_bits_in_octree_with_regions(input_path, binary_string, level_bits_list):
    """
    オクツリーの最下層に、指定したバイナリビットを埋め込む関数。
    埋め込むビット数に応じて区域を分割し、各区域で埋め込むビットが「0」なら1のビットを0に、
    埋め込むビットが「1」なら0のビットを1に変える。
    区域内に1や0がない場合はエラーを表示しプログラムを終了する。

    Parameters:
    input_path (str): テキストファイルのパス。
    binary_string (str): 埋め込むバイナリビット列。
    level_bits_list (list): 各階層のビット情報のリスト（countlayer関数で生成）。
    
    Returns:
    str: 埋め込み後のビット列。
    """
    with open(input_path, 'r') as file:
        bit_stream = list(file.read())  # ビット列を文字リストとして取得

    # 最下層のビット列を取得
    last_level_bits, start_ptr = level_bits_list[-1]
    total_bits = last_level_bits  # 最下層のビット数
    embed_bits = len(binary_string)  # 埋め込むビット数

    # 区域のサイズを計算（余りを考慮）
    region_size = total_bits // embed_bits
    remainder = total_bits % embed_bits

    # 各区域でバイナリビットを埋め込む
    for i, bit in enumerate(binary_string):
        start = start_ptr + i * region_size + min(i, remainder)
        end = start + region_size + (1 if i < remainder else 0)
        region = bit_stream[start:end]

        # エラーチェック：区域内に1や0が存在しない場合
        if bit == '0' and '1' not in region:
            print(f"Error: 区域 {i} 内に1がありません。埋め込みできません。")
            sys.exit(1)
        elif bit == '1' and '0' not in region:
            print(f"Error: 区域 {i} 内に0がありません。埋め込みできません。")
            sys.exit(1)

        if bit == '0':
            # 区域内の1のビットをランダムに0に変える
            one_indices = [j for j, b in enumerate(region) if b == '1']
            if one_indices:
                chosen_index = random.choice(one_indices)
                region[chosen_index] = '0'
        elif bit == '1':
            # 区域内の0のビットをランダムに1に変える
            zero_indices = [j for j, b in enumerate(region) if b == '0']
            if zero_indices:
                chosen_index = random.choice(zero_indices)
                region[chosen_index] = '1'

        # 埋め込んだ後、元のビットストリームに戻す
        bit_stream[start:end] = region

    return ''.join(bit_stream)



def extract_bits_from_octree_with_comparison(original_file, modified_file, embed_bits):
    """
    元のオクツリーと、情報を埋め込んだオクツリーのテキストファイルを比較して、
    埋め込んだビット列を抽出する関数。
    
    Parameters:
    original_file (str): 元のオクツリーデータのテキストファイルパス。
    modified_file (str): 情報を埋め込んだオクツリーデータのテキストファイルパス。
    embed_bits (int): 埋め込まれたビット数。

    Returns:
    str: 抽出したバイナリビット列。
    """
    with open(original_file, 'r') as orig_file:
        original_bit_stream = orig_file.read()

    with open(modified_file, 'r') as mod_file:
        modified_bit_stream = mod_file.read()

    # 比較のための初期化
    total_bits = len(original_bit_stream)
    extracted_bits = []
    
    # 比較して、異なるビットを抽出
    for i in range(total_bits):
        if original_bit_stream[i] != modified_bit_stream[i]:
            if modified_bit_stream[i] == '1':
                extracted_bits.append('1')
            else:
                extracted_bits.append('0')
        
        # 抽出するビット数に達したら終了
        if len(extracted_bits) >= embed_bits:
            break

    return ''.join(extracted_bits)


def encode_octree(node, depth=0, output_path=None, bit_dict=None):
    """Octreeのノードをビット列に符号化する関数"""
    if output_path is None:
        print("Please specify the file path")
        return
    
    if bit_dict is None:
        bit_dict = {}
    
    if isinstance(node, o3d.geometry.OctreeInternalNode):
        children_bits = "".join([str(int(child is not None)) for child in node.children])
        if depth not in bit_dict:
            bit_dict[depth] = []
        bit_dict[depth].append(children_bits)

        for child in node.children:
            if child is not None:
                encode_octree(child, depth + 1, output_path, bit_dict)

    if depth == 0:
        with open(output_path, 'w') as file:
            for depth in sorted(bit_dict.keys()):
                for bits in bit_dict[depth]:
                    file.write(bits)

    return None

def decode_octree(input_path, max_size):
    """オクツリーのファイルをデコードし点群を再構成"""
    with open(input_path, 'r') as file:
        bit_stream = file.read()

    level_ptrs = [0]
    level_bits_list, max_depth = countlayer(bit_stream)
    print("level_bits_list:", level_bits_list)
    max_depth_check = len(level_bits_list)
    if max_depth != max_depth_check:
        print("max_depth calculate error:max_depth=", max_depth, "max_depth_check=", max_depth_check)
        return
    print("Calculated max_depth:", max_depth, " check:", max_depth_check)

    reconstruct_count = [0] * max_depth
    points = []

    reconstruct_octree(bit_stream, level_ptrs, 1, max_depth, level_bits_list, reconstruct_count, points, max_size)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    return pcd

def countlayer(bit_stream):
    """Octreeの各階層のビット数を数える関数"""
    level_bits_list = []
    depth = 1
    bit_ptr = 0
    nodes_in_current_level = 8

    while bit_ptr < len(bit_stream):
        level_bits_list.append((nodes_in_current_level, bit_ptr))
        children_count = sum(1 for i in range(nodes_in_current_level) if bit_ptr + i < len(bit_stream) and bit_stream[bit_ptr + i] == '1')
        if children_count == 0:
            break
        depth += 1
        bit_ptr += nodes_in_current_level
        nodes_in_current_level = children_count * 8

    return level_bits_list, depth - 1


def reconstruct_octree(bit_stream, level_ptrs, current_depth, max_depth, level_bits_list, reconstruct_count, points, size, origin=np.array([0, 0, 0]), num=0):
    """Octreeのビット列から点群を再構成する関数"""
    if current_depth > max_depth:
        return

    nodes_in_current_level, start_ptr = level_bits_list[current_depth - 1]
    
    if len(level_ptrs) < current_depth:
        count = reconstruct_count[current_depth - 2] - 1 if current_depth > 1 else 0
        level_ptrs.append(start_ptr + (count * 8))

    for i in range(8):
        if level_ptrs[current_depth - 1] >= start_ptr + nodes_in_current_level:
            return

        bit = bit_stream[level_ptrs[current_depth - 1]]
        level_ptrs[current_depth - 1] += 1

        if bit == '1':
            voxel_offset = compute_voxel_offset(i, size)
            if current_depth == max_depth:
                point = origin + voxel_offset
                points.append(point)
            else:
                reconstruct_count[current_depth - 1] += 1
                next_origin = origin + voxel_offset
                reconstruct_octree(bit_stream, level_ptrs, current_depth + 1, max_depth, level_bits_list, reconstruct_count, points, size / 2.0, next_origin, num)

    return None

def compute_voxel_offset(i, size):
    """ビットに対応するボクセルのオフセットを計算"""
    return np.array([i & 1, (i >> 1) & 1, (i >> 2) & 1], dtype=np.float32) * (size / 2.0)

def calculate_snr(before_points, after_points):
    """
    点群の位置情報の編集前と後のS/N比を計算します。
    
    Parameters:
    before_points (np.array): 編集前の点群の位置情報（Nx3の配列）。
    after_points (np.array): 編集後の点群の位置情報（Nx3の配列）。
    
    Returns:
    float: S/N比の値。
    """
    o3d.visualization.draw_geometries([before_points, after_points])
    before = np.array(before_points.points)
    after = np.array(after_points.points)
    min_x_before = np.min(before[:, 0])
    min_y_before = np.min(before[:, 1])
    min_z_before = np.min(before[:, 2])
    min_x_after = np.min(after[:, 0])
    min_y_after = np.min(after[:, 1])
    min_z_after = np.min(after[:, 2])
    dif_x = min_x_before - min_x_after
    dif_y = min_y_before - min_y_after
    dif_z = min_z_before - min_z_after
    print(dif_x, dif_y, dif_z)
    transformation = np.array([[1, 0, 0, dif_x],
                               [0, 1, 0, dif_y],
                               [0, 0, 1, dif_z],
                               [0, 0, 0, 1]])
    after_points.transform(transformation)

    o3d.visualization.draw_geometries([before_points, after_points])

    threshold = 0.02  
    trans_init = np.identity(4)  

    reg_p2p = o3d.pipelines.registration.registration_icp(
        before_points, after_points, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    transformation = reg_p2p.transformation
    print("Transformation is:")
    print(transformation)

    after_points.transform(transformation)
    o3d.visualization.draw_geometries([before_points, after_points])

    before_points_np = np.array(before_points.points)
    after_points_np = np.array(after_points.points)

    signal_power = np.sum(before_points_np**2) / len(before_points_np)
    noise = before_points_np - after_points_np
    noise_power = np.sum(noise**2) / len(noise)

    snr = 10 * np.log10(signal_power / noise_power)

    return snr

def Clustering(pcdData,epsm,points):
    """
    点群のノイズ除去耐性チェック用。

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

def sizing_octree(pcd, size_expand):
    """
    点群の範囲を計算(そのまま表示すると範囲がおかしくなる？)
    """
    points = np.asarray(pcd.points)
    max_values = np.max(points, axis=0)
    min_values = np.min(points, axis=0)
    range_x = (max_values[0] - min_values[0])
    range_y = (max_values[1] - min_values[1])
    range_z = (max_values[2] - min_values[2])
    cube_size = max(range_x, range_y, range_z)
    octree_size = (cube_size * (1 + size_expand))
    return octree_size

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