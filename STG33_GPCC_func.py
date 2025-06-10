import random
import numpy as np
import open3d as o3d

def string_to_binary(string):
    # 文字列をバイナリに変換
    return ''.join(format(ord(char), '08b') for char in string)

def embed_watermark(bit_stream, watermark_bits, level_bits_list, max_depth):
    """
    bit_stream の最下層 (max_depth) から 0 のビットをランダムに選んで watermark_bits を埋め込む。
    変更した場所を保持し、変更後のビットストリームを返す。
    """
    # 最下層（max_depth）のビット列を取得
    nodes_in_last_level, start_ptr = level_bits_list[max_depth - 1]
    last_level_bits = bit_stream[start_ptr:start_ptr + nodes_in_last_level]

    zero_indices = [i for i, bit in enumerate(last_level_bits) if bit == '0']
    
    if len(zero_indices) < len(watermark_bits):
        raise ValueError("埋め込み先の葉ノードが足りません。")
    
    # 最下層の0のうち、ランダムに選択
    selected_indices = random.sample(zero_indices, len(watermark_bits))

    # ビット列をリストに変換し、情報を埋め込む
    bit_stream_list = list(bit_stream)
    for i, bit in zip(selected_indices, watermark_bits):
        bit_stream_list[start_ptr + i] = bit  # 0 を watermark のビットで置き換える
    
    return ''.join(bit_stream_list), selected_indices

def extract_watermark(bit_stream, embedded_indices):
    """
    埋め込んだ場所からビット列を抽出して、バイナリを復号する。
    """
    watermark_bits = ''.join(bit_stream[i] for i in embedded_indices)
    return watermark_bits

def binary_to_string(binary_str):
    # 8ビットごとに文字に変換
    chars = [chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8)]
    return ''.join(chars)

def encode_octree(node, depth=0, output_path=None, bit_dict=None):
    """
    Octreeのノードをビット列に符号化する関数
    """
    if output_path is None:
        print("Please specify the path to the file")
        return
    
    if bit_dict is None:
        bit_dict = {}
    
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
                encode_octree(child, depth + 1, output_path, bit_dict)

    if depth == 0:
        with open(output_path, 'w') as file:
            for depth in sorted(bit_dict.keys()):
                for bits in bit_dict[depth]:
                    file.write(bits)

def decode_octree(input_path, max_size):
    """
    エンコードファイルからオクツリーを復元
    """
    with open(input_path, 'r') as file:
        bit_stream = file.read()

    level_ptrs = [0]
    level_bits_list, max_depth = countlayer(bit_stream)

    points = []
    reconstruct_count = [0] * max_depth
    reconstruct_octree(bit_stream, level_ptrs, 1, max_depth, level_bits_list, reconstruct_count, points, max_size)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    return pcd, bit_stream, level_bits_list, max_depth

def countlayer(bit_stream):
    """
    各階層のビット数をカウントし、リストとして返す
    """
    level_bits_list = []
    depth = 1
    bit_ptr = 0
    nodes_in_current_level = 8

    while bit_ptr < len(bit_stream):
        level_bits_list.append((nodes_in_current_level, bit_ptr))

        children_count = sum(1 for i in range(nodes_in_current_level) if bit_stream[bit_ptr + i] == '1')
        if children_count == 0:
            break

        depth += 1
        bit_ptr += nodes_in_current_level
        nodes_in_current_level = children_count * 8
    
    return level_bits_list, depth - 1

def reconstruct_octree(bit_stream, level_ptrs, current_depth, max_depth, level_bits_list, reconstruct_count, points, size, origin=np.array([0,0,0]), num = 0):
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
                reconstruct_octree(bit_stream, level_ptrs, current_depth + 1, max_depth, level_bits_list, reconstruct_count, points, size/2.0, next_origin, num)

def compute_voxel_offset(i, size):
    return np.array([i & 1, (i >> 1) & 1, (i >> 2) & 1], dtype=np.float32) * (size/2.0)

def calculate_snr(before_points, after_points):
    """
    点群の位置情報の編集前と後のS/N比を計算します。
    """
    before = np.array(before_points.points)
    after = np.array(after_points.points)

    # 信号（before_points自体が信号）
    signal_power = np.sum(before**2) / len(before)

    # 雑音（before_pointsとafter_pointsの差分）
    noise = before - after
    noise_power = np.sum(noise**2) / len(noise)

    # S/N比を計算
    snr = 10 * np.log10(signal_power / noise_power)

    return snr
