import numpy as np
import scipy.fftpack
import open3d as o3d

# 文字列をASCIIコードの2進数に変換する関数
def string_to_binary(message):
    return ''.join(format(ord(char), '07b') for char in message)

# ASCIIコードの2進数を文字列に変換する関数
def binary_to_string(binary):
    chars = [binary[i:i+7] for i in range(0, len(binary), 7)]
    return ''.join(chr(int(char, 2)) for char in chars)

# ボクセルリストから全てのボクセルを含む3次元配列を作成
def make_all_voxels(voxels):
    max_index = np.max([voxel.grid_index for voxel in voxels])
    min_index = np.min([voxel.grid_index for voxel in voxels])
    voxel_size = (max_index - min_index) + 1
    voxel_all = np.ones((voxel_size, voxel_size, voxel_size))
    for voxel in voxels:
        index = voxel.grid_index - min_index
        voxel_all[index[0], index[1], index[2]] = voxel.color[0]
    return voxel_all

# 3次元DCT変換を行う関数
def dct_3d(voxel_dct):
    return scipy.fftpack.dctn(voxel_dct, norm='ortho')

# 3次元IDCT変換を行う関数
def idct_3d(voxel_dctcoef):
    return scipy.fftpack.idctn(voxel_dctcoef, norm='ortho')

def generate_positions(n, lower_cut, upper_cut, message_length, same=True, seed_time=None):
    lower_area = int(n * lower_cut)
    upper_area = int(n - n * upper_cut)
    if same:
        np.random.seed(0)
    else:
        np.random.seed(seed_time)
        
    x = np.ones((n, n, n))
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if not (lower_area < i < upper_area and lower_area <= j < upper_area and lower_area <= k < upper_area):
                    x[i, j, k] = 0

    # 埋め込み可能な位置を取得
    possible_positions = np.argwhere(x == 1)
    np.random.shuffle(possible_positions)
    
    # バイナリメッセージの長さ分、埋め込み位置を取得
    embed_positions = possible_positions[:message_length]
    print("Embedding Positions:\n", embed_positions)

    # # 埋め込み領域の可視化
    # vis = o3d.geometry.VoxelGrid()
    # vis.voxel_size = 1 / n
    # for i in range(n):
    #     for j in range(n):
    #         for k in range(n):
    #             voxel_index = np.array([i, j, k], dtype=np.int32)
    #             if x[i, j, k] == 1:
    #                 voxel_color = np.array([1, 1, 1], dtype=np.float64)  # 白
    #             else:
    #                 voxel_color = np.array([0, 0, 0], dtype=np.float64)  # 黒
    #             new_voxel = o3d.geometry.Voxel(grid_index=voxel_index, color=voxel_color)
    #             vis.add_voxel(new_voxel)
    # # 可視化
    # vis_window = o3d.visualization.Visualizer()
    # vis_window.create_window(window_name="埋め込み領域の可視化")
    # vis_window.add_geometry(vis)
    # vis_window.run()
    # vis_window.destroy_window()
    
    return embed_positions


# αを生成する関数
def generate_a(w, psnr):
    N = len(w[0][0])
    A = 1
    n = -1 * (psnr / 10)
    WW = w * w
    a = np.sqrt((np.power(10, n) * N * N * N * A * A) / np.sum(WW))
    return a

def embed_string(dctcoef, message, psnr, lower_cut, upper_cut, seed=0):
    """
    文字列をDCT係数に埋め込む関数。非埋め込み領域を設定し、メッセージを埋め込む。
    
    Args:
        dctcoef (ndarray): DCT係数の配列。
        message (str): 埋め込む文字列。
        psnr (float): PSNR値。
        lower_cut (float): 低周波成分のカット割合。
        upper_cut (float): 高周波成分のカット割合。
        seed (int, optional): ランダムシード。デフォルトは0。
    
    Returns:
        tuple: 埋め込み後のDCT係数、埋め込み位置のリスト。
    """
    np.random.seed(seed)
    binary_message = string_to_binary(message)
    length = len(binary_message)
    
    positions = generate_positions(len(dctcoef[0][0]), lower_cut, upper_cut, length, seed)

    a = generate_a(dctcoef, psnr)

    # メッセージをDCT係数に埋め込む（符号を操作）
    for index, bit in enumerate(binary_message):
        i, j, k = positions[index]
        if bit == '1':
            if dctcoef[i, j, k] < 0:
                dctcoef[i, j, k] = -dctcoef[i, j, k]
        else:
            if dctcoef[i, j, k] > 0:
                dctcoef[i, j, k] = -dctcoef[i, j, k]

    return dctcoef, positions


# 埋め込まれた文字列を検出する関数
def detect_string(dctcoef, positions):
    binary_message = ""
    for pos in positions:
        i, j, k = pos
        if dctcoef[i, j, k] > 0:
            binary_message += '1'
        else:
            binary_message += '0'
    return binary_message

# ボクセルデータを視覚化する関数
def visualize(voxel_dct2, voxel_num, voxel_size):
    import copy
    import open3d as o3d
    import time

    print("視覚化開始")
    start_genvoxel = time.time()
    vis = copy.deepcopy(voxel_dct2)
    iran = -1
    for i in range(vis.size - voxel_num):
        idx = np.unravel_index(np.argmax(vis), vis.shape)
        vis[idx] = iran
    n = len(vis[0][0])
    after_voxels = o3d.geometry.VoxelGrid()
    after_voxels.voxel_size = voxel_size
    for i in range(n):
        for j in range(n):
            for k in range(n):
                voxel_index = np.array([i, j, k], dtype=np.int32)
                voxel_color = np.array([vis[i, j, k], vis[i, j, k], vis[i, j, k]], dtype=np.float64)
                if np.any(voxel_color == iran):
                    continue
                new_voxel = o3d.geometry.Voxel(grid_index=voxel_index, color=voxel_color)
                after_voxels.add_voxel(new_voxel)
    after_num = after_voxels.get_voxels()
    if voxel_num == len(after_num):
        print("ボクセル数：成功")
    else:
        print("ボクセル数：失敗")
    print("視覚化完了")
    print("処理時間：", time.time() - start_genvoxel, "\n")
    return after_voxels

# 圧縮を行う関数
def comp(dctcoef_emb, rate):
    import copy

    com = copy.deepcopy(dctcoef_emb)
    N = com.shape[0]
    indices = np.indices((N, N, N))
    distances = np.sqrt(indices[0]**2 + indices[1]**2 + indices[2]**2)
    sorted_indices = np.unravel_index(np.argsort(distances, axis=None), distances.shape)
    num_voxels = np.prod(com.shape)
    num_compress = int(rate * num_voxels)
    com[sorted_indices[0][-num_compress:], sorted_indices[1][-num_compress:], sorted_indices[2][-num_compress:]] = 0
    return com

# カスタム視覚化関数
def vis_cust(after_voxels, window_name):
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.get_render_option().background_color = np.asarray([0.5372549019607843, 0.7647058823529412, 0.9215686274509804])
    vis.add_geometry(after_voxels)
    vis.run()
    vis.destroy_window()
