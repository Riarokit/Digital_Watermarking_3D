import numpy as np
import scipy.fftpack

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

# 埋め込み場所を生成する関数（上位x%からランダムに選択）
def generate_positions(dctcoef, length, percentage, seed=0):
    np.random.seed(seed)
    # DCT係数の絶対値を大きい順にソート
    flat_indices = np.argsort(-np.abs(dctcoef).flatten())
    num_top_indices = int(len(flat_indices) * (percentage / 100.0))
    top_indices = flat_indices[:num_top_indices]
    selected_indices = np.random.choice(top_indices, size=length, replace=False)
    positions = [np.unravel_index(idx, dctcoef.shape) for idx in selected_indices]
    return positions

# αを生成する関数
def generate_a(w, psnr):
    N = len(w[0][0])
    A = 1
    n = -1 * (psnr / 10)
    WW = w * w
    a = np.sqrt((np.power(10, n) * N * N * N * A * A) / np.sum(WW))
    return a

# 文字列をDCT係数に埋め込む関数
def embed_string(dctcoef, message, percentage, psnr, seed=0):
    binary_message = string_to_binary(message)
    positions = generate_positions(dctcoef, len(binary_message), percentage, seed)
    w = np.abs(dctcoef)  # 重み係数
    a = generate_a(w, psnr)  # 埋め込み強度α

    print(f"埋め込み強度α: {a}")  # デバッグ用に埋め込み強度を表示
    print(f"重み係数 w の最大値: {np.max(w)}, 最小値: {np.min(w)}")  # 重み係数の最大値と最小値を表示

    # 元のDCT係数をコピー
    original_dctcoef = dctcoef.copy()

    # 2進数メッセージをDCT係数に埋め込む
    for index, bit in enumerate(binary_message):
        i, j, k = positions[index]
        if bit == '1':
            dctcoef[i, j, k] += a * w[i, j, k]  # ビットが1なら係数を増加
        else:
            dctcoef[i, j, k] -= a * w[i, j, k]  # ビットが0なら係数を減少
        # if 0 <= index <= 3:
        #     print("dctcoef: ", dctcoef[i,j,k])
        #     print("a: ", a)
        #     print("w: ", w[i, j, k])

    return dctcoef, positions, original_dctcoef, a

# 埋め込まれた文字列を検出する関数
def detect_string(dctcoef_after_idct, positions, original_dctcoef, a):
    binary_message = ""
    
    # DCT係数から2進数メッセージを抽出
    for pos in positions:
        i, j, k = pos
        if dctcoef_after_idct[i, j, k] - original_dctcoef[i, j, k] > 0:
            bit = '1'
        else:
            bit = '0'
        binary_message += bit
        print(f"位置: {pos}, ビット: {bit}, DCT係数: {dctcoef_after_idct[i, j, k]}, 元のDCT係数: {original_dctcoef[i, j, k]}, 差分: {dctcoef_after_idct[i, j, k] - original_dctcoef[i, j, k]}")
    
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
