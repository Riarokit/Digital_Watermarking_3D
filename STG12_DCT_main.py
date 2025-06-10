from modules.sharemodule import o3d, np
from modules import fileread as fr
from modules import preprocess as pp
from modules import tools as t
import STG12_DCT_func as STG12F
import time

# PCDファイルのパスを指定
pcdpath = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240513/extract_people/frame_0.pcd"
pcd = fr.ReadPCD(pcdpath)

# PCDファイルの点群データを可視化
t.VisualizationPCD(pcd, title="PCDファイルの点群データ")

# ボクセルの大きさを指定
voxel_size = 0.05
before_voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

# ボクセル化後の点群を可視化
STG12F.vis_cust(before_voxels, window_name="ボクセル化後の点群")

# 処理開始
start = time.time()

voxels = before_voxels.get_voxels()
voxel_num = len(voxels)

# NxNxN配列を作成
voxel_dct = STG12F.make_all_voxels(voxels)

# make_all_voxelsで作製した、ボクセル化後の点群をN*N*Nの立方体に挿入した立方体を可視化
voxel_dct_vis = o3d.geometry.VoxelGrid()
voxel_dct_vis.voxel_size = voxel_size
n = len(voxel_dct[0][0])
for i in range(n):
    for j in range(n):
        for k in range(n):
            voxel_index = np.array([i, j, k], dtype=np.int32)
            voxel_color = np.array([voxel_dct[i, j, k], voxel_dct[i, j, k], voxel_dct[i, j, k]], dtype=np.float64)
            new_voxel = o3d.geometry.Voxel(grid_index=voxel_index, color=voxel_color)
            voxel_dct_vis.add_voxel(new_voxel)
STG12F.vis_cust(voxel_dct_vis, window_name="N*N*Nの立方体")

# DCT変換
print("dct開始")
start_dct = time.time()
dctcoef = STG12F.dct_3d(voxel_dct)
print("dct完了")
print("処理時間：", time.time() - start_dct, "\n")

# DCT変換後の立方体を可視化
dct_vis = o3d.geometry.VoxelGrid()
dct_vis.voxel_size = voxel_size
for i in range(n):
    for j in range(n):
        for k in range(n):
            voxel_index = np.array([i, j, k], dtype=np.int32)
            voxel_color = np.array([dctcoef[i, j, k], dctcoef[i, j, k], dctcoef[i, j, k]], dtype=np.float64)
            new_voxel = o3d.geometry.Voxel(grid_index=voxel_index, color=voxel_color)
            dct_vis.add_voxel(new_voxel)
STG12F.vis_cust(dct_vis, window_name="DCT変換後の立方体")

# dctcoefの要素数を表示
print("DCT変換後の係数 dctcoef の要素数:")
print(dctcoef.size)

# 立方体のサイズを表示
print("立方体のサイズ (N):", n)

# 文字列「HelloWorld」を埋め込み
print("埋め込み開始")
start_emb = time.time()
message = "HelloWorld"

# 文字列をASCIIコードの2進数に変換した結果を表示
binary_message = STG12F.string_to_binary(message)
print("文字列をASCIIコードの2進数に変換した結果:")
print(binary_message)

seed = 42  # 固定シードを使用して一貫性を保つ
psnr = 40  # PSNR値を設定
lower_cut = 0.3  # 低周波成分のカット割合
upper_cut = 0.3 # 高周波成分のカット割合
rate = 0.01  # 圧縮率(rate*10割カット)
dctcoef_emb, positions = STG12F.embed_string(dctcoef, message, psnr, lower_cut, upper_cut, seed)
print("埋め込み完了")
print("処理時間：", time.time() - start_emb, "\n")

# 埋め込み挿入後の立方体を可視化
emb_vis = o3d.geometry.VoxelGrid()
emb_vis.voxel_size = voxel_size
for i in range(n):
    for j in range(n):
        for k in range(n):
            voxel_index = np.array([i, j, k], dtype=np.int32)
            voxel_color = np.array([dctcoef_emb[i, j, k], dctcoef_emb[i, j, k], dctcoef_emb[i, j, k]], dtype=np.float64)
            new_voxel = o3d.geometry.Voxel(grid_index=voxel_index, color=voxel_color)
            emb_vis.add_voxel(new_voxel)
STG12F.vis_cust(emb_vis, window_name="埋め込み挿入後の立方体")

# dctcoef_embの要素数を表示
print("埋め込み後のDCT係数 dctcoef_emb の要素数:")
print(dctcoef_emb.size)

# positionsの要素数を表示
print("埋め込み位置 positions の要素数:")
print(len(positions))

# 圧縮
com = STG12F.comp(dctcoef_emb, rate)

# 逆DCT変換
print("idct開始")
start_idct = time.time()
voxel_dct2 = STG12F.idct_3d(com)
print("idct完了")
print("処理時間：", time.time() - start_idct, "\n")

# 逆DCT変換後の立方体を可視化
idct_vis = o3d.geometry.VoxelGrid()
idct_vis.voxel_size = voxel_size
for i in range(n):
    for j in range(n):
        for k in range(n):
            voxel_index = np.array([i, j, k], dtype=np.int32)
            voxel_color = np.array([voxel_dct2[i, j, k], voxel_dct2[i, j, k], voxel_dct2[i, j, k]], dtype=np.float64)
            new_voxel = o3d.geometry.Voxel(grid_index=voxel_index, color=voxel_color)
            idct_vis.add_voxel(new_voxel)
STG12F.vis_cust(idct_vis, window_name="逆DCT変換後の立方体")

# 透かし視覚化
after_voxels = STG12F.visualize(voxel_dct2, voxel_num, voxel_size)
STG12F.vis_cust(after_voxels, window_name="透かし入りのボクセル")

# 逆DCT変換後の点群を再度ボクセル化せずにそのままDCT変換
print("再度DCT変換開始")
start_dct2 = time.time()
dctcoef_after_idct = STG12F.dct_3d(voxel_dct2)
print("再度DCT変換完了")
print("処理時間：", time.time() - start_dct2, "\n")

# 検出されたメッセージの詳細を表示
print("検出開始")
start_detect = time.time()
detected_binary_message = STG12F.detect_string(dctcoef_after_idct, positions)

# バイナリメッセージの表示
print("バイナリメッセージの表示:")
print("埋め込みメッセージ（2進数）:", binary_message)
print("検出されたメッセージ（2進数）:", detected_binary_message)

# 埋め込みメッセージと検出されたメッセージのビットの違いを計算
bit_diff = sum(b1 != b2 for b1, b2 in zip(binary_message, detected_binary_message))
bit_diff_percent = (bit_diff / len(binary_message)) * 100
print(f"ビットの違い: {bit_diff_percent:.2f}% ({bit_diff}bit)")

# 検出されたメッセージが2進数であることを確認
if all(bit in '01' for bit in detected_binary_message):
    try:
        ascii_message = STG12F.binary_to_string(detected_binary_message)
        print("2進数から文字列に変換した結果:")
        print(ascii_message)
    except ValueError as e:
        print(f"エラー: {e}")
else:
    print("検出されたメッセージが2進数形式ではありません:")
    print(detected_binary_message)

print("検出完了")
print("処理時間：", time.time() - start_detect, "\n")

# 総処理時間
print("総処理時間：", time.time() - start)
