import open3d as o3d
import numpy as np
import STG33_GPCC_func as STG33F

# 入力ファイルと出力ファイルのパス
input_file = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240513/extract_people/frame_0.pcd"
output_file = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/octree/STG33.txt"

# 点群データを読み込む
pcd = o3d.io.read_point_cloud(input_file)

# 1. 元の点群データを表示
print("元の点群データを表示します")
o3d.visualization.draw_geometries([pcd])

size_expand = 0.01
octree = o3d.geometry.Octree(max_depth=8)
octree.convert_from_point_cloud(pcd, size_expand)

# オクツリーのエンコード
STG33F.encode_octree(octree.root_node, output_path=output_file)

# "HelloWorld" をバイナリに変換
watermark = "HelloWorld"
watermark_bits = STG33F.string_to_binary(watermark)

# オクツリーを復元し、ビット列を取得
pcd_embedded, bit_stream, level_bits_list, max_depth = STG33F.decode_octree(output_file, 1.0)

# バイナリ情報を最下層に埋め込む
modified_bit_stream, embedded_indices = STG33F.embed_watermark(bit_stream, watermark_bits, level_bits_list, max_depth)

# 2. 情報を埋め込んだオクツリーの表示
print("情報を埋め込んだオクツリーを表示します")

# ポイントクラウドのビジュアライゼーション
o3d.visualization.draw_geometries([pcd_embedded])

# SNRの計算
snr = STG33F.calculate_snr(pcd, pcd_embedded)
print("SNR:", snr)

# 復号プロセス
extracted_bits = STG33F.extract_watermark(modified_bit_stream, embedded_indices)
decoded_string = STG33F.binary_to_string(extracted_bits)

# 3. 復元した点群データの表示
print("復元した点群データを表示します")
o3d.visualization.draw_geometries([pcd_embedded])

# 元のバイナリと復号後のバイナリの表示
print("元のバイナリビット列:  ", watermark_bits)
print("復号後のバイナリビット列:", extracted_bits)

# 誤差の計算と表示
def calculate_bit_error(original_bits, extracted_bits):
    # 長さが同じであることを確認
    if len(original_bits) != len(extracted_bits):
        raise ValueError("元のバイナリビット列と復号後のバイナリビット列の長さが一致していません。")

    # ビット誤差の計算
    total_bits = len(original_bits)
    error_bits = sum(1 for o, e in zip(original_bits, extracted_bits) if o != e)
    
    # 誤差率の計算（パーセンテージ）
    error_percentage = (error_bits / total_bits) * 100
    return error_bits, error_percentage

error_bits, error_percentage = calculate_bit_error(watermark_bits, extracted_bits)
print(f"ビット誤差: {error_bits} ビット ({error_percentage:.2f}%)")

# 復号された文字列の表示
print("復号された文字列:", decoded_string)
