import STG32_GPCC_func as STG32F

# メイン処理
input_file = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240513/extract_people/frame_0.pcd"  # 入力の点群データ

# 透かしとして埋め込む文字列をバイナリ形式に変換
watermark_string = "HelloWorld"
watermark_binary = STG32F.string_to_binary(watermark_string)
num_bits = len(watermark_binary)

# 点群データを読み込む
point_cloud = STG32F.load_point_cloud(input_file)

# 元の点群データを表示
STG32F.display_point_cloud(point_cloud, "Original Point Cloud")

# Octreeを使って圧縮
octree_depth = 6  # より細かいオクツリーを作成
octree = STG32F.compress_point_cloud(point_cloud, octree_depth)

# オクツリーを表示
STG32F.display_octree(octree)

# 黒ボクセルの隣にある白ボクセルの色を変更し、埋め込み
black_voxel_positions = STG32F.change_white_voxels_to_black(octree, watermark_binary)

# Octreeから復元（各ボクセル内に10点を生成）
decompressed_pcd = STG32F.decompress_point_cloud(octree, black_voxel_positions, num_points_per_voxel=10)

# 復元された点群データを表示
STG32F.display_point_cloud(decompressed_pcd, "Decompressed Point Cloud")

# 埋め込んだバイナリ情報を復号
extracted_binary = STG32F.extract_watermark_from_octree(octree, black_voxel_positions, num_bits)

# 復号した文字列を表示
extracted_string = STG32F.binary_to_string(extracted_binary)
print(f"Extracted watermark: {extracted_string}")

# 誤差率を計算して表示
error_rate = STG32F.calculate_bit_error_rate(watermark_binary, extracted_binary)
print(f"Bit error rate: {error_rate:.2f}%")

print("Process completed.")
