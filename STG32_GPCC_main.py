import STG32_GPCC_func as STG32F

# メイン処理
input_file = "C:/Users/ryoi1/OneDrive/ドキュメント/B3/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240513/extract_people/frame_0.pcd"  # 入力の点群データ

# 透かしとして埋め込む文字列をバイナリ形式に変換
watermark_string = "HelloWorld"
watermark_binary = STG32F.string_to_binary(watermark_string)
num_bits = len(watermark_binary)

# 1. 点群データを読み込む
point_cloud = STG32F.load_point_cloud(input_file)

# 2. 元の点群データを表示
STG32F.display_point_cloud(point_cloud, "Original Point Cloud")

# 3. Octreeを使ってボクセル化し表示
octree_depth = 8  # より細かいオクツリーを作成
octree = STG32F.compress_point_cloud(point_cloud, octree_depth)
STG32F.display_octree(octree)

# 4. オクツリーの葉ノードをランダムに選び、ビット情報を埋め込み
embedded_voxel_positions = STG32F.embed_data_in_octree(octree, watermark_binary)

# 5. 埋め込まれたボクセルによって埋め込んだデータを復号
extracted_binary = STG32F.extract_data_from_voxels(octree, embedded_voxel_positions, num_bits)

# 6. 元のバイナリビットと復号されたバイナリビットを2行で表示
STG32F.display_bit_comparison(watermark_binary, extracted_binary)

# 7. 復号した文字列を表示
extracted_string = STG32F.binary_to_string(extracted_binary)
print(f"Extracted watermark: {extracted_string}")

# 8. 誤差率を計算して表示
error_rate = STG32F.calculate_bit_error_rate(watermark_binary, extracted_binary)
print(f"Bit error rate: {error_rate:.2f}%")

# 9. 復元された点群データを表示
STG32F.display_point_cloud(point_cloud, "Restored Point Cloud")

print("Process completed.")
