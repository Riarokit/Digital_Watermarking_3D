import numpy as np
import open3d as o3d
import STG51_GFT_func as STG51F
import time

"""
点の埋め込み時にある量子化ステップ幅で強制的に座標を揃えるQIM手法を用いたGFT透かし埋め込み・抽出手法
しかし、結局埋め込みでグラフ構造が変化すると、元のグラフ情報がないと抽出できないため、ブラインド型にできず断念。
"""

if __name__ == "__main__":
    # 定数設定
    GRID_SIZE = 25.0       # ボクセルサイズ L
    GUARD_BAND = 0.2       # ガードバンド幅 epsilon
    QIM_DELTA = 0.01     # QIMステップ幅 (埋め込み強度)
    # グラフ構築設定
    GRAPH_METHOD = 'knn'; GRAPH_PARAM = 8    # k近傍法
    # GRAPH_METHOD = 'radius'; GRAPH_PARAM = 5 # 半径探索
    # 周波数帯域設定
    MIN_SPECTRE = 0.02
    MAX_SPECTRE = 0.8
    # インプットファイルの設定
    IMG_SIZE = 16
    IMAGE_PATH = "watermark16.bmp" # 画像
    INPUT_FILE = "C:/bun_zipper.ply" # 点群

    # 1. データ読み込み
    print(f"Reading {INPUT_FILE}...")
    pcd_before = o3d.io.read_point_cloud(INPUT_FILE)
    pcd_before = STG51F.add_colors(pcd_before, color="grad")
    pcd_before, xyz_orig = STG51F.normalize_point_cloud(
        pcd_before, 
        target_scale=100.0,
        visualize=False
    )
    
    # 2. 透かし画像の準備
    watermark_bits = STG51F.image_to_bitarray(IMAGE_PATH, n=IMG_SIZE)
    print(f"Watermark Bits: {len(watermark_bits)}")

    # 3. クラスタリング
    # ガードバンド処理により、境界付近の点はラベル -1 になる
    print("\n--- Clustering ---")
    start_time = time.time()
    labels = STG51F.voxel_grid_clustering(
        xyz_orig, 
        grid_size=GRID_SIZE, 
        guard_band=GUARD_BAND,
        visualize=False
    )
    
    # 4. 埋め込み (QIM)
    print("\n--- Embedding ---")
    xyz_embedded = STG51F.embed_watermark_qim(
        xyz_orig, labels, watermark_bits,
        grid_size=GRID_SIZE,   # チェック用
        guard_band=GUARD_BAND, # チェック用
        graph_method=GRAPH_METHOD, # グラフ設定
        graph_param=GRAPH_PARAM,   # グラフ設定
        delta=QIM_DELTA,
        min_spectre=MIN_SPECTRE,
        max_spectre=MAX_SPECTRE
    )
    embed_time = time.time() - start_time
    print(f"Embedding Time: {embed_time:.2f}s")
    
    # 5. 埋め込み誤差評価
    pcd_embedded = o3d.geometry.PointCloud()
    pcd_embedded.points = o3d.utility.Vector3dVector(xyz_embedded)
    o3d.visualization.draw_geometries([pcd_embedded], window_name="Embedded")
    STG51F.calc_psnr_xyz(pcd_before, pcd_embedded)

    # 6. ノイズ攻撃
    # print("\n--- Noise Attack ---")
    # xyz_embedded = STG51F.add_noise(xyz_embedded, noise_std=0.02)
    # print(f"noize ratio: {noise_std * 100}%")

    # 7. 切り取り攻撃
    # print("\n--- Crop Attack ---")
    # xyz_embedded = STG51F.crop_point_cloud(xyz_embedded, keep_ratio=0.6)
    # print(f"xyz_embedded Points: {len(xyz_embedded)}")

    # 8. 抽出
    print("\n--- Extraction ---")
    # A. 受信側でのクラスタリング再現
    labels_extracted = STG51F.voxel_grid_clustering(
        xyz_embedded, 
        grid_size=GRID_SIZE, 
        guard_band=GUARD_BAND
    )
    # B. QIM復号
    extracted_bits = STG51F.extract_watermark_qim(
        xyz_embedded, labels_extracted, len(watermark_bits),
        graph_method=GRAPH_METHOD, # グラフ設定
        graph_param=GRAPH_PARAM,   # グラフ設定
        delta=QIM_DELTA,
        min_spectre=MIN_SPECTRE,
        max_spectre=MAX_SPECTRE
    )
    
    # 9. 攻撃耐性評価
    STG51F.evaluate_watermark(watermark_bits, extracted_bits)
    STG51F.bitarray_to_image(extracted_bits, n=IMG_SIZE, save_path="recovered_qim.bmp")
    
    # 10. 可視化 (比較)
    # STG51F.add_colors(pcd_embedded, "grad")
    # o3d.visualization.draw_geometries([pcd_embedded], window_name="Embedded PC")