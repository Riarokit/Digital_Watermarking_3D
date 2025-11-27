import numpy as np
import open3d as o3d
import STG52_GFT_func as STG52F
import time

if __name__ == "__main__":
    # 定数設定
    GRID_SIZE = 4.0        # ボクセルサイズ
    GUARD_BAND = 0.1       # ガードバンド幅
    QIM_DELTA = 0.2        # 埋め込み強度 (重心の移動量)
    # 周波数帯域設定 
    MIN_SPECTRE = 0.0     # DC付近のみカット
    MAX_SPECTRE = 1.0      # 超高周波は避ける
    # インプットファイルの設定
    IMG_SIZE = 16
    IMAGE_PATH = "watermark16.bmp"
    INPUT_FILE = "C:/bun_zipper.ply"

    # 1. 読み込み & 正規化
    print(f"Reading {INPUT_FILE}...")
    pcd_before = o3d.io.read_point_cloud(INPUT_FILE)
    pcd_before = STG52F.add_colors(pcd_before, color="grad")
    pcd_before, xyz_orig = STG52F.normalize_point_cloud(
        pcd_before, 
        target_scale=100.0, 
        visualize=False
    )
    
    # 2. 透かし画像の準備
    watermark_bits = STG52F.image_to_bitarray(IMAGE_PATH, n=IMG_SIZE)
    print(f"Watermark Bits: {len(watermark_bits)}")
    
    # 3. 埋め込み (VGSP)
    print("\n--- Embedding (VGSP) ---")
    start_time = time.time()
    xyz_embedded = STG52F.embed_watermark_vgsp(
        xyz_orig, watermark_bits,
        grid_size=GRID_SIZE,
        guard_band=GUARD_BAND,
        qim_delta=QIM_DELTA,
        min_spectre=MIN_SPECTRE,
        max_spectre=MAX_SPECTRE
    )
    print(f"Time: {time.time() - start_time:.2f}s")
    
    # 4.誤差評価
    pcd_embedded = o3d.geometry.PointCloud()
    pcd_embedded.points = o3d.utility.Vector3dVector(xyz_embedded)
    o3d.visualization.draw_geometries([pcd_embedded], window_name="Embedded")
    STG52F.calc_psnr_xyz(pcd_before, pcd_embedded)
    
    # 5. ノイズ攻撃
    # print("\n--- Noise Attack ---")
    # xyz_embedded = STG52F.add_noise(xyz_embedded, noise_std=0.01)
    # print(f"noize ratio: {noise_std * 100}%")

    # 6. 切り取り攻撃
    # print("\n--- Crop Attack ---")
    # xyz_embedded = STG52F.crop_point_cloud(xyz_embedded, keep_ratio=0.6)
    # print(f"xyz_embedded Points: {len(xyz_embedded)}")
    
    # 7. 抽出
    print("\n--- Extraction ---")
    extracted_bits = STG52F.extract_watermark_vgsp(
        xyz_embedded, len(watermark_bits),
        grid_size=GRID_SIZE,
        guard_band=GUARD_BAND,
        qim_delta=QIM_DELTA,
        min_spectre=MIN_SPECTRE,
        max_spectre=MAX_SPECTRE
    )
    
    # 8. 評価
    STG52F.evaluate_watermark(watermark_bits, extracted_bits)
    STG52F.bitarray_to_image(extracted_bits, n=IMG_SIZE, save_path="recovered_vgsp.bmp")
    
    # 可視化
    # pcd_embedded.colors = pcd_before.colors
    # o3d.visualization.draw_geometries([pcd_embedded])