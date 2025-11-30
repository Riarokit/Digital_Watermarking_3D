import numpy as np
import open3d as o3d
import STG61_DCT_func as STG61F
import time

if __name__ == "__main__":
    # 定数設定
    GRID_DIVS_PER_OCTANT = 16
    QIM_DELTA = 0.2
    
    # 周波数帯域 (1D配列上のインデックス)
    # 固定長K=128個の係数のうち、どこを使うか
    MIN_FREQ = 10      # 低周波の最初(DC近傍)は避ける
    MAX_FREQ = 100     # 128個中の30番目まで使用
    
    ITERATIONS = 10
    
    IMG_SIZE = 16
    IMAGE_PATH = "watermark16.bmp"
    INPUT_FILE = "C:/bun_zipper.ply"

    print(f"Reading {INPUT_FILE}...")
    pcd_before = o3d.io.read_point_cloud(INPUT_FILE)
    pcd_before = STG61F.add_colors(pcd_before, color="grad")
    pcd_before, xyz_orig = STG61F.normalize_point_cloud_exact(pcd_before, target_size=100.0, visualize=False)
    
    watermark_bits = STG61F.image_to_bitarray(IMAGE_PATH, n=IMG_SIZE)
    
    # 埋め込み
    print("\n--- Embedding ---")
    start_time = time.time()
    xyz_embedded = STG61F.embed_watermark_main(
        xyz_orig, watermark_bits,
        grid_divs=GRID_DIVS_PER_OCTANT,
        qim_delta=QIM_DELTA,
        min_freq=MIN_FREQ,
        max_freq=MAX_FREQ,
        iterations=ITERATIONS
    )
    print(f"Time: {time.time() - start_time:.2f}s")
    
    pcd_embedded = o3d.geometry.PointCloud()
    pcd_embedded.points = o3d.utility.Vector3dVector(xyz_embedded)
    STG61F.calc_psnr(pcd_before, pcd_embedded)
    
    # 抽出
    print("\n--- Extraction ---")
    extracted_bits = STG61F.extract_watermark_main(
        xyz_embedded, len(watermark_bits),
        grid_divs=GRID_DIVS_PER_OCTANT,
        qim_delta=QIM_DELTA,
        min_freq=MIN_FREQ,
        max_freq=MAX_FREQ
    )
    
    STG61F.calc_ber(watermark_bits, extracted_bits)
    STG61F.bitarray_to_image(extracted_bits, n=IMG_SIZE, save_path="recovered.bmp")