import numpy as np
import open3d as o3d
import STG15_DCT_func as STG60F
import time

"""
3次元DCTを用いた点群へのデジタル透かし埋め込み手法
点がないボクセルに対してもDCTを計算するため、係数を変調しても点が動かない部分が多く、透かし損失に繋がった。
"""

if __name__ == "__main__":
    # 定数設定
    # 各ボクセルB(オクタント)内の分割数 n
    # ボクセルA(全体)は 2n 分割相当になる
    GRID_DIVS_PER_OCTANT = 12
    
    QIM_DELTA = 0.2
    MIN_FREQ_IDX = 1
    MAX_FREQ_IDX = 7
    ITERATIONS = 15
    
    IMG_SIZE = 16
    IMAGE_PATH = "watermark16.bmp"
    INPUT_FILE = "C:/bun_zipper.ply"

    # 1. 読み込み & 厳密正規化 (Step 1)
    print(f"Reading {INPUT_FILE}...")
    pcd_before = o3d.io.read_point_cloud(INPUT_FILE)
    pcd_before = STG60F.add_colors(pcd_before, color="grad")
    
    pcd_before, xyz_orig = STG60F.normalize_point_cloud_exact(
        pcd_before, target_size=100.0, visualize=False
    )
    STG60F.visualize_hierarchy(
        xyz_orig, 
        grid_divs_per_octant=GRID_DIVS_PER_OCTANT, 
        range_max=50.0
    )
    STG60F.calculate_capacity(GRID_DIVS_PER_OCTANT, MIN_FREQ_IDX, MAX_FREQ_IDX)
    watermark_bits = STG60F.image_to_bitarray(IMAGE_PATH, n=IMG_SIZE)
    
    # 2. 埋め込み (Step 2-8)
    print("\n--- Embedding ---")
    start_time = time.time()
    xyz_embedded = STG60F.embed_watermark_main(
        xyz_orig, watermark_bits,
        grid_divs=GRID_DIVS_PER_OCTANT,
        qim_delta=QIM_DELTA,
        min_freq_idx=MIN_FREQ_IDX,
        max_freq_idx=MAX_FREQ_IDX,
        iterations=ITERATIONS
    )
    print(f"Time: {time.time() - start_time:.2f}s")
    
    # 評価
    pcd_embedded = o3d.geometry.PointCloud()
    pcd_embedded.points = o3d.utility.Vector3dVector(xyz_embedded)
    pcd_embedded.colors = pcd_before.colors
    o3d.visualization.draw_geometries([pcd_embedded], window_name="Embedded")
    STG60F.calc_psnr(pcd_before, pcd_embedded)
    
    # OP. ノイズ攻撃
    # print("\n--- Noise Attack ---")
    # xyz_embedded = STG60F.noise_addition_attack(xyz_embedded, noise_percent=0.1)

    # OP. 切り取り攻撃
    # print("\n--- Crop Attack ---")
    # xyz_embedded = STG60F.cropping_attack(xyz_embedded, keep_ratio=0.9)
    
    # 4. 抽出 (Step 9)
    print("\n--- Extraction ---")
    extracted_bits = STG60F.extract_watermark_main(
        xyz_embedded, len(watermark_bits),
        grid_divs=GRID_DIVS_PER_OCTANT,
        qim_delta=QIM_DELTA,
        min_freq_idx=MIN_FREQ_IDX,
        max_freq_idx=MAX_FREQ_IDX
    )
    
    STG60F.calc_ber(watermark_bits, extracted_bits)
    STG60F.bitarray_to_image(extracted_bits, n=IMG_SIZE, save_path="recovered.bmp")