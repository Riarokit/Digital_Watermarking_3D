import numpy as np
import open3d as o3d
import STG51_GFT_func as STG51F
import time

if __name__ == "__main__":
    # ==========================================
    # 定数設定 (これらが「共通ルール」となる)
    # ==========================================
    GRID_SIZE = 3.5       # ボクセルサイズ L (点群のスケールに合わせて調整が必要)
    GUARD_BAND = 0.15     # ガードバンド幅 epsilon (Lの5-10%程度推奨)
    QIM_DELTA = 0.05      # QIMステップ幅 (埋め込み強度)
    
    # 周波数帯域設定 (低周波のみ利用)
    MIN_SPECTRE = 0.0
    MAX_SPECTRE = 0.3     # 低周波30%のみ使用
    
    # 画像設定
    IMG_SIZE = 16
    IMAGE_PATH = "watermark64.bmp" # 任意の画像
    INPUT_FILE = "C:/bun_zipper.ply" # 点群ファイルパス

    # 1. データ読み込み
    print(f"Reading {INPUT_FILE}...")
    pcd_before = o3d.io.read_point_cloud(INPUT_FILE)
    xyz_orig = np.asarray(pcd_before.points)
    scale = np.max(xyz_orig, axis=0) - np.min(xyz_orig, axis=0)
    print(f"Point Cloud Scale: {scale}")
    # 補足: GRID_SIZEはモデルの大きさに対して適切に設定する必要あり
    # モデル幅が100なら、GRID_SIZE=10.0 (10分割) くらいが目安
    
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
        guard_band=GUARD_BAND
    )
    
    # 4. 埋め込み (QIM)
    print("\n--- Embedding ---")
    xyz_embedded = STG51F.embed_watermark_qim(
        xyz_orig, labels, watermark_bits,
        delta=QIM_DELTA,
        min_spectre=MIN_SPECTRE,
        max_spectre=MAX_SPECTRE
    )
    embed_time = time.time() - start_time
    print(f"Embedding Time: {embed_time:.2f}s")
    
    # 5. 埋め込み誤差評価
    pcd_embedded = o3d.geometry.PointCloud()
    pcd_embedded.points = o3d.utility.Vector3dVector(xyz_embedded)
    STG51F.calc_psnr_xyz(pcd_before, pcd_embedded)

    # 6. ノイズ攻撃
    print("\n--- Attacks ---")
    # ガードバンド幅 (0.15) より小さい移動量ならクラスタは維持されるはず
    xyz_embedded = STG51F.add_noise(xyz_embedded, noise_std=0.02)

    # 7. 切り取り攻撃
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