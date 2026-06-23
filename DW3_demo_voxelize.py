import numpy as np
import open3d as o3d
import DW2_func as DW2F
import os

def create_voxel_wireframe(pcd, voxel_size):
    """
    点群が分割されるボクセルグリッドの境界線をワイヤーフレーム（LineSet）として作成する。
    """
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    voxels = voxel_grid.get_voxels()
    origin = voxel_grid.origin
    
    points = []
    lines = []
    
    # 立方体の12本のエッジの相対インデックス
    cube_edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 天面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 垂直辺
    ]
    
    for v in voxels:
        # ボクセルの最小コーナー座標
        min_pos = origin + v.grid_index * voxel_size
        
        # 8つの頂点
        v_pts = [
            min_pos,
            min_pos + np.array([voxel_size, 0.0, 0.0]),
            min_pos + np.array([voxel_size, voxel_size, 0.0]),
            min_pos + np.array([0.0, voxel_size, 0.0]),
            min_pos + np.array([0.0, 0.0, voxel_size]),
            min_pos + np.array([voxel_size, 0.0, voxel_size]),
            min_pos + np.array([voxel_size, voxel_size, voxel_size]),
            min_pos + np.array([0.0, voxel_size, voxel_size])
        ]
        
        start_idx = len(points)
        points.extend(v_pts)
        for edge in cube_edges:
            lines.append([start_idx + edge[0], start_idx + edge[1]])
            
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    # グリッドの境界線を分かりやすい青色に設定
    colors = [[0.2, 0.6, 0.9] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def main():
    # 1. データの読み込み
    input_file = "C:/bun_zipper.ply"

    voxel_size_percent = 2.0
    
    # テスト環境用にフォールバックを実装
    if not os.path.exists(input_file):
        print(f"入力ファイル {input_file} が見つからないため、球体モデルで代替します。")
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
    else:
        pcd = o3d.io.read_point_cloud(input_file)

    # 2. 点群の正規化
    pcd = DW2F.normalize_point_cloud(pcd)
    xyz = np.asarray(pcd.points)
    
    # 3. ボクセルサイズの計算 (対角線長に対する割合)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    scale_base = np.linalg.norm(xyz_max - xyz_min)
    voxel_size = scale_base * voxel_size_percent / 100
    
    print(f"点群の対角線長: {scale_base:.6f}")
    print(f"ボクセルサイズ割合: {voxel_size_percent:.2f}%")
    print(f"実際のボクセルサイズ: {voxel_size:.6f}")
    
    # 4. ボクセルダウンサンプリングの適用
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"元点数: {len(pcd.points)} -> ダウンサンプリング後点数: {len(pcd_down.points)}")

    # 5. 可視化用の色設定
    # 元の点群を薄いグレーで表示
    pcd_orig_colored = o3d.geometry.PointCloud(pcd)
    pcd_orig_colored.paint_uniform_color([0.7, 0.7, 0.7])
    
    # ダウンサンプリング後の代表点を赤色で表示
    pcd_down_colored = o3d.geometry.PointCloud(pcd_down)
    pcd_down_colored.paint_uniform_color([1.0, 0.0, 0.0])
    
    # ボクセルの境界線(ワイヤーフレーム)を生成
    wireframe = create_voxel_wireframe(pcd, voxel_size)
    
    # 6. Open3Dでの描画
    print("\n可視化ウィンドウを起動します。")
    print(" - グレー: 元の点群")
    print(" - 赤い点: ボクセル内の代表点（ダウンサンプリング結果）")
    print(" - 青い格子: ボクセル分割の境界線 (Voxel Grid)")
    print("※キーボードの '+' / '-' キーを押すと、点の描画サイズを調整できます。")
    
    o3d.visualization.draw_geometries(
        [pcd_orig_colored, pcd_down_colored, wireframe],
        window_name="Voxel Grid Downsampling Visualization",
        width=1024, height=768
    )

if __name__ == "__main__":
    main()
