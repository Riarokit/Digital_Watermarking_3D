import open3d as o3d

input_ply = "C:/dragon_vrip_res1.ply"
output_ply = "dragon_vrip_res2.ply"
target = 50000

pcd = o3d.io.read_point_cloud(input_ply)
print("Original points:", len(pcd.points))

bbox = pcd.get_axis_aligned_bounding_box()
extent = max(bbox.get_extent())
print("Model scale:", extent)

voxel = extent / 500   # ← 初期値（重要）
step = voxel * 0.01

while True:
    pcd_down = pcd.voxel_down_sample(voxel)
    n = len(pcd_down.points)
    print(f"voxel={voxel:.5f}, points={n}")

    if n <= target:
        break

    voxel += step

o3d.io.write_point_cloud(output_ply, pcd_down)
print("Saved:", output_ply)
