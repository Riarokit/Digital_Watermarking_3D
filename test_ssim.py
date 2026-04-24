import numpy as np
import open3d as o3d
import time
try:
    from skimage.metrics import structural_similarity as ssim
    import cv2
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("skimage or cv2 not available")

def capture_pcd_image(pcd, filename="temp.png", point_size=5.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=800)
    vis.add_geometry(pcd)
    
    # 描画オプションの設定
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = point_size
    
    # カメラの設定（適当な視点に合わせる）
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.5)
    vis.capture_screen_image(filename)
    vis.destroy_window()
    return filename

if __name__ == "__main__":
    # サンプル点群の生成
    pcd = o3d.io.read_point_cloud("C:/bun_zipper.ply")
    if len(pcd.points) == 0:
        # ランダム点群
        xyz = np.random.rand(10000, 3)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.paint_uniform_color([1, 0.706, 0])
    else:
        pcd.paint_uniform_color([1, 0.706, 0])
        
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3))) # Dummy
    pcd.estimate_normals()

    # 透かし入り（ノイズ）の点群を生成
    pcd_watermarked = o3d.geometry.PointCloud(pcd)
    xyz = np.asarray(pcd_watermarked.points)
    noise = np.random.normal(0, 0.005, size=xyz.shape)
    pcd_watermarked.points = o3d.utility.Vector3dVector(xyz + noise)
    
    print("Capturing original...")
    capture_pcd_image(pcd, "img_orig.png")
    print("Capturing watermarked...")
    capture_pcd_image(pcd_watermarked, "img_wm.png")
    
    if SKIMAGE_AVAILABLE:
        img_orig = cv2.imread("img_orig.png", cv2.IMREAD_GRAYSCALE)
        img_wm = cv2.imread("img_wm.png", cv2.IMREAD_GRAYSCALE)
        score, diff = ssim(img_orig, img_wm, full=True)
        print(f"SSIM Score: {score}")
    else:
        print("Install scikit-image and opencv-python to run SSIM")
