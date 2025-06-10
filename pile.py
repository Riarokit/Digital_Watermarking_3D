from modules.sharemodule import os,glob
import modules.preprocess as pp
# twowalker
# syned_pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/April/24_with_iso/synthesized_twowalker"
# outputpath = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/April/24_with_iso/piled_twowalker"

# big
# syned_pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/April/24_with_iso/synthesized_big"
# outputpath = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/April/24_with_iso/background/from_livoxSDK"

# 20230609
# syned_pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/June/09/background_sync"
# outputpath = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/piled/June/09/background"
# syned_pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/June/09/1person_sync"
# outputpath = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/piled/June/09/1person_sync"

# 20230620
# background
syned_pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/June/20/background"
outputpath = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/piled/June/20/background"
# background_camra
# syned_pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/syned/June/20/background_camera"
# outputpath = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/piled/June/20/background_camera"
#chair
# syned_pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/syned/June/20/chair"
# outputpath = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/piled/June/20/chair"
#ishi2
# syned_pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/syned/June/20/ishi2"
# outputpath = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/piled/June/20/ishi2"
#iso2
# syned_pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/syned/June/20/iso2"
# outputpath = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/piled/June/20/iso2"
for filename in glob.glob(outputpath+'/'+'*frames.pcd'):
    print(outputpath + '/' + '*frames.pcd')
    os.remove(filename)    
print("finish")

print("FramePiling")
pp.FramePiling(syned_pcddir=syned_pcddir,outputpath=outputpath,split=30)
