import modules.fileconverter as fc
import modules.fileread as fr
import modules.tools as t
import modules.preprocess as pp
import modules.clustering as cl

##### デスク上の読み込み(2024.01.29) #####
# lasをpcdに変換
las_in_path_desk = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/lasdata/20240129/desk.las"
pcd_out_path_desk = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240129/desk"
fc.las2pcd(las_in_path_desk, pcd_out_path_desk)

# pcdファイル読み込み
pcdpath_desk = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240129/desk/frame_0.pcd"
pcd_desk = fr.ReadPCD(pcdpath_desk)

# pcdファイルを可視化
t.VisualizationPCD(pcd_desk, "desk")