from modules import fileread as fr
from modules import preprocess as pp
from modules import tools as t

# pcdファイルの読み込み
backgroundpath = 'C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240129/readbook_background/frame_0.pcd'
pcdpath = 'C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240129/readbook1/frame_1.pcd'
background = fr.ReadPCD(backgroundpath)
pcd = fr.ReadPCD(pcdpath)

# 背景点群除去
remove_bg = pp.RemoveBackground(background, pcd, thresh=0.1)
t.VisualizationPCD(remove_bg, title="Removed Background")