import modules.fileconverter as fc
import modules.fileread as fr
import modules.tools as t
import modules.preprocess as pp
import modules.clustering as cl

# HAPのときは、PCDのフレーム量は0.1秒間で1つくらいがいい。よって、5秒間撮ったら50フレームできるように計算。（背景点群はフレーム１つでいい）
# 例えば、3秒間撮ったデータ数15000のデータは30フレームに分割したいので15000/30=500を引数に設定
# データ数は1回実行しないとわからないため、とりあえず引数をでかめに設定して実行してみる。

##### 本の読み書き(2024.01.29) #####

# ### 背景点群ファイル
# # csvtopcd
# csv_in_path_readbgd = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20240129/readbook_background.csv"
# pcd_out_path_readbgd = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240129/readbook_background"
# fc.csv2pcd(25000, csv_in_path_readbgd, pcd_out_path_readbgd)

# # pcdファイル読み込み
# pcdpath_readbgd = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240129/readbook_background/frame_0.pcd"
# pcd_readbgd = fr.ReadPCD(pcdpath_readbgd)

# # pcdファイルを可視化
# t.VisualizationPCD(pcd_readbgd, "background")

# ### 前景点群(ページめくりなし)
# # csvtopcd
# csv_in_path_all1 = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20240129/readbook1.csv"
# pcd_out_path_all1 = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240129/readbook1"
# fc.csv2pcd(550, csv_in_path_all1, pcd_out_path_all1)

# # pcdファイルの読み込み
# pcdpath_all1 = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240129/readbook1/frame_0.pcd"
# pcd_all1 = fr.ReadPCD(pcdpath_all1)

# # pcdファイルを可視化
# t.VisualizationPCD(pcd_all1, "all2")

# ### 前景点群(ページめくりあり)
# # csvtopcd
# csv_in_path_all2 = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20240129/readbook2.csv"
# pcd_out_path_all2 = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240129/readbook2"
# fc.csv2pcd(495, csv_in_path_all2, pcd_out_path_all2)

# # pcdファイルの読み込み
# pcdpath_all2 = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240129/readbook2/frame_0.pcd"
# pcd_all2 = fr.ReadPCD(pcdpath_all2)

# # pcdファイルを可視化
# t.VisualizationPCD(pcd_all2, "all2")


##### 人物の身長測定(2023.12.04) #####

###背景点群ファイルをcsvからpcdにコンバート
csv_in_path_bgd = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20231204/background.csv"
pcd_out_path_bgd = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20231204/background"
fc.csv2pcd(3000, csv_in_path_bgd, pcd_out_path_bgd)

###全点群ファイルをcsvからpcdにコンバート
csv_in_path_all = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20231204/people.csv"
pcd_out_path_all = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20231204/people"
fc.csv2pcd(3000, csv_in_path_all, pcd_out_path_all)

###pcdファイルの読み込み
# pcd_out_path_all = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20231204/people"
pcdpath_background = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20231204/background/frame_0.pcd"
pcd_background = fr.ReadPCD(pcdpath_background)

pcdpath_all = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20231204/people/frame_0.pcd"
pcd_all = fr.ReadPCD(pcdpath_all)

###pcdファイルを可視化
t.VisualizationPCD(pcd_background, "background")
t.VisualizationPCD(pcd_all, "all")

###背景点群除去
pcd_human = pp.RemoveBackground(pcd_background, pcd_all, 0.05)
t.VisualizationPCD(pcd_human, "human")

###クラスタリング
pcd_human_cl = cl.Clustering(pcd_human, 0.04, 10)
t.VisualizationPCD(pcd_human_cl, "clustering")

###クラスタリングした点群をcsvファイルに出力
