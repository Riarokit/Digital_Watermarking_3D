from modules.sharemodule import o3d,np
from modules import fileread as fr
from modules import preprocess as pp
from modules import tools as t
import STG10_DCT_func as STG10F
# from selfmade import SelectPCD_Ver2 as SPCDV2
# from selfmade import comp
import time

"""
STG10F用のモジュール
dct→透かし→idct→検出・視覚化
"""

# pcdpath = "C:/Users/Public/Pythoncode/LiDAR/Python/data/pcddata/20240427/readbook2/frame_0.pcd"
# pointcloud = o3d.io.read_point_cloud(pcdpath)
# #o3d.visualization.draw_geometries([pointcloud])

# backgroundpath = "C:/Users/Public/Pythoncode/LiDAR/Python/data/pcddata/20240427/background/frame_0.pcd"
pcdpath = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240513/extract_people/frame_0.pcd"
# background = fr.ReadPCD(backgroundpath)
pcd = fr.ReadPCD(pcdpath)

# removed_bg_bef = pp.RemoveBackground(background,pcd,thresh=0.1)
# removed_bg = SPCDV2.SelectPCD(removed_bg_bef,[-100,100],[-5,2],[-100,100])
t.VisualizationPCD(pcd,title="Removed Background")

#ボクセルの大きさ指定
voxel_size = 0.05
before_voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size)
o3d.visualization.draw_geometries([before_voxels])

#処理開始
start = time.time()

voxels = before_voxels.get_voxels()
voxel_num = len(voxels)  #ボクセルの間引きのためにボクセルの数を記録

#NxNxN配列作る
voxel_dct = STG10F.make_all_voxels(voxels)
""""""
voxel_dct_vis = o3d.geometry.VoxelGrid()
voxel_dct_vis.voxel_size=voxel_size
n = len(voxel_dct[0][0])
for i in range(n):
        for j in range(n):
            for k in range(n):
                voxel_index = np.array([i,j,k],dtype=np.int32)
                voxel_color = np.array([voxel_dct[i,j,k],voxel_dct[i,j,k],voxel_dct[i,j,k]],dtype=np.float64)

                new_voxel = o3d.geometry.Voxel(grid_index=voxel_index,color=voxel_color)
                voxel_dct_vis.add_voxel(new_voxel)
STG10F.vis_cust(voxel_dct_vis)
""""""

#DCT変換する
print("dct開始")
start_dct = time.time()
dctcoef = STG10F.dct_3d(voxel_dct)
print("dct完了")
print("処理時間：",time.time()-start_dct,"/n")

""""""
voxel_dct_vis = o3d.geometry.VoxelGrid()
voxel_dct_vis.voxel_size=voxel_size
n = len(dctcoef[0][0])
for i in range(n):
        for j in range(n):
            for k in range(n):
                voxel_index = np.array([i,j,k],dtype=np.int32)
                voxel_color = np.array([dctcoef[i,j,k],dctcoef[i,j,k],dctcoef[i,j,k]],dtype=np.float64)

                new_voxel = o3d.geometry.Voxel(grid_index=voxel_index,color=voxel_color)
                voxel_dct_vis.add_voxel(new_voxel)
STG10F.vis_cust(voxel_dct_vis)
""""""

#透かし埋め込み
print("埋め込み開始")
start_emb = time.time()
nembrate = 0.25  #非埋め込み領域の大きさ([非埋め込み領域/全体]の割合)
dctcoef_emb,a,w = STG10F.embed(dctcoef,nembrate)#！確認のため、a,w追加中
print("埋め込み完了")
print("処理時間：",time.time()-start_emb,"/n")

#GPTに論文から適当に作ってもらった関数（動作未確認）
# quantization_level = 0.1
# compressed_data = comp.compress(dctcoef_emb, quantization_level)
# com = comp.decompress(compressed_data, quantization_level)

#単純に左下のDCT係数を輝度値0に設定するやつ（多分ちゃんと動いてるくさい）
com = STG10F.comp(dctcoef_emb,0.6)

""""""
voxel_dct_vis = o3d.geometry.VoxelGrid()
voxel_dct_vis.voxel_size=voxel_size
n = len(com[0][0])
for i in range(n):
        for j in range(n):
            for k in range(n):
                voxel_index = np.array([i,j,k],dtype=np.int32)
                voxel_color = np.array([com[i,j,k],com[i,j,k],com[i,j,k]],dtype=np.float64)

                new_voxel = o3d.geometry.Voxel(grid_index=voxel_index,color=voxel_color)
                voxel_dct_vis.add_voxel(new_voxel)
STG10F.vis_cust(voxel_dct_vis)
""""""

#逆変換する
print("idct開始")
start_idct = time.time()
voxel_dct2 = STG10F.idct_3d(com)
print("idct完了")
print("処理時間：",time.time()-start_idct,"/n")

""""""
voxel_dct_vis = o3d.geometry.VoxelGrid()
voxel_dct_vis.voxel_size=voxel_size
n = len(voxel_dct2[0][0])
for i in range(n):
        for j in range(n):
            for k in range(n):
                voxel_index = np.array([i,j,k],dtype=np.int32)
                voxel_color = np.array([voxel_dct2[i,j,k],voxel_dct2[i,j,k],voxel_dct2[i,j,k]],dtype=np.float64)

                new_voxel = o3d.geometry.Voxel(grid_index=voxel_index,color=voxel_color)
                voxel_dct_vis.add_voxel(new_voxel)
STG10F.vis_cust(voxel_dct_vis)
""""""

#透かし検出開始
print("検出開始")
start_detect = time.time()
y,p,Tp = STG10F.detect(voxel_dct2,nembrate,False)
 
if p > Tp:
    print("検出器：透かしあり")
else:
    print("検出器：透かし無し")

print("検出完了")
print("処理時間：",time.time()-start_detect,"/n")

#pとかTpとかの値のチェック
#detectで計算したpとTp
print("p:",p,"Tp:",Tp)
#本来のpを求める
n = len(dctcoef_emb[0][0])
x = STG10F.generate_x(n,nembrate)
#！ここからちゃんと動いてるかわからんから要動作確認
yx=dctcoef_emb*x
count = (n*n*n)-int(n*nembrate)*int(n*nembrate)*int(n*nembrate)
sum = np.sum(yx)
truep=sum/count
print("本来のpの値:",truep)

#透かし入れたDCT係数と、透かし入り画像生成後もう一度DCTした係数の差分
if np.allclose(dctcoef_emb,y):
    print("元データと透かし入りデータは近い/n")
else:
    print("元データと透かし入りデータは遠い/n")

#透かし視覚化
after_voxels = STG10F.visualize(voxel_dct2,voxel_num,voxel_size)
#o3d.visualization.draw_geometries([after_voxels])

#すべての処理完了
print("総処理時間：",time.time()-start)

STG10F.vis_cust(after_voxels)

#計算しなおすのめんどいからボクセルを保存したい（o3d.io.write_voxel_grid？でPLY形式で保存）

"""
#値のチェックするやつ
print("チェック開始")
while True:    
    val=input()
    list = val.split(",")
    print(list)
    if list[0] == "b" :
        print(dctcoef[int(list[1]),int(list[2])])
    if list[0] == "a" :
        print(voxel_dct2[int(list[1]),int(list[2])])

#正しいところに色が入ってるか確認用
while True:
    val=input()
    val_list =  val.split(',')
    print(dctcoefficent[int(val_list[0])][int(val_list[1])][int(val_list[2])])

#全部のボクセルのインデックスと色をリストに格納しなおす
for i in voxels:
    grid_index = i.grid_index.tolist()
    color = i.color
    line=[]
    line.extend(grid_index+[color[0]*255])
    voxeldata=[]
    voxeldata.append(line)
"""