import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import copy
import datetime as dt
import pyransac3d as pyrsc
import laspy
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D

############# 2フレームずつフレームを読み込みlasデータをpcdデータに変換するための参考 (Ryuuta) ############# 
# # 背景のlasファイルをまずpcdデータに変換し、for文により、2フレームずつlasファイルを読み込みpcdに変換
# # ここではファイル名がFrame_1.las, Frame_2.las....で保存されているとみなしている
# time =  #読み込みたいフレーム数
# first =  #読み込みたい最初のフレーム番号

# input_path_background = "" #背景入力したいlasデータまでのpath 例 C:\\Users\\Taro\\Documents\\Python\\las\\background.las
# output_path_background = "" #背景pcdデータを出力したいファイルまでのpath 例 C:\\Users\\Taro\\Documents\\Python\\pcd\\background.pcd
# input_dir = "las\\20230424_test1_walking_100\\" #読み込みたいlasデータが存在しているディレクトリ 例 C:\\Users\\Taro\\Documents\\Python\\las\\sample\\
# output_dir = "pcd\\20230424_test1_walking_100\\" #出力したいpcdデータを格納するディレクトリ 例 C:\\Users\\Taro\\Documents\\Python\\pcd\\sample\\
# [background_output_pcd, now_bg] = f.las2pcd(input_path_background, output_path_background) #背景データのlas to pcd変換
# background_pcd = f.ReadPCD(background_output_pcd) #背景pcdデータの読み込み
# for i in range(first,time+first):
#     #Frame_i.pcdとFram_i+1.pcdというデータを読み込み　※iは変数, 1からtime番目までのフレームを2つずつ読み込み
#     [pre_output_pcd, now_pre] = f.las2pcd(input_dir + "Frame_" + str(i) + ".las", output_dir + "Frame_" + str(i) + ".pcd")
#     [post_output_pcd, now_post] = f.las2pcd(input_dir + "Frame_" + str(i+1) + ".las", output_dir + "Frame_" + str(i+1) + ".pcd")
# # 読み込んだ2つのフレームのpcdデータの読み込み
#     pre_pcd = f.ReadPCD(pre_output_pcd)
#     post_pcd = f.ReadPCD(post_output_pcd)
#######################################################################################################

class Person(): #id,coordinate,center(o2d)
    def __init__(self,id,coordinate,center,size,timestump):
        self.id = id
        self.coordinate = coordinate
        self.center = center
        self.size = size
        self.timestump = timestump

# lasからpcdへの変換（o2d）
def las2pcd(lasfile,outputpath):
    with laspy.open(lasfile) as f:
        las = f.read()
        las = np.array(las.xyz)
        # print("Python:{}".format(las))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(las)
        o3d.io.write_point_cloud(outputpath,pcd)
        now = dt.datetime.now()
    return [outputpath , now]

################未完成################
# 全フレームを変換（o2d）
# def las2pcdAll(lasfilepath,outputpath):
#     inFile = laspy.read(lasfilepath)
#     # for i in range(inFile.header.point_records_count):
#     for i in range(5):
#         x = inFile[i].x
#         y = inFile[i].y
#         z = inFile[i].z
#         coordinate = np.concatenate([x,y,z],axis=0)
#         print(type(x))
#         points = np.column_stack((coordinate[0],coordinate[1],coordinate[2]))
#         points = np.asarray(points)
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         o3d.io.write_point_cloud(outputpath+"frame_"+str(i)+".pcd",pcd)
#     now = dt.datetime.now()
#     print("Number of Frame : {}".format(i+1))
#     return [outputpath,now]
################未完成################

# データの読み込み（o2d）
def ReadPCD(filename):
    return o3d.io.read_point_cloud(filename)

# open3dからDataFrame（o2d）
def o3dtoDataFrame(pcd):
    pcd_deepcopied = copy.deepcopy(pcd)
    pcd_coordinate = np.asanyarray(pcd_deepcopied.points)
    # print("MATLAB {}".format(pcd_coordinate))
    pcd_df = pd.DataFrame(data=pcd_coordinate,columns=['x','y','z'])
    return pcd_df

# DataFrameからopen3d(o2d)
# def DataFrametoo3d(pcd_df):
    

#  フィルタリング（Radius Oulier Removal）戻り値はDataFrame型
# def filteringROR(pcd_df,nb_points,radius):（o2d）# 重すぎ
#     print(dt.datetime.now())
#     print(len(pcd_df))
    
#     noiseList = []
#     for i in range(len(pcd_df)):
#         counts = 0
#         print("{}% : {}".format((i/len(pcd_df))*100,dt.datetime.now()))
#         for j in range(len(pcd_df)-1):
#             square = (pcd_df.iloc[i]-pcd_df.iloc[j+1])**2
#             distance = square['x']+square['y']+square['z']
#             if distance < radius:
#                 counts += 1
#             if counts < nb_points:
#                 noiseList.append(i)

#     pcd_df.drop(pcd_df.index[noiseList])
#     print(dt.datetime.now())
#     print(len(pcd_df))
#     return pcd_df

def display_inlier_outlier(cloud, ind):#(o2d)
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],)

#　背景差分法(o2d)
def RemoveBackground(background,target,thresh):
    o3d.visualization.draw_geometries([background],"Include Background")
    distance = target.compute_point_cloud_distance(background)
    distance = np.array(distance)
    moveIndex = np.where(distance>thresh)[0]#初期0.1。なんでこれでうまくいく？なんのリスト？
    # print(len(moveIndex))
    # print(moveIndex)
    
    removedpcd = target.select_by_index(moveIndex)
    # print("OK")
    o3d.visualization.draw_geometries([removedpcd],"Removed Background")
    return removedpcd

# radius outlier removal（o2d）
# これを使うとなぜか動きが不安定になるから非推奨
def FilteringROR(pcd, r, points):
    # o3d.visualization.draw_geometries([pcd],"Before Filtering")
    cl, ind = pcd.remove_radius_outlier(nb_points=points,radius=r)
    # display_inlier_outlier(pcd,ind)
    inlier_cloud = cl.select_by_index(ind)
    # o3d.visualization.draw_geometries([inlier_cloud],"After Flitering")
    print('success!')
    return inlier_cloud
# Ransac（o2d）
def Ransac(pcd,thresh):
    points = np.asarray(pcd.points)
    plane = pyrsc.Plane()

    eq,inliers = plane.fit(points, thresh)
    plane2 = pcd.select_by_index(inliers).paint_uniform_color([1,0,0])
    obb = plane2.get_oriented_bounding_box()
    obb.color = [0,0,1]
    not_plane = pcd.select_by_index(inliers,invert=True)
    # o3d.visualization.draw_geometries([not_plane,obb],"After Ransac")
    return not_plane


# ダウンサンプリング（o2d）
def DownSampling(pcd,size):
    print("Downsample the point cloud with a voxel of {}".format(size))
    print("Size of pcd : {}".format(pcd))
    o3d.visualization.draw_geometries([pcd],"Before DownSampling")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=size)
    o3d.visualization.draw_geometries([voxel_down_pcd],"After DownSampling")
    print("Size of pcd : {}".format(voxel_down_pcd))
    return voxel_down_pcd

# pcdデータの範囲選択（o2d）
def SelectPCD(pcdData,xlim,ylim,zlim):#pcdDataにはopen3dオブジェクトを渡す。DataFrameではない。
    # o3d.visualization.draw_geometries([pcdData], "Before Select")
    pcd_deepcopied = copy.deepcopy(pcdData)
    pcd_coordinate = np.asarray(pcd_deepcopied.points)
    # print("coordinate:{}".format(pcd_coordinate))
    pcd_df = pd.DataFrame(data=pcd_coordinate,columns=['x','y','z'])
    # print("PCD_DataFrame:{}".format(pcd_df))
    if(xlim!=[]):
        pcd_df = pcd_df[(xlim[0] < pcd_df['x']) & (pcd_df['x'] < xlim[1])]
    if(ylim!=[]):
        pcd_df = pcd_df[(ylim[0] < pcd_df['y']) & (pcd_df['y'] < ylim[1])]
    if(zlim!=[]):
        pcd_df = pcd_df[(zlim[0] < pcd_df['z']) & (pcd_df['z'] < zlim[1])]
    # print("PCDSelected_DataFrame:{}".format(pcd_df))
    pcd_df = np.array(pcd_df)
    pcd_deepcopied.points = o3d.utility.Vector3dVector(pcd_df)
    # o3d.visualization.draw_geometries([pcd_deepcopied], "After Selected")
    # print("Chaged DataFrame to o3d.")
    return pcd_deepcopied

# クラスタリング（色付け）されたo3dオブジェクトを返し、可視化まで行う（o2d）
def Clustering(pcdData,epsm,points):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        clusterLabels = np.array(pcdData.cluster_dbscan(eps=epsm, min_points=points,print_progress=True))#クラスタごとにラベル付けしたものを配列で返す
        clusterLabels_noNoise = clusterLabels[clusterLabels>-1]#ノイズでないラベルを抽出
        noiseIndex = np.where(clusterLabels == -1)[0]#ノイズの点（インデックス）
        pcdData = pcdData.select_by_index(np.delete(np.arange(len(pcdData.points)),noiseIndex))#全点群の数分の等間隔のリストから、ノイズのインデックスに対応するものを削除->ノイズ出ない点（インデックス）の抽出
        max_label = clusterLabels_noNoise.max()
        print("<Clustering> point cloud has {} clusters".format(max_label+1))
        colors = plt.get_cmap("tab20")(clusterLabels_noNoise/(max_label if max_label > 0 else 1))
        colors[clusterLabels_noNoise < 0] = 0
        pcdData.colors = o3d.utility.Vector3dVector(colors[:,:3])
        o3d.visualization.draw_geometries([pcdData], "After Clustering")
        return pcdData

#クラスタリング後にノイズ除去(o2d)
def NoiseRemoveClustering(pcdData,epsm,points):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        clusterLabels = np.array(pcdData.cluster_dbscan(eps=epsm, min_points=points,print_progress=True))#クラスタごとにラベル付けしたものを配列で返す
        clusterLabels_noNoise = clusterLabels[clusterLabels>-1]#ノイズラベルを除去
        noiseIndex = np.where(clusterLabels == -1)[0]#ノイズの点（インデックス）
        pcdData = pcdData.select_by_index(np.delete(np.arange(len(pcdData.points)),noiseIndex))#全点群の数分の等間隔のリストから、ノイズのインデックスに対応するものを削除->ノイズ出ない点（インデックス）の抽出
        if clusterLabels_noNoise is None:
            return print('ノイズしかなかった')
        max_label = clusterLabels_noNoise.max()
        print("point cloud has {} clusters".format(max_label+1))
        colors = plt.get_cmap("tab20")(clusterLabels_noNoise/(max_label if max_label > 0 else 1))
        colors[clusterLabels_noNoise < 0] = 0
        pcdData.colors = o3d.utility.Vector3dVector(colors[:,:3])
        o3d.visualization.draw_geometries([pcdData], "After Clustering")
        return pcdData

# フレーム内のクラスタを線形変換(o2d)
def LinTrans(coordinate):#ExtraHumanの中でcluster_dfを受け取る
    print(coordinate[['x','y','z']])
    Get3dGraph(df=coordinate,xlim=[],ylim=[],zlim=[],size=[])
    under_df = coordinate[coordinate['z']<(coordinate['z'].max()+coordinate['z'].min())/2]
    print(f"under_df : {under_df[['x','y','z']]}")
    walkwid = squareform(pdist(under_df[['x','y']]))#下半身の歩幅で向き推定
    maxWalkWid = walkwid.max()
    # 立ち状態でないとき
    if maxWalkWid>0.3:
        print("Situation : Walking")
        maxWidIndex = np.where(walkwid == maxWalkWid)[0]
        print(f"maxWidIndex : {maxWidIndex}")
        vec = under_df.iloc[maxWidIndex[1]]-under_df.iloc[maxWidIndex[0]]
        print(f"vec[y]/vec[x]:{vec[['y']]/vec[['x']]}")
        # centerVector = postPerson.center-prePerson.center
        theta = np.arctan(vec[['y']].values/vec[['x']].values)
        print(f"theta : {theta}")
        # theta = cluster_df[['x','y']].apply(lambda row: math.atan(row['y']/row['x']), axis=1)
        cos = np.cos(theta)
        sin = np.sin(theta)
        # print(f"sin : {sin}")
        # print(f"cos : {cos}")

        R = np.array([[cos,-sin],
                    [sin,cos]])
        print(R)
        LinTransferedCoordinate_df = coordinate[['x','y']].apply(lambda x: np.dot([x['x'],x['y']],R),axis=1)

        # LinTransferedCoordinate_df = R@np.array(cluster_df[['x','y']].T).T
        LinTransferedCoordinate_df = [[sublist[0],sublist[1]] for sublist in LinTransferedCoordinate_df]
        LinTransferedCoordinate_df = pd.DataFrame(LinTransferedCoordinate_df,columns=['x','y'])
        print("LinTransferedCoordinate_df : {}".format(LinTransferedCoordinate_df))
        Get3dGraph(df=coordinate,xlim=[],ylim=[],zlim=[],size=[])
    else:
        print("Situation : Stanging")
        maxWidIndex = np.where(walkwid == maxWalkWid)[0]
        print(f"maxWidIndex : {maxWidIndex}")
        # 傾き
        vec = under_df.iloc[maxWidIndex[1]]-under_df.iloc[maxWidIndex[0]]
        # 垂直の傾き
        tmp = vec['y']
        vec['y'] = vec['x']
        vec['x'] = -tmp
        print(f"vec[y]/vec[x]:{vec[['y']]/vec[['x']]}")
        # centerVector = postPerson.center-prePerson.center
        theta = np.arctan(vec[['y']].values/vec[['x']].values)
        print(f"theta : {theta}")
        # theta = cluster_df[['x','y']].apply(lambda row: math.atan(row['y']/row['x']), axis=1)
        cos = np.cos(theta)
        sin = np.sin(theta)
        # print(f"sin : {sin}")
        # print(f"cos : {cos}")

        R = np.array([[cos,-sin],
                    [sin,cos]])
        print(R)
        LinTransferedCoordinate_df = coordinate[['x','y']].apply(lambda x: np.dot([x['x'],x['y']],R),axis=1)

        # LinTransferedCoordinate_df = R@np.array(cluster_df[['x','y']].T).T
        LinTransferedCoordinate_df = [[sublist[0],sublist[1]] for sublist in LinTransferedCoordinate_df]
        LinTransferedCoordinate_df = pd.DataFrame(LinTransferedCoordinate_df,columns=['x','y'])
        print("LinTransferedCoordinate_df : {}".format(LinTransferedCoordinate_df))
    return LinTransferedCoordinate_df


# 人を抽出(o2d)
def ExtractHumanClustering(pcdData,epsm,points,now,boxsize_x=[0.45,1.2],boxsize_y=[0.45,1.2],boxsize_z=[0.45,2.0],ratio=[2.2,4.0]):
    candidateList = []
    notnoiseList = []
    # noiseList = []
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        cluster_df = pd.DataFrame(data=o3dtoDataFrame(pcdData),columns=['x','y','z'])
        cluster_df["label"] = np.array(pcdData.cluster_dbscan(eps=epsm, min_points=points,print_progress=True))#クラスタごとにラベル付けしたものを配列で返す
        # clusterLabels = np.array(pcdData.cluster_dbscan(eps=epsm, min_points=points,print_progress=True))#クラスタごとにラベル付けしたものを配列で返す
        cluster_df = cluster_df[cluster_df["label"]>-1]#ノイズを除去
        max_label = cluster_df['label'].max()
        print("<Extract>point cloud has {} clusters".format(max_label+1))
        # print(f"cluster_df:{cluster_df}")

        x_center_df = ((cluster_df[['x','label']].groupby('label',as_index=False).max()+cluster_df[['x','label']].groupby('label',as_index=False).min())/2)
        # print(f"x_center_df:{x_center_df}")
        y_center_df = ((cluster_df[['y','label']].groupby('label',as_index=False).max()+cluster_df[['y','label']].groupby('label',as_index=False).min())/2)
        z_center_df = ((cluster_df[['z','label']].groupby('label',as_index=False).max()+cluster_df[['z','label']].groupby('label',as_index=False).min())/2)
        center_df = pd.concat([x_center_df,y_center_df,z_center_df],axis=1)
        center_df = center_df.drop(columns=[center_df.columns[2],center_df.columns[4]])
        center_df["label"]=center_df.index
        # print("center_df:{}".format(center_df))
        # print("center_index:{}".format(center_df.index))
        # print("cluster_columns:{}".format(cluster_df.columns))
        # print(f"center_df_columns : {center_df.columns}")
        # print(f"center_df:{center_df}")

        x_range_df = (cluster_df[['x','label']].groupby('label',as_index=False).max()-cluster_df[['x','label']].groupby('label',as_index=False).min())
        y_range_df = (cluster_df[['y','label']].groupby('label',as_index=False).max()-cluster_df[['y','label']].groupby('label',as_index=False).min())
        z_range_df = (cluster_df[['z','label']].groupby('label',as_index=False).max()-cluster_df[['z','label']].groupby('label',as_index=False).min())
        size_df = pd.concat([x_range_df,y_range_df,z_range_df],axis=1)
        size_df = size_df.drop(columns=[size_df.columns[2],size_df.columns[4]])
        size_df["label"]=size_df.index

        # print(f"size_df_columns : {size_df.columns}")
        # print(f"size_df:{size_df}")
        for L in cluster_df["label"].unique():
            # print(cluster_df[cluster_df["label"]==L])
            if (((boxsize_x[0] <= x_range_df.loc[L,'x'])&(x_range_df.loc[L,'x'] <= boxsize_x[1]))
                &((boxsize_y[0] <= y_range_df.loc[L,'y'])&(y_range_df.loc[L,'y'] <= boxsize_y[1]))
                &((boxsize_z[0] <= z_range_df.loc[L,'z'])&(z_range_df.loc[L,'z'] <= boxsize_z[1]))):
                # lineartrans = LinTrans(coordinate=cluster_df[cluster_df["label"]==L])
                #下限を2.2に設定していたけど、少し厳しいから、1にする
                if((ratio[0] <= (z_range_df.loc[L,'z']/x_range_df.loc[L,'x']))
                   & ((z_range_df.loc[L,'z']/x_range_df.loc[L,'x']) <= ratio[1])):#後々線形変換を利用する
                    notnoiseList.append(L)
                    candidate = Person(id=L,coordinate=cluster_df[cluster_df["label"]==L],center=center_df[center_df["label"]==L],size=size_df[size_df["label"]==L],timestump=now)
                    candidateList.append(candidate)

        notnoiseIndex = cluster_df.index[cluster_df['label'].isin(np.asarray(notnoiseList))]
        
        pcdData = pcdData.select_by_index(notnoiseIndex)
        o3d.visualization.draw_geometries([pcdData], "After Clustering")
        return [pcdData,candidateList] 



# 2フレーム間の同一人物判定(o2d)
def JudgeSamePerson(pcdPre,pcdPost,thresh):
    ##############  Ryuuta  ############## 
    # 今のフレームのクラスタIDを前フレームと異なるものにする
    # 前フレームのクラスタIDのリストを作り、今のフレームのクラスタIDを一度すべて-1とする
    pre_id_lists = []
    for i in range(len(pcdPre)):
        pre_id_lists.append(pcdPre[i].id)
    if len(pre_id_lists) != 0:
    #クラスタIDに0より小さい数字が含まれていた時IDを0,1,2…となるように変更
        if np.min(pre_id_lists) != 0:
            for i in range(len(pre_id_lists)):
                pcdPre[i].id = i
                pre_id_lists[i] = i
        # print(pre_id_lists)
        id_different = np.max(pre_id_lists)
        for i in range(len(pcdPost)):
            pcdPost[i].id = -1
        ##############  Ryuuta  ############## 

        for i in range(len(pcdPre)):
            print("Pre Id:{}".format(pcdPre[i].id))
            for j in range(len(pcdPost)):
                print("Post Id:{}".format(pcdPost[j].id))
                distance = np.sqrt(np.sum(pow(((pcdPre[i].center[['x','y']].values-pcdPost[j].center[['x','y']].values)[0]),2)))
                #リアルタイムでデータを取得できるようになったら
                # tsec = (pcdPost[j].timestump-pcdPre[i].timestump)
                # print("tsec.second:{}".format(tsec.seconds))
                # print("distance:{}".format(distance))
                # if ((tsec.seconds*0.8<distance)&(distance < tsec.seconds*1.0)):
                if(distance<thresh):
                    print("distance<{}".format(thresh))
                    if((pcdPre[i].size['z'].values*0.7<pcdPost[j].size['z'].values)&(pcdPost[j].size['z'].values<pcdPre[i].size['z'].values*1.3)
                    &((pcdPre[i].size['x'].values*0.45<pcdPost[j].size['x'].values)&(pcdPost[j].size['x'].values<pcdPre[i].size['x'].values*2.2))):
                            pcdPost[j] = Person(id=pcdPre[i].id,coordinate=pcdPost[j].coordinate,center=pcdPost[j].center,size=pcdPost[j].size,timestump=pcdPost[j].timestump)
                            print("Same")
                            continue
                    
                else:
                    print(f"distancd of (Id{pcdPre[i].id}Id{pcdPost[j].id}>{thresh})")
        
        ##############  Ryuuta  ############## 
        #現フレームに存在するクラスタのうち前フレームに存在しないクラスタに対して、前フレームには存在しないクラスタIDを割り振る (Ryuuta)
        #IDの重複を防ぐ
        for i in range(len(pcdPost)):
            if pcdPost[i].id == -1:
                id_different += 1
                pcdPost[i].id = id_different
        ##############  Ryuuta  ############## 
    
    else: #pre_id_listsが空の場合
        print("Not exist human in PreFrame")
    return [pcdPre, pcdPost]

#クラスタリングし、クラスタのID, サイズ, 点の個数を表示(Ryuuta)
#追記：重心も表示させておきます(fukui)
def ClusteringInfo(pcdData,epsm,points):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        Ground_Hieght = 1.5 #LiDARセンサーの地面からの高さ
        clusterLabels = np.array(pcdData.cluster_dbscan(eps=epsm, min_points=points,print_progress=True))
        # print("ClusterLabels:{}".format(clusterLabels))
        max_label = clusterLabels.max()
        print("point cloud has {} clusters".format(max_label+1))
        colors = plt.get_cmap("tab20")(clusterLabels/(max_label if max_label > 0 else 1))
        colors[clusterLabels < 0] = 0
        pcdData.colors = o3d.utility.Vector3dVector(colors[:,:3])

        points = np.asarray(pcdData.points)

        lists = {} #添え字がクラスタID,中身がそのクラスタの各点の座標
        x_lists = {} #添え字がクラスタID,中身がそのクラスタの各点のx座標
        y_lists = {} #添え字がクラスタID,中身がそのクラスタの各点のy座標
        z_lists = {} #添え字がクラスタID,中身がそのクラスタの各点のz座標
        for i, j in enumerate(clusterLabels):
            if (j not in lists):
                lists[j] = [points[i]]
                x_lists[j] = [points[i][0]]
                y_lists[j] = [points[i][1]]
                z_lists[j] = [points[i][2]]
            else:
                lists[j].append(points[i])
                x_lists[j].append(points[i][0])
                y_lists[j].append(points[i][1])
                z_lists[j].append(points[i][2])

        for key in lists:
            if(key != -1):
                max_difference = np.max(lists[key], axis = 0) - np.min(lists[key], axis = 0) #xyz座標の最大値と最小値の差
                length = max_difference[0] # 横幅
                width = max_difference[1] # 縦幅
                height = np.max(z_lists[key], axis = 0) + Ground_Hieght # 高さ
                x_centroid = np.sum(x_lists[key], axis = 0) / len(lists[key]) #x座標の重心
                y_centroid = np.sum(y_lists[key], axis = 0) / len(lists[key]) #y座標の重心
                z_centroid = np.sum(z_lists[key], axis = 0) / len(lists[key]) #z座標の重心

                # 情報を表示
                print("クラスタID:", key)
                print("横幅：" + str(length) + " 縦幅：" + str(width) + " 高さ：" + str(height))
                print("点の個数:" + str(len(lists[key])))
                print("重心：(" + str(x_centroid) + ", " + str(y_centroid) + ", " + str(z_centroid) + ")")

        o3d.visualization.draw_geometries([pcdData], "After Clustering")
    return pcdData

#2台のLiDARから得た点群データを1つにまとめる(fukui)
#引数は2つのpcdデータと回転角度，x座標, y座標, z座標の補正値
def pcdsummarize(pcd1, pcd2, angle, corrected_x, corrected_y, corrected_z):
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)

    # pcd1を回転させる
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    pcd1.rotate(R)

    #pcd1のx座標をLiDARの間隔分差し引く
    for i in range(len(pcd1.points)):
        pcd1.points[i][0] -= corrected_x
    
    #pcd1のy座標補正
    for i in range(len(pcd1.points)):
        pcd1.points[i][1] += corrected_y

    #pcd1のz座標補正
    for i in range(len(pcd1.points)):
        pcd1.points[i][2] += corrected_z

    # pcd2と結合する
    merged_pcd = pcd1 + pcd2
    
    return merged_pcd

# 3次元グラフ表示（o2d）
def Get3dGraph(df, xlim=[], ylim=[], zlim=[], size=[]):
    fig = plt.figure()
    data = fig.add_subplot(projection='3d')
    data.scatter(df["x"], df["y"], df["z"])
    data.set_xlabel('X Label')
    data.set_ylabel('Y Label')
    data.set_zlabel('Z Label')

    if (ylim != []):
        plt.ylim(ylim)
    if (xlim != []):
        plt.xlim(xlim)
    if (zlim != []):
        data.set_zlim(zlim[0], zlim[1])

    if (size != []):
        fig.set_size_inches(size[0], size[1])
    plt.show()

# 点群位置補正とフレーム重畳 (Ryuuta)
def FramePiled(Frame_points): #引数はフレームごとの重畳したいPersonクラスの点群データが入っている配列が格納されているリスト(キーはフレーム番号-1の値)
    piled_lists = {} #位置補正した点群データのリスト
    piled_pcd = {} #重畳したpcdデータのリスト
    for i in range(len(Frame_points)):
      for j in range(len((Frame_points[i]))):
        x_average = np.average(Frame_points[i][j].coordinate['x'])
        y_average = np.average(Frame_points[i][j].coordinate['y'])
        z_min = np.min(Frame_points[i][j].coordinate['z'])
        #x,y座標は座標値の平均をとり、その偏差を補正後の座標値とすることでx,y軸中心とするように移動
        #z軸は最小値の値だけ移動させる
        Frame_points[i][j].coordinate['x'] -= x_average 
        Frame_points[i][j].coordinate['y'] -= y_average
        Frame_points[i][j].coordinate['z'] -= z_min
        coordinate_nm = Frame_points[i][j].coordinate.drop('label', axis=1).to_numpy() #DataFrame型のcoordinateをx,y,z座標のみnumpy配列に変換し、格納
        if (Frame_points[i][j].id in piled_lists) == False:
            piled_lists[Frame_points[i][j].id] = coordinate_nm
        else:
            piled_lists[Frame_points[i][j].id] = np.concatenate([piled_lists[Frame_points[i][j].id], coordinate_nm])
    #重畳した点群をnumpy配列からpcdデータに変換し、表示
    for i in range(len(piled_lists)):
        piled_pcd[i] = o3d.geometry.PointCloud()
        piled_pcd[i].points = o3d.utility.Vector3dVector(piled_lists[i])
        o3d.visualization.draw_geometries([piled_pcd[i]], "After Piled")

# 身体の一部を上から階層状に切り出して、各層の最大幅を計算し、最小幅と最大幅の日から人間判定する
# 第1引数：前フレームのPersonリスト(postcandidateList)
# 第2引数：上から何パーセントを抽出(percent)
# 第3引数：抽出した部分を何層で区切るか(layer)
# 第4引数：最大幅と最小幅の比率(ratio)
# 戻り値：人っぽい形のクラスタのみに絞られたリスト
def CutJudgeHuman(postcandidateList,percent,layer,ratio=[2,5]):
    heightline = {}#そのうちDataFrameにする
    wid_df = pd.DataFrame(columns=['heightId','width'])
    minwid = 0 #例えば首など（外れ値を除くために下限を決める必要がある）
    maxwid = 0 #例えば肩（外れ値を除くために上限を決める必要がある）
    print(f"Number of Before postcandidateList:{len(postcandidateList)}")
    for index,candidate in enumerate(postcandidateList):
        print(f"Id : {candidate.id}")
        headCoordinate = candidate.coordinate[(candidate.coordinate['z'].max()-candidate.coordinate['z'].min())*(1-percent)<(candidate.coordinate['z']-candidate.coordinate['z'].min())]
        # Get3dGraph(headCoordinate,[1.8,3.2],[-0.8,0.8],[-1.0,0.4],[10,10])
        for height in range(layer):
            # headCoordinate = candidate.coordinate[(candidate.coordinate['z'].max()-candidate.coordinate['z'].min())*(1-0.5)<(candidate.coordinate['z']-candidate.coordinate['z'].min())]
            # f.Get3dGraph(headCoordinate,[1.8,3.2],[-0.8,0.8],[-1.7,0.7],[])
            heightline[str(height)] = headCoordinate[(((headCoordinate['z'].max()-headCoordinate['z'].min())*(1-(height+1)*0.1))<(headCoordinate['z']-headCoordinate['z'].min()))
                                             &((headCoordinate['z']-headCoordinate['z'].min())<((headCoordinate['z'].max()-headCoordinate['z'].min())*(1-height*0.1)))]
            if len(heightline[str(height)])==0:
                continue
            else:
            # print(f"heightline:{heightline[str(height)][['x','y']]}") 
            # print(f"Index of heightline['x','y'] :{heightline[str(height)][['x','y']].index}")
                newdata = pd.DataFrame({'heightId':[height],'width':[pdist(heightline[str(height)][['x','y']]).max()]})
                wid_df = pd.concat([wid_df,newdata])
            # print(f"wid_df:{wid_df}")
            minwid = wid_df['width'].min()
            maxwid = wid_df['width'].max()
        print(f"minwid : {minwid}")
        print(f"maxwid : {maxwid}")
        print(f"maxwid/minwid:{maxwid/minwid}")
        if not ((ratio[0]<(maxwid/minwid))&((maxwid/minwid)<ratio[1])):
            print(f"NOT maxwid/minwid:{maxwid/minwid}")
            postcandidateList.pop(index)
            print(f"Removed Id : {candidate.id}")
    print(f"Number of After postcandidateList:{len(postcandidateList)}")
    # return postcandidateList# 人っぽい形のものに絞った

# 次のTracking内で使用する(o2d)
# def ExtractIdCenter(candidate):
#     result = [candidate.id,candidate.center[["x","y","z"]].values]
#     return result
def ExtractIdCenter(postcandidateList):
    candidate_df = pd.DataFrame(columns=['id',"x_center","y_center","z_center"])
    for candidate in postcandidateList:
        df = pd.DataFrame(data={'id':candidate.id,'x_center':candidate.center['x'],'y_center':candidate.center['y'],'z_center':candidate.center['z']})
        candidate_df = pd.concat([candidate_df,df],ignore_index=True)
    return candidate_df

#　トラッキング（o2d）
# 第1引数：この関数を使用するときのフレーム番号
# 第2引数：トラッキングの情報を保持するためのDataFrame（columns=["id","center"]）の
# global変数を追加。自分のファイルないでこの変数を使いたいときは
# f.trackng_dfとすると取得できる
tracking_df = pd.DataFrame(columns=["id","x_center","y_center","z_center"])
# 第3引数：前フレーム内のPersonリスト（空で渡す）
# 第4引数：現在のフレーム内のPersonリスト
def Tracking(precandidateList,postcandidateList):
    global tracking_df
    # print(f"tracking_df:{tracking_df}")
    #postcandidateListが要素を1つでも持っているときの処理
    if not len(postcandidateList)==0:
        print(f"postcandidateList:{pd.DataFrame(postcandidateList[0].center)}")
        [precandidateList,postcandidateList] = JudgeSamePerson(precandidateList,postcandidateList,thresh=1)
        candidate_df = ExtractIdCenter(postcandidateList)
        # TrackList = list(map(ExtractIdCenter,postcandidateList))
        for track in candidate_df.iterrows():
            # print(f"track:{track}")
            if not (tracking_df["id"].isin([track[1]['id']])).any():
                tracking_df = pd.concat([tracking_df,pd.DataFrame({"id":track[1]['id'],"x_center":[track[1]['x_center']],"y_center":[track[1]['y_center']],"z_center":[track[1]['z_center']]})],ignore_index=True)
            else:
                x_center = candidate_df[candidate_df['id']==track[1]['id']]['x_center']
                y_center = candidate_df[candidate_df['id']==track[1]['id']]['y_center']
                z_center = candidate_df[candidate_df['id']==track[1]['id']]['z_center']

                # tracking_df.loc[tracking_df["id"]==track[1]['id'],"x_center"] = np.vstack((tracking_df.loc[tracking_df["id"]==track[1]['id'],"x_center"],x_center))
                # tracking_df.loc[tracking_df["id"]==track[1]['id'],"y_center"] = np.vstack((tracking_df.loc[tracking_df["id"]==track[1]['id'],"y_center"],x_center))
                # tracking_df.loc[tracking_df["id"]==track[1]['id'],"z_center"] = np.vstack((tracking_df.loc[tracking_df["id"]==track[1]['id'],"z_center"],x_center))
                tracking_df.loc[tracking_df["id"]==track[1]['id'],"x_center"] = tracking_df.loc[tracking_df["id"]==track[1]['id'],"x_center"].apply(lambda x : np.append(x,x_center))
                tracking_df.loc[tracking_df["id"]==track[1]['id'],"y_center"] = tracking_df.loc[tracking_df["id"]==track[1]['id'],"y_center"].apply(lambda x : np.append(x,y_center))
                tracking_df.loc[tracking_df["id"]==track[1]['id'],"z_center"] = tracking_df.loc[tracking_df["id"]==track[1]['id'],"z_center"].apply(lambda x : np.append(x,z_center))
        #多分必要ないけど、各所のコメントアウトを外せば、DataFrame型ではないもので出力される。    
        # for track in TrackList:
        #     if not (tracking_df["id"].isin([track[0]])).any():
        #         print(f"track[1]{track[1][0]}")
        #         # tracking_df = pd.concat([tracking_df,pd.DataFrame({"id":track[0],"center":track[1]})],ignore_index=True)
        #         tracking_df = pd.concat([tracking_df,pd.DataFrame({"id":track[0],"x_center":[track[1][0][0]],"y_center":[track[1][0][1]],"z_center":[track[1][0][2]]})],ignore_index=True)
        #         print(f"No.{frameNum} tracking_df:{tracking_df}")
        #     else:
        #         print(f"TrackList:{TrackList[int(track[0])]}")
        #         # center = TrackList[int(track[0])][1]
        #         x_center = TrackList[TrackList(track[0])][1][0]
        #         y_center = TrackList[int(track[0])][1][1]
        #         z_center = TrackList[int(track[0])][1][2]
        #         # tracking_df.loc[tracking_df["id"]==track[0],"center"] = tracking_df.loc[tracking_df["id"]==track[0],"center"].apply(lambda x : np.append(x,center))
        #         tracking_df.loc[tracking_df["id"]==track[0],"x_center"] = tracking_df.loc[tracking_df["id"]==track[0],"x_center"].apply(lambda x : np.append(x,x_center))
        #         tracking_df.loc[tracking_df["id"]==track[0],"y_center"] = tracking_df.loc[tracking_df["id"]==track[0],"y_center"].apply(lambda x : np.append(x,y_center))
        #         tracking_df.loc[tracking_df["id"]==track[0],"z_center"] = tracking_df.loc[tracking_df["id"]==track[0],"z_center"].apply(lambda x : np.append(x,z_center))
        #         chunk = 3
        #         # tracking_df['center'] = tracking_df['center'].apply(lambda x : np.array([x[i:i+chunk] for i in range(0, len(x),chunk)]))
        #         precandidateList = postcandidateList
        #         print(f"New ID No.tracking_df:{tracking_df}")
        # print(f"tracking_df:{tracking_df}")
        # print(f"tracking_df[0]:{tracking_df.iloc[0].values}")
        # print(f"tracking_df[1]:{tracking_df.iat[1,1]}")
    #postcandidateListが要素を1つも持っていない場合
    else:
        precandidateList = postcandidateList

#骨格計算(全身の横幅計算)(fukui)
def SkeletalCalculation(pcd):
    ##法線推定
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #
    ##法線データの取得
    #normals = np.asarray(pcd.normals)
    #
    ##主成分分析（PCA）による姿勢推定
    #pca = PCA(n_components=3)
    #pca.fit(normals)
    #skeleton_direction = pca.components_[0]

    #骨格情報計算
    #z座標の0~100パーセンタイルにおいて、体の幅を計算
    print("-----body_width-----")
    for i in range(101):
        #全身の位置推定
        body_height = np.percentile(np.asarray(pcd.points)[:,2], i)# z座標のiパーセンタイルを肩の高さとする
        #左右の位置特定
        body_points = np.asarray(pcd.points)[np.where(np.asarray(pcd.points)[:, 2] > body_height)]

        if len(body_points) > 0:  # body_pointsが空でない場合のみ処理を行う
            left_body = np.percentile(body_points[:, 1], 5)  # y座標の5パーセンタイルを左の位置とする
            right_body = np.percentile(body_points[:, 1], 95)  # y座標の95パーセンタイルを右の位置とする
            body_width = right_body - left_body
            print(body_width)

    print("-----body_left_width-----")
    for i in range(101):
        #全身の位置推定
        body_height = np.percentile(np.asarray(pcd.points)[:,2], i)# z座標のiパーセンタイルを肩の高さとする
        #左右の位置特定
        body_points = np.asarray(pcd.points)[np.where(np.asarray(pcd.points)[:, 2] > body_height)]

        if len(body_points) > 0:  # body_pointsが空でない場合のみ処理を行う
            center_body = np.percentile(body_points[:, 1], 50)  # y座標の50パーセンタイルを中心の位置とする
            left_body = np.percentile(body_points[:, 1], 5)  # y座標の5パーセンタイルを左の位置とする
            right_body = np.percentile(body_points[:, 1], 95)  # y座標の95パーセンタイルを右の位置とする
            body_left_width = center_body - left_body
            print(body_left_width)

    print("-----body_right_width-----")
    for i in range(101):
        #全身の位置推定
        body_height = np.percentile(np.asarray(pcd.points)[:,2], i)# z座標のiパーセンタイルを肩の高さとする
        #左右の位置特定
        body_points = np.asarray(pcd.points)[np.where(np.asarray(pcd.points)[:, 2] > body_height)]

        if len(body_points) > 0:  # body_pointsが空でない場合のみ処理を行う
            center_body = np.percentile(body_points[:, 1], 50)  # y座標の50パーセンタイルを中心の位置とする
            left_body = np.percentile(body_points[:, 1], 5)  # y座標の5パーセンタイルを左の位置とする
            right_body = np.percentile(body_points[:, 1], 95)  # y座標の95パーセンタイルを右の位置とする
            body_right_width = right_body - center_body
            print(body_right_width)
    
#バウンディングボックス生成(fukui)
def bboxcreation(pcd):
    #PCDデータをNumPy配列に変換
    points = np.asarray(pcd.points)

    #XYZ座標の最大値と最小値を計算
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)

    #バウンディングボックスの中心座標を計算
    center = (min_xyz + max_xyz) / 2.0

    #バウンディングボックスのサイズを計算
    size = max_xyz - min_xyz
 
    print("min_xyz")
    print(min_xyz)
    print("max_xyz")
    print(max_xyz)
    print("center_of_bbox")
    print(center)
    print("size_of_bbox")
    print(size)

    #バウンディングボックスの生成
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_xyz, max_bound=max_xyz)

    #バウンディングボックスを可視化
    o3d.visualization.draw_geometries([bbox])

#断面積を対象物の高さごと計算(fukui)
def calculatecrosssection(pcd):
    points = np.asarray(pcd.points)

    #x, y, zの列を取得
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    #z座標のパーセンタイルを計算
    percentiles = np.arange(0, 101, 1)

    #各パーセンタイルごとに断面積を計算
    for i in range(len(percentiles) - 1):
        z_min = np.percentile(z, percentiles[i])
        z_max = np.percentile(z, percentiles[i+1])

        #z座標が範囲内にある点のインデックスを抽出
        indices = np.where((z >= z_min) & (z <= z_max))[0]

        #インデックスに対応するx, y座標を取得
        x_section = x[indices]
        y_section = y[indices]

        #断面積を計算
        section_area = 0.5 * np.abs(np.dot(x_section[:-1], y_section[1:]) - np.dot(x_section[1:], y_section[:-1]))

        print(section_area)

#対象物の高さを分割する(fukui)
def divisioncalculation(pcd, num_intervals):
    points = np.asarray(pcd.points)

    # z座標の最大値と最小値を計算
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])

    # 最小値と最大値に対応するxyz座標を取得
    min_xyz = points[np.argmin(points[:, 2])]
    max_xyz = points[np.argmax(points[:, 2])]

    # z座標を指定された区間数に分割
    z_intervals = np.linspace(z_min, z_max, num_intervals + 1)

    # 分割点のz座標と平均座標を格納するリスト
    division_info_list = []

    # 最小値のxyz座標を分割点に追加
    division_info_list.append([min_xyz[0], min_xyz[1], min_xyz[2]])

    for i in range(1, num_intervals):
        z_division_point = (z_intervals[i] + z_intervals[i - 1]) / 2

        # 分割点に該当する座標を取得
        division_points = points[abs(points[:, 2] - z_division_point) < 0.001]

        # 平均座標を計算
        if len(division_points) > 0:
            avg_x = np.mean(division_points[:, 0])
            avg_y = np.mean(division_points[:, 1])
            avg_z = np.mean(division_points[:, 2])

            division_info_list.append([avg_x, avg_y, avg_z])

    # 最大値のxyz座標を分割点に追加
    division_info_list.append([max_xyz[0], max_xyz[1], max_xyz[2]])

    # 分割点の情報をPandas DataFrameに変換
    df = pd.DataFrame(division_info_list, columns=["x", "y", "z"])

    return df

#df形式の3次元座標とPCDデータを3Dグラフ描画(fukui)
def plot_points_3d(division_df, pcd):
    # 3Dグラフの描画
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ## PCDデータから得られる3次元座標を白抜きの水色でプロット
    #points = np.asarray(pcd.points)
    #ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='lightblue', marker='o', label='PCD Data')

    # DataFrameから得られる3次元座標を赤でプロット
    ax.scatter(division_df['x'], division_df['y'], division_df['z'], c='red', marker='o', label='DataFrame')

    # グラフの設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()

    plt.show()

#傾き計算(fukui)
def calculate_slope_3d(dataframe):
    # DataFrameからx, y, z列を取得
    x = dataframe['x']
    y = dataframe['y']
    z = dataframe['z']

    # 隣接する点との傾きを計算して "slope" 列に追加
    slopes_zy = [None]  # 最初の行には傾きが存在しないため None を追加
    slopes_zx = [None]
    slopes_yx = [None]

    for i in range(1, len(x)):
        delta_x = x[i] - x[i - 1]
        delta_y = y[i] - y[i - 1]
        delta_z = z[i] - z[i - 1]

        if delta_y != 0:
            slope_zy = (delta_z / delta_y) if delta_y != 0 else float('inf')
            slopes_zy.append(slope_zy)
        else:
            slopes_zy.append(None)

        if delta_x != 0:
            slope_zx = (delta_z / delta_x) if delta_x != 0 else float('inf')
            slopes_zx.append(slope_zx)
        else:
            slopes_zx.append(None)

        if delta_x != 0:
            slope_yx = (delta_y / delta_x) if delta_x != 0 else float('inf')
            slopes_yx.append(slope_yx)
        else:
            slopes_yx.append(None)

    # DataFrameに "slope" 列を追加
    dataframe['slope_zy'] = slopes_zy
    dataframe['slope_zx'] = slopes_zx
    dataframe['slope_yx'] = slopes_yx

    return dataframe

#支持基底面を抽出し重心と比較(fukui)
def sijikiteimen(dataframe, z_lim, h):
    sijikiteimen_df = pd.DataFrame(data=None,columns=[])

    # 引数のDataFrameからx, y, z列を取得
    x = dataframe['x']
    y = dataframe['y']
    z = dataframe['z']

    # sijikiteiのxyzを宣言
    x_sijikiteimen = [None]
    y_sijikiteimen = [None]
    z_sijikiteimen = [None]

    # 重心計算
    x_centroid = np.mean(x)
    y_centroid = np.mean(y)
    z_centroid = np.mean(z)

    z_min = z_lim - h
    z_max = z_lim + h

    # 任意のz座標±hの範囲内のxyz座標を抽出(支持基底面上の点抽出)
    for i in range(1, len(z)):
        if(z_min <= z[i] and z[i] <= z_max):
            x_sijikitei = x[i]
            y_sijikitei = y[i]
            z_sijikitei = z[i]
            x_sijikiteimen.append(x_sijikitei)
            y_sijikiteimen.append(y_sijikitei)
            z_sijikiteimen.append(z_sijikitei)

    sijikiteimen_df['x'] = x_sijikiteimen
    sijikiteimen_df['y'] = y_sijikiteimen
    sijikiteimen_df['z'] = z_sijikiteimen

    # 支持基底面上の点を表示
    #print(sijikiteimen_df)

    # 支持基底面上の重心計算
    x_sijikiteimen_centroid = np.mean(x_sijikitei)
    y_sijikiteimen_centroid = np.mean(y_sijikitei)
    z_sijikiteimen_centroid = np.mean(z_sijikitei)

    # 重心を表示
    print("centroid:(," + str(x_centroid) + ", " +  str(y_centroid) + ", " + str(z_centroid) +",)" )

    # 支持基底面の重心を表示
    print("sentroid_of_sijikitei:(," + str(x_sijikiteimen_centroid) + ", " +  str(y_sijikiteimen_centroid) + ", " + str(z_sijikiteimen_centroid) +",)" )

    #　重心と支持基底面のずれを計算
    length_x = x_sijikiteimen_centroid - x_centroid
    length_y = y_sijikiteimen_centroid - y_centroid
    #length_z = z_sijikiteimen_centroid - z_centroid
    length_squares = (length_x * length_x) + (length_y * length_y)
    length = np.sqrt(length_squares)

    print("zahyou_zure:(," + str(length_x) + ", " +  str(length_y) +",)" )
    print("zure:,"+ str(length))

    return sijikiteimen_df

#分散共分散行列・相関行列・標準偏差の計算(fukui)
def variance_covariance(dataframe):
    # 引数のDataFrameからx, y, z列を取得
    x = dataframe['x']
    y = dataframe['y']
    z = dataframe['z']

    # 分散共分散行列の計算
    cov_matrix = np.cov([x, y, z], bias=True)

    # 相関行列の計算
    corr_matrix = np.corrcoef([x, y, z])

    # 標準偏差の計算
    x_std = np.std(x, ddof=1)
    y_std = np.std(y, ddof=1)
    z_std = np.std(z, ddof=1)

    # 結果の表示
    print("bunsankyoubunsan:")
    print(cov_matrix)
    print("soukan")
    print(corr_matrix)
    print("hyoujyunhensa:")
    print("x," +  str(x_std))
    print("y," +  str(y_std))
    print("z," +  str(z_std))

#特定の座標をz軸の範囲で抽出し、特徴量を計算する(fukui)
def extraction(dataframe, upper, lower):
    extraction_df = pd.DataFrame(data=None,columns=[])

    #引数のDataFrameからx, y, z列を取得
    x = dataframe['x']
    y = dataframe['y']
    z = dataframe['z']
    cf = dataframe['cumulative_relative_frequency']

    #z座標〇％～〇％の座標を格納する配列を宣言
    extraction_x = [None]
    extraction_y = [None]
    extraction_z = [None]

    #z座標の上位z_lim%の座標を取得
    for i in range(1, len(cf)):
        if(cf[i]<=upper and cf[i]>=lower):
            extraction_z.append(z[i])
            extraction_x.append(x[i])
            extraction_y.append(y[i])

    extraction_df['x'] = extraction_x
    extraction_df['y'] = extraction_y
    extraction_df['z'] = extraction_z

    return extraction_df

def frequency_distribution(df, bin_width):
    # NaNを含む行を削除
    df = df.dropna(subset=['z'])

    # 階級幅を定数化
    bins = [i * bin_width for i in range(int(min(df['z']) // bin_width), int(max(df['z']) // bin_width) + 2)]
    df['z_class'] = pd.cut(df['z'], bins=bins, include_lowest=True)

    # 度数分布表を作成
    frequency_table = pd.value_counts(df['z_class']).sort_index().reset_index()
    frequency_table.columns = ['z_class', 'frequency']

    # 'z_class' 列の Interval オブジェクトから数値を取り出す
    frequency_table['z_class'] = frequency_table['z_class'].apply(lambda interval: interval.left)

    # 相対度数を計算
    frequency_table['relative_frequency'] = frequency_table['frequency'] / frequency_table['frequency'].sum()

    # 累計相対度数を計算
    frequency_table['cumulative_relative_frequency'] = frequency_table['relative_frequency'].cumsum()

    # 各点の累計相対度数を取得
    coords_with_cumulative_relative_frequency = pd.DataFrame(columns=['x', 'y', 'z', 'cumulative_relative_frequency'])

    for index, row in df.iterrows():
        z_value = row['z']
        try:
            cumulative_relative_frequency = frequency_table.loc[frequency_table['z_class'] == z_value, 'cumulative_relative_frequency'].iloc[0]
        except IndexError:
            cumulative_relative_frequency = None

        coords_with_cumulative_relative_frequency = coords_with_cumulative_relative_frequency.append({
            'x': row['x'],
            'y': row['y'],
            'z': row['z'],
            'cumulative_relative_frequency': cumulative_relative_frequency
        }, ignore_index=True)

    return coords_with_cumulative_relative_frequency