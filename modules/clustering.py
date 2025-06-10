from modules.sharemodule import o3d,np,plt
from modules.shareclass import Person
# クラスタリング（色付け）されたo3dオブジェクトを返し、可視化まで行う（o2d）
def Clustering(pcdData,epsm,points):
    o3d.visualization.draw_geometries([pcdData], "Before Clustering")
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
        newpcdData = pcdData.select_by_index(np.delete(np.arange(len(pcdData.points)),noiseIndex))#全点群の数分の等間隔のリストから、ノイズのインデックスに対応するものを削除->ノイズ出ない点（インデックス）の抽出
        print(f"pcdData:{pcdData}")
        print(f"newpcdData:{newpcdData}")
        if len(clusterLabels_noNoise)==0:
            return print('ノイズしかなかった')
        max_label = clusterLabels_noNoise.max()
        print("point cloud has {} clusters".format(max_label+1))
        colors = plt.get_cmap("tab20")(clusterLabels_noNoise/(max_label if max_label > 0 else 1))
        colors[clusterLabels_noNoise < 0] = 0
        newpcdData.colors = o3d.utility.Vector3dVector(colors[:,:3])
        o3d.visualization.draw_geometries([newpcdData], "After Clustering")
        return newpcdData
    
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

