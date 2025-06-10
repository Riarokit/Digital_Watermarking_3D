from modules.sharemodule import o3d,np,pyrsc,copy,pd,os
from modules.shareclass import Person
import modules.fileread as fr
import modules.tools as t

#　背景差分法(o2d)
# 第1引数：背景のpcdData(pathではなくてDataそのものを与える)
# 第2引数：背景に加えてターゲットを含む(pathではなくてDataそのものを与える)
def RemoveBackground(background,target,thresh):
    # o3d.visualization.draw_geometries([background],"Only Background")
    # o3d.visualization.draw_geometries([background+target],"Include Background")
    distance = target.compute_point_cloud_distance(background)
    distance = np.array(distance)
    moveIndex = np.where(distance>thresh)[0]#初期0.1。なんでこれでうまくいく？なんのリスト？
    # print(len(moveIndex))
    # print(moveIndex)
    
    removedpcd = target.select_by_index(moveIndex)
    # print("OK")
    # o3d.visualization.draw_geometries([removedpcd],"Removed Background")
    return removedpcd

#2台のLiDARから得た点群データを1つにまとめる(fukui)
#引数は2つのpcdデータと回転角度，x座標とy座標の補正値
def pcdsummarize(pcd1, pcd2, angle, corrected_x, corrected_y, corrected_z=0):
    rad = np.radians(angle)
    points1 = np.asarray(pcd1.points)

    # pcd1を回転させる
    R = np.array([[np.cos(rad), -np.sin(rad), 0],
                  [np.sin(rad), np.cos(rad), 0],
                  [0, 0, 1]])
    # pcd1.rotate(R)
    points1 = np.dot(points1, R)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)

    #pcd1のx座標をLiDARの間隔分差し引く
    for i in range(len(pcd1.points)):
        pcd1.points[i][0] += corrected_x
    
    #pcd1のy座標補正
    for i in range(len(pcd1.points)):
        pcd1.points[i][1] += corrected_y

    #pcd1のz座標補正
    for i in range(len(pcd1.points)):
        pcd1.points[i][2] += corrected_z
        
    # pcd2と結合する
    merged_pcd = pcd1 + pcd2
    
    return merged_pcd

# radius outlier removal（o2d）
def display_inlier_outlier(cloud, ind):#(o2d)
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],)
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
    # o3d.visualization.draw_geometries([pcd],"Before DownSampling")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=size)
    # o3d.visualization.draw_geometries([voxel_down_pcd],"After DownSampling")
    print("Size of pcd : {}".format(voxel_down_pcd))
    return voxel_down_pcd

# pcdデータの範囲選択（o2d）
# 第1引数：pcdのデータ(ReadPCDで受け取ったもの)
# 第2,3,4引数：抽出したい範囲をリストで指定する
# 戻り値：抽出されたpcdData
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

# o2d
#　処理する内容はFramePiledと同じ（多分）
#  FramePiledはcsvファイルで変換したときに用いるのに対して、FramePilingはLivox_SDKで変換されたpcdファイルを、conver
def FramePiling(syned_pcddir,outputpath,split):
    pcdlist = os.listdir(syned_pcddir)
    pcdlist = [pcdlist[i:i+split] for i in range(0, len(pcdlist), split)]
    print(len(pcdlist))
    for ind,name in enumerate(pcdlist):
        pcd = o3d.geometry.PointCloud()
        print(name)
        for i in range(len(name)):
            readpcd = fr.ReadPCD(syned_pcddir + "/" + name[i])
            pcd += readpcd
            # そのうちファイル数に応じてやる
        if len(str(split+ind*split))==1:
            o3d.io.write_point_cloud(outputpath + "/000" + str(split+ind*split) + "frames.pcd",pcd)
            print(outputpath + "/" + str(split+ind*split) + "frames.pcd")
        elif len(str(split+ind*split))==2:
            o3d.io.write_point_cloud(outputpath + "/00" + str(split+ind*split) + "frames.pcd",pcd)
            print(outputpath + "/" + str(split+ind*split) + "frames.pcd")
        elif len(str(split+ind*split))==3:
            o3d.io.write_point_cloud(outputpath + "/0" + str(split+ind*split) + "frames.pcd",pcd)
            print(outputpath + "/" + str(split+ind*split) + "frames.pcd")
        elif len(str(split+ind*split))>3:
            o3d.io.write_point_cloud(outputpath + "/" + str(split+ind*split) + "frames.pcd",pcd)
            print(outputpath + "/" + str(split+ind*split) + "frames.pcd")

# angle（角度調整用ファイル）から法線を計算する（o2d）
def CalcNormal(anglepcd):#,normal_rad=1,max_nn=3000
    angle_df = t.o3dtoDataFrame(anglepcd)
    angle_df_len = len(angle_df)
    # 検索半径を決定するため
    range_df = pd.DataFrame()
    range_df['max'] = angle_df.max()
    range_df['min'] = angle_df.min()
    range_df['range'] = range_df['max']-range_df['min']
    # print(range_df)
    # check = input(f"Length of angle_df:{angle_df_len}¥n Max value of df:{angle_df_max}¥n Min value of df:{angle_df_min}")
    
    # 検索半径と最大最近傍
    anglepcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=range_df['range'].max()/2, max_nn=int(angle_df_len/2)))
    # t.VisualizationPCD(anglepcd,normal=True)
    # t.Get3dGraph(angle_df)
    normals = np.asarray(anglepcd.normals)
    normal_df = pd.DataFrame({'x':normals[:,0],
                            'y':normals[:,1],
                            'z':normals[:,2]})
    # print(normal_df.mean())
    # s = pd.Series([0, 0, 0],index=['x','y','z'])
    vec_df = normal_df.mean()
    # vec_df = pd.concat([s,vec_df],axis=1).T
    # t.Get3dGraph(vec_df)
    # 初期化
    return vec_df

# CalcNormalで計算した法線を使って補正（o2d）
def HorizontalAdjust(pcd,angle_df):
    # print(angle_df)
    # 下に傾けた場合
    adjusted_df = t.o3dtoDataFrame(pcd)
    xz_rad = np.arctan(angle_df['z']/angle_df['x'])# 水平からの角度
    # -90<xz_rad<0のとき
    if (-90<np.rad2deg(xz_rad)) and (np.rad2deg(xz_rad)<0):
        rotate_rad = np.deg2rad(-90) -xz_rad
        # print(rotate_rad)
    else:
        check = input(f"角度が-90<xz_rad<0でない")
    x = adjusted_df['x']
    z = adjusted_df['z']
    adjusted_df['x'] = x * np.cos(rotate_rad) - z * np.sin(rotate_rad)
    adjusted_df['z'] = x * np.sin(rotate_rad) + z * np.cos(rotate_rad)
    # print(f"x:{x}, z:{z}")
    # print(f"x_r:{vec['x']}, z:{vec['z']}")
    # vec = pd.concat([s,vec],axis=1).T
    # t.Get3dGraph(vec)
    # rotate_rad = np.arctan(vec['z']/vec['x'])
    # print(rotate_rad)
    adjusted_df['z'] = adjusted_df['z']-adjusted_df['z'].min()
    print(adjusted_df)
    t.Get3dGraph(adjusted_df,zlim=[adjusted_df['z'].mean()-1,adjusted_df['z'].mean()+1],title="adjusted_df")
    adjustedpcd = t.DataFrametoO3d(adjusted_df)
    """
    # 法線確認用
    adjustedpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=3000))
    # t.VisualizationPCD(adjustedpcd,normal=True)
    normals = np.asarray(adjustedpcd.normals)
    normal_df = pd.DataFrame({'x':normals[:,0],
                            'y':normals[:,1],
                            'z':normals[:,2]})
    # t.VisualizationPCD(adjustedpcd)
    t.Get3dGraph(angle_df,zlim=[-1.5,0.5])
    """
    return adjustedpcd