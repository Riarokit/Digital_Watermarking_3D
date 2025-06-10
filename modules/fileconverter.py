from modules.sharemodule import laspy,dt,np,pd,o3d,math,os
import modules.fileread as fr
import modules.preprocess as pp

# lasからpcdへの変換（o2d）
# 第1引数：lasfileまでのパス
# 第2引数：pcdファイルの出力先+ファイル名
# 戻り値：出力先までのパスと変換終了時の時刻を持つリスト
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


############ csvデータから２つのLiDARを合成する用のプログラムの実行ファイル例　############
# divide =  # divide [ms]ごとに重畳
# folderpath = ""

# # ２つのLidar用
# f.CsvLidarDivide("csv\\"+folderpath+".csv","csv\\"+folderpath+"_lidar1.csv", "csv\\"+folderpath+"_lidar2.csv")
# f.csv2pcd(divide,"csv\\"+folderpath+"_lidar1.csv","pcd\\"+folderpath+"_lidar1_divide="+str(divide))
# f.csv2pcd(divide,"csv\\"+folderpath+"_lidar2.csv","pcd\\"+folderpath+"_lidar2_divide="+str(divide))
# f.Lidarsyn_output("pcd\\"+folderpath+"_lidar1_divide="+str(divide), "pcd\\"+folderpath+"_lidar2_divide="+str(divide), "pcd\\"+folderpath+"_lidarsyn_"+str(divide), angle, x,y,z)
############ csvデータから２つのLiDARを合成する用のプログラムの実行ファイル例　############

#pcdデータをベクトル回転させる(Ryuuta)
def vectorrotate(points_df,degree,x_distance,y_distance,z_distance):
    points_df['X'] = points_df['X']*math.cos(math.radians(degree)) - points_df['Y']*math.sin(math.radians(degree))
    points_df['Y'] = points_df['X']*math.sin(math.radians(degree)) + points_df['Y']*math.cos(math.radians(degree))
    points_df['X'] += x_distance
    points_df['Y'] += y_distance
    points_df['Z'] += z_distance
    return points_df

#csvデータから二つのLidarデータを分割して出力(Ryuuta)
#csvData_path：処理したいcsvファイルまでのパス
#lidar1_outpath：分割したLidar片方のみを出力したいcsvファイルまでのパス
#lidar2_outpath：分割したLidarもう一方のみを出力したいcsvファイルまでのパス
def CsvLidarDivide(csvData_path,lidar1_outpath,lidar2_outpath):
    data = (pd.read_csv(csvData_path, usecols = ['Timestamp', 'X', 'Y', 'Z']))[::]
    lidar_timestamp = data['Timestamp'][10]%10000

    data_nm = data.to_numpy()
    coordinate_nm_2 = data_nm[np.any(data_nm%10000 == lidar_timestamp, axis=1)]
    data_nm = np.delete(data_nm, np.where(np.any(data_nm%10000 == lidar_timestamp, axis=1)), axis=0)
    points2_df = pd.DataFrame(data = coordinate_nm_2, columns=['Timestamp', 'X', 'Y', 'Z'])
    points2_df.to_csv(lidar2_outpath)

    points1_df = pd.DataFrame(data = data_nm, columns=['Timestamp', 'X', 'Y', 'Z'])
    points1_df.to_csv(lidar1_outpath)



#HAP用csvデータから二つのLidarデータを分割して出力(Ryuuta)
#csvData_path：処理したいcsvファイルまでのパス
#lidar1_inpath：分割したいLidar片方だけのcsvファイルまでのパス(0フレーム目のみ)
#lidar2_inpath：分割したいLidarもう一方のみのcsvファイルまでのパス(0フレーム目のみ)
#lidar1_outpath：分割したLidar片方のみを出力したいcsvファイルまでのパス
#lidar2_outpath：分割したLidarもう一方のみを出力したいcsvファイルまでのパス
def CsvLidarDivide_h(csvData_path,lidar1_outpath,lidar2_outpath):
    data = (pd.read_csv(csvData_path, usecols = ['Handle','Timestamp', 'X', 'Y', 'Z']))[::]
    lidar_handle1 = data['Handle'][1]
    print(lidar_handle1)
    data_nm = data.to_numpy()
    coordinate_nm_1 = data_nm[np.any(data_nm == lidar_handle1, axis=1)]
    data_nm = np.delete(data_nm, np.where(np.any(data_nm == lidar_handle1, axis=1)), axis=0)
    points1_df = pd.DataFrame(data = coordinate_nm_1, columns=['Handle','Timestamp', 'X', 'Y', 'Z'])
    # points1_df = points1_df.apply(vectorrotate, axis=1, degree=180, x_distance=4.45, y_distance=0, z_distance=0)
    points1_df.to_csv(lidar1_outpath)

    data_df = pd.DataFrame(data = data_nm, columns=['Handle','Timestamp', 'X', 'Y', 'Z'])
    lidar_handle2 = data_df['Handle'][2]
    print(lidar_handle2)
    coordinate_nm_2 = data_nm[np.any(data_nm == lidar_handle2, axis=1)]
    points2_df = pd.DataFrame(data = coordinate_nm_2, columns=['Handle','Timestamp', 'X', 'Y', 'Z'])
    points2_df.to_csv(lidar2_outpath)



# csvからpcdへの変換(Ryuuta)
# divide：1つのTimestampが1ms間隔であるため、divide[ms]ずつまとめて出力
# read_csv_path：読み込みたいcsvファイルのパス(ファイル名まで)
# output_folder_path：出力したいpcdファイルを保存するフォルダー　ファイル名はframe_i(i:0,1,2,3....)
def csv2pcd(divide, read_csv_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    data = (pd.read_csv(read_csv_path, usecols = ['Timestamp', 'X', 'Y', 'Z']))[::]
    data = data.drop([0])
    timestamp_nm = (data.drop('X',axis=1).drop('Y',axis=1).drop('Z',axis=1).to_numpy())
    timestamp_nm_uni = np.unique(timestamp_nm)
    timestamp_nm_uni = np.sort(timestamp_nm_uni)[::-1]
    coordinate_nm = data.to_numpy()

    a = len(timestamp_nm_uni)
    print(a)
    for i in range(a//divide-1):
        frame = coordinate_nm[np.any(coordinate_nm > timestamp_nm_uni[(i+1)*divide], axis=1), 1:4]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame)
        o3d.io.write_point_cloud(output_folder_path + "//frame_" + str(a//divide-i-1) + ".pcd", pcd)
        coordinate_nm = np.delete(coordinate_nm, np.where(np.any(coordinate_nm > timestamp_nm_uni[(i+1)*divide], axis=1)), axis=0)
    frame = coordinate_nm[np.any(coordinate_nm >= timestamp_nm_uni[len(timestamp_nm_uni)-1], axis=1), 1:4]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame)
    o3d.io.write_point_cloud(output_folder_path + "//frame_" + str(0) + ".pcd", pcd)

#二つのLiDARから取得したpcdデータを合成する(Ryuuta)
#folder_inpath1：合成したいpcdデータが存在しているフォルダまでのパス
#folder_inpath2：合成したいpcdデータが存在しているフォルダまでのパス
#folder_outputpath：合成したpcdデータを出力したいフォルダまでのパス
def Lidarsyn_output(folder_inpath1, folder_inpath2, folder_outpath, degree=0, x_distance=0, y_distance=0, z_distance=0):
    if not os.path.exists(folder_outpath):
        os.makedirs(folder_outpath)
    for i in range(sum(os.path.isfile(os.path.join(folder_inpath1, name)) for name in os.listdir(folder_inpath1))):
        pcd_1 = fr.ReadPCD(folder_inpath1+"//frame_"+str(i)+".pcd")
        pcd_2 = fr.ReadPCD(folder_inpath2+"//frame_"+str(i)+".pcd")
        pcd = pp.pcdsummarize(pcd_2, pcd_1, degree, x_distance, y_distance, z_distance)
        # o3d.visualization.draw_geometries([pcd], "syn")
        o3d.io.write_point_cloud(folder_outpath + "//frame_" + str(i) + ".pcd", pcd)

# o2d
# ファイルを合成するという処理内容はLidarsynと同じ
# FileSynは2台のLivox_SDKを用いて変換されたpcdを、位置補正を行ったうえで、合成したファイルを出力する
# 第1引数:Livox_SDKで出力したファイルがあるフォルダまでのパス
# 第2引数:出力したいフォルダまでのパス
# 第3引数:xの補正値
# 第4引数:yの補正値
def FileSyn(pcddir,outputpath,angle,x,y):
    tflist = []
    count = {}
    # judge = 0
    lv1list = []
    lv2list = []
    for name in os.listdir(pcddir):
        if not name[-8:] in count.keys():
            # tflist.append(name[-8:])
            count[name[-8:]]=1
            # print(tflist)
            print(count)
        else:
            count[name[-8:]] += 1
    print(count)
    for key in count:
        if count.get(key) > 2:
            tflist.append(key)
    # print(tflist)


    for i in range(sum(os.path.isfile(os.path.join(pcddir, name)) for name in os.listdir(pcddir))):
        lv=os.listdir(pcddir)[i]
        # if lv[-8:] == "0000.pcd":
        #     judge += 1
        #     print(lv)
        #     print("judge changed")
        # if judge == 1:
        if lv[-8:]==tflist[0]:
            pcd = fr.ReadPCD(pcddir+lv)
            lv1list.append(pcd)
        elif lv[-8:]==tflist[1]:
            pcd = fr.ReadPCD(pcddir+lv)
            lv2list.append(pcd)
    del lv1list[0]
    del lv2list[0]
    print("finish")
    print(f"len of 1:{len(lv1list)}")
    print(f"len of 2:{len(lv2list)}")
    # そのうちファイル数に応じてやる
    for j in range(len(lv1list)):
        pcd = pp.pcdsummarize(lv1list[j],lv2list[j],angle=angle, corrected_x=x,corrected_y=y)
        if len(str(j))==1:
            o3d.io.write_point_cloud(outputpath + "/frame_" + "000" + str(j) + ".pcd",pcd)
            print(outputpath + "/frame_" + "000" + str(j) + ".pcd")
        elif len(str(j))==2:
            o3d.io.write_point_cloud(outputpath + "/frame_" + "00" + str(j) + ".pcd",pcd)
            print(outputpath + "/frame_" + "00" + str(j) + ".pcd")
        elif len(str(j))==3:
            o3d.io.write_point_cloud(outputpath + "/frame_" + "0" + str(j) + ".pcd",pcd)
            print(outputpath + "/frame_" + "0" + str(j) + ".pcd")
        elif len(str(j))>3:
            o3d.io.write_point_cloud(outputpath + "/frame_" + str(j) + ".pcd",pcd)
            print(outputpath + "/frame_" + str(j) + ".pcd")
        

    # for i in range(sum(os.path.isfile(os.path.join(pcddir, name)) for name in os.listdir(pcddir))-1):
    #     print("---------------")
    #     lv1=os.listdir(pcddir)[i]
    #     lv2=os.listdir(pcddir)[i+1]
    #     print(f"lv1:{lv1}")
    #     print(f"lv2:{lv2}")
    #     if lv1[-8:]==tflist[0] and lv2[-8:]==tflist[1]:
    #         #重畳の処理
    #         pcd1 = fr.ReadPCD(pcddir+lv1)
    #         pcd2 = fr.ReadPCD(pcddir+lv2)
    #         pcd = pp.pcdsummarize(pcd1,pcd2,angle=math.pi,corrected_x=4.45,corrected_y=0)
    #         print(i)
    #         print(outputpath)
    #         print(f"lv1:{lv1}")
    #         print(f"lv2:{lv2}")
    #         # if i>999:
    #         #     pcddf = t.o3dtoDataFrame(pcd)
    #         #     print(pcddf['x'].max())
    #         #     print(pcddf['y'].max())
    #         #     print(pcddf['x'].min())
    #         #     print(pcddf['y'].min())
    #         #     o3d.visualization.draw_geometries([pcd],str(i))
    #         # print(f"{i}")
    #         if len(str(i))==1:
    #             o3d.io.write_point_cloud(outputpath + "/frame_" + "000" + str(i) + ".pcd",pcd)
    #         elif len(str(i))==2:
    #             o3d.io.write_point_cloud(outputpath + "/frame_" + "00" + str(i) + ".pcd",pcd)
    #         elif len(str(i))==3:
    #             o3d.io.write_point_cloud(outputpath + "/frame_" + "0" + str(i) + ".pcd",pcd)
    #         elif len(str(i))>3:
    #             o3d.io.write_point_cloud(outputpath + "/frame_" + str(i) + ".pcd",pcd)
    #         # i+=1#同じファイルを重畳させないため
    #         # lv1=os.listdir(pcddir)[i]
    #         # lv2=os.listdir(pcddir)[i+1]
    #         # print(f"lv1:{lv1}")
    #         # print(f"lv2:{lv2}")

    print("finish")

