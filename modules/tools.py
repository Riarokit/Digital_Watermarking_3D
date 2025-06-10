from modules.sharemodule import copy,np,pd,plt,squareform,pdist,o3d
from modules.shareclass import Person
from modules import preprocess  as pp
from modules import tools as t

#PCALinTransで使用する
from sklearn.preprocessing  import MinMaxScaler
from sklearn.decomposition import PCA

# open3dからDataFrame（o2d）
def o3dtoDataFrame(pcd):
    pcd_deepcopied = copy.deepcopy(pcd)
    pcd_coordinate = np.asanyarray(pcd_deepcopied.points)
    # print("MATLAB {}".format(pcd_coordinate))
    pcd_df = pd.DataFrame(data=pcd_coordinate,columns=['x','y','z'])
    return pcd_df

# DataFrameからo3d（o2d）
#　第1引数：x,y,z座標を持つDataFrame
#  戻り値：pcd
# 例えばこんな感じに使う
# for L in cluster_df['label'].unique():
#         df = cluster_df[cluster_df['label']==L]
#         pcd = t.DataFrametoO3d(df)
#         o3d.visualization.draw_geometries([pcd],"Check")
def DataFrametoO3d(pcd_df):
    print("Converting Data to o3d")
    pcd = o3d.geometry.PointCloud()
    coordinate = pcd_df[['x','y','z']]
    pcd.points = o3d.utility.Vector3dVector(coordinate.values)
    # o3d.visualization.draw_geometries([pcd], "Check")
    return pcd

def VisualizationPCD(pcdData,title="",normal=False):
    o3d.visualization.draw_geometries([pcdData],title,point_show_normal=normal)

# 3次元グラフ表示（o2d,ishida）
def Get3dGraph(df, title="", plotsize=3, xlim=[], ylim=[], zlim=[], size=[]):
    fig = plt.figure()
    data = fig.add_subplot(projection='3d')
    data.scatter(df["x"], df["y"], df["z"],s=plotsize)
    data.set_xlabel('X Label')
    data.set_ylabel('Y Label')
    data.set_zlabel('Z Label')
    if xlim==[] and ylim==[]:
        range_y = df['y'].max()-df['y'].min()
        range_x = df['x'].max()-df['x'].min()
        if range_y > range_x:
            xlim = [df['x'].max()-range_y-1.0, df['x'].max()+1.0]
            ylim = [df['y'].min()-1.0, df['y'].max()+1.0]
            plt.xlim(xlim)
            plt.ylim(ylim)
        else:
            xlim = [df['x'].min()-1.0, df['x'].max()+1.0]
            ylim = [df['y'].max()-range_x-1.0, df['y'].max()+1.0]
            plt.xlim(xlim)
            plt.ylim(ylim)
    else:
        if (ylim != []):
            plt.ylim(ylim)
            range_y = max(ylim)-min(ylim)
            xlim = [df['x'].max()-range_y,df['x'].max()]
            plt.xlim(xlim)
        if (xlim != []):
            plt.xlim(xlim)
            range_x = max(xlim)-min(xlim)
            ylim = [df['y'].max()-range_x,df['y'].max()]
            plt.ylim(ylim)
    if (zlim != []):
        data.set_zlim(zlim)
    if (size != []):
        fig.set_size_inches(size[0], size[1])
    else:
        fig.set_size_inches(10, 10)
    if (title !=""):
        plt.title(title)
    plt.show()

# バウンディングボックスるを表示する(o2d)
def Bounding_box(pcdData):
    #テスト用で鳥のデータを使ってる.実際に使うときはコメントアウト
    dataset = o3d.data.EaglePointCloud()
    pcd = o3d.io.read_point_cloud(dataset.path)

    # 実際に使うときはコメントアウトを外す
    # pcd = pcdData

    # Flip it, otherwise the pointcloud will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    print(pcd)
    axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box.color = (1, 0, 0)
    oriented_bounding_box = pcd.get_oriented_bounding_box()
    oriented_bounding_box.color = (0, 1, 0)
    print(
        "Displaying axis_aligned_bounding_box in red and oriented bounding box in green ..."
    )
    o3d.visualization.draw_geometries(
        [pcd, axis_aligned_bounding_box, oriented_bounding_box])

    # pcd = backgroundpcd + pcd
    # o3d.visualization.draw_geometries(
    #     [pcd, axis_aligned_bounding_box, oriented_bounding_box])

# 位置合わせ確認用（o2d）
# 第1引数：pcd
# 第2~4引数：範囲選択
def PositionJustification(pcd,eps=0.1, min_points=10,xlim=[],ylim=[],zlim=[]):
    df = o3dtoDataFrame(pcd)
    pcd = pp.SelectPCD(pcd,xlim=xlim,ylim=ylim,zlim=zlim)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        cluster_df = pd.DataFrame(data=o3dtoDataFrame(pcd),columns=['x','y','z'])
        cluster_df["label"] = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points,print_progress=True))#クラスタごとにラベル付けしたものを配列で返す
        # clusterLabels = np.array(pcdData.cluster_dbscan(eps=epsm, min_points=points,print_progress=True))#クラスタごとにラベル付けしたものを配列で返す
        cluster_df = cluster_df[cluster_df["label"]>-1]#ノイズを除去
        for L in cluster_df['label'].unique():
            df = cluster_df[cluster_df['label']==L]
            pcd = DataFrametoO3d(df)
            o3d.visualization.draw_geometries([pcd],"Check")

# フレーム内のクラスタを線形変換(o2d)
# 第1引数:人間クラスタのpcd
#ExtraHumanの中でcluster_dfを受け取る
def LinTrans(pcd,layer,percent,xlim=[-5,5],ylim=[-5,5],zlim=[-5,5],size=[10,10]):#0200用
    coordinate = o3dtoDataFrame(pcd)
    LinTransferedCoordinate_df = pd.DataFrame(columns=['x','y','z'])
    coord1 = None
    coord2 = None
    if pcd is None:
        enter = input("Can't LinTrans, because pcd is NoneType.Please press Enter: ")
        return LinTransferedCoordinate_df
    print(f"coordinate:{coordinate[['x','y','z']]}")
    # Get3dGraph(df=coordinate,xlim=[xlim[0],xlim[1]],ylim=[ylim[0],ylim[1]],zlim=[zlim[0],zlim[1]],size=[size[0],size[1]])
    under_df = coordinate[(coordinate['z']-coordinate['z'].min())<(coordinate['z'].max()-coordinate['z'].min())/3]
    # Get3dGraph(df=under_df,xlim=[xlim[0],xlim[1]],ylim=[ylim[0],ylim[1]],zlim=[zlim[0],zlim[1]],size=[size[0],size[1]])
    underlay = {}
    underlay_df = pd.DataFrame(columns=[['x','y']])
    walkwid = 0
    for underheight in range(layer):
        underlay = pcd.coordinate[(pcd.coordinate['z'].max()-pcd.coordinate['z'].min())*(1-percent)<(pcd.coordinate['z']-pcd.coordinate['z'].min())]
        underlay[str(underheight)] = under_df[(((under_df['z'].max()-under_df['z'].min())*(1-((underheight/layer)+1)*0.1))<(under_df['z']-under_df['z'].min()))
                                            &((under_df['z']-under_df['z'].min())<((under_df['z'].max()-under_df['z'].min())*(1-(underheight/layer)*0.1)))]
        if len(underlay[str(underheight)])==0:
            print(f"Length of underlay[{underheight}]:{len(underlay[str(underheight)])}")
            continue
        else:
            underlay_df = underlay[str(underheight)]
            # print(f"underlay_df:{underlay_df}")
            tmpwalkwid = squareform(pdist(underlay_df[['x','y']]))#下半身の歩幅で向き推定
            maxWid = np.max(tmpwalkwid)
            # print(f"maxWid:{maxWid}")
            if maxWid > walkwid:
                walkwid = maxWid
                maxInd = np.unravel_index(np.argmax(tmpwalkwid),tmpwalkwid.shape)
                coord1 = underlay_df.iloc[maxInd[0]][['x','y']]# 最大幅を与える座標1
                coord2 = underlay_df.iloc[maxInd[1]][['x','y']]# 最大幅を与える座標2
    print(walkwid)
    if coord1 is None or coord2 is None:
        print("len(coord1)==0 or len(coord2)==0")
        return LinTransferedCoordinate_df
    # 立ち状態でないとき
    if walkwid>0.45:
        print("Situation : Walking")
        vec = coord2-coord1
        print(f"vec:{vec}")
        theta = np.arctan(vec['y']/vec['x'])
        print(f"theta:{theta}")
        cos = np.cos(theta)
        sin = np.sin(theta)
        # 反時計回り
        R = np.array([[cos,-sin],
                    [sin,cos]])
        R = np.linalg.inv(R)
        # LinTransferedCoordinate_df = pd.DataFrame(columns=['x','y'])
        LinTransferedCoordinate_df['x'] = coordinate[['x','y']].apply(lambda x: np.dot(R,[x['x'],x['y']])[0],axis=1)
        LinTransferedCoordinate_df['y'] = coordinate[['x','y']].apply(lambda x: np.dot(R,[x['x'],x['y']])[1],axis=1)
        LinTransferedCoordinate_df['z'] = coordinate['z']
        print(f"LinTransferedCoordinate_df:{LinTransferedCoordinate_df}")
        # Get3dGraph(df=LinTransferedCoordinate_df,xlim=[xlim[0],xlim[1]],ylim=[ylim[0],ylim[1]],zlim=[zlim[0],zlim[1]],size=[size[0],size[1]])
    else:
        print("Situation : Stanging")
        # maxWidIndex = np.where(walkwid == walkwid)[0]
        # print(f"maxWidIndex : {maxWidIndex}")
        # 傾き
        vec = coord2-coord1
        # 垂直の傾き
        tmp = vec['y']
        vec['y'] = vec['x']
        vec['x'] = -tmp
        print(f"vec:{vec}")
        theta = np.arctan(vec['y']/vec['x'])
        print(f"theta:{theta}")
        
        cos = np.cos(theta)
        sin = np.sin(theta)
        
        R = np.array([[cos,-sin],
                    [sin,cos]])
        R = np.linalg.inv(R)
        # LinTransferedCoordinate_df = pd.DataFrame(columns=['x','y'])
        LinTransferedCoordinate_df['x'] = coordinate[['x','y']].apply(lambda x: np.dot(R,[x['x'],x['y']])[0],axis=1)
        LinTransferedCoordinate_df['y'] = coordinate[['x','y']].apply(lambda x: np.dot(R,[x['x'],x['y']])[1],axis=1)
        # print(f"LinTransferedCoordinate_df:{LinTransferedCoordinate_df}")
        LinTransferedCoordinate_df['z'] = coordinate['z']
        print(f"LinTransferedCoordinate_df:{LinTransferedCoordinate_df}")
        # Get3dGraph(df=LinTransferedCoordinate_df,xlim=[xlim[0],xlim[1]],ylim=[ylim[0],ylim[1]],zlim=[zlim[0],zlim[1]],size=[size[0],size[1]])
    return LinTransferedCoordinate_df

# 人を抽出(o2d)
# 第1引数:pcd

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
                    candidate_pcd = t.DataFrametoO3d(candidate.coordinate)
                    o3d.visualization.draw_geometries([candidate_pcd], "Candidate pcd")
        notnoiseIndex = cluster_df.index[cluster_df['label'].isin(np.asarray(notnoiseList))]
        
        pcdData = pcdData.select_by_index(notnoiseIndex)
        # o3d.visualization.draw_geometries([pcdData], "After Extracting")
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
            heightline[str(height)] = headCoordinate[(((headCoordinate['z'].max()-headCoordinate['z'].min())*(1-((height/layer)+1)*0.1))<(headCoordinate['z']-headCoordinate['z'].min()))
                                             &((headCoordinate['z']-headCoordinate['z'].min())<((headCoordinate['z'].max()-headCoordinate['z'].min())*(1-(height/layer)*0.1)))]
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
    return postcandidateList # 人っぽい形のものに絞った

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

# o2d
# 胸部の点群に対して主成分分析を行い、第2軸（進行方向）のベクトルを州出する。
# そのベクトルを線形変換の回転に用いる
# 第1引数：クラスタのpcdのDataFrame
# 第2引数：肩までのパーセンタイル
def PCALinTrans(pcd_df,percent=0.3):
    LinTransferedCoordinate_df = pd.DataFrame(columns=['x','y','z'])
    # pcd_df = t.o3dtoDataFrame(pcd)
    # t.Get3dGraph(df=pcd_df,title="Before PCALinTrans",xlim=[],ylim=[],zlim=[],size=[10,10])
    # plt.scatter(x=pcd_df['x'],y=pcd_df['y'])
    sholder_df = pcd_df[((pcd_df['z'].max()-pcd_df['z'].min())*(1-percent)<(pcd_df['z']-pcd_df['z'].min())) & ((pcd_df['z']-pcd_df['z'].min())<(pcd_df['z'].max()-pcd_df['z'].min())*(1-(percent-0.1)))]
    # 肩周りの抽出確認
    # t.Get3dGraph(df=sholder_df,title="Sholder",size=[10,10])
    # sholder_df = sholder_df.iloc[:, :2].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    # print(sholder_df.head())

    # print(f"sholder_df:{sholder_df}")
    # plt.figure(figsize=(6, 6))
    # plt.scatter(sholder_df['x'], sholder_df['y'], alpha=0.8)
    # plt.grid()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()
    print(f"sholder_df:{sholder_df}")
    if len(sholder_df) < 2:
        return pcd_df
    pca = PCA(n_components=2)
    pca.fit(sholder_df[['x','y']])
    # feature = pca.transform(sholder_df[['x','y']])
    # print(f"feature:{feature}")
    # plt.figure(figsize=(6, 6))
    # plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
    # plt.grid()
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.show()

    pca_df = pd.DataFrame(pca.components_, columns=['x','y'], index=["PC{}".format(x + 1) for x in range(2)])
    print(pca_df)
    theta = np.arctan(pca_df.at['PC2','y']/pca_df.at['PC2','x'])
    print(f"theta:{theta}")
    cos = np.cos(theta)
    sin = np.sin(theta)
    # 反時計回り
    R = np.array([[cos,-sin],
                [sin,cos]])
    R = np.linalg.inv(R)
    LinTransferedCoordinate_df['x'] = pcd_df[['x','y']].apply(lambda x: np.dot(R,[x['x'],x['y']])[0],axis=1)
    LinTransferedCoordinate_df['y'] = pcd_df[['x','y']].apply(lambda x: np.dot(R,[x['x'],x['y']])[1],axis=1)
    LinTransferedCoordinate_df['z'] = pcd_df['z']
    # t.Get3dGraph(df=LinTransferedCoordinate_df,title="After PCALinTrans",xlim=[],ylim=[],zlim=[],size=[10,10])
    return LinTransferedCoordinate_df