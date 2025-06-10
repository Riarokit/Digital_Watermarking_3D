import open3d as o3d
import numpy as np
import math
from mpmath import mp
import scipy.fftpack
import copy
import time

"""
埋め込みがあるかどうかを判別するのみ
dctとidctのライブラリ使った版
処理速いしこっちのほうが安定してそう
"""

def make_all_voxels(voxels):
    """
    引数はget_voxelsで取得した点群のリスト。
    DCTするために、点群の存在しない場所についても輝度値を設定する。
    この戻り値はNxNxNの各ボクセルの輝度値を格納した配列(ndarray)
    """
    #ボクセルインデックスの最大値、最小値を求めて範囲を決める
    max_index = np.max([voxel.grid_index for voxel in voxels])
    min_index = np.min([voxel.grid_index for voxel in voxels])
    voxel_size=(max_index-min_index)+1
    #print("ボクセルサイズ：",voxel_size)

    #voxel_sizeの1で初期化したndarrayを用意
    voxel_all = np.ones((voxel_size,voxel_size,voxel_size))
    
    #voxel_dctにvoxelsの値を代入
    for voxel in voxels:
        #インデックスを範囲にシフト
        index = voxel.grid_index-min_index
        voxel_all[index[0],index[1],index[2]]=voxel.color[0]
        #print("0-1色",voxel_all[index[0],index[1],index[2]])
    return(voxel_all)

def dct_3d(voxel_dct):
    """
    引数はmake_all_voxelsで作ったNxNxNの配列(ndarray)
    この戻り値はNxNxNのDCT変換した配列(ndarray)
    ！引数をボクセルグリッド直接渡せるようにした方がいいかも
    """
    voxel_dctcoef = scipy.fftpack.dctn(voxel_dct,norm='ortho')
    return(voxel_dctcoef)

def idct_3d(voxel_dctcoef):
    voxel_dct = scipy.fftpack.idctn(voxel_dctcoef,norm='ortho')
    return(voxel_dct)

def generate_x(n,nembrate,same=True):
    """
    引数はDCT係数が入ってる配列の大きさNと非埋め込み領域の割合[非埋め込み領域/全体]
    第三引数で乱数が同じもの使うかランダムにするか決められる。
    NxNxNの、埋め込み領域に2値疑似乱数を格納した配列(ndarray)を生成する。
    非埋め込み領域には0が入ってる。
    """
    seed_time = int(time.time())
    nembregion = int(n*nembrate)
    if same==True:
        np.random.seed(0)
    else:
        np.random.seed(seed_time)
    x = np.random.choice([+1,-1],size=(n,n,n))
    for i in range(nembregion):
        for j in range(nembregion):
            for k in range(nembregion):
                x[i,j,k] = 0
    return(x)

def generate_a(x,w,psnr):
    """
    psnrの値に対応するαを生成する。
    ！論文の式(17)をΣを増やすだけで3次元に拡張できるかは謎
    (近似式でα^3とかなる可能性ある？)
    """
    N = len(w[0][0])
    A = 1  #pcdは輝度値を0-1で扱うので、多分ダイナミックレンジは1
    n = -1*(psnr/10)
    W = idct_3d(x*w)
    WW = W*W
    a = math.sqrt((math.pow(10,n)*N*N*N*A*A)/np.sum(WW))
    return(a)

def embed(dctcoef,nembrate):
    """
    透かしの埋め込みする関数
    ！a,wはもとの論文とかたどってもう一回ちゃんと考えたほうがいい
    """
    n = len(dctcoef[0][0])
    x = generate_x(n,nembrate)
    psnr = 40  #aを生成するときのPSNR[dB]
    w = np.abs(dctcoef)  #透かしの重み係数w(i,j)、W=|Y|らしい
    a = generate_a(x,w,psnr)  #埋め込み強度α、よくわからんけど輝度値が255までだから合わせるためにこれにしてる
    x = a*w*x
    dctcoef_emb = dctcoef+x
    return(dctcoef_emb,a,w)

def detect(voxel_dct2,nembrate,same=True):
    """
    第一引数は透かし埋め込み後のボクセルデータ(ndarray)、第二引数は非埋め込み領域の割合[非埋め込み領域/全体]
    第三引数で乱数が同じもの使うかランダムにするか決められる。
    透かしを検出する。
    戻り値はとりあえず、透かし埋め込み後のDCT係数、検出の値p、閾値Tp
    """
    #pを求める
    y = dct_3d(voxel_dct2)
    n = len(y[0][0])
    x = generate_x(n,nembrate,same)
    #！ここからちゃんと動いてるかわからんから要動作確認
    yx=y*x
    count = (n*n*n)-int(n*nembrate)*int(n*nembrate)*int(n*nembrate)
    sum = np.sum(yx)
    p=sum/count

    #Tpを求める
    kara = np.ones((n,n,n))
    nembregion = int(n*nembrate)
    for i in range(nembregion):
        for j in range(nembregion):
            for k in range(nembregion):
                kara[i,j,k] = 0
    y_tilde = y*kara
    y_tilde = y_tilde*y_tilde
    sum = np.sum(y_tilde)
    bunnsann =  (1/math.pow((3*n*n*n),2))*sum
    Tp = 3.97*math.sqrt(2*bunnsann)  #しきい値
    return(y,p,Tp)

def visualize(voxel_dct2,voxel_num,voxel_size):
    print("視覚化開始")
    print("透かし入りボクセル生成開始")
    start_genvoxel = time.time()
    vis = copy.deepcopy(voxel_dct2)

    #間引きするボクセルの印として設定する輝度値
    #(輝度値は大きい順に除外するのでこの値は0より小さい値)
    iran = -1

    #ボクセルの間引きのためにいらないボクセルの輝度値をiranの値に変更
    for i in range(vis.size-voxel_num):
        idx = np.unravel_index(np.argmax(vis),vis.shape)
        vis[idx] = iran

    n = len(vis[0][0])
    # 空のVoxelGridを作成
    after_voxels = o3d.geometry.VoxelGrid()
    after_voxels.voxel_size=voxel_size

    """
    #ボクセルを除外するときのしきい値
    threshold = 0.1
    """

    #空のボクセルグリッドに逆変換したボクセル入れる
    for i in range(n):
        for j in range(n):
            for k in range(n):
                voxel_index = np.array([i,j,k],dtype=np.int32)
                voxel_color = np.array([vis[i,j,k],vis[i,j,k],vis[i,j,k]],dtype=np.float64)
                if np.any(voxel_color == iran):
                    continue
                else:
                    #voxel_color = np.array([0,0,0],dtype=np.float64)   #！なんかうまくいかないから応急処置
                    new_voxel = o3d.geometry.Voxel(grid_index=voxel_index,color=voxel_color)
                    after_voxels.add_voxel(new_voxel)
                """
                #しきい値でボクセルを間引き
                if np.all(voxel_color < threshold):
                    continue
                else:
                    new_voxel = o3d.geometry.Voxel(grid_index=voxel_index,color=voxel_color)
                    after_voxels.add_voxel(new_voxel)
                """

    after_num = after_voxels.get_voxels()
    if voxel_num == len(after_num):
        print("ボクセル数：成功")
    else:
        print("ボクセル数：失敗")
    
    print("視覚化完了")
    print("処理時間：",time.time()-start_genvoxel,"\n")
    
    return(after_voxels)

def comp(dctcoef_emb,rate):
    """
    圧縮の簡単なサンプル
    """
    com = copy.deepcopy(dctcoef_emb)
    # データの形状を取得
    N = com.shape[0]

    # 各ボクセルの原点からの距離を計算
    indices = np.indices((N,N,N))
    distances = np.sqrt(indices[0]**2 + indices[1]**2 + indices[2]**2)

    # 距離とボクセルのインデックスを一緒にソート
    sorted_indices = np.unravel_index(np.argsort(distances, axis=None), distances.shape)

    # 圧縮するボクセル数を計算
    num_voxels = np.prod(com.shape)
    num_compress = int(rate * num_voxels)

    # 最も距離が大きいボクセルから順に0に置き換え
    com[sorted_indices[0][-num_compress:], sorted_indices[1][-num_compress:], sorted_indices[2][-num_compress:]] = 0

    return com

def vis_cust(after_voxels) :
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    vis.get_render_option().background_color = np.asarray([0.5372549019607843,0.7647058823529412,0.9215686274509804])

    vis.add_geometry(after_voxels)

    vis.run()

    vis.destroy_window()