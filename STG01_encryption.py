import modules.fileconverter as fc
import modules.fileread as fr
import modules.tools as t
import modules.preprocess as pp
import modules.clustering as cl
import pandas as pd
import random as rd
import numpy as np
import STG01_global_value as g
    

### csvファイルの読み込み
csv_in_path_bef_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20240513/extract_people.csv"
df = g.to_df(csv_in_path_bef_stego)

### 共通鍵の確認
char_bin_list = []      #ステガノグラフィで送信したい文字列のビットを1文字ずつリストに格納
print("ランダムなリスト（受信側との共通鍵）の作成")
key_list = g.key(df)

### 文字列をASCIIコードに変換
print("10個の文字列の7ビット表示")
for i in range(len(g.char)):
    bin_char = bin(ord(g.char[i]))[2:]       #ord関数で文字を数字に変換したものを、bin関数で2進数に変換
    print(bin_char)
    for j in range(7):                       #rangeの7は英語1文字がASCIIで7ビットだから
        char_bin_list.append(bin_char[j])    #char_bin_listは送信したい文字列を2進数に変換したものを1ビットずつ格納するリスト
print("charのバイナリのリスト")
print(char_bin_list)

### 値の交換
for i in range(len(key_list)):
    key_now = key_list[i]                  #key_listは送信したい文字列を隠す行を格納したリスト
    col_Orix = df.columns.get_loc("Ori_x")   #col_Orixは"Ori_x"列が何列目かを格納する変数
    col_X = df.columns.get_loc("X")          #col_Xは"X"列が何列目かを格納する変数
    if(char_bin_list[i] == '0'):
        if(df.iloc[key_now, col_Orix] % 2 == 1):
            df.iloc[key_now, col_Orix] += 1
            df.iloc[key_now, col_X] = df.iloc[key_now, col_Orix] / 1000
    elif(char_bin_list[i] == '1'):
        if(df.iloc[key_now, col_Orix] % 2 == 0):
            df.iloc[key_now, col_Orix] += 1
            df.iloc[key_now, col_X] = df.iloc[key_now, col_Orix] / 1000

### できてるか確認
print("1個目: " + char_bin_list[0], df.iloc[key_list[0], col_Orix], df.iloc[key_list[0], col_X])
print("2個目: " + char_bin_list[1], df.iloc[key_list[1], col_Orix], df.iloc[key_list[1], col_X])
print("3個目: " + char_bin_list[2], df.iloc[key_list[2], col_Orix], df.iloc[key_list[2], col_X])


### csvファイルの出力
df.to_csv("C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20240513/extract_people_stego.csv", index=False)

### stego前ファイルをcsvからpcdにコンバート
csv_in_path_bef_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20240513/extract_people.csv"
pcd_out_path_bef_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240513/extract_people"
fc.csv2pcd(1, csv_in_path_bef_stego, pcd_out_path_bef_stego)

### stego後ファイルをcsvからpcdにコンバート
csv_in_path_aft_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20240513/extract_people_stego.csv"
pcd_out_path_aft_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240513/extract_people_stego"
fc.csv2pcd(1, csv_in_path_aft_stego, pcd_out_path_aft_stego)

### pcdファイルの読み込み
pcdpath_bef_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240513/extract_people/frame_0.pcd"
pcd_bef_stego = fr.ReadPCD(pcdpath_bef_stego)
pcdpath_aft_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240513/extract_people_stego/frame_0.pcd"
pcd_aft_stego = fr.ReadPCD(pcdpath_aft_stego)

### stego前のpcdファイルを可視化
t.VisualizationPCD(pcd_bef_stego, "before_stego")

### stego後のpcdファイルを可視化
t.VisualizationPCD(pcd_aft_stego, "after_stego")






