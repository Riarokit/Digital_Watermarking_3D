import modules.fileconverter as fc
import modules.fileread as fr
import modules.tools as t
import modules.preprocess as pp
import modules.clustering as cl
import pandas as pd
import random as rd
import numpy as np
import STG03_global_value as g
import json
import io
from PIL import Image
from bitarray import bitarray

### csvファイルの読み込み
csv_in_path_bef_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20240513/extract_people.csv"
df = g.to_df(csv_in_path_bef_stego)

### 写真をビット列に変換
print("写真のビット列への変換")
file_path = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/london.jpg"
img = Image.open(file_path, mode='r')
resized_img = img.resize((32, 32))  # 32x32に変形
img_bytes = io.BytesIO()
resized_img.save(img_bytes, format='png')
img_bytes = img_bytes.getvalue()
# print("写真のバイト列: ", img_bytes)
print("写真のバイト列の長さ: ", len(img_bytes))
print(img_bytes)
byte_length = len(img_bytes)

### 共通鍵の確認
char_bin_list = []      #ステガノグラフィで送信したい文字列のビットを1文字ずつリストに格納
key_list = g.key(df, byte_length)

### numpyのデータ型を標準のPythonデータ型に変換してからリストをファイルに書き出し(jsonではnumpy型で書き出しできない)
def convert_to_builtin_type(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError
with open('shared_variable.json', 'w') as f:
    json.dump([list(map(convert_to_builtin_type, item)) for item in key_list], f)

### 文字列をASCIIコードに変換
format_bit = "0"
format_bit += str(g.bit_num)
format_bit += "b"
for i in range(len(img_bytes)):
    bin_char = format(ord(chr(img_bytes[i])), format_bit)       #ord関数で文字を数字に変換したものを、bin関数で2進数に変換
    if 0 <= i <= 10:
        print(i)
        print(bin_char)
        print(chr(int(bin_char, 2)))
    for j in range(g.bit_num):                       #rangeの7は英語1文字がASCIIで7ビットだから
        char_bin_list.append(bin_char[j])    #char_bin_listは送信したい文字列を2進数に変換したものを1ビットずつ格納するリスト

### 値の交換
for i in range(len(key_list)):                  #key_listは送信したい文字列を隠す行を格納したリスト
    if(char_bin_list[i] == '0'):
        if(df.iloc[key_list[i][0], key_list[i][1]] % 2 == 1):
            df.iloc[key_list[i][0], key_list[i][1]] += 1
            df.iloc[key_list[i][0], key_list[i][1]-5] = df.iloc[key_list[i][0], key_list[i][1]] / 1000
    elif(char_bin_list[i] == '1'):
        if(df.iloc[key_list[i][0], key_list[i][1]] % 2 == 0):
            df.iloc[key_list[i][0], key_list[i][1]] += 1
            df.iloc[key_list[i][0], key_list[i][1]-5] = df.iloc[key_list[i][0], key_list[i][1]] / 1000

### できてるか確認
print("1個目: ", char_bin_list[0], df.iloc[key_list[0][0], key_list[0][1]], df.iloc[key_list[0][0], key_list[0][1]-5])
print("2個目: ", char_bin_list[1], df.iloc[key_list[1][0], key_list[1][1]], df.iloc[key_list[1][0], key_list[1][1]-5])
print("3個目: ", char_bin_list[2], df.iloc[key_list[2][0], key_list[2][1]], df.iloc[key_list[2][0], key_list[2][1]-5])


### csvファイルの出力
df.to_csv("C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20240513/extract_people_stego.csv", index=False)

# ### stego前ファイルをcsvからpcdにコンバート
# csv_in_path_bef_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20240513/extract_people.csv"
# pcd_out_path_bef_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240513/extract_people"
# fc.csv2pcd(1, csv_in_path_bef_stego, pcd_out_path_bef_stego)

# ### stego後ファイルをcsvからpcdにコンバート
# csv_in_path_aft_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20240513/extract_people_stego.csv"
# pcd_out_path_aft_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240513/extract_people_stego"
# fc.csv2pcd(1, csv_in_path_aft_stego, pcd_out_path_aft_stego)

# ### pcdファイルの読み込み
# pcdpath_bef_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240513/extract_people/frame_0.pcd"
# pcd_bef_stego = fr.ReadPCD(pcdpath_bef_stego)
# pcdpath_aft_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/pcddata/20240513/extract_people_stego/frame_0.pcd"
# pcd_aft_stego = fr.ReadPCD(pcdpath_aft_stego)

# ### stego前のpcdファイルを可視化
# t.VisualizationPCD(pcd_bef_stego, "before_stego")

# ### stego後のpcdファイルを可視化
# t.VisualizationPCD(pcd_aft_stego, "after_stego")