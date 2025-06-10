import modules.fileconverter as fc
import modules.fileread as fr
import modules.tools as t
import modules.preprocess as pp
import modules.clustering as cl
import pandas as pd
import random as rd
import numpy as np
import STG02_global_value as g
import sys
import json

### csvファイルの読み込み
csv_in_path_aft_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20240513/extract_people_stego.csv"
df = g.to_df(csv_in_path_aft_stego)

### 共通鍵の確認
with open('shared_variable.json', 'r') as f:
    key_list = json.load(f)

key_list = [tuple(item) for item in key_list]

### 値の抽出
char_bin_list = []
for i in range(len(key_list)):
    if(df.iloc[key_list[i][0], key_list[i][1]] % 2 == 1):   #この値が示すビットは１なので
        char_bin_list.append(1)
    elif(df.iloc[key_list[i][0], key_list[i][1]] % 2 == 0):   #この値が示すビットは0なので
        char_bin_list.append(0)

print(char_bin_list)
print(len(char_bin_list))

### ビット列から文字列に変換
dec_char = ""                                    #復号したい文字列
if len(char_bin_list) % g.bit_num != 0:                  #文字列の文字数が割り切れない場合、英語以外が使われている可能性があるためエラー処理
    print("英語以外の文字が使われているか、プログラムの途中でエラーが発生しています。")
    sys.exit(1)
dec_char_len = int(len(char_bin_list) / g.bit_num)       #char_bin_listの長さから文字数がわかる
for i in range(dec_char_len):
    bin_char = ""                                #復号したい文字列をビット列で表記する文字列
    for j in range(g.bit_num):
        bin_char += str(char_bin_list[i*7+j])    #str関数はchar_bin_listを文字に変換する
    print("%d文字目のビット表記: %s" % (i+1, bin_char))
    dec_char += chr(int(bin_char, 2))            #int関数は文字列bin_charを2進数から10進数に変換する
    print("%d文字目まで: %s" % (i+1, dec_char))
print("復号された文字列は: %s" % dec_char)