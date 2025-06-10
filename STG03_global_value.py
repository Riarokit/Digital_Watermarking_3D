import random as rd
import pandas as pd
import math

# 03___02のアルゴリズムを用いて、写真を埋め込む
# だが、\ のASCIIでの扱い方がわからない

bit_num = 8

### 送信側と同じデータをDataFrameに変換
def to_df(str):
    df = pd.read_csv(str, encoding="shift-jis")
    return df

def key(df,byte_length):
    df["Index"] = range(0, len(df.index))
    df_x = pd.DataFrame({"Index" : df["Index"], "Value" : df["Ori_x"]}); df_x["Key"] = df.columns.get_loc("Ori_x")
    df_y = pd.DataFrame({"Index" : df["Index"], "Value" : df["Ori_y"]}); df_y["Key"] = df.columns.get_loc("Ori_y")
    df_z = pd.DataFrame({"Index" : df["Index"], "Value" : df["Ori_z"]}); df_z["Key"] = df.columns.get_loc("Ori_z")
    df_sort = pd.concat([df_x, df_y, df_z])
    df_sort = df_sort.sort_values("Value", ascending=False)
    print("ソート後のデータフレーム")
    print(df_sort.head())

    N = math.ceil(len(df_sort) / 4)
    splited_df = [df_sort[i:i+N] for i in range(0, len(df_sort), N)]
    classA_bit = math.ceil(bit_num*byte_length*0.6)
    classB_bit = math.ceil(bit_num*byte_length*0.3)
    classC_bit = bit_num*byte_length-classA_bit-classB_bit
    rd.seed(0)
    make_key_classA = [rd.randint(0, len(splited_df[0])) for i in range(classA_bit)]    #ASCII文字コードは1英字7ビット
    rd.seed(0)
    make_key_classB = [rd.randint(0, len(splited_df[1])) for i in range(classB_bit)]
    rd.seed(0)
    make_key_classC = [rd.randint(0, len(splited_df[2])) for i in range(classC_bit)]
    make_key = []
    col_Key = splited_df[0].columns.get_loc("Key")       #col_Keyは"Key"列が何列目かを格納する変数
    col_Index = splited_df[0].columns.get_loc("Index")   #col_Indexは"Index"列が何列目かを格納する変数
    for i in range(len(make_key_classA)):             #ClassA,B,Cの選抜された行のIndexと属性情報(x,y,z)をタプルで格納
        make_key.append((splited_df[0].iloc[make_key_classA[i], col_Index], splited_df[0].iloc[make_key_classA[i], col_Key]))
    for i in range(len(make_key_classB)):             
        make_key.append((splited_df[1].iloc[make_key_classB[i], col_Index], splited_df[1].iloc[make_key_classB[i], col_Key]))
    for i in range(len(make_key_classC)):             
        make_key.append((splited_df[2].iloc[make_key_classC[i], col_Index], splited_df[2].iloc[make_key_classC[i], col_Key]))
    # rd.shuffle(make_key)          #Classがばらばらになるようにする(そのままだとClassA,B,Cの順になってしまうため)
    print("キーのリストの長さ: ", len(make_key))
    key_list = make_key
    return key_list