import random as rd
import pandas as pd

# 01___完全にランダムに埋め込む行を決定する方法

char = "Helloworld"     #文字列Helloworldを送信したい

### 送信側と同じデータをDataFrameに変換
def to_df(str):
    df = pd.read_csv(str, encoding="shift-jis")
    print(df.head())
    return df

def key(df):
    rd.seed(0)
    make_key = [rd.randint(0, len(df)) for i in range(7*len(char))]    #ASCII文字コードは1英字7ビット
    key_list = make_key
    print(key_list)
    return key_list