import random as rd
import test_global as tg
import stegano_02_global_value as g

# make_key = rd.randint(1,100)
# a = tg.Key(make_key)
# print(a.get_key())


csv_in_path_bef_stego = "C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/csvdata/20240513/extract_people.csv"
df = g.to_df(csv_in_path_bef_stego)
g.key(df)