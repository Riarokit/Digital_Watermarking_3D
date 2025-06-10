from modules.sharemodule import laspy,np,o3d,dt

# データの読み込み（o2d）
# 第1引数：pcdファイルまでのパス
# 戻り値：pcdを処理できるようにしたもの
def ReadPCD(filename):
    return o3d.io.read_point_cloud(filename)

