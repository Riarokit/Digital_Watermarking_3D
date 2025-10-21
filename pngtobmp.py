from PIL import Image

# 元の画像を読み込み
img = Image.open("watermark16.png").convert("1")  # ← 1bitに変換

# 32×32にリサイズ
img = img.resize((16, 16), Image.NEAREST)

# 保存（1bitビットマップとして）
img.save("watermark16.bmp")
