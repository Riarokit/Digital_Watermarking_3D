from PIL import Image

# 元の画像を読み込み
img = Image.open("watermark8.png").convert("1")  # ← 1bitに変換

# 32×32にリサイズ
img = img.resize((64, 64), Image.NEAREST)

# 保存（1bitビットマップとして）
img.save("watermark8.bmp")
