import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 3次元点群データの読み込み関数
def load_point_cloud(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

# データディレクトリの設定
train_dir = "data_object_velodyne/training/velodyne"
test_dir = "data_object_velodyne/testing/velodyne"

# 訓練データとテストデータのファイルリストを取得
train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.bin')][:100]
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.bin')][:100]

# 点群データの読み込み
train_point_clouds = [load_point_cloud(f) for f in train_files]
test_point_clouds = [load_point_cloud(f) for f in test_files]

# 最長の点群の長さを取得
max_length = max(pc.shape[0] for pc in train_point_clouds)

# データをサンプルする関数
def sample_point_cloud(point_cloud, num_samples):
    if point_cloud.shape[0] > num_samples:
        indices = np.random.choice(point_cloud.shape[0], num_samples, replace=False)
        return point_cloud[indices]
    return point_cloud

# サンプルサイズを指定
num_samples = 10000
train_point_clouds = [sample_point_cloud(pc, num_samples) for pc in train_point_clouds]
test_point_clouds = [sample_point_cloud(pc, num_samples) for pc in test_point_clouds]

# メッセージをバイナリ形式に変換する関数
def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)

# 点群データにメッセージを埋め込む関数
def embed_message_in_point_cloud(point_cloud, message):
    binary_message = text_to_binary(message)
    binary_message = np.array([int(bit) for bit in binary_message])
    binary_message = np.pad(binary_message, (0, point_cloud.shape[0] - len(binary_message)), 'constant')
    point_cloud[:, 3] = point_cloud[:, 3] + binary_message
    return point_cloud

# 点群データをパディングする関数
def pad_point_cloud(point_cloud, max_length):
    if point_cloud.shape[0] < max_length:
        padding = np.zeros((max_length - point_cloud.shape[0], 4), dtype=np.float32)
        point_cloud = np.vstack((point_cloud, padding))
    return point_cloud

# メッセージ "Hello World" を埋め込み、パディング
message = "Hello World"
embedded_train_point_clouds = [pad_point_cloud(embed_message_in_point_cloud(pc, message), max_length) for pc in train_point_clouds]

# 埋め込みモデルの定義
class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc = nn.Linear(256, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.fc(x)
        return x

# 抽出モデルの定義
class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.fc(x)
        return torch.sigmoid(x)

# モデルの初期化
embedder = Embedder()
extractor = Extractor()

# データローダーの作成
# 点群データの形状を (N, 4) から (N, 4, 1) に変換
train_data = torch.tensor(embedded_train_point_clouds, dtype=torch.float32).permute(0, 2, 1)
train_loader = DataLoader(TensorDataset(train_data), batch_size=4, shuffle=True)

# 損失関数と最適化アルゴリズムの定義
criterion = nn.MSELoss()
optimizer = optim.Adam(embedder.parameters(), lr=0.001)

# モデルの訓練
num_epochs = 10
for epoch in range(num_epochs):
    for data in train_loader:
        inputs = data[0]
        optimizer.zero_grad()  # 勾配の初期化
        outputs = embedder(inputs)  # モデルの順伝播
        loss = criterion(outputs, inputs)  # 損失の計算
        loss.backward()  # 逆伝播
        optimizer.step()  # パラメータの更新
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# バイナリメッセージをテキストに変換する関数
def binary_to_text(binary_message):
    binary_message = ''.join(map(str, binary_message))
    text = ''.join(chr(int(binary_message[i:i+8], 2)) for i in range(0, len(binary_message), 8))
    return text

# テストデータに対してメッセージの確認と抽出を行う
for pc in test_point_clouds:
    pc_padded = pad_point_cloud(pc, max_length)
    pc_tensor = torch.tensor(pc_padded, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
    extracted_message = extractor(pc_tensor).detach().numpy().round().astype(int).flatten()
    text_message = binary_to_text(extracted_message)
    print(f"Extracted message: {text_message}")

# カバー点群とステゴ点群の表示関数
def plot_point_clouds(original_pc, stego_pc):
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_pc[:, 0], original_pc[:, 1], original_pc[:, 2], s=1)
    ax1.set_title('Cover Point Cloud')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(stego_pc[:, 0], stego_pc[:, 1], stego_pc[:, 2], s=1, c='r')
    ax2.set_title('Stego Point Cloud')

    plt.show()

# カバー点群とステゴ点群の表示
cover_pc = train_point_clouds[0]  # カバー点群
stego_pc = embedded_train_point_clouds[0]  # ステゴ点群
plot_point_clouds(cover_pc, stego_pc)
