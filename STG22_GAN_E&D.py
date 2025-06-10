import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import trimesh
import matplotlib.pyplot as plt
import torch.nn.functional as F

# データセットの読み込み
class PointCloudDataset(Dataset):
    def __init__(self, root_folder):
        self.files = []
        for subdir in os.listdir(root_folder):
            model_path = os.path.join(root_folder, subdir, 'models', 'model_normalized.obj')
            if os.path.exists(model_path):
                self.files.append(model_path)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        mesh = trimesh.load(self.files[idx])
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        points = np.array(mesh.vertices, dtype=np.float32)
        return points / 10.0

# Collate function for DataLoader
def collate_fn(batch):
    max_points = max(len(points) for points in batch)
    padded_batch = []
    masks = []
    
    for points in batch:
        num_points = points.shape[0]
        padded = np.pad(points, ((0, max_points - num_points), (0, 0)), mode='constant', constant_values=0)
        mask = np.zeros((max_points,), dtype=np.float32)
        mask[:num_points] = 1
        padded_batch.append(padded)
        masks.append(mask)
    
    padded_batch = np.array(padded_batch)  # リストからNumPy配列に変換
    masks = np.array(masks)  # リストからNumPy配列に変換
    
    return torch.tensor(padded_batch, dtype=torch.float32), torch.tensor(masks, dtype=torch.float32)

# データローダーの設定
dataset = PointCloudDataset(root_folder='C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/02747177')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)  # バッチサイズを8に設定

# 生成器の定義
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(3 + message_length, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048, 3)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))
        return x

# 識別器の定義
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(3, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.sigmoid(self.fc4(x))
        return x

# 復号器の定義
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(3, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, message_length)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

# パラメータの設定
batch_size = 16  # バッチサイズを8に設定
lr = 0.0002  # 学習率を0.0002に設定
num_epochs = 3  # エポック数を100に設定
message = "HelloWorld"  # メッセージを "HelloWorld" に設定
message_length = len(message)  # メッセージの長さを計算

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの初期化
generator = Generator().to(device)
discriminator = Discriminator().to(device)
decoder = Decoder().to(device)

# オプティマイザーの設定
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr, betas=(0.5, 0.999))

# 損失関数
criterion = nn.BCELoss()
decoder_criterion = nn.MSELoss()

# 生成した点群データの表示
def display_point_cloud(point_cloud, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    ax.set_title(title)
    plt.show()

# Combinedネットワークの定義
class Combined(nn.Module):
    def __init__(self, generator, discriminator):
        super(Combined, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
    
    def forward(self, x):
        generated_data = self.generator(x)
        validity = self.discriminator(generated_data)
        return validity

combined = Combined(generator, discriminator).to(device)

# 学習ループ
for epoch in range(num_epochs):
    for i, (point_clouds, masks) in enumerate(dataloader):
        point_clouds = point_clouds.to(device)
        masks = masks.to(device)
        batch_size, num_points, _ = point_clouds.shape

        # メッセージをテンソルに変換
        message_tensor = torch.tensor([ord(c) for c in message], dtype=torch.float32).view(1, -1).to(device)
        message_tensor = message_tensor.repeat(batch_size, num_points, 1)

        # 本物データのラベルを作成
        real_labels = torch.ones(batch_size * num_points, 1).to(device)
        # 偽物データのラベルを作成
        fake_labels = torch.zeros(batch_size * num_points, 1).to(device)

        # 識別器の学習
        D_optimizer.zero_grad()
        outputs = discriminator(point_clouds.view(-1, 3))
        D_loss_real = criterion(outputs, real_labels)
        D_loss_real.backward()
        
        input_tensor = torch.cat((point_clouds.view(batch_size, num_points, 3), message_tensor), dim=2).view(-1, 3 + message_length)
        fake_point_cloud = generator(input_tensor).view(batch_size, num_points, 3)
        outputs = discriminator(fake_point_cloud.view(-1, 3).detach())
        D_loss_fake = criterion(outputs, fake_labels)
        D_loss_fake.backward()
        D_optimizer.step()
        
        D_loss = D_loss_real + D_loss_fake

        # 識別器の重みを固定
        for param in discriminator.parameters():
            param.requires_grad = False
        
        # 生成器の学習
        G_optimizer.zero_grad()
        validity = combined(input_tensor)
        G_loss = criterion(validity, real_labels)  # 生成された偽物に対して本物のラベルを使用
        G_loss.backward()
        G_optimizer.step()

        # 識別器の重みを解放
        for param in discriminator.parameters():
            param.requires_grad = True

        # 復号器の学習
        decoder_optimizer.zero_grad()
        decoded_message = decoder(fake_point_cloud.view(-1, 3).detach())
        decoder_loss = decoder_criterion(decoded_message, message_tensor.view(-1, message_length))
        decoder_loss.backward()
        decoder_optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], D_loss: {D_loss.item()}, G_loss: {G_loss.item()}, Decoder_loss: {decoder_loss.item()}')

# 学習済み生成器でカバー点群にメッセージを埋め込む
cover_point_cloud, masks = next(iter(dataloader))  # 2つの値を受け取るように修正
cover_point_cloud = cover_point_cloud.to(device)
display_point_cloud(cover_point_cloud[0].cpu().numpy(), "Cover Point Cloud")

message_tensor = torch.tensor([ord(c) for c in message], dtype=torch.float32).view(1, -1).to(device)
message_tensor = message_tensor.repeat(cover_point_cloud.shape[1], 1)
input_tensor = torch.cat((cover_point_cloud.view(-1, 3), message_tensor), dim=1)
fake_point_cloud = generator(input_tensor).detach().cpu().numpy().reshape(-1, 3)
display_point_cloud(fake_point_cloud, "Generated Stego Point Cloud")

# 生成したステゴ点群データからメッセージを復号
fake_point_cloud_tensor = torch.tensor(fake_point_cloud).to(device)
decoded_message_tensor = decoder(fake_point_cloud_tensor.view(-1, 3))
decoded_message = ''.join([chr(max(32, min(126, int(c)))) for c in decoded_message_tensor.detach().cpu().numpy().flatten()])
print("Decoded Message:", decoded_message)
print("Decoded Message Length:", len(decoded_message))
