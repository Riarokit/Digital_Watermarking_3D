import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import trimesh
import matplotlib.pyplot as plt

# データセットの読み込み
class PointCloudDataset(Dataset):
    def __init__(self, root_folder, num_points=4096):
        self.files = []
        for subdir in os.listdir(root_folder):
            model_path = os.path.join(root_folder, subdir, 'models', 'model_normalized.obj')
            if os.path.exists(model_path):
                self.files.append(model_path)
        self.num_points = num_points
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        mesh = trimesh.load(self.files[idx])
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)  # シーンを単一のメッシュに変換
        points = np.array(mesh.vertices, dtype=np.float32)  # データ型をfloat32に強制
        if points.shape[0] > self.num_points:
            points = points[:self.num_points, :]
        elif points.shape[0] < self.num_points:
            padding = np.zeros((self.num_points - points.shape[0], 3), dtype=np.float32)
            points = np.vstack((points, padding))
        return points

# データローダーの設定
dataset = PointCloudDataset(root_folder='C:/Users/ryoi1/OneDrive/デスクトップ/3-2.PDF一覧/情報通信ゼミナール/2023.12_GitHub/LiDAR-1/Python/data/02747177')   # フォルダーパスは適宜変更
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)   # バッチサイズは適宜変更

# 生成器の定義
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.fc(x)
        return x.view(x.size(0), 4096, 3)  # フラット化したデータを元の形に戻す

# 識別器の定義
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 点群データをフラット化する
        x = self.fc(x)
        return x

# 復号器の定義
class Decoder(nn.Module):
    def __init__(self, input_dim, message_length):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, message_length)
        )
    
    def forward(self, stego_point_cloud):
        batch_size = stego_point_cloud.shape[0]
        x = stego_point_cloud.view(batch_size, -1)
        x = self.fc(x)
        return x

# パラメータの設定
batch_size = 64
lr = 0.0002
num_epochs = 20
message = "Hello World"
message_length = len(message)
input_dim_G = 12288 + message_length  # 12288はカバー点群の次元
output_dim_G = 12288  # 4096 * 3
input_dim_D = 12288

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの初期化
generator = Generator(input_dim_G, output_dim_G).to(device)
discriminator = Discriminator(input_dim_D).to(device)
decoder = Decoder(input_dim_D, message_length).to(device)

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

# 学習ループ
for epoch in range(num_epochs):
    for i, point_cloud in enumerate(dataloader):
        point_cloud = point_cloud.to(device)
        batch_size = point_cloud.size(0)
        
        # 本物のデータラベル
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # 識別器の学習
        D_optimizer.zero_grad()
        outputs = discriminator(point_cloud)
        D_loss_real = criterion(outputs, real_labels)
        D_loss_real.backward()
        
        message_tensor = torch.tensor([ord(c) for c in message], dtype=torch.float32).repeat(batch_size, 1).to(device)
        input_tensor = torch.cat((point_cloud.view(batch_size, -1), message_tensor), dim=1)
        fake_point_cloud = generator(input_tensor)
        outputs = discriminator(fake_point_cloud.detach())
        D_loss_fake = criterion(outputs, fake_labels)
        D_loss_fake.backward()
        D_optimizer.step()
        
        D_loss = D_loss_real + D_loss_fake
        
        # 生成器の学習
        G_optimizer.zero_grad()
        outputs = discriminator(fake_point_cloud)
        G_loss = criterion(outputs, real_labels)
        G_loss.backward()
        G_optimizer.step()
        
        # 復号器の学習
        decoder_optimizer.zero_grad()
        decoded_message = decoder(fake_point_cloud.detach())
        decoder_loss = decoder_criterion(decoded_message, message_tensor)
        decoder_loss.backward()
        decoder_optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], D_loss: {D_loss.item()}, G_loss: {G_loss.item()}, Decoder_loss: {decoder_loss.item()}')

# 学習済み生成器でカバー点群にメッセージを埋め込む
cover_point_cloud = next(iter(dataloader))[0].to(device)
display_point_cloud(cover_point_cloud[0].cpu().numpy(), "Cover Point Cloud")

message_tensor = torch.tensor([ord(c) for c in message], dtype=torch.float32).view(1, -1).to(device)
input_tensor = torch.cat((cover_point_cloud.view(1, -1), message_tensor), dim=1)
fake_point_cloud = generator(input_tensor).detach().cpu().numpy()
display_point_cloud(fake_point_cloud[0], "Stego Point Cloud")

# 生成したステゴ点群データからメッセージを復号
fake_point_cloud_tensor = torch.tensor(fake_point_cloud).to(device)
decoded_message_tensor = decoder(fake_point_cloud_tensor)
decoded_message = ''.join([chr(int(c)) for c in decoded_message_tensor.detach().cpu().numpy().flatten()])
print("Decoded Message:", decoded_message)
