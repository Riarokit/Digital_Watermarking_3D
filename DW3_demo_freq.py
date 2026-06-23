import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh

"""
近似平面からの高さを信号としたGFT（Graph Fourier Transform）において、
低周波成分と高周波成分の変調がそれぞれどのような形状変化をもたらすのかを
最もわかりやすく可視化するためのデモスクリプト。
"""

def visualize_gft_height_from_plane(show_axis=False):
    # 1. データの生成 (3D上の曲面パッチ)
    np.random.seed(42)
    rows, cols = 20, 20
    N = rows * cols
    
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    xx, yy = np.meshgrid(x, y)
    
    # 緩やかな波打つ曲面を作成 (ノイズを少しマイルドに)
    zz = 0.2 * np.sin(np.pi * xx) + 0.1 * np.cos(np.pi * yy)
    
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    
    # 2. 直交回帰 / PCAによる近似平面の計算
    centroid = np.mean(points, axis=0)
    X_centered = points - centroid
    cov = (X_centered.T @ X_centered) / N
    eigvals_pca, eigvecs_pca = np.linalg.eigh(cov)
    
    # 最小固有値の固有ベクトルが平面の法線ベクトル
    normal = eigvecs_pca[:, 0]
    if normal[2] < 0: # 法線を常にZ軸正の向きに揃える
        normal = -normal

    # 3. グラフ信号「近似平面からの高さ（h）」の計算
    # 内積を用いて平面からの符号付き距離(高さ)を計算
    h = np.sum((points - centroid) * normal, axis=1)

    # 4. 点群上のグラフ構築 (k-NN)
    k = 8
    adjacency = kneighbors_graph(points, k, mode='distance', include_self=False)
    W = adjacency.toarray()
    
    # ガウスカーネルで距離を重みに変換 (距離が近いほど重みが大きい)
    mask = W > 0
    sigma = np.mean(W[mask])
    W[mask] = np.exp(-W[mask]**2 / (sigma**2))
    W = np.maximum(W, W.T) # 対称化 (無向グラフ)

    # 5. GFTの準備（グラフラプラシアン行列の固有値分解）
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigenvalues, eigenvectors = eigh(L)

    # 6. 高さのグラフフーリエ変換 (GFT)
    gft_coeffs = eigenvectors.T @ h

    # 7. --- 低周波成分の変調 (Low-Frequency Modulation) ---
    # 固有値が小さい（滑らかな変化を持つ）成分を強調する
    coeffs_low = gft_coeffs.copy()
    target_idx_low = [1, 2, 3, 4, 5] # スライドで目立つように複数の低周波成分をとる
    strength_low = 1.6
    for idx in target_idx_low:
        coeffs_low[idx] += strength_low
    
    # 逆GFTで変調された高さ h' を取得し、法線方向に沿って座標を更新
    h_low = eigenvectors @ coeffs_low
    points_low = points + (h_low - h)[:, None] * normal

    # 8. --- 高周波成分の変調 (High-Frequency Modulation) ---
    # 固有値が大きい（激しい変化を持つ）成分を強調する
    coeffs_high = gft_coeffs.copy()
    target_idx_high = range(N - 30, N) # 最も高い周波数帯域を広げる
    strength_high = 0.3 # トゲトゲをスライド用に強調
    for idx in target_idx_high:
        coeffs_high[idx] += strength_high

    # 逆GFTで変調された高さ h' を取得し、法線方向に沿って座標を更新
    h_high = eigenvectors @ coeffs_high
    points_high = points + (h_high - h)[:, None] * normal

    # 9. 可視化
    fig = plt.figure(figsize=(20, 7))
    fig.suptitle('GFT based on "Heights from Approximated Plane"\nHow Low/High Frequencies affect the Mesh Configuration', 
                 fontsize=18, y=0.98, fontweight='bold')
    
    # 高さの色分けの正規化 (全員でスケールを統一する)
    norm = plt.Normalize(-1.5, 1.5)
    cmap = plt.cm.coolwarm

    def plot_surface(ax, pts, h_vals, title):
        # メッシュの形状に戻す
        grid_x = pts[:, 0].reshape(rows, cols)
        grid_y = pts[:, 1].reshape(rows, cols)
        grid_z = pts[:, 2].reshape(rows, cols)
        
        # 色を高さ h_vals に対応させる
        face_colors = cmap(norm(h_vals.reshape(rows, cols)))
        
        # 表面を描画
        surf = ax.plot_surface(grid_x, grid_y, grid_z, 
                               facecolors=face_colors, edgecolor='k', linewidth=0.2, alpha=0.9)
        
        # 近似平面（半透明）を描画
        plane_x, plane_y = np.meshgrid(np.linspace(-1.2, 1.2, 2), np.linspace(-1.2, 1.2, 2))
        plane_z = centroid[2] - (normal[0] * (plane_x - centroid[0]) + normal[1] * (plane_y - centroid[1])) / normal[2]
        ax.plot_surface(plane_x, plane_y, plane_z, alpha=0.15, color='gray')
        
        # 平面の中心から法線ベクトルを描画
        ax.quiver(centroid[0], centroid[1], centroid[2], 
                  normal[0]*0.5, normal[1]*0.5, normal[2]*0.5, 
                  color='black', arrow_length_ratio=0.15, linewidth=3, label="Normal Vector")
        
        ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        ax.set_zlim(-1.5, 1.5)
        
        # 見やすくするために軸の数字を消す
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        if not show_axis:
            ax.axis('off') # 軸、目盛り、背景のグレーのパネルを完全に非表示にする
        
        # 横斜めからの視点 (平面からの浮き沈みがわかりやすい)
        ax.view_init(elev=15, azim=-70)

    # Subplot 1: Original
    ax1 = fig.add_subplot(131, projection='3d')
    plot_surface(ax1, points, h, "1. Original Surface")
    
    # Subplot 2: Low Freq
    ax2 = fig.add_subplot(132, projection='3d')
    plot_surface(ax2, points_low, h_low, "2. Low-Frequency Modulation")

    # Subplot 3: High Freq
    ax3 = fig.add_subplot(133, projection='3d')
    plot_surface(ax3, points_high, h_high, "3. High-Frequency Modulation")

    # カラーバーの追加
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2, ax3], orientation='horizontal', 
                        fraction=0.03, pad=0.1, aspect=40)
    cbar.set_label('Height (h) from Approximated Plane', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)
    plt.show()

if __name__ == "__main__":
    # スライド向けに背景目盛りや枠線を消す場合は show_axis=False (デフォルト) を使用
    visualize_gft_height_from_plane(show_axis=False)