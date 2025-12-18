import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from scipy.linalg import eigh

def visualize_gft_low_vs_high():
    # 1. データの生成
    np.random.seed(42)
    rows, cols = 7,7
    N = rows * cols
    
    # 格子点を作成
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    xx, yy = np.meshgrid(x, y)
    
    # 1次元配列になおして結合
    points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # 少しノイズを加えて「スキャンされた点群」っぽくする
    points += np.random.normal(0, 0.02, points.shape)
    
    x_coord = points[:, 0] # 信号としてX座標を使用

    # 2. グラフ構築（k近傍法）
    k = 6
    adjacency = kneighbors_graph(points, k, mode='connectivity', include_self=False)
    adjacency = 0.5 * (adjacency + adjacency.T) # 対称化

    # 3. GFTの準備（ラプラシアン行列の固有値分解）
    laplacian = csgraph.laplacian(adjacency, normed=False)
    eigenvalues, eigenvectors = eigh(laplacian.toarray())

    # 4. GFT（グラフフーリエ変換）
    gft_coeffs = eigenvectors.T @ x_coord

    # 5. 変調（低周波 vs 高周波）
    
    # --- 低周波成分の変調 ---
    # index=1 (Fiedler vector付近) を操作
    coeffs_low = gft_coeffs.copy()
    target_idx_low = 1 
    strength_low = 0.3
    coeffs_low[target_idx_low] += strength_low
    
    new_x_low = eigenvectors @ coeffs_low
    points_low = points.copy()
    points_low[:, 0] = new_x_low

    # --- 高周波成分の変調 ---
    # index=N-1 (最大固有値に対応する成分) を操作
    coeffs_high = gft_coeffs.copy()
    target_idx_high = N - 1
    strength_high = 0.3
    coeffs_high[target_idx_high] += strength_high
    
    new_x_high = eigenvectors @ coeffs_high
    points_high = points.copy()
    points_high[:, 0] = new_x_high

    # 6. 可視化（横に並べて比較）
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 共通の描画関数
    def plot_graph(ax, pts_original, pts_modified, title):
        # グラフの接続関係を描画
        rows_adj, cols_adj = adjacency.nonzero()
        for i, j in zip(rows_adj, cols_adj):
            if i < j:
                if np.linalg.norm(pts_original[i] - pts_original[j]) < 0.3:
                    ax.plot([pts_original[i, 0], pts_original[j, 0]], 
                            [pts_original[i, 1], pts_original[j, 1]], 
                            'k-', alpha=0.1, zorder=1)
        
        # 点の描画
        ax.scatter(pts_original[:, 0], pts_original[:, 1], c='blue', label='Original', alpha=0.6, s=50, zorder=2)
        ax.scatter(pts_modified[:, 0], pts_modified[:, 1], c='red', label='Modified', alpha=0.6, s=50, zorder=3)

        # 移動ベクトルの描画
        for i in range(N):
            ax.arrow(pts_original[i, 0], pts_original[i, 1], 
                      pts_modified[i, 0] - pts_original[i, 0], 
                      0, 
                      color='gray', alpha=0.8, head_width=0.015, length_includes_head=True, zorder=4)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()

    plot_graph(axes[0], points, points_low, "Low-Frequency Modulation\n(Smooth Shift / Coherent Movement)")
    plot_graph(axes[1], points, points_high, "High-Frequency Modulation\n(Jagged Shift / Incoherent Movement)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_gft_low_vs_high()