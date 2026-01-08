import numpy as np
import matplotlib.pyplot as plt

def generate_paper_figure_final():
    # ==========================================
    # 【設定エリア】ここを変えると図の見た目が変わります
    # ==========================================
    
    # 1. 曲面の形状設定
    # 半径を大きく、角度の範囲を狭くすると「緩やかな曲面」になります
    R = 2.0             # 曲率半径 (大きいほど平らに近づく。2.0 -> 4.0に変更)
    t_start = np.pi/6   # 開始角度
    t_end = 5*np.pi/6   # 終了角度
    num_points = 20     # 点の数

    # 2. 変位(移動)の大きさ設定
    # この値を小さくすると、矢印や変形が小さくなります
    scale_factor = 0.35

    # 3. 比較用のノイズ（従来手法）の大きさ
    noise_level = 0.015  # ノイズの振幅 (小さく設定)

    # 4. 描画スタイル
    arrow_width = 0.005  # 矢印の太さ
    arrow_head = 3.5     # 矢印の頭のサイズ

    # ==========================================
    # データ生成・計算プロセス
    # ==========================================
    np.random.seed(42)
    
    # --- ベースとなる曲面の生成 ---
    t = np.linspace(t_start, t_end, num_points)
    # 円弧の計算 (yの位置を見やすい高さに調整)
    x = R * np.cos(t)
    y = R * np.sin(t) - R * np.sin(t_start) + 0.2
    points = np.column_stack((x, y))
    
    # 法線ベクトルの計算（中心から外向き）
    normals = np.column_stack((np.cos(t), np.sin(t)))

    # --- Step 1: 滑らかな変位ベクトル v (GFT低周波成分のシミュレーション) ---
    # 右上方向へゆっくり流れるようなベクトル場を作成
    # scale_factor を掛けて全体を微小にする
    v_base_x = 0.5  # 右方向成分
    v_base_y = 0.5  # 上方向成分
    
    # 場所によって少し向きを変えて「滑らかな波」っぽくする
    v_x = (v_base_x + 0.2 * np.cos(t * 2)) * scale_factor
    v_y = (v_base_y + 0.2 * np.sin(t * 2)) * scale_factor
    vectors_v = np.column_stack((v_x, v_y))
    
    # --- Step 2: 法線への射影 s = v . n ---
    scalars_s = np.sum(vectors_v * normals, axis=1)
    projected_vectors = normals * scalars_s[:, np.newaxis]
    
    # --- Step 3: 最終的な変形 ---
    # 提案手法 (Proposed)
    points_proposed = points + projected_vectors
    
    # 従来手法 (Conventional: High-Freq Noise)
    rand_noise = np.random.uniform(-noise_level, noise_level, size=t.shape)
    points_noisy = points + normals * rand_noise[:, np.newaxis]

    # ==========================================
    # 描画処理 (論文掲載用レイアウト)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # 共通設定関数
    def setup_ax(ax, title):
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_aspect('equal')
        # 表示範囲をデータに合わせて自動調整しつつ、少し余裕を持たせる
        ax.set_xlim(np.min(x)-0.2, np.max(x)+0.2)
        ax.set_ylim(0, np.max(y) + 0.5)
        ax.axis('off') # 軸を非表示
        
        # 元の形状（Surface）を薄い点線で描画
        ax.plot(x, y, 'k--', linewidth=1.2, alpha=0.3, label='Original Surface')
        ax.scatter(x, y, c='black', s=10, alpha=0.2)

    # --- (a) Step 1: Smooth Vector Field ---
    ax1 = axes[0]
    setup_ax(ax1, "(a)")
    
    ax1.quiver(x, y, v_x, v_y, angles='xy', scale_units='xy', scale=1, 
               color='blue', width=arrow_width, headwidth=arrow_head, label='Vector $\\vec{v}$')
    
    # 注釈
    # ax1.text(np.mean(x), np.mean(y) + 0.3, "Smooth but NOT aligned\nwith normals", 
    #          ha='center', va='center', color='blue', fontsize=11,
    #          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    ax1.legend(loc='lower center', frameon=True)


    # --- (b) Step 2: Projection onto Normal ---
    ax2 = axes[1]
    setup_ax(ax2, "(b)")
    
    # 法線ガイド（薄いグレー）
    ax2.quiver(x, y, normals[:,0]*0.2, normals[:,1]*0.2, angles='xy', scale_units='xy', scale=1, 
               color='gray', alpha=0.2, width=arrow_width*0.8, headwidth=2)
    
    # 射影ベクトル（赤）
    ax2.quiver(x, y, projected_vectors[:,0], projected_vectors[:,1], angles='xy', scale_units='xy', scale=1, 
               color='#D93025', width=arrow_width, headwidth=arrow_head, label='Projected Vector $s\\vec{n}$')
    
    # # 注釈
    # ax2.text(np.mean(x), np.mean(y) + 0.3, "Constrained to\nNormal Direction", 
    #          ha='center', va='center', color='#D93025', fontsize=11,
    #          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    ax2.legend(loc='lower center', frameon=True)


    # --- (c) Step 3: Final Result Comparison ---
    # ax3 = axes[2]
    # setup_ax(ax3, "(c)")
    
    # # 提案手法（赤実線）
    # ax3.plot(points_proposed[:,0], points_proposed[:,1], '-', color='#D93025', linewidth=2.5, alpha=0.9, label='Embedded Points')
    
    # # # 従来手法（緑点線）
    # # ax3.plot(points_noisy[:,0], points_noisy[:,1], ':', color='green', linewidth=1.5, alpha=0.7, label='Conventional (Noisy)')
    
    # # 注釈
    # ax3.text(np.mean(x), np.max(y) + 0.2, "Proposed: Smooth", color='#D93025', fontsize=11, fontweight='bold', ha='center')
    # # ax3.text(np.mean(x), np.min(y) - 0.1, "Conventional: Jagged", color='green', fontsize=11, fontweight='bold', ha='center')

    # ax3.legend(loc='lower center', frameon=True)

    plt.tight_layout()
    plt.savefig("STG50_demo.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("画像を 'STG50_demo.png' として保存しました。")

if __name__ == "__main__":
    generate_paper_figure_final()