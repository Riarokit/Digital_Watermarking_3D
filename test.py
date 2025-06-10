import numpy as np
from scipy.spatial import KDTree

def find_points_in_range(point_cloud, point_X, min_distance, max_distance):
    """
    指定された点Xの周囲にmin~maxの距離範囲内の点を探索。
    
    Args:
        point_cloud (numpy.ndarray): 点群データ（Nx3の配列）
        point_X (numpy.ndarray): 検索対象の点（1x3の配列）
        min_distance (float): 最小距離
        max_distance (float): 最大距離

    Returns:
        list: 条件に一致する点と点Xの座標情報を保持するリスト
    """
    # KDTreeを構築
    kdtree = KDTree(point_cloud)
    
    # max_distance以内の点を取得
    indices = kdtree.query_ball_point(point_X, max_distance)
    
    # min_distanceを満たす点のみをフィルタリング
    matching_points = []
    for idx in indices:
        distance = np.linalg.norm(point_cloud[idx] - point_X)
        if min_distance <= distance <= max_distance:
            matching_points.append({
                'point_X': point_X.tolist(),
                'matching_point': point_cloud[idx].tolist(),
                'distance': distance
            })
    
    return matching_points

# 点群データの例 (numpy配列)
point_cloud = np.random.rand(1000, 3)  # 1000点の3次元点群 (例)

# 検索対象の点X
point_X = np.array([0.5, 0.5, 0.5])

# 距離範囲を指定
min_distance = 0.1
max_distance = 0.3

# 範囲内の点を探索
result = find_points_in_range(point_cloud, point_X, min_distance, max_distance)

# 結果を表示
for entry in result:
    print(f"点X: {entry['point_X']}, 対象点: {entry['matching_point']}, 距離: {entry['distance']:.4f}")
