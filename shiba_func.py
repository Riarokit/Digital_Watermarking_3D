from modules.sharemodule import o3d,np
import copy
import math
# from modules import fileread as fr
# from modules import preprocess as pp
# from modules import tools as t
# from selfmade import dctV2
# from selfmade import dct
# from selfmade import SelectPCD_Ver2 as SPCDV2
# from selfmade import comp
# import time

def vis_cust_bound(points,bounding_box):
    """
    バウンディングボックスも表示できちゃう点群表示するやつ
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    vis.get_render_option().background_color = np.asarray([0.5372549019607843,0.7647058823529412,0.9215686274509804])

    vis.add_geometry(points)
    vis.add_geometry(bounding_box)

    vis.run()

    vis.destroy_window()
    
    return

def split8(pcd,cube_min,cube_size):
    """
    octree自作で作ろうとした残骸
    点群を8つに分けてボックスを表示する
    """
    div = cube_size/2
    cube_size = div
    # ブロック0
    cube0_min = copy.deepcopy(cube_min)
    cube0_max = cube0_min+div
    bounding_box0 = o3d.geometry.AxisAlignedBoundingBox(cube0_min,cube0_max)
    # mt.vis_cust_bound(pcd,bounding_box0)
    # ブロック1
    cube1_min = copy.deepcopy(cube_min)
    cube1_min[1] += div
    cube1_max = cube1_min+div
    bounding_box1 = o3d.geometry.AxisAlignedBoundingBox(cube1_min,cube1_max)
    # mt.vis_cust_bound(pcd,bounding_box1)
    # ブロック2
    cube2_min = copy.deepcopy(cube_min)
    cube2_min[0] += div
    cube2_max = cube2_min+div
    bounding_box2 = o3d.geometry.AxisAlignedBoundingBox(cube2_min,cube2_max)
    # mt.vis_cust_bound(pcd,bounding_box2)
    # ブロック3
    cube3_min = copy.deepcopy(cube_min)
    cube3_min[0] += div
    cube3_min[1] += div
    cube3_max = cube3_min+div
    bounding_box3 = o3d.geometry.AxisAlignedBoundingBox(cube3_min,cube3_max)
    # mt.vis_cust_bound(pcd,bounding_box3)
    # ブロック4
    cube4_min = copy.deepcopy(cube_min)
    cube4_min[2] += div
    cube4_max = cube4_min+div
    bounding_box4 = o3d.geometry.AxisAlignedBoundingBox(cube4_min,cube4_max)
    # mt.vis_cust_bound(pcd,bounding_box4)
    # ブロック5
    cube5_min = copy.deepcopy(cube_min)
    cube5_min[2] += div
    cube5_min[1] += div
    cube5_max = cube5_min+div
    bounding_box5 = o3d.geometry.AxisAlignedBoundingBox(cube5_min,cube5_max)
    # mt.vis_cust_bound(pcd,bounding_box5)
    # ブロック6
    cube6_min = copy.deepcopy(cube_min)
    cube6_min[2] += div
    cube6_min[0] += div
    cube6_max = cube6_min+div
    bounding_box6 = o3d.geometry.AxisAlignedBoundingBox(cube6_min,cube6_max)
    # mt.vis_cust_bound(pcd,bounding_box6)
    # ブロック7
    cube7_min = copy.deepcopy(cube_min)
    cube7_min[2] += div
    cube7_min[1] += div
    cube7_min[0] += div
    cube7_max = cube7_min+div
    bounding_box7 = o3d.geometry.AxisAlignedBoundingBox(cube7_min,cube7_max)
    # mt.vis_cust_bound(pcd,bounding_box7)

    bounding_box_list = []
    cube_min_list = [cube0_min,cube1_min,cube2_min,cube3_min,cube4_min,cube5_min,cube6_min,cube7_min]
    cube_max_list = [cube0_max,cube1_max,cube2_max,cube3_max,cube4_max,cube5_max,cube6_max,cube7_max]
    for i in range(8):
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(cube_min_list[i],cube_max_list[i])
        bounding_box_list.append(bounding_box)
    vis_cust_bound(pcd,bounding_box_list)
    
    return cube_min_list,cube_max_list



def jud(points,cube_min_list,cube_max_list):
    """
     octree自作で作ろうとした残骸
     ボックス内に点があるか判定する
    """
    value=[]
    for i in range(8):
        if np.any((points>=cube_min_list[i])&(points<=cube_max_list[i])):
            value.append(1)
        else:
            value.append(0)
    return value


def encode_octree(node, output_path_location=None, output_path_color=None, depth=0, bit_dict=None, color_list=None):
    """
    Octreeのノードをビット列に符号化する関数
    第一引数はroot_node,depthとbit_dictは指定しなくていい,optput_pathはファイルのパス(テキスト形式で指定)
    """
    if output_path_location is None:
        print("Specify the path to the location information file")
        return None
    
    if output_path_location is None:
        print("Specify the path to the color information file")
        return None
    
    if bit_dict is None:
        bit_dict = {}

    if color_list is None:
        color_list = []
    
    # 内部ノードのチェック（個々の子ノードを判定）
    if isinstance(node, o3d.geometry.OctreeInternalNode):
        # 子ノードが存在するかをビット列で表現
        children_bits = "".join([str(int(child is not None)) for child in node.children])
        
        if depth not in bit_dict:
            bit_dict[depth] = []
        bit_dict[depth].append(children_bits)

        # 各子ノードを再帰的に処理
        for child in node.children:
            if child is not None:
                encode_octree(child, output_path_location, output_path_color, depth + 1, bit_dict, color_list)

    elif isinstance(node,o3d.geometry.OctreePointColorLeafNode):
        color_list.append(node.color)

    if depth == 0:
        with open(output_path_location,'w') as file:
            for depth in sorted(bit_dict.keys()):
                for bits in bit_dict[depth]:
                    file.write(bits)
        with open(output_path_color,'w') as file:
            for color in color_list:
                file.write(f"{color[0]},{color[1]},{color[2]}\n")

    return None

def decode_octree(input_path_location,input_path_color,max_size):
    """
    第一引数はエンコーダで作ったファイルのパス
    第二引数はoctreeの最初のボックス(ルートノード)の大きさ
    本来デコーダは第一引数のファイルパスだけのほうがいい気がするけど、工夫すればエンコードされたデータから
    ルートノードの大きさは計算できる気がするから研究段階ならこれでいい気がする
    あとは実装する人に任せる
    """
    with open(input_path_location, 'r') as file:
        bit_stream = file.read()  # 0と1のみを取り出す

    color_list = []

    with open(input_path_color,'r') as file:
        for line in file:
            values = line.strip().split(',')
            if len (values)== 3:
                try:
                    color = list(map(float, values))
                    color_list.append(color)
                except ValueError:
                    print(f"Invalid color data: {line}")
                    continue

    level_ptrs = [0] # 現在地

    level_bits_list,max_depth = countlayer(bit_stream)
    print("level_bits_list:",level_bits_list)
    max_depth_check = len(level_bits_list)
    if max_depth != max_depth_check:
        print("max_depth calculate error:max_depth=",max_depth,"max_depth_check=",max_depth_check)
        return None
    print("Calculated max_depth:",max_depth," check:",max_depth_check)

    reconstruct_count = [0] * max_depth
    points = []

    # voxel_size = 1.0 / (2 ** max_depth)
    # max_size = voxel_size * (2 ** max_depth)
    reconstruct_pointcloud(bit_stream,level_ptrs,1,max_depth,level_bits_list,reconstruct_count,points,max_size)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(color_list))

    return pcd

def countlayer(bit_stream):
    level_bits_list = []
    depth = 1
    bit_ptr = 0
    nodes_in_current_level = 8

    while bit_ptr < len(bit_stream):
        # この階層のビット数を保持
        level_bits_list.append((nodes_in_current_level,bit_ptr))

        # 次の階層のビット数を計算
        children_count = sum(1 for i in range(nodes_in_current_level) if bit_ptr + i < len(bit_stream) and bit_stream[bit_ptr + i] == '1')

        # 次の階層に子ノードがない場合、終了
        if children_count == 0:
            break

        # 次の階層に移動
        depth += 1
        bit_ptr += nodes_in_current_level
        nodes_in_current_level = children_count * 8
    
    return level_bits_list,depth - 1

def reconstruct_pointcloud(bit_stream,level_ptrs,current_depth,max_depth,level_bits_list,reconstruct_count,points,size,origin=np.array([0,0,0]),num = 0):
    if current_depth > max_depth:
        return
    
    # 今の階層のノードの数と読み込み地点を取得
    nodes_in_current_level, start_ptr = level_bits_list[current_depth -1]
    
    # 1階層目以外は現在の階層のlevel_ptrsを読み取り位置にセット（start_ptrは各階層の最初の位置、countはすでに読み込んだビット数）
    if len(level_ptrs) < current_depth:
        # reconstruct_countは再帰した回数＝8ビットずつ読み込んだ回数
        count = reconstruct_count[current_depth - 2] - 1 if current_depth > 1 else 0
        level_ptrs.append(start_ptr + (count * 8))
    
    # 8ビットずつ1を走査。最深階層以外は再帰処理、最深階層は点を生成。
    for i in range(8):
        if level_ptrs[current_depth - 1] >= start_ptr + nodes_in_current_level:
            return

        # ポインタで現在地のビットを取り出し
        bit = bit_stream[level_ptrs[current_depth - 1]]
        
        # ポインタを１進める
        level_ptrs[current_depth - 1] += 1

        # 現在地が1なら処理開始
        if bit == '1':
            # オフセット計算
            voxel_offset = compute_voxel_offset(i,size)

            #現在最深階層なら点を追加
            if current_depth == max_depth:
                point = origin + voxel_offset
                points.append(point)
            # 現在最深階層以外なら再帰処理で階層を進む
            else:
                reconstruct_count[current_depth - 1] += 1
                next_origin = origin + voxel_offset
                reconstruct_pointcloud(bit_stream, level_ptrs,current_depth + 1,max_depth,level_bits_list,reconstruct_count,points,size/2.0,next_origin,num)
    return None

def compute_voxel_offset(i,size):
    """
    ビットに対応するボクセルのオフセットを計算する
    i は 0 から 7 の値を取り、それに応じたオフセットを返す
    """
    return np.array([i & 1, (i >> 1) & 1, (i >> 2) & 1],dtype=np.float32) * (size/2.0)

def psnr_color(before_points, after_points, max_value=1.0):
    
    o3d.visualization.draw_geometries([before_points, after_points])
    before = np.array(before_points.points)
    after = np.array(after_points.points)
    min_x_before = np.min(before[:, 0])
    min_y_before = np.min(before[:, 1])
    min_z_before = np.min(before[:, 2])
    min_x_after = np.min(after[:, 0])
    min_y_after = np.min(after[:, 1])
    min_z_after = np.min(after[:, 2])
    dif_x = min_x_before-min_x_after
    dif_y = min_y_before-min_y_after
    dif_z = min_z_before-min_z_after
    print(dif_x,dif_y,dif_z)
    transformation = np.array([[1, 0, 0, dif_x],
                                [0, 1, 0, dif_y],
                                [0, 0, 1, dif_z],
                                [0, 0, 0, 1]])
    after_points.transform(transformation)
    # 初期の視覚化
    o3d.visualization.draw_geometries([before_points, after_points])

    # ICPによる位置合わせ
    threshold = 0.02  # 対応点を探索する距離のしきい値
    trans_init = np.identity(4)  # 初期の変換行列（単位行列）

    # 点群同士のICP位置合わせ
    reg_p2p = o3d.pipelines.registration.registration_icp(
        before_points, after_points, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # 最適な変換行列を取得
    transformation = reg_p2p.transformation
    print("Transformation is:")
    print(transformation)

    # 変換後の視覚化
    after_points.transform(transformation)
    o3d.visualization.draw_geometries([before_points, after_points])

    # 点群の座標と色情報を取得
    bef_coords = np.asarray(before_points.points)
    aft_coords = np.asarray(after_points.points)
    
    bef_colors = np.asarray(before_points.colors)
    aft_colors = np.asarray(after_points.colors)
    
    # 点の順序を座標に基づいてソート
    bef_sorted_idx = np.lexsort((bef_coords[:, 2], bef_coords[:, 1], bef_coords[:, 0]))
    aft_sorted_idx = np.lexsort((aft_coords[:, 2], aft_coords[:, 1], aft_coords[:, 0]))

    # ソート後の色情報
    bef_colors_sorted = bef_colors[bef_sorted_idx]
    aft_colors_sorted = aft_colors[aft_sorted_idx]
    
    # 点の数が一致しているか確認
    if len(bef_colors_sorted) != len(aft_colors_sorted):
        raise ValueError("点群の数が一致していません")

    # 差分を計算
    sub = aft_colors_sorted - bef_colors_sorted
    # print([f"{aft_colors_sorted[i]} - {bef_colors_sorted[i]} = {sub[i]}" for i in range(len(sub))])

    # 平均二乗誤差（MSE）を計算（R, G, B全てのチャンネルを考慮）
    MSE = np.mean(sub ** 2)

    # PSNRを計算
    PSNR = 10 * math.log10((max_value ** 2) / MSE)
    print("PSNR:", PSNR)
    return PSNR