import os
import cv2
import numpy as np

def read_labels(txt_path, ignore_class=True, decimal_precision=1):
    """ 读取标注信息，返回一个集合，格式为 {(x1, y1, x2, y2, x3, y3, x4, y4)} """
    labels = set()
    if not os.path.exists(txt_path):
        print(f"Warning: File not found {txt_path}")
        return labels
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                print(f"Invalid format in file {txt_path}: {line.strip()}")
                continue

            # 处理坐标，减少浮点数精度误差
            coords = tuple(round(float(parts[i]), decimal_precision) for i in range(8))

            # 选择是否忽略类别匹配
            if ignore_class:
                labels.add(coords)
            else:
                labels.add(coords + (parts[8],))  # 包含类别信息进行匹配
    
    return labels

def draw_boxes(A_txt_path, B_txt_path, img_path, output_path):
    # 读取图像
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Unable to read image {img_path}.")
        return

    # 读取 A 和 B 的标注
    labels_A = read_labels(A_txt_path)
    labels_B = read_labels(B_txt_path)

    print(f"A 标注数量: {len(labels_A)}, B 标注数量: {len(labels_B)}")  # 调试信息

    matched_count = 0  # 统计绿色目标数

    # 遍历 B 的标注
    for label in labels_B:
        points = list(label[:8])  # 获取坐标

        # 转换为整数坐标
        points = [(int(points[i]), int(points[i + 1])) for i in range(0, len(points), 2)]

        # 判断颜色（绿色 = A 中存在，红色 = 仅 B 中存在）
        if label in labels_A:
            color = (0, 255, 0)
            matched_count += 1
            thickness = 4
        else:
            color = (0, 0, 255)
            thickness = 2

        # 画多边形
        cv2.polylines(image, [np.array(points)], isClosed=True, color=color, thickness=thickness)

    print(f"绿色框数量: {matched_count}, 红色框数量: {len(labels_B) - matched_count}")  # 调试信息

    # 保存结果
    cv2.imwrite(output_path, image)

"""
B表示all
A表示sparse
在A和B中的元素为绿色
其他的B中的为红色
"""

# 示例调用
# A_txt_path = "/workspace/Dataset/DOTAv1_Split/sparse/image_annotation_split_percent10/labeled_annotation/P1393__1024__4120___0.txt"  # A 标注文件路径
A_txt_path = "/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/annfiles_obb/P1393__1024__4120___0.txt"  # B 标注文件路径
B_txt_path = "/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/annfiles_obb/P1393__1024__4120___0.txt"  # B 标注文件路径
img_path = "/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/images/P1393__1024__4120___0.png"  # 对应的图像路径
output_path = "/workspace/MCL/tools/draw_gtbox/P1393__1024__4120___0_all.png"  # 结果保存路径

draw_boxes(A_txt_path, B_txt_path, img_path, output_path)
