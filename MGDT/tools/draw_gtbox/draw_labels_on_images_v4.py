import os
import cv2
import numpy as np
import random

def read_labels(txt_path, decimal_precision=1):
    """ 读取标注信息，返回一个列表，格式为 [(x1, y1, ..., x4, y4)] """
    labels = []
    if not os.path.exists(txt_path):
        print(f"Warning: File not found {txt_path}")
        return labels
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                print(f"Invalid format in file {txt_path}: {line.strip()}")
                continue

            coords = tuple(round(float(parts[i]), decimal_precision) for i in range(8))
            labels.append(coords)
    
    return labels

def draw_random_n_green(txt_path, img_path, output_path, n=10, seed=42):
    # 设置随机种子确保可重复
    random.seed(seed)

    # 读取图像
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Unable to read image {img_path}.")
        return

    # 读取标注框
    labels = read_labels(txt_path)
    total = len(labels)
    print(f"总标注数量: {total}")

    if total == 0:
        return

    # 随机选取 n 个索引作为绿色框
    green_indices = set(random.sample(range(total), min(n, total)))

    for idx, label in enumerate(labels):
        points = [(int(label[i]), int(label[i + 1])) for i in range(0, 8, 2)]

        if idx in green_indices:
            color = (0, 255, 0)  # Green
            thickness = 4
        else:
            color = (0, 0, 255)  # Red
            thickness = 2

        cv2.polylines(image, [np.array(points)], isClosed=True, color=color, thickness=thickness)

    print(f"绿色框数量: {len(green_indices)}, 红色框数量: {total - len(green_indices)}")

    # 保存结果图
    cv2.imwrite(output_path, image)

# 示例调用
txt_path = "/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/annfiles_obb/P1393__1024__1648___4120.txt"
img_path = "/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/images/P1393__1024__1648___4120.png"
output_path = "/workspace/MCL/tools/draw_gtbox/P1393__1024__1648___4120_percent1.png"

draw_random_n_green(txt_path, img_path, output_path, n=1)
