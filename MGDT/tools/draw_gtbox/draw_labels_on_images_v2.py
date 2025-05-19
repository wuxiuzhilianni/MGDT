# 颜色变成绿色且线条细一些，且不要写上类别信息

import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_single_image(txt_path, img_path, output_path):
    # 读取图像
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Unable to read image {img_path}.")
        return

    # 读取txt文件
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # 遍历每一行标注
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            print(f"Invalid format in file {txt_path}: {line.strip()}")
            continue

        # 提取坐标点并转换为整数
        points = list(map(float, parts[:8]))
        points = [(int(points[i]), int(points[i + 1])) for i in range(0, len(points), 2)]

        # 画绿色细线多边形（thickness=1）
        cv2.polylines(image, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=1)

    # 保存结果
    cv2.imwrite(output_path, image)

def draw_labels_on_images(txt_folder, img_folder, output_folder, num_threads=8):
    # 检查输出文件夹是否存在，不存在则创建
    os.makedirs(output_folder, exist_ok=True)

    # 获取txt和png文件列表
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    img_files = {f: os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png')}

    tasks = []
    for txt_file in txt_files:
        # 获取txt对应的图像文件名
        img_file = txt_file.replace('.txt', '.png')
        if img_file not in img_files:
            print(f"Warning: Image file {img_file} not found for {txt_file}.")
            continue

        # 路径设置
        txt_path = os.path.join(txt_folder, txt_file)
        img_path = img_files[img_file]
        output_path = os.path.join(output_folder, img_file)

        tasks.append((txt_path, img_path, output_path))

    # 多线程处理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(lambda x: process_single_image(*x), tasks), total=len(tasks), desc="Processing images"))

# 使用示例
txt_folder = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/annfiles_obb'
img_folder = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/images'
output_folder = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/images_with_gtv2'

draw_labels_on_images(txt_folder, img_folder, output_folder, num_threads=8)
