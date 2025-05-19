import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def crop_and_save_objects(txt_path, img_path, output_folder):
    """根据TXT标注裁剪图像中的目标对象"""
    # 读取图像
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Unable to read image {img_path}.")
        return

    # 读取txt文件
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # 为每个图像创建单独的输出文件夹
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img_output_dir = os.path.join(output_folder, img_name)
    os.makedirs(img_output_dir, exist_ok=True)

    # 遍历每一行标注
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 9:
            print(f"Invalid format in file {txt_path}: {line.strip()}")
            continue

        # 提取坐标点和类别名称
        points = list(map(float, parts[:8]))
        label = parts[8]

        # 将点转换为numpy数组
        points = np.array([(points[i], points[i+1]) for i in range(0, len(points), 2)], dtype=np.float32)

        # 计算旋转矩形
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 获取旋转矩形的宽度和高度
        width = int(rect[1][0])
        height = int(rect[1][1])

        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(rect[0], rect[2], 1.0)
        
        # 执行旋转
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # 裁剪旋转后的图像
        cropped = cv2.getRectSubPix(rotated, (width, height), rect[0])

        # 保存裁剪结果
        output_path = os.path.join(img_output_dir, f"{label}_{idx}.png")
        cv2.imwrite(output_path, cropped)

def crop_objects_from_dataset(txt_folder, img_folder, output_folder, num_threads=8):
    """批量处理数据集中的所有图像"""
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

        tasks.append((txt_path, img_path, output_folder))

    # 多线程处理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(lambda x: crop_and_save_objects(*x), tasks), 
                 total=len(tasks), 
                 desc="Cropping objects"))

# 使用示例
txt_folder = '/workspace/GeoChat/data/dota/pseudo_box/grounding_method3/weakly_box0.4'
img_folder = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/images'
output_folder = '/workspace/GeoChat/data/dota/pseudo_box/grounding_method3/weakly_box0.4_cropped'

crop_objects_from_dataset(txt_folder, img_folder, output_folder, num_threads=8)