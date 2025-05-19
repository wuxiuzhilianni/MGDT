import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import cv2  # 应该加在顶部

def analyze_bbox_sizes(txt_folder, img_folder):
    """统计不同类别目标框大小占原图百分比的范围"""
    # 存储统计结果 {类别: [面积百分比列表]}
    class_stats = defaultdict(list)
    
    # 获取txt和png文件列表
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    img_files = {f: os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png')}

    for txt_file in tqdm(txt_files, desc="Analyzing bbox sizes"):
        # 获取对应的图像文件
        img_file = txt_file.replace('.txt', '.png')
        if img_file not in img_files:
            continue
            
        img_path = img_files[img_file]
        
        # 获取图像尺寸
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        img_area = img_width * img_height

        # 读取标注文件
        txt_path = os.path.join(txt_folder, txt_file)
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            # 提取坐标点和类别名称
            points = list(map(float, parts[:8]))
            label = parts[8]

            # 计算多边形面积
            polygon = np.array([(points[i], points[i+1]) for i in range(0, len(points), 2)], dtype=np.float32)
            area = cv2.contourArea(polygon)
            
            # 计算占图像面积的百分比
            area_ratio = (area / img_area) * 100
            
            # 存储统计结果
            class_stats[label].append(area_ratio)

    # 计算每个类别的百分比范围
    result = {}
    for label, ratios in class_stats.items():
        if ratios:  # 确保列表不为空
            result[label] = {
                'min': round(min(ratios), 2),
                'max': round(max(ratios), 2),
                'mean': round(np.mean(ratios), 2),
                'median': round(np.median(ratios), 2),
                'count': len(ratios)
            }
    
    return result

# 使用示例
txt_folder = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/annfiles_obb'
img_folder = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/images'

stats = analyze_bbox_sizes(txt_folder, img_folder)

# 打印统计结果
print("目标框大小占图像百分比统计:")
for label, data in stats.items():
    print(f"类别 '{label}':")
    print(f"  数量: {data['count']}")
    print(f"  最小占比: {data['min']}%")
    print(f"  最大占比: {data['max']}%")
    print(f"  平均占比: {data['mean']}%")
    print(f"  中位数占比: {data['median']}%")
    print("-" * 40)