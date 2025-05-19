import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# 指定文件夹路径
folder_path = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/labelTxt_0.01'

def process_file(file_path):
    category_count = Counter()
    total_annotations = 0
    
    # 批量读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 8:  # 确保是一个annotation
            category = parts[-2]  # 获取类别
            category_count[category] += 1
            total_annotations += 1  # 增加总annotation的计数
            
    return category_count, total_annotations

# 用于统计每个类别的数量
overall_category_count = Counter()
total_annotations = 0

# 遍历文件夹中的所有txt文件
with ProcessPoolExecutor() as executor:
    futures = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            futures.append(executor.submit(process_file, file_path))

    for future in futures:
        category_count, annotations = future.result()
        overall_category_count.update(category_count)
        total_annotations += annotations

# 打印每个类别出现的次数
for category, count in overall_category_count.items():
    print(f"Category: {category}, Count: {count}")

# 打印总的annotation数量
print(f"Total annotations: {total_annotations}")

