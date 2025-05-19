import os
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_label_file(label_file, label_dir):
    """
    处理单个标注文件，提取图像对应的类别信息。

    Args:
        label_file (str): 标注文件名。
        label_dir (str): 标注文件所在目录。

    Returns:
        tuple: 图像名称和类别列表。
    """
    file_path = os.path.join(label_dir, label_file)
    image_name = label_file.replace('.txt', '.png')  # 假设对应图像为 .jpg 格式

    classes = set()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            category = parts[-2]  # 假设类别位于倒数第二列
            classes.add(category)

    return image_name, list(classes)

def extract_annotations_to_pt(label_dir, output_file, max_workers=8):
    """
    提取每张图像中所有类别并保存到 .pt 文件，支持多线程和进度条。

    Args:
        label_dir (str): 标注文件所在目录。
        output_file (str): 输出文件路径，用于保存类别映射。
        max_workers (int): 最大线程数，默认为 8。
    """
    annotations = {}

    # 获取标注文件列表
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    # 使用线程池并显示进度条
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_label_file, label_file, label_dir): label_file for label_file in label_files}
        for future in tqdm(futures, desc="Processing annotations", unit="file"):
            try:
                image_name, classes = future.result()
                annotations[image_name] = classes
            except Exception as e:
                print(f"Error processing file: {futures[future]} - {e}")

    
    # 保存为 .pt 文件
    torch.save(annotations, output_file)

# 示例调用
label_dir = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/annfiles_obb'  # 替换为标注文件的路径
output_file = '/workspace/animax/MCL/tools/Assinger_Assistent/image_class_prompt.pt'  # 保存的映射文件路径
extract_annotations_to_pt(label_dir, output_file)
