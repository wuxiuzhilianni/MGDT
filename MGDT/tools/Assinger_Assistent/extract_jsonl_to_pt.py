import json
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 预定义的有效类别
VALID_CLASSES = {
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
}

# def process_jsonl_line(line):
#     """
#     处理单行 JSONL 数据，提取图像名称和类别信息。
    
#     Args:
#         line (str): JSONL 文件中的一行。
    
#     Returns:
#         tuple: 图像名称和类别列表。如果没有有效类别，则返回空列表。
#     """
#     # 解析 JSON 行
#     record = json.loads(line.strip())
#     image_name = record['image_id']
#     answer = record['answer']

#     # 如果 answer 为 'none' 或不在有效类别中，则返回空列表表示没有有效类别
#     if answer == 'none' or answer not in VALID_CLASSES:
#         classes = []  # 返回空列表，表示该图像没有有效类别
#     else:
#         classes = [answer]

#     return image_name, classes

def process_jsonl_line(line):
    """
    处理单行 JSONL 数据，提取图像名称和类别信息。
    
    Args:
        line (str): JSONL 文件中的一行。
    
    Returns:
        tuple: 图像名称和类别列表。如果没有有效类别，则返回空列表。
    """
    # 解析 JSON 行
    record = json.loads(line.strip())
    image_name = record['image_id']
    answer = record['answer']

    # 如果 answer 为 'none'，返回空列表
    if answer == 'none':
        classes = []
    else:
        # 将 answer 按逗号分割为列表并过滤无效类别
        classes = [cls.strip() for cls in answer.split(',') if cls.strip() in VALID_CLASSES]

    return image_name, classes

def extract_annotations_from_jsonl(jsonl_file, output_file, max_workers=8):
    """
    从 JSONL 文件中提取每张图像的类别信息并保存到 .pt 文件，支持多线程和进度条。
    
    Args:
        jsonl_file (str): JSONL 文件的路径。
        output_file (str): 输出文件路径，用于保存类别映射。
        max_workers (int): 最大线程数，默认为 8。
    """
    annotations = {}

    # 使用线程池并显示进度条
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        with open(jsonl_file, 'r') as file:
            for line in file:
                futures[executor.submit(process_jsonl_line, line)] = line

        # 处理所有任务
        for future in tqdm(futures, desc="Processing annotations", unit="file"):
            try:
                image_name, classes = future.result()
                annotations[image_name] = classes
            except Exception as e:
                print(f"Error processing file: {futures[future]} - {e}")

    # 保存为 .pt 文件
    torch.save(annotations, output_file)

# 示例调用
jsonl_file = '/workspace/animax/GeoChat/data/dota/dota_trainval_answerv1_with_percent1_label.jsonl'  # 替换为 JSONL 文件路径
output_file = '/workspace/animax/MCL/tools/Assinger_Assistent/image_class_prompt_from_chat_with_percent1_label_modified.pt'  # 替换为保存的 .pt 文件路径
extract_annotations_from_jsonl(jsonl_file, output_file)
