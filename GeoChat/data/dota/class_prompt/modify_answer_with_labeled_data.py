import os
import json
from tqdm import tqdm

def extract_classes_from_annotation(file_path):
    """
    从标注文件中提取类别。
    
    Args:
        file_path (str): 标注文件路径。
    
    Returns:
        set: 提取的类别集合。
    """
    classes = set()
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 9:  # 确保行格式正确
                category = parts[8]  # 第9列为类别
                classes.add(category)
    return classes

def update_answers_in_jsonl(jsonl_file, annotation_folder, output_file):
    """
    根据标注文件夹中的类别信息更新 JSONL 文件的 `answer` 字段。
    
    Args:
        jsonl_file (str): 原始 JSONL 文件路径。
        annotation_folder (str): 标注文件夹路径，其中包含多个 .txt 文件。
        output_file (str): 调整后的 JSONL 文件保存路径。
    """
    updated_records = []
    
    # 遍历 JSONL 文件中的每一行
    with open(jsonl_file, 'r') as infile:
        for line in tqdm(infile, desc="Processing JSONL records"):
            record = json.loads(line.strip())
            image_id = record['image_id']
            base_name = os.path.splitext(image_id)[0] + ".txt"  # 将 .png 替换为 .txt
            
            # 构造标注文件路径
            annotation_path = os.path.join(annotation_folder, base_name)
            
            if os.path.exists(annotation_path):
                # 提取标注文件中的类别
                annotation_classes = extract_classes_from_annotation(annotation_path)
                existing_answer = set(record['answer'].split(',')) if record['answer'] != 'none' else set()
                
                if not existing_answer:  # 原 `answer` 为 `none`
                    record['answer'] = ",".join(sorted(annotation_classes)) if annotation_classes else "none"
                else:  # 原 `answer` 不为 `none`
                    # 将标注文件中的类别补充到原 `answer` 中，并按字典序排序
                    combined_classes = sorted(existing_answer.union(annotation_classes))
                    record['answer'] = ",".join(combined_classes)
            else:
                record['answer'] = record['answer']  # 如果没有对应的标注文件，保持原值
            
            updated_records.append(record)
    
    # 保存更新后的 JSONL 文件
    with open(output_file, 'w') as outfile:
        for record in updated_records:
            json.dump(record, outfile)
            outfile.write('\n')

# 示例调用
jsonl_file = '/workspace/animax/GeoChat/data/dota/dota_trainval_answerv1.jsonl'  # 替换为输入 JSONL 文件路径
annotation_folder = '/workspace/Dataset/DOTAv1_Split/sparse/image_annotation_split_percent2/labeled_annotation'  # 替换为标注文件夹路径
output_file = '/workspace/animax/GeoChat/data/dota/dota_trainval_answerv1_with_percent2_label.jsonl'  # 替换为输出 JSONL 文件路径

update_answers_in_jsonl(jsonl_file, annotation_folder, output_file)
