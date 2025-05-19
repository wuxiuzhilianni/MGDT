import json
import os
from pathlib import Path

# 定义类别列表
CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter')

def process_files(json_path, txt_input_dir, txt_output_dir):
    """处理JSON和TXT文件"""
    # 创建输出目录
    Path(txt_output_dir).mkdir(parents=True, exist_ok=True)
    
    # 首先读取JSON文件，建立question_id到answer的映射
    id_to_answer = {}
    with open(json_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                question_id = data['question_id']
                answer = data.get('answer', '').lower()
                id_to_answer[question_id] = answer
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")
    
    # 处理每个TXT文件
    for txt_file in os.listdir(txt_input_dir):
        if not txt_file.endswith('.txt'):
            continue
        
        # 获取对应的answer
        question_id = txt_file
        answer = id_to_answer.get(question_id, 'none')
        
        input_path = os.path.join(txt_input_dir, txt_file)
        output_path = os.path.join(txt_output_dir, txt_file)
        
        # 处理文件内容
        with open(input_path, 'r') as f_in:
            lines = f_in.readlines()
        
        new_lines = []
        if answer == 'none':
            # answer为none，清空文件
            pass
        else:
            # 检查每行是否符合answer类别
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 9:  # 确保是有效标注行
                    line_class = parts[8].lower()
                    if line_class in CLASSES and line_class == answer:
                        new_lines.append(line)
        
        # 写入新文件
        with open(output_path, 'w') as f_out:
            f_out.writelines(new_lines)
        
        print(f"Processed: {txt_file} (answer: {answer}) - kept {len(new_lines)} lines")

# 使用示例
if __name__ == "__main__":
    JSON_PATH = '/workspace/GeoChat/data/dota/class_prompt/dota_trainval_answerv1.jsonl'  # 替换为你的JSON文件路径
    TXT_INPUT_DIR = '/workspace/GeoChat/data/dota/weak_box/step3_use_remoteclip_to_refine_raw_box/bbox_after_clip'  # 替换为原始TXT文件夹路径
    TXT_OUTPUT_DIR = '/workspace/GeoChat/data/dota/weak_box/step4_use_class_prompt_to_refine_box/bbox_after_class_prompt'  # 替换为输出TXT文件夹路径
    
    process_files(JSON_PATH, TXT_INPUT_DIR, TXT_OUTPUT_DIR)