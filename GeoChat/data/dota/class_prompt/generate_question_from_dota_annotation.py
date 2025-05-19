import os
import json
import argparse

def extract_ground_truth(txt_file):
    """ 从txt文件中提取类别信息（类别是文件中的最后一个字段） """
    with open(txt_file, 'r') as file:
        content = file.readlines()
    
    # 获取最后一列作为类别
    labels = set()
    for line in content:
        parts = line.strip().split()
        if len(parts) >= 2:
            labels.add(parts[-2])  # 类别在倒数第二个字段
        
    # 返回类别（如果没有类别，则返回空）
    return list(labels) if labels else None

def process_files(input_folder, output_file, custom_text):
    """ 处理文件夹中的所有txt文件，并将数据写入jsonl文件 """
    with open(output_file, 'w') as jsonl_file:
        # 遍历文件夹中的所有txt文件
        for txt_filename in os.listdir(input_folder):
            if txt_filename.endswith('.txt'):
                # 构建每个txt文件的路径
                txt_filepath = os.path.join(input_folder, txt_filename)
                
                # 提取ground_truth（类别）
                ground_truth = extract_ground_truth(txt_filepath)
                
                # 获取问题ID和图像文件名（对应 .png 文件）
                question_id = txt_filename
                image_id = txt_filename.replace('.txt', '.png')
                
                # 使用自定义的text内容
                text = custom_text

                # 创建jsonl格式数据
                data = {
                    "question_id": question_id,
                    "image": image_id,
                    "text": text,
                    "ground_truth": ", ".join(ground_truth) if ground_truth else "",  # 如果没有ground_truth，设置为空字符串
                }

                # 写入jsonl文件
                jsonl_file.write(json.dumps(data) + "\n")
                print(f"Processed {question_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-folder", type=str, default="/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/annfiles_obb", help="路径到txt文件夹"
    )
    parser.add_argument(
        "--output-file", type=str, default="/workspace/GeoChat/data/dota/test_different_question/dota_trainval_questionv2.jsonl", help="输出jsonl文件的路径"
    )
    parser.add_argument(
        "--text", type=str, default="What objects are in this image? Choose from the following: plane, baseball-diamond, bridge, ground-track-field, small-vehicle, large-vehicle, ship, tennis-court, basketball-court, storage-tank, soccer-ball-field, roundabout, harbor, swimming-pool, helicopter, none. \n Answer in one word or a short phrase.", help="自定义的text内容"
    )
    args = parser.parse_args()

    # 调用处理函数
    process_files(args.input_folder, args.output_file, args.text)
