import os
import json
import cv2  # 用于读取图像尺寸

# 类别 ID 到类别名称的映射
class_id_to_name = {
    '0': 'car',
    '1': 'truck',
    '2': 'van',
    '3': 'long vehicle',
    '4': 'bus',
    '5': 'airliner',
    '6': 'propeller aircraft',
    '7': 'trainer aircraft',
    '8': 'charted aircraft',
    '9': 'fighter aircraft',
    '10': 'others',
    '11': 'stair truck',
    '12': 'pushback truck',
    '13': 'helicopter',
    '14': 'boat'
}

def convert_txt_to_jsonl(input_folder, output_file):
    question_id_counter = 1  # 初始化 question_id 计数器
    
    with open(output_file, 'w') as outfile:
        # 遍历输入文件夹中的每个文件
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                image_id = filename.split('.')[0]  # 从文件名中提取图像 ID
                
                # 构建对应的图像文件路径
                image_path = os.path.join(input_folder, image_id + '.jpg')
                
                # 检查图像文件是否存在
                if not os.path.exists(image_path):
                    print(f"Warning: Image file {image_path} does not exist.")
                    continue

                # 读取图像的宽度和高度
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error: Could not open or find the image {image_path}.")
                    continue

                height, width = img.shape[:2]
                
                # 读取 txt 文件的内容
                with open(os.path.join(input_folder, filename), 'r') as infile:
                    for line in infile:
                        parts = line.strip().split()
                        
                        # 提取边界框的中心坐标和宽高，并转换为浮点数
                        class_id = parts[0]
                        x_center, y_center, box_width, box_height = map(float, parts[1:])

                        # 计算实际边界框坐标
                        x_center *= width
                        y_center *= height
                        box_width *= width
                        box_height *= height

                        x1 = int(x_center - box_width / 2)
                        y1 = int(y_center - box_height / 2)
                        x2 = int(x_center + box_width / 2)
                        y2 = int(y_center + box_height / 2)

                        # 创建格式化的 question 字符串
                        question = f"{{<{x1}><{y1}><{x2}><{y2}>}}"
                        
                        # 将类别 ID 转换为类别名称
                        ground_truth = class_id_to_name[class_id]
                        
                        # 创建输出字典
                        output_dict = {
                            "image_id": image_id,
                            "question": question,
                            "dataset": "simd",
                            "question_id": f"simd_{question_id_counter:05d}",
                            "ground_truth": ground_truth
                        }
                        
                        # 将输出字典写入 JSONL 文件
                        json.dump(output_dict, outfile)
                        outfile.write('\n')
                        
                        # 更新 question_id 计数器
                        question_id_counter += 1

# 设置输入文件夹和输出文件
input_folder = '/workspace/Project/OVDSAT/data/simd/training'  # 替换为您的 txt 文件夹路径
output_file = 'output.jsonl'

# 调用函数
convert_txt_to_jsonl(input_folder, output_file)
