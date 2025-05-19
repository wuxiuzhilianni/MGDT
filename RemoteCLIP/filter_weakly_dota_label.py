import os
import cv2
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 初始化CLIP模型并预计算text_features
def init_clip_model():
    model_name = 'ViT-L-14'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    
    # 加载预训练权重
    ckpt = torch.load("/workspace/RemoteCLIP/RemoteCLIP-ViT-L-14.pt", map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    
    # 定义类别和文本查询
    categories = [
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank', 'soccer-ball-field',
        'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    ]
    text_queries = [f"A image of {category}" for category in categories]
    
    # 预计算text_features
    with torch.no_grad(), torch.cuda.amp.autocast():
        text = tokenizer(text_queries).cuda()
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return model, preprocess, text_features, categories

# 裁剪图像中的目标对象
def crop_object(image, points):
    """根据给定的点裁剪图像中的目标对象"""
    try:
        points = np.array([(points[i], points[i+1]) for i in range(0, len(points), 2)], dtype=np.float32)
        
        # 检查坐标点是否有效
        if len(points) < 3:
            print(f"Warning: Not enough points for cropping ({len(points)} points)")
            return None
            
        # 计算旋转矩形
        rect = cv2.minAreaRect(points)
        width, height = int(rect[1][0]), int(rect[1][1])
        
        # 检查裁剪尺寸是否合理
        if width <= 0 or height <= 0:
            print(f"Warning: Invalid crop size ({width}x{height})")
            return None
            
        # 获取旋转矩阵并执行旋转
        M = cv2.getRotationMatrix2D(rect[0], rect[2], 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # 裁剪旋转后的图像
        cropped = cv2.getRectSubPix(rotated, (width, height), rect[0])
        
        # 检查裁剪结果是否有效
        if cropped is None or cropped.size == 0:
            print("Warning: Cropping returned empty image")
            return None
            
        return cropped
    except Exception as e:
        print(f"Error in crop_object: {e}")
        return None

def classify_with_clip(model, preprocess, text_features, cropped_image, threshold=0.5):
    """使用CLIP模型对裁剪的图像进行分类"""
    if cropped_image is None:
        print("Warning: Received None image in classify_with_clip")
        return None, 0.0
        
    try:
        # 预处理图像
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        image = preprocess(pil_image).unsqueeze(0)
        
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            # 提取图像特征
            image_features = model.encode_image(image.cuda())
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度概率
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
        
        max_prob_idx = np.argmax(text_probs)
        max_prob = text_probs[max_prob_idx]
        
        if max_prob >= threshold:
            return max_prob_idx, max_prob
        else:
            return None, max_prob
    except Exception as e:
        print(f"Error in classify_with_clip: {e}")
        return None, 0.0

def process_single_file(args):
    """处理单个图像和对应的TXT文件"""
    txt_path, img_path, output_path, model, preprocess, text_features, categories, threshold = args
    
    # 读取图像
    try:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Unable to read image {img_path}")
            # 仍然创建空文件
            open(output_path, 'w').close()
            return None
    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
        # 仍然创建空文件
        open(output_path, 'w').close()
        return None
    
    # 读取TXT文件内容
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading txt file {txt_path}: {e}")
        # 仍然创建空文件
        open(output_path, 'w').close()
        return None
    
    new_lines = []
    for line_idx, line in enumerate(lines):
        try:
            parts = line.strip().split()
            if len(parts) < 9:
                print(f"Invalid format in file {txt_path}: {line.strip()}")
                continue
            
            # 提取坐标点
            points = list(map(float, parts[:8]))
            
            # 裁剪对象
            cropped = crop_object(image, points)
            if cropped is None:
                print(f"Skipping invalid crop in {txt_path} line {line_idx+1}")
                continue
            
            # 使用CLIP分类
            class_idx, max_prob = classify_with_clip(model, preprocess, text_features, cropped, threshold)
            
            original_label = parts[8]
            if class_idx is not None:
                new_label = categories[class_idx]
                if original_label != 'unknown':
                    # 高置信度，但原标签有效 -> 保留原标签
                    new_line = ' '.join(parts[:8] + [original_label] + parts[9:]) + '\n'
                else:
                    # 原标签是 unknown，使用CLIP结果替换
                    new_line = ' '.join(parts[:8] + [new_label] + parts[9:]) + '\n'
                new_lines.append(new_line)
            # 如果 class_idx 为 None（即低置信度），则不写入任何内容，即删除该行

        except Exception as e:
            print(f"Error processing line {line_idx+1} in {txt_path}: {e}")
            continue
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 写入文件（即使为空）
    try:
        with open(output_path, 'w') as f:
            if new_lines:
                f.writelines(new_lines)
            # 如果没有有效行，则创建空文件
    except Exception as e:
        print(f"Error writing to {output_path}: {e}")
    
    return txt_path

# 批量处理数据集
def process_dataset(txt_folder, img_folder, output_folder, threshold=0.5, num_threads=8):
    """批量处理数据集中的所有图像"""
    # 初始化CLIP模型并获取预计算的text_features
    model, preprocess, text_features, categories = init_clip_model()
    
    # 获取txt和png文件列表
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    img_files = {f: os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png')}
    
    # 准备任务列表
    tasks = []
    for txt_file in txt_files:
        img_file = txt_file.replace('.txt', '.png')
        if img_file not in img_files:
            print(f"Warning: Image file {img_file} not found for {txt_file}.")
            continue
        
        txt_path = os.path.join(txt_folder, txt_file)
        img_path = img_files[img_file]
        output_path = os.path.join(output_folder, txt_file)
        
        tasks.append((txt_path, img_path, output_path, model, preprocess, text_features, categories, threshold))
    
    # 多线程处理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(process_single_file, tasks), 
                      total=len(tasks), 
                      desc="Processing images"))
    
    print(f"\nProcessing completed. Results saved to {output_folder}")

# 使用示例
if __name__ == "__main__":
    # 输入输出路径设置
    txt_folder = '/workspace/GeoChat/data/dota/weak_box/step2_get_answer_from_geochat_grounding_task_and_format_answer_to_txt/raw_box_txt_format'
    img_folder = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/images'
    output_folder = '/workspace/GeoChat/data/dota/weak_box/step3_use_remoteclip_to_refine_raw_box/bbox_after_clip'
    
    # 参数设置
    threshold = 0.95  # 分类置信度阈值
    num_threads = 8  # 并行线程数
    
    # 运行处理
    process_dataset(txt_folder, img_folder, output_folder, threshold, num_threads)