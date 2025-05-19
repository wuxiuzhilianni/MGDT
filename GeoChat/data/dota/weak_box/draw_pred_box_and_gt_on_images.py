import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def draw_annotations(image, txt_path, color, label_tag):
    if not os.path.exists(txt_path):
        return image

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue

        points = list(map(float, parts[:8]))
        points = [(int(points[i]), int(points[i + 1])) for i in range(0, len(points), 2)]
        label = parts[8]

        cv2.polylines(image, [np.array(points)], isClosed=True, color=color, thickness=2)
        text = f"{label_tag}: {label}"
        cv2.putText(image, text, points[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)
    return image

def process_single_image(gt_txt, pred_txt, img_path, output_path):
    # 判断预测文件是否存在且非空
    if not os.path.exists(pred_txt):
        return
    with open(pred_txt, 'r') as f:
        if all(line.strip() == '' for line in f.readlines()):
            return

    image = cv2.imread(img_path)
    if image is None:
        print(f"Error reading image: {img_path}")
        return

    image = draw_annotations(image, gt_txt, color=(0, 0, 255), label_tag="GT")
    image = draw_annotations(image, pred_txt, color=(255, 0, 0), label_tag="Pred")

    cv2.imwrite(output_path, image)

def draw_gt_and_pred(gt_folder, pred_folder, img_folder, output_folder, num_threads=8):
    os.makedirs(output_folder, exist_ok=True)

    txt_files = [f for f in os.listdir(gt_folder) if f.endswith('.txt')]
    img_files = {f: os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png')}

    tasks = []
    for txt_file in txt_files:
        img_file = txt_file.replace('.txt', '.png')
        if img_file not in img_files:
            print(f"Warning: Image {img_file} not found for {txt_file}")
            continue

        gt_txt = os.path.join(gt_folder, txt_file)
        pred_txt = os.path.join(pred_folder, txt_file)
        img_path = img_files[img_file]
        output_path = os.path.join(output_folder, img_file)

        tasks.append((gt_txt, pred_txt, img_path, output_path))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(lambda x: process_single_image(*x), tasks),
                  total=len(tasks), desc="Processing images"))

# 使用示例
gt_folder = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/annfiles_obb'
pred_folder = '/workspace/GeoChat/data/dota/pseudo_box/grounding_method3/weakly_box0.4'
img_folder = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/images'
output_folder = '/workspace/GeoChat/data/dota/pseudo_box/grounding_method3/visual_weakly_box0.4'

draw_gt_and_pred(gt_folder, pred_folder, img_folder, output_folder, num_threads=8)
