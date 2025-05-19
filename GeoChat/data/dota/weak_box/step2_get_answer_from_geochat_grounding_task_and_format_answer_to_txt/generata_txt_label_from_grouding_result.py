import json
import re
import os
import math
import numpy as np
from pathlib import Path
import cv2

def parse_bbox_string(answer):
    """解析answer字符串中的边界框信息"""
    pattern = r'\{<(\d+)><(\d+)><(\d+)><(\d+)>\|<(-?\d+)>\}'
    matches = re.findall(pattern, answer)
    
    bboxes = []
    for match in matches:
        x1, y1, x2, y2, angle = map(float, match)
        bboxes.append((x1, y1, x2, y2, angle))
    
    return bboxes

# 定义类别列表
CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter')

def process_condition1(answer):
    pattern_seg = r'(<p>.*?</p>)(.*?)(?=<p>|$)'
    matches_seg = re.findall(pattern_seg, answer, re.DOTALL)

    results = []

    for i, (category_text, bbox_text) in enumerate(matches_seg):
        normalized_text = category_text.lower().replace(' ', '-')
        category = 'unknown'
        for c in CLASSES:
            if c in normalized_text:
                category = c
                break

        pattern_bbox = r'\{<(\d+)><(\d+)><(\d+)><(\d+)>\|<(-?\d+)>\}'
        matches_bbox = re.findall(pattern_bbox, bbox_text)

        for bbox in matches_bbox:
            x1, y1, x2, y2, angle = map(float, bbox)
            results.append({
                'category': category,
                'bbox': (x1, y1, x2, y2, angle)
            })

    return results

def process_condition2(answer):
    results = []

    # 正则匹配类别和 bbox 格式：类别 {<x1><y1><x2><y2>|<angle>}
    pattern_bbox = r'([a-zA-Z\- ]+)\s*\{<(\d+)><(\d+)><(\d+)><(\d+)>\|<(-?\d+)>\}'
    matches = re.findall(pattern_bbox, answer)

    for match in matches:
        raw_category, x1, y1, x2, y2, angle = match
        # 统一格式：小写 + 去空格 + 转连字符
        normalized = raw_category.strip().lower().replace(' ', '-')

        # 匹配到类别
        category = 'unknown'
        for c in CLASSES:
            if c == normalized:
                category = c
                break

        results.append({
            'category': category,
            'bbox': (float(x1), float(y1), float(x2), float(y2), float(angle))
        })

    return results

def process_condition3(answer):
    results = []

    # 使用给定的正则模式匹配所有边界框
    pattern = r'\{<(\d+)><(\d+)><(\d+)><(\d+)>\|<(-?\d+)>\}'
    matches = re.findall(pattern, answer)

    # 遍历所有的匹配项，将其作为 'unknown' 类别
    for match in matches:
        x1, y1, x2, y2, angle = map(float, match)  # 将坐标和角度转换为浮动数
        results.append({
            'category': 'unknown',
            'bbox': (x1, y1, x2, y2, angle)
        })

    return results

import json
import os
import cv2
import numpy as np
from pathlib import Path

CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter')

def process_bboxes(answer, image_size=1024):
    """处理answer中的每段信息，提取类别和边界框"""
    
    if '<p>' in answer:
        results = process_condition1(answer)
    elif re.search(r'\w+\s*\{<\d+><\d+><\d+><\d+>\|<[-\d]+>\}', answer):
        results = process_condition2(answer)
    else:
        results = process_condition3(answer)
    
    # 设定缩放比例
    scale = image_size / 100
    boxes = []
    seen_boxes = set()

    # 遍历提取的结果
    for result in results:
        category = result['category']
        bbox = result['bbox']

        try:
            # 提取坐标和角度
            x1, y1, x2, y2, angle = bbox
            
            # 坐标缩放
            x1_orig = x1 * scale
            y1_orig = y1 * scale
            x2_orig = x2 * scale
            y2_orig = y2 * scale
            
            # 计算中心坐标和宽高
            center_x = (x1_orig + x2_orig) / 2
            center_y = (y1_orig + y2_orig) / 2
            width = abs(x2_orig - x1_orig)
            height = abs(y2_orig - y1_orig)
            
            # 创建旋转矩形
            rotated_rect = ((center_x, center_y), (width, height), angle)
            box_points = cv2.boxPoints(rotated_rect).astype(np.float32)
            
            # 保证边界框在图像尺寸范围内
            box_points[:, 0] = np.clip(box_points[:, 0], 0, image_size-1)
            box_points[:, 1] = np.clip(box_points[:, 1], 0, image_size-1)
            
            box_tuple = tuple(np.round(box_points, 1).flatten())
            
            # 确保每个框唯一
            if box_tuple not in seen_boxes:
                seen_boxes.add(box_tuple)
                boxes.append({
                    'category': category,
                    'points': box_points.tolist()
                })

        except Exception as e:
            print(f"Error processing box: {e}")
    
    # 限制框的数量
    if len(boxes) >= 8:
        print(f"Too many boxes ({len(boxes)}), only keeping first 4")
        boxes = boxes[:4]
    
    return boxes


def write_label_file(output_path, boxes):
    """写入标签文件"""
    with open(output_path, 'w') as f:
        for box in boxes:
            points = box['points']
            line = (f"{points[0][0]:.1f} {points[0][1]:.1f} "
                   f"{points[1][0]:.1f} {points[1][1]:.1f} "
                   f"{points[2][0]:.1f} {points[2][1]:.1f} "
                   f"{points[3][0]:.1f} {points[3][1]:.1f} "
                   f"{box['category']} 1\n")
            f.write(line)


def process_json_file(json_path, output_dir, image_size=1024, verbose=True):
    """处理JSON文件"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                question_id = data['question_id']
                answer = data.get('answer', '')
                
                boxes = process_bboxes(answer, image_size) if answer else []
                
                output_path = os.path.join(output_dir, question_id)
                write_label_file(output_path, boxes)
                
                if verbose:
                    original_count = len(parse_bbox_string(answer)) if answer else 0
                    box_count = len(boxes)
                    status = f"with {box_count} boxes (original: {original_count})"
                    print(f"Processed: {question_id} {status}")
                    
            except Exception as e:
                print(f"Error processing line: {e}")
                if 'question_id' in locals():
                    open(os.path.join(output_dir, question_id), 'w').close()

if __name__ == "__main__":
    INPUT_JSON = '/workspace/GeoChat/data/dota/weak_box/step2_get_answer_from_geochat_grounding_task_and_format_answer_to_txt/box_answer_dota.jsonl'
    OUTPUT_DIR = '/workspace/GeoChat/data/dota/weak_box/step2_get_answer_from_geochat_grounding_task_and_format_answer_to_txt/raw_box_txt_format'
    IMAGE_SIZE = 1024
    VERBOSE = True
    
    process_json_file(INPUT_JSON, OUTPUT_DIR, IMAGE_SIZE, VERBOSE)





    # answer = '<p>1 white plane</p> {<49><48><57><56>|<90>}<delim>{<35><54><43><62>|<90>}<delim>{<24><56><32><64>|<90>}<delim>{<18><60><26><68>|<90>}<delim>{<13><63><21><71>|<90>}  at the center\n<p>1 silver large-vehicle</p> {<70><93><74><97>|<90>} at the bottom\n<p>1 white large-vehicle</p> {<70><90><74><94>|<90>} at the bottom\n<p>1 white large-vehicle</p> {<70><87><74><91>|<90>} at the bottom\n<p>1 white large-vehicle</p> {<70><8'
    # answer = '1. <p>1 white small vehicle</p> {<45><93><49><93>|<90>}\n2. <p>1 white small vehicle</p> {<45><95><49><95>|<90>}\n3. <p>1 white small vehicle</p> {<45><97><49><97>|<90>}\n4. <p>1 white small vehicle</p> {<45><99><49><99>|<90>}\n5. <p>1 white small vehicle</p> {<45><96><49><96>|<90>}\n6. <p>1 white small vehicle</p> {<45><98><49><98>|<90>}\n7. <p>1 white small vehicle</p> {<45><99><49><99>|<90>}\n8. <p>1 white small vehicle</p> {<45><96><49><9'
    # answer = '1. plane {<88><3><92><9>|<90>}\n2. tennis-court {<5><60><21><76>|<90>}\n3. tennis-court {<10><70><26><86>|<90>}\n4. tennis-court {<25><55><41><69>|<90>}\n5. tennis-court {<40><55><56><69>|<90>}\n6. tennis-court {<56><45><72><59>|<90>}\n7. tennis-court {<71><45><87><59>|<90>}\n8. tennis-court {<84><45><100><59>|<90>}\n9. tennis-court {<10><55><26><69>|<90>}\n10. tennis-court {<18><55><34><69>|<90>'
    # answer = 'bridge {<68><69><76><73>|<90>}\n\nplane {<70><67><74><71>|<90>}\n\n1 bridge at the bottom right\n\n1 bridge at the center\n\n1 bridge at the left\n\n1 bridge at the top right\n\n1 bridge at the top left\n\n1 plane at the bottom\n\n1 plane at the bottom right\n\n1 plane at the center\n\n1 plane at the left\n\n1 plane at the top left\n\n1 soccer-ball-field at the bottom right\n\n1 soccer-ball-field at the top left\n\n1 storage-tank at the top right\n\n1 tennis-court at the top left'
    # answer = 'ship {<30><86><38><90>|<90>}'
    # answer = '{<27><60><39><72>|<90>}'
    # if '<p>' in answer:
    #     print('condition1')
    #     results = process_condition1(answer)
    #     for result in results:
    #         print(result)
    # elif re.search(r'\w+\s*\{<\d+><\d+><\d+><\d+>\|<[-\d]+>\}', answer):
    #     print('condition2')
    #     results = process_condition2(answer)
    #     for result in results:
    #         print(result)
    # else:
    #     print('condition3')
    #     results = process_condition3(answer)
    #     for result in results:
    #         print(result)


