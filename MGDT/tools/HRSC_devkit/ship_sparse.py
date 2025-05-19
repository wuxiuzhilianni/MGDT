import os
import xml.etree.ElementTree as ET
import argparse
import random
from collections import defaultdict
from tqdm import tqdm  # 添加进度条支持

def process_xml(file_path, output_path, ratio, stats):
    tree = ET.parse(file_path)
    root = tree.getroot()
    objects = root.find("HRSC_Objects")

    if objects is None:
        return  # 无目标，跳过处理

    obj_list = objects.findall("HRSC_Object")
    total = len(obj_list)

    if total == 0:
        return  # 无目标，跳过处理

    # 确保至少保留 1 个目标
    num_keep = max(1, int(total * ratio) + 1)
    random.shuffle(obj_list)
    keep_objs = obj_list[:num_keep]
    delete_objs = obj_list[num_keep:]

    # 统计数据
    stats["total"] += total
    stats["kept"] += len(keep_objs)

    # 删除 XML 中的多余目标
    for obj in delete_objs:
        objects.remove(obj)

    # 保存新的 XML
    os.makedirs(output_path, exist_ok=True)
    new_file_path = os.path.join(output_path, os.path.basename(file_path))
    tree.write(new_file_path)

def main(input_dir, output_dir, ratio):
    stats = {"total": 0, "kept": 0}

    xml_files = [f for f in os.listdir(input_dir) if f.endswith(".xml")]

    # 处理所有 XML 文件，添加进度条
    for filename in tqdm(xml_files, desc="Processing XMLs"):
        file_path = os.path.join(input_dir, filename)
        process_xml(file_path, output_dir, ratio, stats)

    # 计算删除的数量
    deleted = stats["total"] - stats["kept"]

    # 打印统计信息
    print("\n=== 统计结果 ===")
    print(f"{'总目标数':<10}{'保留数':<10}{'删除数':<10}")
    print("-" * 30)
    print(f"{stats['total']:<10}{stats['kept']:<10}{deleted:<10}")
    print("================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="随机保留 XML 目标标注")
    parser.add_argument("--input", default='/workspace/Dataset/HRSC2016/Train/Annotations', help="输入 XML 文件夹")
    parser.add_argument("--output", default='/workspace/animax/MCL/tools/HRSC_devkit/Annotations1%', help="输出 XML 文件夹")
    parser.add_argument("--ratio", type=float, default=0.05, help="保留比例（0~1）")

    args = parser.parse_args()
    main(args.input, args.output, args.ratio)
