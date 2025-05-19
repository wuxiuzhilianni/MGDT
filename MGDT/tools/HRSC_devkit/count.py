import os
import xml.etree.ElementTree as ET
import argparse
from collections import defaultdict

def count_annotations(file_path, stats):
    """
    统计 XML 文件中的目标标注数量。

    参数：
    - file_path: XML 文件路径
    - stats: 统计信息字典
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    objects = root.find("HRSC_Objects")
    
    num_objects = len(objects.findall("HRSC_Object")) if objects is not None else 0
    stats[num_objects] += 1

def main(input_dir, txt_file):
    """
    读取 txt 文件，统计 XML 文件中目标的个数。

    参数：
    - input_dir: XML 文件所在文件夹
    - txt_file: 指定需要统计的 XML 文件名列表
    """
    stats = defaultdict(int)
    
    # 读取 txt 文件，获取 XML 文件名（不包含后缀）
    with open(txt_file, "r") as f:
        xml_files_to_count = {line.strip() + ".xml" for line in f}
    
    # 处理 XML 文件
    for filename in xml_files_to_count:
        file_path = os.path.join(input_dir, filename)
        if os.path.exists(file_path):
            count_annotations(file_path, stats)
    
    # 按标注数量排序并打印结果
    print("\n=== XML 标注统计结果 ===")
    print(f"{'标注个数':<10}{'XML 文件数':<10}")
    print("-" * 25)
    
    for num_objects in sorted(stats.keys()):
        print(f"{num_objects:<10}{stats[num_objects]:<10}")
    
    print("======================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 TXT 指定的 XML 文件中标注的个数")
    parser.add_argument("--input", default='/workspace/Dataset/HRSC2016/FullDataSet/Annotations5%', help="输入 XML 文件夹")
    parser.add_argument("--txt", default='/workspace/Dataset/HRSC2016/ImageSets/trainval.txt', help="包含 XML 文件名的 TXT 文件")
    
    args = parser.parse_args()
    main(args.input, args.txt)