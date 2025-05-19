import os
import xml.etree.ElementTree as ET
import argparse
import random
from collections import defaultdict

def process_xml(file_path, output_path, ratio, stats):
    tree = ET.parse(file_path)
    root = tree.getroot()
    objects = root.find("HRSC_Objects")

    if objects is None:
        return  # 没有目标，直接跳过

    # 按类别统计所有目标
    categories = defaultdict(list)
    for obj in objects.findall("HRSC_Object"):
        class_id = obj.find("Class_ID").text
        categories[class_id].append(obj)

    # 统计信息
    category_stats = {}

    # 随机删除部分标注
    for class_id, obj_list in categories.items():
        total = len(obj_list)
        num_keep = max(1, int(total * ratio) + 1)  # 至少保留 1 个
        random.shuffle(obj_list)
        keep_objs = obj_list[:num_keep]
        delete_objs = obj_list[num_keep:]

        # 记录统计数据
        category_stats[class_id] = {
            "total": total,
            "kept": len(keep_objs),
            "deleted": len(delete_objs),
        }

        # 删除 XML 中的多余目标
        for obj in delete_objs:
            objects.remove(obj)

    # 保存新的 XML
    os.makedirs(output_path, exist_ok=True)
    new_file_path = os.path.join(output_path, os.path.basename(file_path))
    tree.write(new_file_path)

    # 更新全局统计信息
    for class_id, stats_dict in category_stats.items():
        stats[class_id]["total"] += stats_dict["total"]
        stats[class_id]["kept"] += stats_dict["kept"]
        stats[class_id]["deleted"] += stats_dict["deleted"]

def main(input_dir, output_dir, ratio):
    stats = defaultdict(lambda: {"total": 0, "kept": 0, "deleted": 0})

    # 处理所有 XML 文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".xml"):
            file_path = os.path.join(input_dir, filename)
            process_xml(file_path, output_dir, ratio, stats)

    # 计算所有类别的总和
    total_all = sum(data["total"] for data in stats.values())
    kept_all = sum(data["kept"] for data in stats.values())
    deleted_all = sum(data["deleted"] for data in stats.values())

    # 打印统计信息（美观格式）
    print("\n=== 统计结果 ===")
    print(f"{'类别':<10}{'总数':<10}{'保留数':<10}{'删除数':<10}")
    print("-" * 40)
    for class_id, data in sorted(stats.items()):
        print(f"{class_id:<10}{data['total']:<10}{data['kept']:<10}{data['deleted']:<10}")
    print("-" * 40)
    print(f"{'总计':<10}{total_all:<10}{kept_all:<10}{deleted_all:<10}")
    print("================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="随机删除 XML 目标标注")
    parser.add_argument("--input", default='/workspace/Dataset/HRSC2016/Train/Annotations', help="输入 XML 文件夹")
    parser.add_argument("--output", default='/workspace/animax/MCL/tools/HRSC_devkit/Annotations10%', help="输出 XML 文件夹")
    parser.add_argument("--ratio", type=float, default=0.3, help="保留比例（0~1）")

    args = parser.parse_args()
    main(args.input, args.output, args.ratio)



"""
=== 统计结果 ===
类别        总数        保留数       删除数       
----------------------------------------
100000001 386       199       187       
100000002 1         1         0         
100000003 44        33        11        
100000004 1         1         0         
100000005 50        44        6         
100000006 20        14        6         
100000007 266       146       120       
100000008 66        56        10        
100000009 132       78        54        
100000010 30        28        2         
100000011 160       102       58        
100000013 8         8         0         
100000015 51        39        12        
100000016 99        71        28        
100000018 34        31        3         
100000019 47        42        5         
100000020 15        15        0         
100000022 55        26        29        
100000024 8         3         5         
100000025 167       108       59        
100000026 6         6         0         
100000027 44        18        26        
100000028 5         5         0         
100000029 15        15        0         
100000030 25        24        1         
100000032 13        13        0         
----------------------------------------
总计      1748      1126      622       
================



=== 统计结果 ===
类别        总数        保留数       删除数       
----------------------------------------
100000001 386       199       187       
100000002 1         1         0         
100000003 44        33        11        
100000004 1         1         0         
100000005 50        44        6         
100000006 20        14        6         
100000007 266       146       120       
100000008 66        56        10        
100000009 132       78        54        
100000010 30        28        2         
100000011 160       102       58        
100000013 8         8         0         
100000015 51        39        12        
100000016 99        71        28        
100000018 34        31        3         
100000019 47        42        5         
100000020 15        15        0         
100000022 55        26        29        
100000024 8         3         5         
100000025 167       108       59        
100000026 6         6         0         
100000027 44        18        26        
100000028 5         5         0         
100000029 15        15        0         
100000030 25        24        1         
100000032 13        13        0         
----------------------------------------
总计        1748      1126      622       
================
"""