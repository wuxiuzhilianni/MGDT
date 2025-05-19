import os
import shutil
import xml.etree.ElementTree as ET
import argparse
import random
from collections import defaultdict

def process_xml(file_path, output_path, ratio, stats):
    """
    处理 XML 文件，按给定比例随机删除目标并保存修改后的文件。

    参数：
    - file_path: 原始 XML 文件路径
    - output_path: 处理后 XML 保存的路径
    - ratio: 保留目标的比例
    - stats: 统计信息字典
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    objects = root.find("HRSC_Objects")

    if objects is None:
        return False  # 没有目标，跳过处理

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

    return True  # 处理成功

def main(input_dir, output_dir, txt_file, ratio):
    """
    读取 txt 文件，按照指定比例处理 XML 文件并保存。

    参数：
    - input_dir: 原始 XML 文件所在文件夹
    - output_dir: 处理后的 XML 保存文件夹
    - txt_file: 指定需要处理的 XML 文件名列表
    - ratio: 保留目标的比例
    """
    stats = defaultdict(lambda: {"total": 0, "kept": 0, "deleted": 0})

    # 读取 txt 文件，获取 XML 文件名（不包含后缀）
    with open(txt_file, "r") as f:
        xml_files_to_modify = {line.strip() + ".xml" for line in f}

    # 获取 input 目录下的所有 XML 文件
    all_xml_files = {f for f in os.listdir(input_dir) if f.endswith(".xml")}

    total_input_files = len(all_xml_files)  # `input` 目录中的 XML 总数
    modified_count = 0  # 记录 `txt` 里列出的 XML 实际处理数量

    os.makedirs(output_dir, exist_ok=True)

    # 遍历 `input` 目录中的 XML
    for filename in all_xml_files:
        file_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if filename in xml_files_to_modify:
            if process_xml(file_path, output_dir, ratio, stats):
                modified_count += 1
        else:
            shutil.copy(file_path, output_path)  # 未列出的 XML 直接复制

    total_output_files = len(os.listdir(output_dir))  # `output` 目录中的 XML 数量

    # 计算所有类别的总和
    total_all = sum(data["total"] for data in stats.values())
    kept_all = sum(data["kept"] for data in stats.values())
    deleted_all = sum(data["deleted"] for data in stats.values())

    # 打印统计信息
    print("\n=== 处理完成 ===")
    print(f"📂 `input` 目录 XML 文件总数: {total_input_files}")
    print(f"📌 `txt` 中列出的 XML 文件数量: {len(xml_files_to_modify)}")
    print(f"✅ 实际修改了 {modified_count} 个 XML")
    print(f"📂 `output` 目录 XML 文件总数: {total_output_files}\n")

    print("=== 统计结果 ===")
    print(f"{'类别':<10}{'总数':<10}{'保留数':<10}{'删除数':<10}")
    print("-" * 40)
    for class_id, data in sorted(stats.items()):
        print(f"{class_id:<10}{data['total']:<10}{data['kept']:<10}{data['deleted']:<10}")
    print("-" * 40)
    print(f"{'总计':<10}{total_all:<10}{kept_all:<10}{deleted_all:<10}")
    print("================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据 TXT 列表随机删除 XML 目标标注")
    parser.add_argument("--input", default='/workspace/Dataset/HRSC2016/FullDataSet/Annotations', help="输入 XML 文件夹")
    parser.add_argument("--output", default='/workspace/animax/MCL/tools/HRSC_devkit/Annotations10%', help="输出 XML 文件夹")
    parser.add_argument("--txt", default='/workspace/Dataset/HRSC2016/ImageSets/trainval.txt', help="包含 XML 文件名的 TXT 文件")
    parser.add_argument("--ratio", type=float, default=0.3, help="保留比例（0~1）")

    args = parser.parse_args()
    main(args.input, args.output, args.txt, args.ratio)
