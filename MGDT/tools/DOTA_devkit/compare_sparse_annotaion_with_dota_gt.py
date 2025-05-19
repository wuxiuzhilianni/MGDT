import os

def extract_classes_from_annotation(file_path):
    """
    从标注文件中提取类别。
    
    Args:
        file_path (str): 标注文件路径。
    
    Returns:
        set: 提取的类别集合。
    """
    classes = set()
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 9:  # 确保行格式正确
                category = parts[8]  # 第9列为类别
                classes.add(category)
    return classes

def compare_folders(folder_a, folder_b):
    """
    对比两个文件夹中的类别差异。
    
    Args:
        folder_a (str): 文件夹 A 的路径。
        folder_b (str): 文件夹 B 的路径。
    
    Returns:
        dict: 包含类别统计结果的字典。
    """
    # 获取文件夹中所有文件名
    files_a = set(os.listdir(folder_a))
    files_b = set(os.listdir(folder_b))
    
    # 确保两个文件夹中的文件名称一致
    common_files = files_a.intersection(files_b)
    if len(common_files) != len(files_a) or len(common_files) != len(files_b):
        print("Warning: 两个文件夹中的文件名不完全一致！")
    
    # 初始化统计结果
    stats = {
        "a_extra": 0,  # A 中存在多余类别的文件数
        "b_extra": 0,  # B 中存在多余类别的文件数
        "same": 0,     # 两者类别完全相同的文件数
    }
    
    # 遍历所有共同文件
    for file_name in common_files:
        path_a = os.path.join(folder_a, file_name)
        path_b = os.path.join(folder_b, file_name)
        
        # 提取类别
        classes_a = extract_classes_from_annotation(path_a)
        classes_b = extract_classes_from_annotation(path_b)
        
        # 统计类别差异
        if classes_a == classes_b:
            stats["same"] += 1
        else:
            if classes_a - classes_b:  # A 中存在 B 中没有的类别
                stats["a_extra"] += 1
            if classes_b - classes_a:  # B 中存在 A 中没有的类别
                stats["b_extra"] += 1
    
    return stats

# 示例调用
folder_a = '/workspace/Dataset/DOTAv1_Split/sparse/labelTxt_0.05'  # 替换为文件夹 A 的路径
folder_b = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/annfiles_obb'  # 替换为文件夹 B 的路径

results = compare_folders(folder_a, folder_b)

print("统计结果：")
print(f"A 中存在多余类别的文件数: {results['a_extra']}")
print(f"B 中存在多余类别的文件数: {results['b_extra']}")
print(f"两者类别完全相同的文件数: {results['same']}")
