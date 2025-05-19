import os
import numpy as np
from collections import defaultdict
import argparse
from tqdm import tqdm
"""
对于每个 class, 统计 annotation 的数量, 并随机选取 x%
"""
def split_dota_annotations_by_gt(input_dir, percent=10.0, seed=1, output_dir='split_annotations', log_file='log.md'):
    """
    Split DOTA annotations by retaining a percentage of GT boxes per category.
    
    Args:
        input_dir: Directory with DOTA label txt files.
        percent: Percentage of GT boxes to retain.
        seed: Random seed for reproducibility.
        output_dir: Directory for saving the split annotations.
        log_file: File for logging statistics.
    """
    np.random.seed(seed)  # Ensure reproducibility
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Counters and storage for annotations
    total_counts = defaultdict(int)
    remaining_counts = defaultdict(int)
    annotations_by_category = defaultdict(list)
    log_messages = []

    # Get all .txt files from the input directory
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    # Process each annotation file
    for file_name in tqdm(all_files, desc='Processing annotations'):
        with open(os.path.join(input_dir, file_name), 'r') as f:
            lines = f.readlines()

            for line in lines:
                if line.startswith('imagesource') or line.startswith('gsd'):
                    continue  # 跳过冗余信息行
                parts = line.strip().split()
                if len(parts) == 10:  # 有效行包含8个坐标、类别和难度
                    category = parts[8]
                    total_counts[category] += 1
                    annotations_by_category[category].append((file_name, line))
                else:
                    log_messages.append(f'Invalid line in {file_name}: {line.strip()}')

    # Select annotations to retain based on the percentage
    selected_annotations = defaultdict(list)
    for category, annotations in annotations_by_category.items():
        total = len(annotations)
        retain_count = max(1, int(total * percent / 100.0))  # Always retain at least one annotation if annotations of the class < 100
        selected_indices = np.random.choice(total, retain_count, replace=False)
        
        remaining_counts[category] = retain_count
        for idx in selected_indices:
            file_name, line = annotations[idx]
            selected_annotations[file_name].append(line)

    # Write the selected annotations to new files, or create empty files
    for file_name in all_files:
        with open(os.path.join(output_dir, file_name), 'w') as f:
            f.writelines(selected_annotations.get(file_name, []))  # Write annotations or leave empty

    # Log category-wise statistics
    log_messages.append(f'Category-wise statistics after {percent}% split:')
    for category in total_counts:
        log_messages.append(f'{category}: {remaining_counts[category]} retained out of {total_counts[category]}')

    # 确保日志目录存在
    log_file_path = os.path.join(output_dir, 'log.md')
    with open(log_file_path, 'w') as log_f:
        log_f.write('\n'.join(log_messages))

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Split DOTA annotations by ground truth boxes.')
    # parser.add_argument('--input_dir', type=str, default='/opt/data/private/LiaoWei/MCL/data/split_ss_dota/train/annfiles_obb', help='Directory containing the DOTA label txt files.')
    parser.add_argument('--input_dir', type=str, default='/opt/data/private/LiaoWei/OpenDataLab___DOTA_V1_dot_0/raw/DOTA_V1.0/train/labelTxt-v1.0/labelTxt', help='Directory containing the DOTA label txt files.')
    parser.add_argument('--percent', type=float, default=70.0, help='Percentage of GT boxes to retain.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility.')
    # parser.add_argument('--output_dir', type=str, default='/opt/data/private/LiaoWei/MCL/data/split_ss_dota/train/annfiles_obb_split_annotations', help='Directory to save the split annotations.')
    parser.add_argument('--output_dir', type=str, default='/opt/data/private/LiaoWei/OpenDataLab___DOTA_V1_dot_0/raw/DOTA_V1.0/train/labelTxt-v1.0/labelTxt_split_annotations', help='Directory to save the split annotations.')

    args = parser.parse_args()

    # Add percentage to output directory name
    output_dir_with_percent = f"{args.output_dir}_percent_{int(args.percent)}"
    
    # Call the function with parsed arguments
    split_dota_annotations_by_gt(args.input_dir, args.percent, args.seed, output_dir_with_percent)
