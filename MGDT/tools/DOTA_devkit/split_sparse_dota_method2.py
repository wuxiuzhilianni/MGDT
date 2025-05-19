import os
import numpy as np
from collections import defaultdict
import argparse
from tqdm import tqdm

def split_dota_annotations_per_image(input_dir, percent=10.0, seed=1, output_dir='split_annotations', log_file='log.md'):
    """
    Split DOTA annotations per image by retaining a percentage of GT boxes per category.
    
    Args:
        input_dir: Directory with DOTA label txt files.
        percent: Percentage of GT boxes to retain.
        seed: Random seed for reproducibility.
        output_dir: Directory for saving the split annotations.
        log_file: File for logging statistics.
    """
    np.random.seed(seed)  # Ensure reproducibility
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Counters for total annotations and retained annotations per category
    total_counts = defaultdict(int)
    retained_counts = defaultdict(int)

    # Get all .txt files from the input directory
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    # Process each annotation file
    for file_name in tqdm(all_files, desc='Processing annotations'):
        annotations_by_category = defaultdict(list)

        with open(os.path.join(input_dir, file_name), 'r') as f:
            lines = f.readlines()

            for line in lines:
                if line.startswith('imagesource') or line.startswith('gsd'):
                    continue  # Skip redundant info lines
                parts = line.strip().split()
                if len(parts) == 10:  # Valid line with coordinates, category, and difficulty
                    category = parts[8]
                    total_counts[category] += 1
                    annotations_by_category[category].append(line)

        # Select annotations to retain based on the percentage for this image
        selected_annotations = []
        for category, annotations in annotations_by_category.items():
            total = len(annotations)
            retain_count = max(1, int(total * percent / 100.0))  # Always retain at least one annotation
            selected_indices = np.random.choice(total, retain_count, replace=False)
            
            retained_counts[category] += retain_count  # Update retained count for the category
            for idx in selected_indices:
                selected_annotations.append(annotations[idx])

        # Write the selected annotations to new files
        with open(os.path.join(output_dir, file_name), 'w') as f:
            f.writelines(selected_annotations)  # Write selected annotations

    # Log category-wise statistics
    log_messages = []
    log_messages.append(f'Category-wise statistics after {percent}% split:')
    for category in total_counts:
        log_messages.append(f'{category}: {retained_counts[category]} retained out of {total_counts[category]}')

    # Log the total retained and original counts
    total_retained = sum(retained_counts.values())
    total_original = sum(total_counts.values())
    log_messages.append(f'Total retained annotations: {total_retained}')
    log_messages.append(f'Total original annotations: {total_original}')
    log_messages.append(f'Retention ratio: {total_retained / total_original:.2f}' if total_original > 0 else 'No original annotations.')

    # Ensure log directory exists
    log_file_path = os.path.join(output_dir, 'log.md')
    with open(log_file_path, 'w') as log_f:
        log_f.write('\n'.join(log_messages))


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Split DOTA annotations by ground truth boxes per image.')
    # parser.add_argument('--input_dir', type=str, default='/opt/data/private/LiaoWei/OpenDataLab___DOTA_V1_dot_0/raw/DOTA_V1.0/train/labelTxt-v1.0/labelTxt', help='Directory containing the DOTA label txt files.')
    parser.add_argument('--input_dir', type=str, default='/workspace/Dataset/DOTAv1_Split/sparse/image_annotation_split_percent5/unlabeled_annotation_with_label', help='Directory containing the DOTA label txt files.')
    parser.add_argument('--percent', type=float, default=10.0, help='Percentage of GT boxes to retain.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility.')
    # parser.add_argument('--output_dir', type=str, default='/opt/data/private/LiaoWei/OpenDataLab___DOTA_V1_dot_0/raw/DOTA_V1.0/train/labelTxt-v1.0/Split_Method2/labelTxt_split_annotations', help='Directory to save the split annotations.')
    parser.add_argument('--output_dir', type=str, default='/workspace/Dataset/DOTAv1_Split/sparse/image_annotation_split_percent5/unlabeled_annotation_with_label_split', help='Directory to save the split annotations.')

    args = parser.parse_args()

    # Add percentage to output directory name
    output_dir_with_percent = f"{args.output_dir}_percent_{int(args.percent)}"
    
    # Call the function with parsed arguments
    split_dota_annotations_per_image(args.input_dir, args.percent, args.seed, output_dir_with_percent)
