import os
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt


def process_file(args):
    txt_path, img_path = args
    area_ratios, width_ratios, height_ratios = [], [], []

    img = cv2.imread(img_path)
    if img is None:
        return area_ratios, width_ratios, height_ratios

    img_height, img_width = img.shape[:2]
    img_area = img_width * img_height

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue

        points = list(map(float, parts[:8]))
        polygon = np.array([(points[i], points[i + 1]) for i in range(0, 8, 2)], dtype=np.float32)
        area = cv2.contourArea(polygon)
        if area <= 0:
            continue

        # Get rotated bbox size
        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)

        area_ratios.append((area / img_area) * 100)
        width_ratios.append((width / img_width) * 100)
        height_ratios.append((height / img_height) * 100)

    return area_ratios, width_ratios, height_ratios


def analyze_ratios(txt_folder, img_folder, bins=100):
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    args_list = []

    for txt_file in txt_files:
        img_file = txt_file.replace('.txt', '.png')
        txt_path = os.path.join(txt_folder, txt_file)
        img_path = os.path.join(img_folder, img_file)
        if os.path.exists(img_path):
            args_list.append((txt_path, img_path))

    print(f"开始处理 {len(args_list)} 个样本，使用 {cpu_count()} 核心多进程")

    area_all, width_all, height_all = [], [], []
    with Pool(cpu_count()) as pool:
        for area, width, height in tqdm(pool.imap_unordered(process_file, args_list), total=len(args_list)):
            area_all.extend(area)
            width_all.extend(width)
            height_all.extend(height)

    hist_area, bin_area = np.histogram(area_all, bins=bins, range=(0, 100))
    hist_width, bin_width = np.histogram(width_all, bins=bins, range=(0, 100))
    hist_height, bin_height = np.histogram(height_all, bins=bins, range=(0, 100))

    return (hist_area, bin_area), (hist_width, bin_width), (hist_height, bin_height)


def plot_histogram(hist, bin_edges, output_path, title, xlabel):
    plt.figure(figsize=(14, 6))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.bar(bin_centers, hist, width=1.0, edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel("目标框数量")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_histogram_txt(hist, bin_edges, output_txt_path):
    with open(output_txt_path, 'w') as f:
        for i in range(len(hist)):
            f.write(f"{bin_edges[i]:.2f}~{bin_edges[i+1]:.2f} {hist[i]}\n")


if __name__ == "__main__":
    txt_folder = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/annfiles_obb'
    img_folder = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/images'
    output_dir = '/workspace/MCL/tools/count_gt_size_distribution'
    os.makedirs(output_dir, exist_ok=True)

    (hist_area, bin_area), (hist_width, bin_width), (hist_height, bin_height) = analyze_ratios(txt_folder, img_folder)

    # 绘图 + 保存 txt
    plot_histogram(hist_area, bin_area,
                   os.path.join(output_dir, 'bbox_area_percent_distribution.png'),
                   '目标框面积占图像百分比分布', '面积百分比（%）')
    save_histogram_txt(hist_area, bin_area,
                       os.path.join(output_dir, 'bbox_area_percent_distribution.txt'))

    plot_histogram(hist_width, bin_width,
                   os.path.join(output_dir, 'bbox_width_percent_distribution.png'),
                   '目标框宽度占图像百分比分布', '宽度百分比（%）')
    save_histogram_txt(hist_width, bin_width,
                       os.path.join(output_dir, 'bbox_width_percent_distribution.txt'))

    plot_histogram(hist_height, bin_height,
                   os.path.join(output_dir, 'bbox_height_percent_distribution.png'),
                   '目标框高度占图像百分比分布', '高度百分比（%）')
    save_histogram_txt(hist_height, bin_height,
                       os.path.join(output_dir, 'bbox_height_percent_distribution.txt'))
