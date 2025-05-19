import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def separate_labeled_unlabeled_images(img_folder, txt_folder, output_root):
    # 定义输出文件夹路径
    label_img_folder = os.path.join(output_root, "labeled_image")
    labeled_annotation_folder = os.path.join(output_root, "labeled_annotation")
    unlabel_img_folder = os.path.join(output_root, "unlabeled_image")
    unlabeled_annotation_folder = os.path.join(output_root, "unlabeled_annotation")

    # 创建输出文件夹
    os.makedirs(label_img_folder, exist_ok=True)
    os.makedirs(labeled_annotation_folder, exist_ok=True)
    os.makedirs(unlabel_img_folder, exist_ok=True)
    os.makedirs(unlabeled_annotation_folder, exist_ok=True)

    # 获取所有图片和标注文件名（不含后缀）
    img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_folder) if f.endswith(".png")}
    txt_files = {os.path.splitext(f)[0]: f for f in os.listdir(txt_folder) if f.endswith(".txt")}

    def process_file(img_name, img_file):
        img_path = os.path.join(img_folder, img_file)
        txt_path = os.path.join(txt_folder, txt_files[img_name]) if img_name in txt_files else None

        if txt_path and os.path.exists(txt_path):
            # 检查txt文件是否为空
            if os.path.getsize(txt_path) > 0:
                # txt文件不为空，属于标注文件
                shutil.copy(img_path, os.path.join(label_img_folder, img_file))
                shutil.copy(txt_path, os.path.join(labeled_annotation_folder, txt_files[img_name]))
            else:
                # txt文件为空，属于未标注文件
                shutil.copy(img_path, os.path.join(unlabel_img_folder, img_file))
                shutil.copy(txt_path, os.path.join(unlabeled_annotation_folder, txt_files[img_name]))
        else:
            # 没有对应的txt文件，视为未标注
            shutil.copy(img_path, os.path.join(unlabel_img_folder, img_file))

    # 使用多线程加速处理
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda item: process_file(*item), img_files.items()), total=len(img_files), desc="Processing files", unit="file"))

    # 统计文件数量
    count_files(label_img_folder, labeled_annotation_folder, unlabel_img_folder, unlabeled_annotation_folder)

def count_files(label_img_folder, labeled_annotation_folder, unlabel_img_folder, unlabeled_annotation_folder):
    print("\nFile counts after separation:")
    print(f"Labeled images: {len(os.listdir(label_img_folder))}")
    print(f"Labeled annotations: {len(os.listdir(labeled_annotation_folder))}")
    print(f"Unlabeled images: {len(os.listdir(unlabel_img_folder))}")
    print(f"Unlabeled annotations: {len(os.listdir(unlabeled_annotation_folder))}")

# 使用示例
img_folder = "/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/images"  # 替换为存放png图片的文件夹路径
txt_folder = "/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/labelTxt_0.05_with_weakly_pesudo_label"    # 替换为存放txt文件的文件夹路径
output_root = "/workspace/Dataset/DOTAv1_Split/sparse/image_annotation_split_percent5_with_weakly_pesudo_label"  # 替换为输出文件夹的路径

separate_labeled_unlabeled_images(img_folder, txt_folder, output_root)
