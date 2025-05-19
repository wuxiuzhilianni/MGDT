import os
import shutil

def copy_matching_png(txt_dir, png_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 txt 文件的文件名（不带扩展名）
    txt_names = {os.path.splitext(f)[0] for f in os.listdir(txt_dir) if f.endswith('.txt')}

    # 遍历 png 文件夹，将名字匹配的文件复制到目标文件夹
    for png_file in os.listdir(png_dir):
        name, ext = os.path.splitext(png_file)
        if ext.lower() == '.png' and name in txt_names:
            src_path = os.path.join(png_dir, png_file)
            dst_path = os.path.join(output_dir, png_file)
            shutil.copy(src_path, dst_path)

# 示例调用
txt_folder = '/workspace/Dataset/DOTAv1_Split/sparse/weakly_pesudo_annotation'
png_folder = '/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/images'
output_folder = '/workspace/Dataset/DOTAv1_Split/sparse/weakly_pesudo_image'

copy_matching_png(txt_folder, png_folder, output_folder)
