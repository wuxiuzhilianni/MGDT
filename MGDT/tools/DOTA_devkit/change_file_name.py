import os
import concurrent.futures
from tqdm import tqdm

# 指定文件夹路径
folder_path = '/workspace/Dataset/DOTAv1_Split/sparse/labelTxt_0.05'

def rename_file(filename):
    parts = filename.split('__')
    if len(parts) >= 2 and parts[1] == '1.0':
        new_filename = f"{parts[0]}__1024__{parts[2]}__{parts[3]}"
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        os.rename(old_filepath, new_filepath)

# 获取所有文件名并筛选出需要处理的文件
filenames = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 使用多线程处理文件重命名，并显示进度条
with concurrent.futures.ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(rename_file, filenames), total=len(filenames)))
