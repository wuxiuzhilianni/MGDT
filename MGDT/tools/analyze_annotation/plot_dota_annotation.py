import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import argparse
from tqdm import tqdm

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description='Count annotations in .txt files.')
parser.add_argument('--folder', type=str, default='/opt/data/private/LiaoWei/MCL/data/luzihan/labelTxt_0.05', help='Path to the folder containing .txt files')
parser.add_argument('--output', type=str, default='/opt/data/private/LiaoWei/MCL/tools/output.png', help='Output image file name')
args = parser.parse_args()

# 统计包含不同数量annotations的文件个数
annotation_counts = {}

def count_annotations(file_path):
    """读取文件并统计annotations的数量"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return len(lines)

# 使用线程池来并行处理
with ThreadPoolExecutor() as executor:
    futures = []
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(args.folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(args.folder, filename)
            futures.append(executor.submit(count_annotations, file_path))
    
    # 收集结果，使用tqdm显示进度
    for future in tqdm(futures, desc="Counting annotations"):
        num_annotations = future.result()
        if num_annotations not in annotation_counts:
            annotation_counts[num_annotations] = 0
        annotation_counts[num_annotations] += 1

# 准备数据用于绘图
x = sorted(annotation_counts.keys())  # X轴为不同的annotation数量
y = [annotation_counts[count] for count in x]  # Y轴为对应的文件个数

# 绘制柱状图
plt.bar(x, y)
plt.xlabel('Number of Annotations')
plt.ylabel('Number of .txt Files')
plt.title('Number of .txt Files by Annotation Count')
plt.xticks(x)  # 显示X轴刻度

# 在柱状图上显示数量
for i, count in enumerate(y):
    plt.text(x[i], count, str(count), ha='center', va='bottom')

# 保存图像到output.png
plt.savefig(args.output)

# 显示图像
plt.show()
