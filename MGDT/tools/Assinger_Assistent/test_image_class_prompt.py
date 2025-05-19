import torch
from pprint import pprint

# 加载 .pt 文件
pt_file = '/workspace/animax/MCL/tools/Assinger_Assistent/dota_trainval_answerv1_with_percent5_label.pt'
data = torch.load(pt_file)

# 打印字典信息
total_images = len(data)
print(f"Loaded {total_images} image-class mappings.\n")

# 打印前 500 条映射
print("Example mappings (first 500 entries):")
pprint(list(data.items())[:50], width=120)

# 打印总数量
print(f"\nTotal number of images: {total_images}")
