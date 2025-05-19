def sum_10_to_100(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    total = 0
    for line in lines:
        range_str, count = line.strip().split()
        start = float(range_str.split('~')[0])
        if 1 <= start < 100:
            total += int(count)

    print(f'Total from 10~100: {total}')

# 使用方法
sum_10_to_100('/workspace/MCL/tools/count_gt_size_distribution/bbox_area_percent_distribution.txt')  # 替换为你的文件路径
