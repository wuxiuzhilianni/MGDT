# 文件路径
stat_file = '/workspace/animax/MCL/mmdetection/tools/px_distribution/px_distribution.txt'

def read_px_values(file_path):
    """读取 px 值并返回一个 numpy 数组。"""
    px_values = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                px_values.append(float(line.strip()))  # 转换为浮点数
            except ValueError:
                continue  # 跳过无法解析的行
    return px_values

def calculate_threshold_stats(px_values, thr):
    """统计总数、低于阈值和高于阈值的数量。"""
    total_count = len(px_values)
    below_thr_count = sum(1 for px in px_values if px < thr)
    above_thr_count = total_count - below_thr_count

    return total_count, below_thr_count, above_thr_count

if __name__ == '__main__':
    # 读取 px 数据
    px_values = read_px_values(stat_file)

    # 指定阈值
    thr = 0.005  # 用户可以修改此值

    # 计算统计数据
    total, below_thr, above_thr = calculate_threshold_stats(px_values, thr)

    # 打印结果
    print(f"总数: {total}")
    print(f"低于阈值({thr})的数量: {below_thr}")
    print(f"高于阈值({thr})的数量: {above_thr}")