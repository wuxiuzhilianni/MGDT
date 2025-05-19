import argparse
import json
import os

def extract_val_logs(input_file, output_file):
    """提取 val 模式的记录并保存到新的 JSON 文件中。"""
    val_logs = []
    with open(input_file, 'r') as f:
        for line in f:
            log = json.loads(line.strip())
            if log.get("mode") == "val":
                val_logs.append(log)

    # 更新 epoch 值
    for i, log in enumerate(val_logs):
        log["epoch"] = i + 1  # 从1开始递增

    # 写入新的 JSON 文件
    with open(output_file, 'w') as f:
        for log in val_logs:
            f.write(json.dumps(log) + '\n')

def main():
    parser = argparse.ArgumentParser(description='提取 val 模式的日志')
    parser.add_argument('--input_file', type=str, default='/workspace/animax/MCL/workdirs/Method2/Dense_Teacher/dense_teacher_fcos_percent5_with_sparse_focal_loss/20241127_070903.log.json', help='输入 JSON 文件路径')
    parser.add_argument('--output_file', type=str, default='/workspace/animax/MCL/tools/analyze_logs/dense_teacher_analyze/dense_teacher_fcos_percent5_with_sparse_focal_loss.json', help='输出 JSON 文件路径')
    args = parser.parse_args()

    # 提取 val 模式的记录
    extract_val_logs(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
