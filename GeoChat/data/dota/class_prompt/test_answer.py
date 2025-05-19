import json

# 初始化计数
correct_none_empty = 0  # 预测为 none 且 ground_truth 为空
correct_single_match = 0  # 预测为 x 且 ground_truth 仅为 x
correct_subset_match = 0  # 预测为 x 且 ground_truth 是 x 的超集
total_count = 0  # 总数

file_path = '/workspace/Qwen-VL-master/class_prompt/answer_phi3.jsonl'

with open(file_path, 'r') as file:
    for line in file:
        # 逐行读取并解析 JSON
        record = json.loads(line.strip())

        # 标准化 answer 和 ground_truth（按逗号分割，去除空格，排序）
        answer = sorted(map(str.strip, record['answer'].split(','))) if record['answer'] != 'none' else []
        ground_truth = sorted(map(str.strip, record['ground_truth'].split(','))) if record['ground_truth'] else []

        # 更新总数
        total_count += 1

        # 分类统计
        if not answer and not ground_truth:
            correct_none_empty += 1  # 预测为 'none' 且 ground_truth 为空
        elif answer:
            if answer == ground_truth:
                correct_single_match += 1  # 预测与 ground_truth 完全匹配
            elif all(item in ground_truth for item in answer):
                correct_subset_match += 1  # 预测为 ground_truth 的子集

# 输出结果
print("总计统计结果：")
print(f"总数: {total_count}")
print(f"正确预测 - 预测为 'none' 且 ground_truth 为空: {correct_none_empty}")
print(f"正确预测 - 预测为 x 且 ground_truth 仅为 x: {correct_single_match}")
print(f"正确预测 - 预测为 x 且 ground_truth 为 x 的超集: {correct_subset_match}")
print(f"正确总数: {correct_none_empty + correct_single_match + correct_subset_match}")
print(f"错误总数: {total_count - (correct_none_empty + correct_single_match + correct_subset_match)}")

"""
orignal answer:
总数: 21046
正确预测 - 预测为 'none' 且 ground_truth 为空: 6943
正确预测 - 预测为 x 且 ground_truth 仅为 x: 4861
正确预测 - 预测为 x 且 ground_truth 为 x 的超集: 3771
正确总数: 15575
错误总数: 5471

modify labeled answer 1%:
总数: 21046
正确预测 - 预测为 'none' 且 ground_truth 为空: 6819
正确预测 - 预测为 x 且 ground_truth 仅为 x: 7539
正确预测 - 预测为 x 且 ground_truth 为 x 的超集: 2259
正确总数: 16617
错误总数: 4429

modify labeled answer 2%:
总数: 21046
正确预测 - 预测为 'none' 且 ground_truth 为空: 6821
正确预测 - 预测为 x 且 ground_truth 仅为 x: 7755
正确预测 - 预测为 x 且 ground_truth 为 x 的超集: 2139
正确总数: 16715
错误总数: 4331

modify labeled answer 5%:
总数: 21046
正确预测 - 预测为 'none' 且 ground_truth 为空: 6829
正确预测 - 预测为 x 且 ground_truth 仅为 x: 8075
正确预测 - 预测为 x 且 ground_truth 为 x 的超集: 1976
正确总数: 16880
错误总数: 4166

modify labeled answer 10%:
"""