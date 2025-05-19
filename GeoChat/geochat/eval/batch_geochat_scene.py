import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle
from geochat.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init
from geochat.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
from collections import defaultdict
import pdb


def evaluation_metrics(data_path):
    # Read the JSONL file and parse each line into a dictionary
    base = [json.loads(q) for q in open(data_path, "r")]

    # Initialize counters for total correct and incorrect answers
    total_correct = 0
    total_incorrect = 0

    # Initialize a dictionary to store correct and incorrect counts for each category
    category_counts = defaultdict(lambda: {'correct': 0, 'incorrect': 0})

    for answers in tqdm(base):
        gt=answers['question_id'].split('/')[0].lower()
        answer=answers['answer'].replace(' ','').lower().replace('.','')

        # Check if the predicted answer is correct
        if gt==answer:
            total_correct += 1
            category_counts[gt]['correct'] += 1  # Increment correct count for the category
        else:
            total_incorrect += 1
            category_counts[gt]['incorrect'] += 1  # Increment incorrect count for the category

    # Print overall results
    print(f'{"Overall correct:":<20} {total_correct}')
    print(f'{"Overall incorrect:":<20} {total_incorrect}')
    print(f'{"Overall Total:":<20} {total_correct + total_incorrect}')
    print(f'{"Overall Acc:":<20} {total_correct / (total_correct + total_incorrect):.2f}')

    # Print accuracy per category
    print('\nAccuracy per category:')
    header = f'{"Category":<20} {"Correct":<10} {"Incorrect":<10} {"Total":<10} {"Acc":<10}'
    print(header)
    print('-' * len(header))

    for category, counts in category_counts.items():
        category_total = counts['correct'] + counts['incorrect']
        category_acc = counts['correct'] / category_total if category_total > 0 else 0
        print(f'{category:<20} {counts["correct"]:<10} {counts["incorrect"]:<10} {category_total:<10} {category_acc:<10.2f}')



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # 禁用 Pytorch 的权重初始化，加速模型加载
    disable_torch_init()

    # 修改路径格式
    model_path = os.path.expanduser(args.model_path)

    # 根据模型路径获取模型名称(geochat_weights)
    model_name = get_model_name_from_path(model_path)

    # 加载模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    # pdb.set_trace()

    # 读取 questions.jsonl 文件
    questions=[]
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")] # len=2100
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx) # 使用 get_chunk 函数对 list 进行分块处理，n=1&k=0 时依然是原 list
    # questions[i] 包括，'question_id','image','text' 和 'ground_truth'

    # 确保 answer.jsonal 文件存在
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")  # 打开答案文件，准备写入模型生成的答案。
    
    # 开始遍历问题列表，每次处理一个批次（由 `batch_size` 参数决定）
    for i in tqdm(range(0,len(questions),args.batch_size)):
        input_batch=[]  # 存储当前批次的文本输入张量
        input_image_batch=[]    # 存储当前批次的图像输入张量
        count=i     # 用于追踪当前问题索引
        image_folder=[]        # 存储当前批次的图像数据
        batch_end = min(i + args.batch_size, len(questions))    # 计算当前批次的结束索引

        # 处理当前批次内的每个问题
        for j in range(i,batch_end):
            image_file=questions[j]['image']    # 获取当前问题对应的图像文件名
            qs=questions[j]['text']     # 获取当前问题的文本内容
            
            # 根据模型配置，添加图像标记（image token）到问题文本
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            # 根据 mode 构建 Conversation 类
            conv = conv_templates[args.conv_mode].copy()     #  args.conv_mode = llava_v1
            conv.append_message(conv.roles[0], qs)      # 给 Message 属性赋值
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()  # str 类型
            """
            A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>
            Classify the given image in one of the following classes. Classes: dense residential, river, overpass, medium residential, tennis court, agricultural, intersection, buildings, freeway, runway, chaparral, storage tanks, parking lot, sparse residential, beach, forest, baseball diamond, golf course, mobile home park, airplane, harbor. 
            Answer in one word or a short phrase. ASSISTANT:
            """

            # 使用 tokenizer_image_token 函数将文本 prompt 转换为输入tensor
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            input_batch.append(input_ids)

            # 打开并加载当前问题对应的图像
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_folder.append(image) # 将图像添加到图像批次

            # 设置停止生成的条件（例如，当遇到特定的结束标记时停止）
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # 对于当前批次，计算输入张量的最大长度，并用零填充较短的张量以匹配最长的张量
        max_length = max(tensor.size(1) for tensor in input_batch)
        final_input_list = [torch.cat((torch.zeros((1,max_length - tensor.size(1)), dtype=tensor.dtype,device=tensor.get_device()), tensor),dim=1) for tensor in input_batch]
        final_input_tensors=torch.cat(final_input_list,dim=0) # 将所有张量组合成一个批次张量

        # 预处理图像批次，将其转换为模型所需的张量格式
        image_tensor_batch = image_processor.preprocess(image_folder,crop_size ={'height': 504, 'width': 504},size = {'shortest_edge': 504}, return_tensors='pt')['pixel_values']

        # 在推理模式下，生成模型的输出 tensor
        with torch.inference_mode():
            output_ids = model.generate( final_input_tensors, images=image_tensor_batch.half().cuda(), do_sample=False , temperature=args.temperature, top_p=args.top_p, num_beams=1, max_new_tokens=256,length_penalty=2.0, use_cache=True)

        # 检查生成的输出张量与输入张量的前缀是否相同，以确保模型正确地接收了输入
        input_token_len = final_input_tensors.shape[1]
        n_diff_input_output = (final_input_tensors != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        # 将 output_ids 解码为可读文本
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

        # 处理生成的每个答案，去除停止符并进行格式化
        for k in range(0,len(final_input_list)):
            output = outputs[k].strip() # 去除两端的空白字符
            # 如果答案以停止符结尾，移除停止符
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()

            # 生成一个唯一的答案 ID（使用 shortuuid）
            ans_id = shortuuid.uuid()
            
            # 将生成的答案及其相关信息写入答案文件
            ans_file.write(json.dumps({
                                    "question_id": questions[count]["question_id"],
                                    "image_id": questions[count]["image"],
                                    "answer": output,
                                    "ground_truth": questions[count]['ground_truth']
                                    }) + "\n")
            count=count+1
            ans_file.flush() # 刷新输出缓冲区，将内容写入文件
    ans_file.close()

    # 调用 evaluation_metrics 函数，计算模型生成的答案的准确性
    evaluation_metrics(answers_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/workspace/GeoChat/geochat_weights")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/images")
    parser.add_argument("--question-file", type=str, default="/workspace/GeoChat/data/dota/test_different_question/dota_trainval_questionv2.jsonl")
    parser.add_argument("--answers-file", type=str, default="/workspace/GeoChat/data/dota/test_different_question/dota_trainval_answerv2.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size",type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
    # evaluation_metrics("/workspace/GeoChat/data/ucmerced/ans.jsonl")
