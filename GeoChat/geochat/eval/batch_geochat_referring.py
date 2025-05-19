import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from geochat.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from geochat.conversation import conv_templates, SeparatorStyle
from geochat.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init
from geochat.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from collections import defaultdict

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    # print(model)
    questions = []
    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")

    for i in tqdm(range(0, len(questions), args.batch_size)):
        input_batch = []
        input_image_batch = []
        count = i
        image_folder = []
        batch_end = min(i + args.batch_size, len(questions))

        for j in range(i, batch_end):
            image_file = questions[j]["image_id"] + ".jpg"
            # qs="[identify] What is the object present at " + questions[j]['question']
            # qs="[identify] Classify the image within one of the given classes: car,truck,van,long vehicle,bus,airliner,propeller aircraft,trainer aircraft,charted aircraft,figther aircraft,others,stair truck,pushback truck,helicopter,boat.Answer with one word or short phrase." + questions[j]['question']
            qs = (
                "[identify] Classify the image within one of the given classes: airplane,airport,baseballfield,basketballcourt,bridge,chimney,dam,Expressway-Service-area,Expressway-toll-station,golffield,groundtrackfield,harbor,overpass,ship,stadium,storagetank,tenniscourt,trainstation,vehicle,windmill.Answer with one word or short phrase."
                + questions[j]["question"]
            )

            if model.config.mm_use_im_start_end:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )
            input_batch.append(input_ids)

            image = Image.open(os.path.join(args.image_folder, image_file))

            image_folder.append(image)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        max_length = max(tensor.size(1) for tensor in input_batch)

        final_input_list = [
            torch.cat(
                (
                    torch.zeros(
                        (1, max_length - tensor.size(1)),
                        dtype=tensor.dtype,
                        device=tensor.get_device(),
                    ),
                    tensor,
                ),
                dim=1,
            )
            for tensor in input_batch
        ]
        final_input_tensors = torch.cat(final_input_list, dim=0)
        image_tensor_batch = image_processor.preprocess(
            image_folder,
            crop_size={"height": 504, "width": 504},
            size={"shortest_edge": 504},
            return_tensors="pt",
        )["pixel_values"]

        # 调用model.generate方法获取输出结果
        with torch.inference_mode():
            output_ids = model.generate(
                final_input_tensors,
                images=image_tensor_batch.half().cuda(),
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=1,
                max_new_tokens=256,
                length_penalty=2.0,
                use_cache=True,
            )

        input_token_len = final_input_tensors.shape[1]
        n_diff_input_output = (
            (final_input_tensors != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )
        for k in range(0, len(final_input_list)):
            output = outputs[k].strip()
            if output.endswith(stop_str):
                output = output[: -len(stop_str)]
            output = output.strip()

            ans_id = shortuuid.uuid()

            ans_file.write(
                json.dumps(
                    {
                        "question_id": questions[count]["question_id"],
                        "image_id": questions[count]["image_id"],
                        "answer": output,
                        "ground_truth": questions[count]["ground_truth"],
                        "question": questions[count]["question"],
                        # "type": questions[count]['type'],
                        "dataset": questions[count]["dataset"],
                        # "obj_ids": questions[count]['obj_ids'],
                        # "size_group": questions[count]['size_group'],
                    }
                )
                + "\n"
            )
            count = count + 1
            ans_file.flush()
    ans_file.close()


def evaluation_metrics(data_path):
    # Read the JSONL file and parse each line into a dictionary
    base = [json.loads(q) for q in open(data_path, "r")]

    # Initialize counters for total correct and incorrect answers
    total_correct = 0
    total_incorrect = 0

    # Initialize dictionaries to store correct and incorrect counts for each category and each answer
    gt_counts = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    answer_counts = defaultdict(lambda: {"correct": 0, "incorrect": 0})

    for answers in tqdm(base):
        gt = answers["ground_truth"].lower()  # Ground truth
        answer = answers["answer"].lower()  # Predicted answer

        # Check if the predicted answer is correct
        if answer in gt:
            total_correct += 1
            gt_counts[gt][
                "correct"
            ] += 1  # Increment correct count for the ground truth category
            answer_counts[answer][
                "correct"
            ] += 1  # Increment correct count for the predicted answer category
        else:
            total_incorrect += 1
            gt_counts[gt][
                "incorrect"
            ] += 1  # Increment incorrect count for the ground truth category
            answer_counts[answer][
                "incorrect"
            ] += 1  # Increment incorrect count for the predicted answer category

    # Print overall results
    print(f'{"Overall correct:":<20} {total_correct}')
    print(f'{"Overall incorrect:":<20} {total_incorrect}')
    print(f'{"Overall Total:":<20} {total_correct + total_incorrect}')
    print(
        f'{"Overall Acc:":<20} {total_correct / (total_correct + total_incorrect):.2f}'
    )

    # Print accuracy per ground truth category
    print("\nAccuracy per ground truth category:")
    header = (
        f'{"Category":<25} {"Correct":<10} {"Incorrect":<10} {"Total":<10} {"Acc":<10}'
    )
    print(header)
    print("-" * len(header))

    for category, counts in gt_counts.items():
        category_total = counts["correct"] + counts["incorrect"]
        category_acc = counts["correct"] / category_total if category_total > 0 else 0
        print(
            f'{category:<25} {counts["correct"]:<10} {counts["incorrect"]:<10} {category_total:<10} {category_acc:<10.2f}'
        )

    # Print accuracy per answer category
    print("\nAccuracy per answer category:")
    header = (
        f'{"Category":<25} {"Correct":<10} {"Incorrect":<10} {"Total":<10} {"Acc":<10}'
    )
    print(header)
    print("-" * len(header))

    for category, counts in answer_counts.items():
        category_total = counts["correct"] + counts["incorrect"]
        category_acc = counts["correct"] / category_total if category_total > 0 else 0
        print(
            f'{category:<25} {counts["correct"]:<10} {counts["incorrect"]:<10} {category_total:<10} {category_acc:<10.2f}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="/workspace/GeoChat/geochat_weights"
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--image-folder",
        type=str,
        default="/workspace/Dataset/OpenDataLab___DIOR/raw/DIOR/JPEGImages-trainval",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default="/workspace/GeoChat/data/dior/dior_tranval_horizontal.jsonl",
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        default="/workspace/GeoChat/dior_tranval_horizontal_answer_XX.jsonl",
    )
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
    # evaluation_metrics(
    #     "/workspace/GeoChat/data/dior/dior_tranval_horizontal_answer.jsonl"
    # )
