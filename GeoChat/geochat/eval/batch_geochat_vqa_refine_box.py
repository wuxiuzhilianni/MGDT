import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import cv2
import numpy as np
from PIL import Image

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle
from geochat.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init
from geochat.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# 定义类别列表
CLASSES = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter']

def crop_object(image, points):
    """根据给定的点裁剪图像中的目标对象"""
    points = np.array([(points[i], points[i+1]) for i in range(0, len(points), 2)], dtype=np.float32)
    rect = cv2.minAreaRect(points)
    width, height = int(rect[1][0]), int(rect[1][1])
    if width <= 0 or height <= 0:
        return None
    M = cv2.getRotationMatrix2D(rect[0], rect[2], 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    cropped = cv2.getRectSubPix(rotated, (width, height), rect[0])
    return cropped

def eval_model(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    os.makedirs(args.output_txt_folder, exist_ok=True)
    ans_file = open(args.answers_file, "w")

    txt_files = [f for f in os.listdir(args.txt_folder) if f.endswith('.txt')]

    for txt_file in tqdm(txt_files):
        img_file = txt_file.replace('.txt', '.png')
        img_path = os.path.join(args.image_folder, img_file)

        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Unable to read image {img_path}")
            continue

        txt_path = os.path.join(args.txt_folder, txt_file)
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            points = list(map(float, parts[:8]))
            label = parts[8].lower()
            if label not in CLASSES:
                continue

            try:
                cropped = crop_object(image, points)
                if cropped is None or cropped.size == 0:
                    continue

                cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                # v1
                # qs = f"Is there a whole {label} in the image.\nAnswer with yes or no."
                # v2
                qs = f"Is there a {label} in the center of the image.\nAnswer with yes or no."

                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

                image_tensor = image_processor.preprocess(
                    [cropped_pil], 
                    crop_size={'height': 504, 'width': 504},
                    size={'shortest_edge': 504}, 
                    return_tensors='pt'
                )['pixel_values']

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.half().cuda(),
                        do_sample=False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=256,
                        length_penalty=2.0,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria]
                    )

                input_token_len = input_ids.shape[1]
                output = tokenizer.decode(output_ids[0, input_token_len:], skip_special_tokens=True).strip()
                if output.endswith(stop_str):
                    output = output[:-len(stop_str)]
                output = output.strip().lower()

                if output == "yes":
                    new_lines.append(line)

                ans_file.write(json.dumps({
                    "question_id": txt_file,
                    "image_id": img_file,
                    "bbox": parts[:8],
                    "label": label,
                    "answer": output,
                    "prompt": qs
                }) + "\n")
                ans_file.flush()

            except Exception as e:
                print(f"Error processing {txt_file}: {e}")
                continue

        output_txt_path = os.path.join(args.output_txt_folder, txt_file)
        with open(output_txt_path, 'w') as fw:
            fw.writelines(new_lines)

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/workspace/GeoChat/geochat_weights")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/workspace/Dataset/DOTAv1_Split/split_ss_dota/trainval/images")
    parser.add_argument("--txt-folder", type=str, default="/workspace/GeoChat/data/dota/weak_box/step4_use_class_prompt_to_refine_box/bbox_after_class_prompt")
    parser.add_argument("--output-txt-folder", type=str, default="/workspace/GeoChat/data/dota/weak_box/step5_use_several_VQA_task_to_refine_box/bbox_after_vqa")
    parser.add_argument("--answers-file", type=str, default="/workspace/GeoChat/data/dota/weak_box/step5_use_several_VQA_task_to_refine_box/bbox_after_vqa.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
