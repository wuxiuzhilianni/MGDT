import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from open_clip import tokenizer

import open_clip
list = open_clip.list_pretrained()  # clip.available_models() will list the names of available CLIP models.
# for str in list:
#     print(str)

model, _, preprocess = open_clip.create_model_and_transforms(
        model_name='ViT-L-14',
        pretrained='openai',
        # device= "cuda:2",
        cache_dir='cache/weights/open_clip'
    )

model.eval()
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Context length:", context_length)
print("Vocab size:", vocab_size)


tokenizer.tokenize("Hello World!")