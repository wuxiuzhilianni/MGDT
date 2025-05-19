import torch
import open_clip
from PIL import Image
import os
import argparse

# 定义完整的模板列表
template_list = [
    "This is a {}",
    "There is a {}",
    "a photo of a {} in the scene",
    "a photo of a small {} in the scene",
    "a photo of a medium {} in the scene",
    "a photo of a large {} in the scene",
    "a photo of a {}",
    "a photo of a small {}",
    "a photo of a medium {}",
    "a photo of a large {}",
    "This is a photo of a {}",
    "This is a photo of a small {}",
    "This is a photo of a medium {}",
    "This is a photo of a large {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "There is one {} in the scene",
    "This is a {} in the scene",
    "This is the {} in the scene",
    "This is one {} in the scene",
    "This is one small {} in the scene",
    "This is one medium {} in the scene",
    "This is one large {} in the scene",
    "There is a small {} in the scene",
    "There is a medium {} in the scene",
    "There is a large {} in the scene",
    "There is a {} in the photo",
    "There is the {} in the photo",
    "There is one {} in the photo",
    "There is a small {} in the photo",
    "There is the small {} in the photo",
    "There is one small {} in the photo",
    "There is a medium {} in the photo",
    "There is the medium {} in the photo",
    "There is one medium {} in the photo",
    "There is a large {} in the photo",
    "There is the large {} in the photo",
    "There is one large {} in the photo",
    "There is a {} in the picture",
    "There is the {} in the picture",
    "There is one {} in the picture",
    "There is a small {} in the picture",
    "There is the small {} in the picture",
    "There is one small {} in the picture",
    "There is a medium {} in the picture",
    "There is the medium {} in the picture",
    "There is one medium {} in the picture",
    "There is a large {} in the picture",
    "There is the large {} in the picture",
    "There is one large {} in the picture",
    "This is a {} in the photo",
    "This is the {} in the photo",
    "This is one {} in the photo",
    "This is a small {} in the photo",
    "This is the small {} in the photo",
    "This is one small {} in the photo",
    "This is a medium {} in the photo",
    "This is the medium {} in the photo",
    "This is one medium {} in the photo",
    "This is a large {} in the photo",
    "This is the large {} in the photo",
    "This is one large {} in the photo",
    "This is a {} in the picture",
    "This is the {} in the picture",
    "This is one {} in the picture",
    "This is a small {} in the picture",
    "This is the small {} in the picture",
    "This is one small {} in the picture",
    "This is a medium {} in the picture",
    "This is the medium {} in the picture",
    "This is one medium {} in the picture",
    "This is a large {} in the picture",
    "This is the large {} in the picture",
    "This is one large {} in the picture",
]

# 定义类别
classes = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
)

def load_model_and_preprocess(model_name, ckpt_path):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    return model.cuda().eval(), tokenizer, preprocess

def extract_and_save_text_features(model, tokenizer, template_list, classes, path):
    if not os.path.exists(path):
        text_features = []
        for template in template_list:
            texts = [template.format(c) for c in classes]
            text = tokenizer(texts)
            with torch.no_grad(), torch.cuda.amp.autocast():
                features = model.encode_text(text.cuda())
                features = features / features.norm(dim=-1, keepdim=True)
                text_features.append(features)
        text_features = torch.stack(text_features).mean(dim=0)
        torch.save(text_features, path)
    return torch.load(path)

def calculate_similarity(model, preprocess, image_path, text_features):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image.cuda())
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return similarity.cpu().numpy()[0]

def print_similarity_scores(classes, similarity):
    print("\n图像与各类别的相似度分数:")
    for cls, score in zip(classes, similarity):
        print(f"{cls:<20} {score*100:5.1f}%")

def print_top_k_classes(classes, similarity, top_k=3):
    top_probs, top_labels = torch.topk(torch.tensor(similarity), top_k)
    print(f"\n相似度最高的{top_k}个类别:")
    for i in range(top_k):
        print(f"{classes[top_labels[i]]:<20} {top_probs[i]*100:5.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Calculate image-text similarity using CLIP model.")
    parser.add_argument('--model_name', type=str, default='ViT-B-32', help='Name of the model to use.')
    parser.add_argument('--ckpt_path', type=str, default='/workspace/Project/Remote-VILD/RemoteCLIP/pt/RemoteCLIP-ViT-B-32.pt', help='Path to the model checkpoint.')
    parser.add_argument('--image_path', type=str, default='/workspace/Project/Remote-VILD/RemoteCLIP/assets/plane1.jpg', help='Path to the input image.')
    parser.add_argument('--text_features_path', type=str, default='/workspace/Project/Remote-VILD/RemoteCLIP/pt/text_features.pt', help='Path to save/load text features.')
    args = parser.parse_args()

    model, tokenizer, preprocess = load_model_and_preprocess(args.model_name, args.ckpt_path)
    text_features = extract_and_save_text_features(model, tokenizer, template_list, classes, args.text_features_path)
    similarity = calculate_similarity(model, preprocess, args.image_path, text_features)
    
    print_similarity_scores(classes, similarity)
    print_top_k_classes(classes, similarity)

if __name__ == "__main__":
    main()