import os
import torch
import open_clip
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

# Load the model and preprocessing pipeline
model_name = 'RN50'  # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

# Load the checkpoint
ckpt = torch.load("/workspace/Project/Remote-VILD/RemoteCLIP/pt/RemoteCLIP-RN50.pt", map_location="cpu")
message = model.load_state_dict(ckpt)
print(message)

# Move the model to GPU and set it to evaluation mode
model = model.cuda().eval()

# Define categories
categories = [
    'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
    'windmill'
]

# Define multiple text templates
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

# Compute text features for all templates and average them
text_features_list = []

for template in tqdm(template_list):
    # Create text queries for all categories with the current template
    text_queries = [template.format(category) for category in categories] # [N] N：类别数量

    # Tokenize the text queries
    text = tokenizer(text_queries).cuda() # [N,L] L:标记序列的长度

    # Encode text queries to obtain text features
    with torch.no_grad():
        text_features = model.encode_text(text) # [N, D] D：文本特征的维度
        text_features /= text_features.norm(dim=-1, keepdim=True) # 除以L2范数
        text_features_list.append(text_features)

# Average the text features across all templates
text_features_avg = torch.stack(text_features_list).mean(dim=0) # [T, N, D] T：模板数量，mean 之后得到 [N,D]

# Normalize the averaged text features
text_features_avg /= text_features_avg.norm(dim=-1, keepdim=True) # [N,D]

# Initialize counters for accuracy calculation
total_images = 0
correct_predictions = 0
category_counts = defaultdict(lambda: {'correct': 0, 'total': 0})

# Define the input folder containing the images
input_folder = "/workspace/GeoChat/data/dior/crop_with_gt_from1to1000"

# Process each image in the folder
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith('.jpg'):
        total_images += 1
        # Extract the ground truth category from the filename
        gt_category = filename.split('_')[1]

        # Load and preprocess the image
        image_path = os.path.join(input_folder, filename)
        image = preprocess(Image.open(image_path)).unsqueeze(0).cuda()

        # Predict using the model
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity between image and text features
            text_probs = (100.0 * image_features @ text_features_avg.T).softmax(dim=-1).cpu().numpy()[0]

        # Get the predicted category with the highest probability
        predicted_category = categories[text_probs.argmax()]

        # Update counters for accuracy
        category_counts[gt_category]['total'] += 1
        if predicted_category == gt_category:
            correct_predictions += 1
            category_counts[gt_category]['correct'] += 1

        # print(f"Image: {filename} - Ground Truth: {gt_category} - Predicted: {predicted_category} - Confidence: {text_probs.max() * 100:.2f}%")

# Calculate and print overall accuracy
overall_accuracy = correct_predictions / total_images
print(f"\nOverall accuracy: {overall_accuracy:.2f} ({correct_predictions} out of {total_images})")

# Print per-category accuracy
print("\nAccuracy per category:")
print(f'{"Category":<25} {"Correct":<10} {"Total":<10} {"Acc":<10}')
print('-' * 60)
for category, counts in category_counts.items():
    category_accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
    print(f'{category:<25} {counts["correct"]:<10} {counts["total"]:<10} {category_accuracy:<10.2f}')
