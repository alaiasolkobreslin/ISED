import os
import torch
import clip
from PIL import Image

from dataset import scenes

scene_names = ['basement', 'bathroom', 'bedroom', 'dining room', 'kitchen', 'lab', 'living room', 'home lobby', 'office']

def test(test_dataset):
    correct = 0
    total = len(test_dataset)
    for (img, target) in test_dataset:
        pred = compute_prediction(img)
        if pred == target:
            correct += 1
    print(correct, total, (correct/total))

def compute_prediction(img):
    image = preprocess(img).unsqueeze(0).to(device)
    text= torch.cat([clip.tokenize(f"a {c}") for c in scene_names]).to(device)

    # Calculate features
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return probs.argmax()

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Prepare the inputs
samples = []
data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data/scene"))
data_dir = os.path.join(data_root, "test")
data_dirs = os.listdir(data_dir)
for sample_group in data_dirs:
    sample_group_dir = os.path.join(data_dir, sample_group)
    if not os.path.isdir(sample_group_dir) or not sample_group in scenes:
        continue
    label = scenes.index(sample_group)
    sample_group_files = os.listdir(sample_group_dir)
    for idx in range(len(sample_group_files)):
        sample_img_path = os.path.join(sample_group_dir, sample_group_files[idx])
        if sample_img_path.endswith('jpg'):
            samples.append((Image.open(sample_img_path).resize((768, 512)), label))

test(samples)