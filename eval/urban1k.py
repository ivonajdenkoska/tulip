from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer, get_model_config
import torch
import torch.utils.data as data
import os
import numpy as np
from tqdm import tqdm


image_root = '/Datasets/Urban1k/image/'
caption_root = '/Datasets/Urban1k/caption/'

class local_dataset(data.Dataset):
    def __init__(self, data_path):
        self.image_root = f"{data_path}{image_root}"
        self.caption_root = f"{data_path}{caption_root}"
        self.total_image = os.listdir(self.image_root)
        self.total_caption = os.listdir(self.caption_root)

    def __len__(self):
        return len(self.total_caption)

    def __getitem__(self, index):
        caption_name = self.total_caption[index]
        image_name = self.total_caption[index][:-4] + '.jpg'
        image = Image.open(self.image_root + image_name)
        f=open(self.caption_root + caption_name)
        caption = f.readlines()[0]
        
        return image, caption

class OptimizedLocalDataset(data.Dataset):
    def __init__(self, data_path, processor):
        self.image_root = f"{data_path}{image_root}"
        self.caption_root = f"{data_path}{caption_root}"
        self.total_image = os.listdir(self.image_root)
        self.total_caption = os.listdir(self.caption_root)
        self.processor = processor    
    def __len__(self):
        return len(self.total_caption)
    
    def __getitem__(self, index):
        caption_name = self.total_caption[index]
        image_name = self.total_caption[index][:-4] + '.jpg'

        with Image.open(self.image_root + image_name) as img:
            img_tensor = self.processor(img)

        with open(self.caption_root + caption_name) as f:
            caption = f.readlines()[0]

        return img_tensor, caption
    

def run_urban1k_openclip(model, distilled_model, processor, data_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    distilled_model.eval()

    # Create DataLoader
    batch_size = 512  # Adjust based on your GPU memory
    dataset = OptimizedLocalDataset(data_path, processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    img_feature_list = []
    text_feature_list = []

    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="Processing batches"):
            # Process batch of images
            images = images.to(device)
            image_features = model.encode_image(images)
            img_feature_list.append(image_features)

            # Process batch of captions
            text_encoded = processor.tokenizer(captions).to(device)
            text_features = distilled_model.encode_text(text_encoded)
            text_feature_list.append(text_features)

        # Concatenate all features
        image_embeds = torch.cat(img_feature_list, dim=0)
        text_feature = torch.cat(text_feature_list, dim=0)

        # Normalize features
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        logit_scale = 100
        
        # Calculate metrics
        metrics = get_clip_metrics(image_embeds, text_feature, logit_scale)
        
        # Print metrics
        for k in [1, 5, 10]:
            print(f"Text to Image - R@{k}: {metrics[f'text_to_image_R@{k}']}")
            print(f"Image to Text - R@{k}: {metrics[f'image_to_text_R@{k}']}")

        return metrics

def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
