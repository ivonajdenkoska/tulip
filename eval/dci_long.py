import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import os
from PIL import Image
import pandas as pd
import numpy as np

from open_clip import create_model_and_transforms, get_tokenizer, get_model_config
from tqdm import tqdm

class local_dataset(data.Dataset):
    def __init__(self, data_path):
        self.annotations_path = os.path.join(data_path, 'dci_long.csv')
        self.annotations = pd.read_csv(self.annotations_path, sep='\t')
        self.image_root = data_path
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_path = self.annotations.iloc[index]['filepath']
        image_path = os.path.join(self.image_root, image_path)
        image = Image.open(image_path)
        caption = self.annotations.iloc[index]['title']

        return image, caption

class OptimizedLocalDataset(data.Dataset):
    def __init__(self, data_path, processor):
        self.annotations_path = os.path.join(data_path, 'dci_long.csv')
        self.annotations = pd.read_csv(self.annotations_path, sep='\t')
        self.image_root = data_path
        self.processor = processor
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.annotations.iloc[index]['filepath'])
        caption = self.annotations.iloc[index]['title']

        with Image.open(image_path) as img:
            img_tensor = self.processor(img)

        return img_tensor, caption


def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    return images, list(captions)


def run_dci_long_openclip(model, distilled_model, processor, data_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    distilled_model.eval()
    data_path = '/data/dci_long'

    dataset = OptimizedLocalDataset(data_path, processor)
    batch_size = 512  # Adjust based on your GPU memory
    num_workers = 4  # Adjust based on your CPU cores

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2
    )
    
    img_feature_list = []
    text_list = []
    logit_scale = 100

    with torch.no_grad():
        # iterate through the dataset using tqdm
        for images, captions in tqdm(dataloader, desc="Processing batches"):
            texts_encoded = processor.tokenizer(captions).to(device)
            images_encoded = images.to(device)
            image_features = model.encode_image(images_encoded)
            text_features = distilled_model.encode_text(texts_encoded)

            text_list.append(text_features)
            img_feature_list.append(image_features)
        
        # Text feature extraction
        text_feature = torch.cat(text_list, dim=0)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
            
        image_embeds = torch.cat(img_feature_list, dim=0)
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        
        # Use the second snippet's function to calculate metrics
        metrics = get_clip_metrics(image_embeds, text_feature, logit_scale)
        
        # Print or return the metrics
        for k in [1, 5, 10]:
            print(f"Text to Image - R@{k}: {metrics[f'text_to_image_R@{k}']}")

        for k in [1, 5, 10]:
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