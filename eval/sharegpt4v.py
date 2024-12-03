import json
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer, get_model_config

import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

image_root = '/data/ShareGPT4V/data/'
caption_root = '/Datasets/ShareGPT4V/'

class local_dataset(data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_root = image_root
        json_name = f"{data_path}{caption_root}share-captioner_coco_lcs_sam_1246k_1107.json" 

        with open(json_name, 'r') as f:
            self.total_caption = json.load(f)[:1000]

    def __len__(self):
        return len(self.total_caption)

    def __getitem__(self, index):
        caption = self.total_caption[index]['conversations'][1]["value"]
        image_name = self.total_caption[index]["image"]
        image = Image.open(self.image_root + image_name)
           
        return image, caption

class OptimizedLocalDataset(data.Dataset):
    def __init__(self, data_path, processor):
        self.data_path = data_path
        self.image_root = image_root
        self.processor = processor
        json_name = f"{data_path}{caption_root}share-captioner_coco_lcs_sam_1246k_1107.json" 
        # read the json file
        with open(json_name, 'r') as f:
            self.total_caption = json.load(f)[:1000]
        

    def __len__(self):
        return len(self.total_caption)
    
    def __getitem__(self, index):
        caption = self.total_caption[index]['conversations'][1]["value"]
        image_name = self.total_caption[index]["image"]
           
        with Image.open(self.image_root + image_name) as img:
            img_tensor = self.processor(img)

        return img_tensor, caption
    
def run_sharegpt4v(model, distilled_model, processor, data_path):
    dataset = local_dataset(data_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    
    img_feature_list = []
    text_list = []

    correct_t2i = 0
    total_t2i = 0
    correct_i2t = 0
    total_i2t = 0
    
    with torch.no_grad():
        for i, (image, caption) in enumerate(dataset):
            text_list.append(caption)

        # Text feature extraction
        text_feature = processor.tokenizer(text_list).to(device)
        text_feature = distilled_model.encode_text(text_feature)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        
        for i, (image, caption) in enumerate(dataset):            
            image = processor(image).unsqueeze(0).to(device)
            img_feature = model.encode_image(image)
            img_feature_list.append(img_feature)
            
        image_embeds = torch.cat(img_feature_list, dim=0)
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        
        # Text to image similarity
        print("text 2 image")
        i = 0
        correct_t2i = 0
        total_t2i = 0
        for i in range(text_feature.shape[0]):
            text = text_feature[i]

            sim = 100 * text @ image_embeds.T
            sim = sim.squeeze()
            correct_i = torch.argmax(sim)

            if i==correct_i:
                correct_t2i = correct_t2i + 1
            total_t2i = total_t2i + 1
        print(total_t2i)
        print(correct_t2i)
        print(correct_t2i/total_t2i)
        
        # Image to text similarity
        print("image to text")
        i = 0
        correct_i2t = 0
        total_i2t = 0
        for i in range(image_embeds.shape[0]):
            img = image_embeds[i]
            sim = 100 * img @ text_feature.T
            sim = sim.squeeze()
            correct_i = torch.argmax(sim)

            if i==correct_i:
                correct_i2t = correct_i2t + 1
            total_i2t = total_i2t + 1
        print(total_i2t)
        print(correct_i2t)
        print(correct_i2t/total_i2t)
        return { "text2image": correct_t2i/total_t2i, "image2text": correct_i2t/total_i2t }    

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    return images, list(captions)


def run_sharegpt4v_openclip(model, distilled_model, processor, data_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    distilled_model.eval()
    
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

    logit_scale = 100
    img_feature_list = []
    text_list = []

    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="Processing batches"):
            text_encoded = processor.tokenizer(captions).to(device)
            image_encoded = images.to(device)
            
            image_features = model.encode_image(image_encoded)
            text_features = distilled_model.encode_text(text_encoded)

            text_list.append(text_features)
            img_feature_list.append(image_features)
        
        text_feature = torch.cat(text_list, dim=0)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
            
        image_embeds = torch.cat(img_feature_list, dim=0)
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        
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
