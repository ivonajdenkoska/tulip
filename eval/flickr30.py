import torch
from torchvision.datasets import CocoCaptions
from PIL import Image
import numpy as np
import pandas as pd


device = "cuda" if torch.cuda.is_available() else "cpu"

def get_text_feature(model, distilled_model, processor, data_path):
    text_list = []
    feature_list = []
    with torch.no_grad():
        df = pd.read_csv(data_path+"Datasets/flickr/results.csv", sep='|')
        df.columns = ['image_name', 'comment_number', 'comment']
        for i, data in df.iterrows():
            text = data['comment']
            if isinstance(text, float):
                text = " "
            text_list.append(text)
        len_list = len(text_list)
        print(len_list)

    #avoid OOM
    with torch.no_grad():
        for i in range(2000):
            text = text_list[i*len_list//2000: (i+1)*len_list//2000]
            text = processor.tokenizer(text).to(device)
            text_features = distilled_model.encode_text(text).to(device)
            feature_list.append(text_features)
    
    text_feature = torch.concatenate(feature_list, dim=0)
    return text_feature
    

def get_image_feature(model, processor, data_path):
    data_root = data_path+"Datasets/flickr/flickr30k_images/flickr30k_images/"
    img_feature_list = []
    with torch.no_grad():
        with open(data_path+"Datasets/flickr/results.csv", 'r') as f:
            next(f)
            dataset = f.readlines()
            data_len = len(dataset)
            for i in range(data_len//5):
                #1 image corresponding to 5 captions
                data = dataset[5*i]
                image_name = data.split("|")[0]
                image = Image.open(data_root + image_name)
                image = processor(image).unsqueeze(0).to(device)
                img_feature = model.encode_image(image).to(device)
                img_feature_list.append(img_feature)
                torch.cuda.empty_cache()
                del img_feature, image

            img_feature = torch.concatenate(img_feature_list, dim=0)
            return img_feature

def get_accuracy_t2i(text_feature, image_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

        text_feature = text_feature.cuda()
        image_feature = image_feature.cuda()

        pred_true = 0

        sim = (text_feature @ image_feature.T).softmax(dim=-1)

        for i in range(text_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            true_index = i//5
            if true_index in topk:
                pred_true = pred_true + 1

        print(pred_true/text_feature.shape[0])
        return pred_true/text_feature.shape[0]

def get_accuracy_i2t(text_feature, image_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

        text_feature = text_feature.cuda()
        image_feature = image_feature.cuda()

        pred_true = 0

        sim = (image_feature @ text_feature.T).softmax(dim=-1)
        for i in range(image_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            for j in range(5):
                true_index = 5*i + j
                if true_index in topk:
                    pred_true = pred_true + 1
                    break

        print(pred_true/image_feature.shape[0])
        return pred_true/image_feature.shape[0]

def run_flickr30(model, distilled_model, processor, data_path):
    model.eval()
    distilled_model.eval()

    image_feature = get_image_feature(model, processor, data_path)
    text_feature = get_text_feature(model, distilled_model, processor, data_path)

    image_to_text_R_1 = get_accuracy_i2t(text_feature, image_feature, 1)
    image_to_text_R_5 = get_accuracy_i2t(text_feature, image_feature, 5)
    image_to_text_R_10 = get_accuracy_i2t(text_feature, image_feature, 10)
    text_to_image_R_1 = get_accuracy_t2i(text_feature, image_feature, 1)
    text_to_image_R_5 = get_accuracy_t2i(text_feature, image_feature, 5)
    text_to_image_R_10 = get_accuracy_t2i(text_feature, image_feature, 10)

    return { "text_to_image_R@1": text_to_image_R_1, "text_to_image_R@5": text_to_image_R_5, "text_to_image_R@10": text_to_image_R_10,
             "image_to_text_R@1": image_to_text_R_1, "image_to_text_R@5": image_to_text_R_5, "image_to_text_R@10": image_to_text_R_10 }    
