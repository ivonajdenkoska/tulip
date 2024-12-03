import torch
from torchvision.datasets import CocoCaptions
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_coco(model, distilled_model, processor, data_path):
    coco = CocoCaptions(root=data_path+"Datasets/coco/val2017/", annFile=data_path+"Datasets/coco/annotations/captions_val2017.json", transform=None)

    image_features = []
    text_features = []
    pred_true = 0

    with torch.no_grad():
        for image, captions in tqdm(coco, desc="Processing batches"):
            image_input = processor(image).unsqueeze(0).to(device)
            image_features.append(model.encode_image(image_input))

            captions = captions[0:5]
            caption_input = processor.tokenizer(captions).to(device)
            text_feature = distilled_model.encode_text(caption_input)
            text_features.extend(text_feature)

        image_features = torch.stack(image_features).squeeze()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = torch.stack(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = image_features.squeeze() @ text_features.squeeze().T
    
        print("I2T")
        for i in range(5000):
            pred = similarity[i]
            b = pred.argsort()[-1:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        print(pred_true / 5000)
        image_to_text_R_1 = pred_true / 5000
        pred_true = 0

        for i in range(5000):
            pred = similarity[i]
            b = pred.argsort()[-5:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        print(pred_true / 5000)
        image_to_text_R_5 = pred_true / 5000
        pred_true = 0

        for i in range(5000):
            pred = similarity[i]
            b = pred.argsort()[-10:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        print(pred_true / 5000)
        image_to_text_R_10 = pred_true / 5000
        pred_true = 0

        print("T2I")
        similarity = similarity.T
        for i in range(25000):
            pred = similarity[i]
            b = pred.argsort()[-1:]
            true_index = i//5
            if true_index in b:
                pred_true = pred_true + 1

        print(pred_true/25000)
        text_to_image_R_1 = pred_true/25000
        pred_true = 0

        for i in range(25000):
            pred = similarity[i]
            b = pred.argsort()[-5:]
            true_index = i//5
            if true_index in b:
                pred_true = pred_true + 1

        print(pred_true/25000) 
        text_to_image_R_5 = pred_true/25000
        pred_true = 0

        for i in range(25000):
            pred = similarity[i]
            b = pred.argsort()[-10:]
            true_index = i//5
            if true_index in b:
                pred_true = pred_true + 1

        print(pred_true/25000)
        text_to_image_R_10 = pred_true/25000
        
        return { "text_to_image_R@1": text_to_image_R_1, "text_to_image_R@5": text_to_image_R_5, "text_to_image_R@10": text_to_image_R_10,
                 "image_to_text_R@1": image_to_text_R_1, "image_to_text_R@5": image_to_text_R_5, "image_to_text_R@10": image_to_text_R_10 }    
