import numpy as np
import torch
import argparse
from open_clip import create_model_and_transforms, get_tokenizer, get_model_config
import glob
import re
import wandb
from datetime import datetime

from open_clip.transformer_rope import TextTransformerRoPE
from open_clip.transformer_cope import TextTransformerCoPE
from open_clip.transformer import TextTransformer

from eval.urban1k import run_urban1k_openclip
from eval.coco import run_coco
from eval.flickr30 import run_flickr30
from eval.sharegpt4v import run_sharegpt4v_openclip
from eval.dci_long import run_dci_long_openclip


def reprodicibility(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def get_latest_checkpoint(path: str):
    checkpoints = glob.glob(path + "**/*.pt", recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def get_parser():
    parser = argparse.ArgumentParser("TULIP evaluation")
    parser.add_argument("--model_name", type=str, default="coca_ViT-L-14")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="mscoco_finetuned_laion2b_s13b_b90k",
    )
    parser.add_argument("--distilled_model_path", type=str, default=None)
    parser.add_argument("--pos_encodings", type=str, choices=['cope', 'rope', 'learnable'], default="rope")
    parser.add_argument("--context_length", type=int, default=248)

    parser.add_argument("--run_urban1k", action="store_true")
    parser.add_argument("--run_coco", action="store_true")
    parser.add_argument("--run_flickr", action="store_true")
    parser.add_argument("--run_sharegpt4v", action="store_true")
    parser.add_argument("--run_dci_long", action="store_true")
    parser.add_argument("--lit_style", action="store_true")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dense-cap-eval")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    return args


def run_eval_clip(args):
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    args.name = "-".join(
        [
            date_str,
            f"model_{args.model_name}",
            f"pretrained_{args.pretrained}",
        ]
    )

    print("[-] Loading base CLIP")
    accumulative_results = {}
    
    for seed in args.seeds:
        # fix seed
        reprodicibility(seed)
        base_clip_model, _, processor = create_model_and_transforms(
            args.model_name, pretrained=args.pretrained
        )
        tokenizer = get_tokenizer(args.model_name, context_length=args.context_length)  
        teacher_cfg = get_model_config(args.model_name)

        if args.distilled_model_path is None:            
            distilled_model = base_clip_model.cuda() 
            base_clip_model = base_clip_model.cuda()
        else:
            if args.pos_encodings == "cope":
                distilled_model = TextTransformerCoPE(context_length=args.context_length,  
                                                        vocab_size=teacher_cfg["text_cfg"]["vocab_size"],
                                                        width=teacher_cfg["text_cfg"]["width"],
                                                        heads=teacher_cfg["text_cfg"]["heads"],
                                                        layers=teacher_cfg["text_cfg"]["layers"],
                                                        output_dim=teacher_cfg["text_cfg"]["width"])
            elif args.pos_encodings == "rope":
                distilled_model = TextTransformerRoPE(context_length=args.context_length,  
                                                        vocab_size=teacher_cfg["text_cfg"]["vocab_size"],
                                                        width=teacher_cfg["text_cfg"]["width"],
                                                        heads=teacher_cfg["text_cfg"]["heads"],
                                                        layers=teacher_cfg["text_cfg"]["layers"],
                                                        output_dim=teacher_cfg["text_cfg"]["width"])
            elif args.pos_encodings == "learnable":
                distilled_model = TextTransformer(context_length=args.context_length, 
                                                vocab_size=teacher_cfg["text_cfg"]["vocab_size"],
                                                width=teacher_cfg["text_cfg"]["width"],
                                                heads=teacher_cfg["text_cfg"]["heads"],
                                                layers=teacher_cfg["text_cfg"]["layers"],
                                                output_dim=teacher_cfg["text_cfg"]["width"])
                
            checkpoint = torch.load(args.distilled_model_path)
            
            # remove all the module. prefix from the state_dict of the checkpoint
            checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            
            # Load the text part
            filtered_state_dict = {}
            for name, param in distilled_model.named_parameters():
                if name in checkpoint['state_dict']:
                    filtered_state_dict[name] = checkpoint['state_dict'][name]
                else:
                    print(f"Warning: {name} not found in checkpoint")

            # Load the filtered state dict
            missing_keys, unexpected_keys = distilled_model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")

            # Verify loading
            for name, param in distilled_model.named_parameters():
                if name in filtered_state_dict:
                    print(f"Parameter {name} loaded successfully.")
                    print(f"Loaded value: {param.data.mean().item():.4f}")
                    print(f"Checkpoint value: {filtered_state_dict[name].mean().item():.4f}")
                else:
                    print(f"Parameter {name} not found in checkpoint.")
                    raise ValueError(f"Parameter {name} not found in checkpoint.")
            
            distilled_model.cuda()
            if not args.lit_style:
                visual_state_dict = {k.replace('visual.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('visual.')}
                # Load the visual part
                missing_keys, unexpected_keys = base_clip_model.visual.load_state_dict(visual_state_dict, strict=False)
                print(f"[-] Missing keys: {missing_keys}")
                print(f"[-] Unexpected keys: {unexpected_keys}")

                # Verify a few key parameters
                for name, param in base_clip_model.visual.named_parameters():
                    if name in visual_state_dict:
                        print(f"Parameter {name} loaded successfully.")
                        print(f"Loaded value: {param.data.mean().item():.4f}")
                        print(f"Checkpoint value: {visual_state_dict[name].mean().item():.4f}")
                    else:
                        print(f"Parameter {name} not found in checkpoint.")
                        raise ValueError(f"Parameter {name} not found in checkpoint.")
                
            base_clip_model = base_clip_model.cuda()

        base_clip_model.eval()
        processor.tokenizer = tokenizer
        print("[-] Loaded model")

        if args.run_urban1k: 
            print(f"[-] Running Urban1k - seed: {seed}")
            urban1k_results = run_urban1k_openclip(base_clip_model, distilled_model, processor, args.data_path)
            if "urban1k" not in accumulative_results:
                accumulative_results["urban1k"] = {}
            accumulative_results["urban1k"][seed] = urban1k_results
        if args.run_coco:
            print(f"[-] Running COCO retrieval - seed: {seed}")
            coco_results = run_coco(base_clip_model, distilled_model, processor, args.data_path)
            if "coco" not in accumulative_results:
                accumulative_results["coco"] = {}
            accumulative_results["coco"][seed] = coco_results
        if args.run_flickr:
            print(f"[-] Running Flickr retrieval - seed: {seed}")
            flickr_results = run_flickr30(base_clip_model, distilled_model, processor, args.data_path)
            if "flickr" not in accumulative_results:
                accumulative_results["flickr"] = {}
            accumulative_results["flickr"][seed] = flickr_results
        if args.run_sharegpt4v:
            print(f"[-] Running ShareGPT4V - seed: {seed}")
            sharegpt4v_results = run_sharegpt4v_openclip(base_clip_model, distilled_model, processor, args.data_path)
            if "sharegpt4v" not in accumulative_results:
                accumulative_results["sharegpt4v"] = {}
            accumulative_results["sharegpt4v"][seed] = sharegpt4v_results
        if args.run_dci_long:
            print(f"[-] Running DCI-Long - seed: {seed}")
            dci_long_results = run_dci_long_openclip(base_clip_model, distilled_model, processor, args.data_path)
            if "dci_long" not in accumulative_results:
                accumulative_results["dci_long"] = {}
            accumulative_results["dci_long"][seed] = dci_long_results
    
    print(accumulative_results)
    final_results = {}
    for task, results in accumulative_results.items():
        final_results[task] = {}
        for seed, result in results.items():
            for metric, value in result.items():
                if metric not in final_results[task]:
                    final_results[task][metric] = []
                final_results[task][metric].append(value)

    for task, results in final_results.items():
        for metric, values in results.items():
            final_results[task][metric] = np.mean(values)

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.name,
            config=vars(args),
        )
        wandb.log(final_results)
        wandb.finish()

if __name__ == "__main__":
    args = get_parser()
    run_eval_clip(args)
