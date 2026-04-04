import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import json
import tqdm
import torch
import shutil
import logging
import argparse
import warnings
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from metrics import MetricCalculator

# Assuming these are your custom modules
from dataset.LavalObjaverseDataset import EvalDataset

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
os.environ['HF_HOME'] = './hf_cache'

logger = logging.getLogger(__name__)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def log_validation(dataloader, pipeline, args, metric_fn):
    device = get_device()
    output_dir = Path(args.output_dir).resolve()
    res_json_path = output_dir / f"{args.baseline}_{args.task}_results.json"
    
    # Resume Logic
    if args.skip_exist and res_json_path.exists():
        with open(res_json_path, 'r') as f:
            evaluation_results = json.load(f)
        if isinstance(evaluation_results["data_pair"], list):
            evaluation_results["data_pair"] = {str(item["sample_idx"]): item for item in evaluation_results["data_pair"]}
        logger.info(f"📦 Resumed: {len(evaluation_results['data_pair'])} samples already processed.")
    else:
        evaluation_results = {"average": {}, "data_pair": {}}

    # Load Metadata
    with open(args.pair_info, 'r') as f:
        data_pairs = json.load(f)

    bar = tqdm.tqdm(dataloader, desc="Evaluating")
    
    for batch in bar:
        if batch is None: continue

        indices = batch['idx'].tolist()
        keep_indices = [i for i, idx in enumerate(indices) if str(idx) not in evaluation_results["data_pair"]]
        
        if not keep_indices: continue
        
        # Filter batch for unprocessed samples only
        k_idx = torch.tensor(keep_indices, device=device)
        filtered_batch = {
            k: v[k_idx.to(v.device)] if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()
        }
        # Inference
        with torch.autocast("cuda"):
            outputs = pipeline(filtered_batch) # Expected shape (B, S, C, H, W)
            targets = filtered_batch['target_images'] # Adjust key based on your dataset

        # Calculate and Store
        B_filtered = outputs.shape[0]
        for b in range(B_filtered):
            sample_idx = filtered_batch['idx'][b].item()
            meta = data_pairs[sample_idx] # Ensure key matching
            
            # Metrics
            p, sp, s, l, _, _, _ = metric_fn(outputs[b:b+1], targets[b:b+1], average=True)
            
            current_eval = {
                "sample_idx": sample_idx,
                "psnr": p, "spsnr": sp, "ssim": s, "lpips": l,
                "pred_image": [],
                "pred_depth": []
            }

            # Save per view
            for v in range(outputs.shape[1]):
                view_name = meta["view"][v].split('.')[0]
                view_dir = output_dir / str(sample_idx) / view_name
                view_dir.mkdir(parents=True, exist_ok=True)
                
                img_path = view_dir / "pred_relight.png"
                save_image(outputs[b, v], img_path)
                if args.save_gt:
                    save_image(filtered_batch["target_images"][b, v], view_dir / "gt.png")
                if args.save_ref:
                    save_image(filtered_batch["source_images"][b, v], view_dir / "ref.png")
                current_eval["pred_image"].append(str(img_path))

            evaluation_results["data_pair"][str(sample_idx)] = current_eval

            # Update averages
            all_vals = list(evaluation_results["data_pair"].values())
            for k in ["psnr", "spsnr", "ssim", "lpips"]:
                evaluation_results["average"][k] = np.mean([x[k] for x in all_vals])

            bar.set_postfix({k.upper(): f"{v:.2f}" for k, v in evaluation_results["average"].items()})

            # Atomic JSON save
            with open(res_json_path, 'w') as f:
                json.dump(evaluation_results, f, indent=4)

    return evaluation_results

def main(args):
    logging.basicConfig(level=logging.INFO)
    device = get_device()

    # Pipeline selection
    if args.baseline == "LightSwitch":
        from pipeline.LightSwitch import LightSwitchPipeline
        pipeline = LightSwitchPipeline()
    elif args.baseline == "DiffusionRenderer":
        from pipeline.DiffusionRenderer import DiffusionRendererPipeline
        pipeline = DiffusionRendererPipeline()
    elif args.baseline == "NeuralGaffer":
        from pipeline.NeuralGaffer import NeuralGafferPipeline
        pipeline = NeuralGafferPipeline()
        dataset = EvalDataset(args.dataset_path, args.pair_info, black_background=True)

    elif args.baseline == "Trained-NeuralGaffer":
        from pipeline.NeuralGaffer import NeuralGafferPipeline
        raise NotImplementedError(f"Baseline {args.baseline} not supported.")
        # pipeline = DiffusionRendererPipeline(resume_from_checkpoint)
    else:
        raise NotImplementedError(f"Baseline {args.baseline} not supported.")
    
    # Dataset Setup
    # dataset = EvalDataset(args.dataset_path, args.pair_info)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    metric_fn = MetricCalculator(device)

    log_validation(dataloader, pipeline, args, metric_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_pretrained", type=str, default='models/2d_training/relight')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='./output/')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, default="/media/HDD1/hejun/LavalObjaverseDataset")
    parser.add_argument("--baseline", type=str, default="LightSwitch")
    parser.add_argument("--skip_exist", action='store_true')
    parser.add_argument("--pair_info", type=str, default='/media/HDD1/hejun/LavalObjaverseDataset/experimental_pair/1_to_1_mapping_pairs.json')
    parser.add_argument("--save_gt", action='store_true')
    parser.add_argument("--save_ref", action='store_true')

    args = parser.parse_args()
    
    if args.baseline == "LightSwitch" and args.batch_size != 1:
        raise ValueError(f"Invalid batch size {args.batch_size} for Baseline f{args.baseline}")
    # Path preparation
    args.task = args.pair_info.split('/')[-1].split('.')[0]
    args.output_dir = os.path.join(args.output_dir, args.baseline, args.task)
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)

'''
dataset_path : /media/HDD1/hejun/LavalObjaverseDataset on 0823
dataset_path : /media/HDD2/hejun/LavalObjaverseDataset on 0422

'''