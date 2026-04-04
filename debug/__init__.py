import os
import torch
from torchvision.utils import save_image

DEBUG_DIR = "debug_tensors"
os.makedirs(DEBUG_DIR, exist_ok=True)

def save_debug(tensor: torch.Tensor, filename: str = "debug.png", max_batches: int = 2):
    """
    Universal saver for 4D [B, C, H, W] or 5D [B, F, C, H, W] tensors.
    Assumes input range is [-1, 1] and rescales to [0, 1].
    """
    if tensor.ndim not in (4, 5):
        raise ValueError(f"Expected 4D or 5D tensor, got {tensor.ndim}D")
        
    # Limit samples to save memory & keep grids readable
    B_save = min(tensor.shape[0], max_batches)
    subset = tensor[:B_save].detach().float().cpu()
    
    # Handle 5D -> flatten to 4D, set grid columns = Frames
    if subset.ndim == 5:
        B, F, C, H, W = subset.shape
        subset = subset.view(B * F, C, H, W)
        nrow = F
    else:
        # 4D: default to 4 columns
        nrow = 4
        
    # Rescale [-1, 1] -> [0, 1] & clamp
    img = subset * 0.5 + 0.5
    img = torch.clamp(img, 0.0, 1.0)
    
    save_image(img, os.path.join(DEBUG_DIR, filename), nrow=nrow, normalize=False)
    print(f"✅ Saved {filename} | Shape: {tensor.shape[:3]}... -> Grid cols: {nrow}")