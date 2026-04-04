import torch
import numpy as np
from PIL import Image
import cv2
from rembg import remove
from segment_anything import sam_model_registry, SamPredictor
from typing import Tuple, Union

def sam_init(path="sam/sam_vit_h——4b8939.pth", device_id=0):
    '''
    mkdir sam
    cd sam
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    '''
    sam_checkpoint = path
    model_type = "vit_h"

    device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [C, H, W] tensor (0-1 or 0-255) to PIL Image (RGB)"""
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor [C, H, W], got {tensor.ndim}D")
    
    # Normalize to [0, 255] if in [0, 1] range
    if tensor.max() <= 1.0:
        tensor = tensor * 255
    tensor = tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
    return Image.fromarray(tensor).convert("RGB")

def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image (RGB/RGBA/L) to [C, H, W] tensor in [0, 1] range.
    Handles:
      - RGB: [H, W, 3] → [3, H, W]
      - RGBA: [H, W, 4] → [4, H, W]  
      - L (grayscale): [H, W] → [1, H, W]
    """
    arr = np.array(pil_img).astype(np.float32) / 255.0
    
    if arr.ndim == 2:
        # Grayscale mask: [H, W] → [1, H, W]
        arr = arr[None, ...]  # Add channel dimension
    elif arr.ndim == 3:
        # RGB/RGBA: [H, W, C] → [C, H, W]
        arr = arr.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unexpected image dimensions: {arr.ndim}. Expected 2 or 3.")
    
    return torch.from_numpy(arr)  # Already on CPU, no need for permute

def pred_bbox_pil(image: Image.Image) -> Tuple[int, int, int, int]:
    """Predict bounding box using rembg (same as reference script)"""
    image_nobg = remove(image.convert('RGBA'), alpha_matting=True)
    alpha = np.asarray(image_nobg)[:, :, -1]
    x_nonzero = np.nonzero(alpha.sum(axis=0))
    y_nonzero = np.nonzero(alpha.sum(axis=1))
    
    if len(x_nonzero[0]) == 0 or len(y_nonzero[0]) == 0:
        # Fallback: return full image bbox
        return 0, 0, image.width - 1, image.height - 1
    
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    return x_min, y_min, x_max, y_max

def sam_segment_single(predictor: SamPredictor, pil_img: Image.Image) -> Image.Image:
    """Run SAM segmentation on a single PIL image (RGB), return RGBA PIL image"""
    image_np = np.asarray(pil_img.convert("RGB"))
    bbox = pred_bbox_pil(pil_img)
    
    predictor.set_image(image_np)
    masks, scores, _ = predictor.predict(box=np.array(bbox), multimask_output=True)
    
    # Select best mask (highest score)
    best_mask = masks[np.argmax(scores)]
    
    # Compose RGBA output
    out_image = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image_np
    out_image[:, :, 3] = (best_mask * 255).astype(np.uint8)
    
    return Image.fromarray(out_image, mode='RGBA')

def image_preprocess(pil_rgba: Image.Image, lower_contrast: bool = True, 
                     target_size: int = 256, ratio: float = 0.75) -> Tuple[Image.Image, Image.Image]:
    """Preprocess RGBA PIL image: contrast adjust, center-pad, resize to target_size"""
    image_arr = np.array(pil_rgba)  # [H, W, 4]
    in_h, in_w = image_arr.shape[:2]

    # Optional contrast reduction
    if lower_contrast:
        alpha_val = 0.8  # Contrast control (1.0-3.0)
        beta_val = 0     # Brightness control
        # Apply to RGB channels only, preserve alpha
        rgb = cv2.convertScaleAbs(image_arr[:, :, :3], alpha=alpha_val, beta=beta_val)
        image_arr[:, :, :3] = rgb
        # Ensure solid alpha for foreground
        image_arr[image_arr[:, :, -1] > 200, -1] = 255

    # Extract alpha mask and get bounding rect
    mask = (np.array(pil_rgba.split()[-1]) > 0).astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(mask)
    
    # Compute padded square size
    max_size = max(w, h)
    side_len = int(max_size / ratio)
    side_len = max(side_len, target_size)  # Ensure at least target_size
    
    # Center-pad the RGBA image
    padded = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded[center - h//2 : center - h//2 + h, 
           center - w//2 : center - w//2 + w] = image_arr[y:y+h, x:x+w]
    
    # Resize to target_size
    rgba_resized = Image.fromarray(padded).resize((target_size, target_size), Image.LANCZOS)
    
    # Separate RGB (with white background) and alpha mask
    rgba_arr = np.array(rgba_resized).astype(np.float32) / 255.0
    rgb = rgba_arr[:, :, :3] * rgba_arr[:, :, -1:] + (1 - rgba_arr[:, :, -1:])  # Composite on white
    alpha = rgba_arr[:, :, -1]
    
    return (Image.fromarray((rgb * 255).astype(np.uint8)), 
            Image.fromarray((alpha * 255).astype(np.uint8), mode="L"))

def segment_images(
    predictor: SamPredictor, 
    images: torch.Tensor, 
    lower_contrast: bool = True,
    target_size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Segment foreground objects from a batch of images using SAM + rembg.
    
    Args:
        predictor: Initialized SamPredictor instance
        images: Input tensor of shape [B, F, C, H, W] or [BF, C, H, W], 
                values in [0, 1] or [0, 255], RGB order
        lower_contrast: Whether to reduce contrast during preprocessing
        target_size: Output resolution (default: 256)
    
    Returns:
        rgb_batch: [B, F, 3, target_size, target_size] tensor, RGB images composited on white background
        mask_batch: [B, F, 1, target_size, target_size] tensor, alpha masks in [0, 1]
    """
    # Handle input shape: [B, F, C, H, W] -> [BF, C, H, W]
    original_shape = images.shape
    if images.ndim == 5:
        B, F, C, H, W = original_shape
        images = images.reshape(-1, C, H, W)
    else:
        BF, C, H, W = images.shape
        B = F = None  # Will return flattened
    
    device = images.device
    dtype = images.dtype
    results_rgb = []
    results_mask = []
    
    for i in range(images.shape[0]):
        # 1. Tensor → PIL (RGB)
        pil_rgb = tensor_to_pil(images[i])
        
        # 2. SAM segmentation (PIL RGB → PIL RGBA)
        pil_rgba = sam_segment_single(predictor, pil_rgb)
        
        # 3. Preprocess: center-pad, resize, composite
        rgb_pil, mask_pil = image_preprocess(
            pil_rgba, 
            lower_contrast=lower_contrast, 
            target_size=target_size
        )
        
        # 4. PIL → Tensor [C, H, W]
        rgb_tensor = pil_to_tensor(rgb_pil)[:3]  # [3, H, W]
        mask_tensor = pil_to_tensor(mask_pil)[0:1]  # [1, H, W]
        
        results_rgb.append(rgb_tensor)
        results_mask.append(mask_tensor)
        
        # Clear CUDA cache periodically to avoid OOM
        if (i + 1) % 4 == 0:
            torch.cuda.empty_cache()
    
    # Stack results: [BF, C, H, W]
    rgb_batch = torch.stack(results_rgb, dim=0).to(device=device, dtype=dtype)
    mask_batch = torch.stack(results_mask, dim=0).to(device=device, dtype=dtype)
    
    # Restore original batch/frame dimensions if input was 5D
    if B is not None and F is not None:
        rgb_batch = rgb_batch.reshape(B, F, 3, target_size, target_size)
        mask_batch = mask_batch.reshape(B, F, 1, target_size, target_size)
    
    return rgb_batch, mask_batch