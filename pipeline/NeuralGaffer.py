import torch
import torch.nn.functional as F
import torch.nn as nn
from safetensors.torch import load_file
import os
from pipeline import BaselinePipeline
from Neural_Gaffer.pipeline_neural_gaffer import Neural_Gaffer_StableDiffusionPipeline
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection

from argparse import Namespace

from pipeline import BaselinePipeline
from Neural_Gaffer.pipeline_neural_gaffer import Neural_Gaffer_StableDiffusionPipeline

config = Namespace(
    guidance_scale=3.0,
    seed=42,
    resolution=256,
    enable_xformers_memory_efficient_attention=False,
    mixed_precision='fp16',
    pretrained_model_name_or_path="kxic/zero123-xl",
    revision=None,
    ckpt_dir="Neural_Gaffer/neural_gaffer_res256"
)

class NeuralGafferPipeline(BaselinePipeline):
    def __init__(self, device='cuda', dtype=torch.float32, resume_from_checkpoint="Neural_Gaffer/neural_gaffer_res256/checkpoint-80000"):
        super(NeuralGafferPipeline, self).__init__(device, dtype)
        
        # Standard seeding
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            
        # Load base models
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="image_encoder", revision=config.revision
        )
        vae = AutoencoderKL.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision
        )
        unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="unet", revision=config.revision
        )
        
        vae.requires_grad_(False)
        image_encoder.requires_grad_(False)
        
        # Expand conv_in channels: 8 -> 16 (zero-init new channels)
        conv_in_16 = nn.Conv2d(16, unet.conv_in.out_channels, kernel_size=unet.conv_in.kernel_size, padding=unet.conv_in.padding)
        conv_in_16.requires_grad_(False)
        unet.conv_in.requires_grad_(False)
        nn.init.zeros_(conv_in_16.weight)
        conv_in_16.weight[:, :8, :, :].copy_(unet.conv_in.weight)
        conv_in_16.bias.copy_(unet.conv_in.bias)
        unet.conv_in = conv_in_16
        unet.requires_grad_(False)

        # --- CHECKPOINT LOADING WITHOUT ACCELERATE ---
        if resume_from_checkpoint:
            ckpt_path = resume_from_checkpoint
            if ckpt_path == "latest" and os.path.exists(config.ckpt_dir):
                ckpt_dirs = [d for d in os.listdir(config.ckpt_dir) if d.startswith("checkpoint-")]
                if ckpt_dirs:
                    ckpt_dirs.sort(key=lambda x: int(x.split("-")[1]))
                    ckpt_dir = os.path.join(config.ckpt_dir, ckpt_dirs[-1])
                    # Prefer safetensors, fallback to bin
                    ckpt_path = os.path.join(ckpt_dir)
                else:
                    ckpt_path = None

            if ckpt_path and os.path.exists(ckpt_path):
                print(f"✅ Resuming UNet from checkpoint: {ckpt_path}")
                state_dict = load_file(os.path.join(ckpt_path, "model.safetensors"), device=device)
                
                # If saved via accelerate, keys are usually prefixed with "unet."
                if any(k.startswith("unet.") for k in state_dict.keys()):
                    state_dict = {k.replace("unet.", ""): v for k, v in state_dict.items()}
                    
                unet.load_state_dict(state_dict, strict=False)
            else:
                print("⚠️ Checkpoint not found. Using base weights.")

        # Build pipeline (no unwrap_model needed)
        scheduler = DDIMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
        self.pipeline = Neural_Gaffer_StableDiffusionPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            vae=vae.eval(),
            image_encoder=image_encoder.eval(),
            feature_extractor=None,
            unet=unet.eval(),
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=self.dtype,
        )
        self.pipeline = self.pipeline.to(device)
        self.pipeline.set_progress_bar_config(disable=True)
        if config.enable_xformers_memory_efficient_attention:
            self.pipeline.enable_xformers_memory_efficient_attention()

    def __call__(self, batch):
        batch_size = batch["source_images"].size()[0]
        batch = self.batch_preprocess(batch)
        bf, _, h, w = input_image.shape
        input_image = batch["image_cond"].to(dtype=self.weight_dtype)

        target_envmap_ldr = batch["envir_map_target_ldr"].to(dtype=self.weight_dtype)
        target_envmap_hdr = batch["envir_map_target_hdr"].to(dtype=self.weight_dtype)

        generartor_list = [torch.Generator(device=self.accelerator.device).manual_seed(config.seed) for _ in range(bf)]
        with torch.autocast("cuda"):
            # todo: change the name of "cond_envir_map" to "target_envmap_hdr"
            output = self.pipeline(input_imgs=input_image, prompt_imgs=input_image, 
                            first_target_envir_map=target_envmap_hdr, second_target_envir_map=target_envmap_ldr, poses=None, 
                            height=h, width=w,
                            guidance_scale=config["guidance_scale"], num_inference_steps=50, generator=generartor_list).images

        output = 0.5 * (1.0 + output) # [-1, 1] -> [0, 1]
        output = output.reshape(batch_size, -1, 3, h, w)
        return output

    def batch_preprocess(self, batch):
        """
        Processes the raw batch into a flattened BF (Batch*Frames) format
        ready for the Specific Pipeline.
        """
        
        reference_image = batch["source_images"]
        batch_size, F, C, H, W = reference_image.size()
        reference_image = reference_image.reshape(batch_size * F, C, H, W)
        reference_image = F.interpolate(reference_image, size=(256, 256), mode='bilinear', align_corners=False)

        raw_lighting = batch["target_lighting"]
        raw_lighting = raw_lighting.expand(-1, F, -1, -1, -1)
        raw_lighting = raw_lighting.reshape(batch_size * F, C, H, W)
        reference_image = F.interpolate(raw_lighting, size=(256, 256), mode='bilinear', align_corners=False)

        source_view = batch["source_view"]
        batch_size, F, _, _ = raw_lighting.size()
        source_view = source_view.reshape(batch_size * F, 4, 4)

        ldr, hdr = tunemap(rotate_lighting(raw_lighting))

        reference_image = 2.0 * reference_image - 1.0
        ldr = 2.0 * ldr - 1.0
        hdr = 2.0 * hdr - 1.0

        processed_batch = {
            "image_cond": reference_image,
            "envir_map_target_ldr": ldr,
            "envir_map_target_hdr": hdr,

        }
        return processed_batch

def rotate_lighting(lighting, RT):
    """
    lighting: [B, C, H, W]
    RT: [B, 4, 4] (c2w)
    """
    B, C, H, W = lighting.shape
    device = lighting.device

    # 1. Get rotation from World to Camera
    # R_c2w is the top-left 3x3. R_w2c is the transpose.
    R_w2c = RT[:, :3, :3].transpose(1, 2)

    # 2. Recreate your specific grid (matching generate_envir_map_dir)
    lat_step = torch.pi / H
    lng_step = 2 * torch.pi / W
    
    # Latitude: pi/2 to -pi/2
    theta_range = torch.linspace(torch.pi/2 - 0.5*lat_step, -torch.pi/2 + 0.5*lat_step, H, device=device)
    # Longitude: pi to -pi
    phi_range = torch.linspace(torch.pi - 0.5*lng_step, -torch.pi + 0.5*lng_step, W, device=device)
    
    theta, phi = torch.meshgrid(theta_range, phi_range, indexing='ij')

    # 3. Map to Cartesian (matching your view_dirs stack)
    # x = cos(phi)cos(theta), y = sin(phi)cos(theta), z = sin(theta)
    x = torch.cos(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.cos(theta)
    z = torch.sin(theta)
    
    # [H, W, 3] -> [B, N, 3]
    world_dirs = torch.stack([x, y, z], dim=-1).view(1, -1, 3).expand(B, -1, -1)

    # 4. Rotate: New_Dir = R_w2c @ World_Dir
    rotated_dirs = torch.bmm(world_dirs, R_w2c) # [B, N, 3]
    
    # 5. Convert back to your theta/phi space
    # theta = arcsin(z)
    # phi = atan2(y, x)
    rx, ry, rz = rotated_dirs[..., 0], rotated_dirs[..., 1], rotated_dirs[..., 2]
    
    r_theta = torch.asin(rz.clamp(-1, 1))
    r_phi = torch.atan2(ry, rx)

    # 6. Normalize to [-1, 1] for grid_sample
    # Map theta [pi/2, -pi/2] -> [-1, 1]
    grid_v = - (r_theta / (torch.pi / 2)) 
    # Map phi [pi, -pi] -> [-1, 1]
    grid_u = r_phi / torch.pi

    grid = torch.stack([grid_u, grid_v], dim=-1).view(B, H, W, 2)

    # 7. Sample original map
    return F.grid_sample(lighting, grid, mode='bilinear', padding_mode='reflection', align_corners=True)


def tunemap(lighting):
    """
    lighting: [B, C, H, W] tensor, typically float32
    Returns:
        envir_map_ldr: [B, C, H, W] tensor (0-255, uint8)
        envir_map_hdr: [B, C, H, W] tensor (0-255, uint8)
    """
    # --- 1. LDR (Linear to Gamma space) ---
    # Clip to [0, 1] and apply gamma 2.2
    envir_map_ldr = torch.clamp(lighting, 0, 1)
    envir_map_ldr = torch.pow(envir_map_ldr, 1/2.2)
    
    # Scale to [0, 255] and convert to uint8
    envir_map_ldr = (envir_map_ldr * 255).to(torch.uint8)

    # --- 2. HDR (Log transform) ---
    # Using torch.log1p(x) which is log(1 + x)
    envir_map_hdr = torch.log1p(10 * lighting)
    
    # Global Rescale to [0, 1] per batch 
    # (If you want per-image rescaling, use dim=(1,2,3) inside the max calls)
    batch_max = envir_map_hdr.view(envir_map_hdr.shape[0], -1).max(dim=1)[0]
    batch_max = batch_max.view(-1, 1, 1, 1) + 1e-8 # Prevent division by zero
    
    envir_map_hdr = envir_map_hdr / batch_max
    
    # Scale to [0, 255] and convert to uint8
    envir_map_hdr = (envir_map_hdr * 255).to(torch.uint8)

    return envir_map_ldr, envir_map_hdr
