import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
import tempfile
import PIL.Image
from pipeline import BaselinePipeline

from LightSwitch.produce_gs_relightings import *
from LightSwitch.dataset_colmap import *
from .segment import segment_images, sam_init

env_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 512), antialias=True),
        transforms.Normalize([0.5], [0.5]),
    ])

class LightSwitchPipeline(BaselinePipeline):
    def __init__(self, device='cuda'):
        super(BaselinePipeline).__init__()
        self.config = {
            "pretrained_model": "thebluser/lightswitch",
            "pretrained_model_sm": "thebluser/stable-material-mv",
            "guidance_scale": 3.0,
            "sm_guidance_scale": 3.0,
            "seed": 42,
            "resolution": 512,
        }
        
        # Set Seed
        torch.manual_seed(self.config["seed"])
        
        # Determine Dtype
        self.weight_dtype = torch.float32
        self.device = device

        # Initialize Models
        scheduler = DDIMScheduler.from_pretrained(
            "Manojb/stable-diffusion-2-1-base", 
            subfolder="scheduler", 
            prediction_type="v_prediction"
        )
        vae = AutoencoderKL.from_pretrained(
            "Manojb/stable-diffusion-2-1-base", 
            subfolder="vae"
        )
        unet = UNet2DConditionModel.from_pretrained(
            self.config["pretrained_model"],
            use_safetensors=True,
        )

        # Forward Pipeline (Relighting)
        self.forward = RelightingPipelineMVVAE(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None
        ).to(device=self.device, dtype=self.weight_dtype)

        # Inverse Pipeline (Stable Material)
        self.inverse = StableMaterialPipelineMV.from_pretrained(
            self.config["pretrained_model_sm"], 
            torch_dtype=self.weight_dtype, 
            trust_remote_code=True
        ).to(self.device)

        self.sam = sam_init(
            "sam/sam_vit_h_4b8939.pth",
        )

    def inverse_process(self, input_image, pose, mask, num_inference_loops=1, num_inference_steps=35):
        """
        Inverse process for material estimation with robust CFG & channel alignment.
        Returns: pred_albedo, pred_orm (both [B, C, H, W])
        """
        stable_material = self.inverse
        h, w = input_image.shape[2:]
        batch_size = input_image.shape[0]
        do_cfg = self.config["sm_guidance_scale"] > 1.0

        # 1. Encode prompt (底層已處理 CFG 翻倍)
        with torch.no_grad():
            prompt_embeds = stable_material._encode_image_with_pose(
                input_image, pose, self.device, 1, do_cfg
            )

        # 2. Setup Scheduler
        stable_material.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = stable_material.scheduler.timesteps

        # 3. Prepare Latents
        # 注意：標準 VAE Latents 為 4 通道，若你的模型預期 8 通道請保留原值
        latents = stable_material.prepare_latents(
            batch_size, 8, h, w, prompt_embeds.dtype, self.device, None
        )
        img_latents = stable_material.prepare_img_latents(
            input_image, batch_size, prompt_embeds.dtype, self.device, None, do_cfg
        )


        pred_albedo, pred_orm = [], []

        chunk_size = 16  # 🔑 可依顯存調整 (建議 8 / 16 / 32)
        # 4. Denoising Loop
        for _ in range(num_inference_loops):
            for i, t in enumerate(tqdm(timesteps, disable=True)):
                # 1. 全局隨機打亂 (保持與原邏輯完全一致)
                latents_B = latents.shape[0]
                perm_indices = torch.randperm(latents_B)
                latents = latents[perm_indices]
                prompt_embeds = prompt_embeds[perm_indices]
                img_latents = img_latents[perm_indices]

                updated_latents_chunks = []

                # 2. 分塊處理 (避免 OOM)
                for start in range(0, latents_B, chunk_size):
                    end = min(start + chunk_size, latents_B)
                    lat_chunk = latents[start:end]
                    prompt_chunk = prompt_embeds[start:end]
                    img_chunk = img_latents[start:end]

                    # 針對當前 chunk 執行 CFG 翻倍
                    if do_cfg:
                        img_in = torch.cat([torch.zeros_like(img_chunk), img_chunk], dim=0)
                        prompt_in = torch.cat([torch.zeros_like(prompt_chunk), prompt_chunk], dim=0)
                    else:
                        img_in = img_chunk
                        prompt_in = prompt_chunk

                    with torch.no_grad():
                        out = stable_material.call_1_denoise_permute(
                            latents=lat_chunk,
                            img_latents=img_in,
                            prompt_embeds=prompt_in,
                            guidance_scale=self.config["sm_guidance_scale"],
                            t=t
                        ).images[0]  # 預期輸出 Batch 大小與 lat_chunk 一致

                    # 立即移至 CPU 暫存，釋放 GPU 顯存
                    updated_latents_chunks.append(out.cpu())

                # 3. 拼接回完整 Batch 並還原原始順序
                latents = torch.cat(updated_latents_chunks, dim=0).to(latents.device)
                latents = reverse_order(latents, [perm_indices])
                prompt_embeds = reverse_order(prompt_embeds, [perm_indices])
                img_latents = reverse_order(img_latents, [perm_indices])

            # 4. Decode (保持原位置不變)
            albedo, orm = stable_material.decode_latents(latents, permute=True)
            pred_albedo.append(albedo)
            pred_orm.append(orm)
        
        pred_albedo = np.mean(np.stack(pred_albedo), axis=0)
        pred_orm = np.mean(np.stack(pred_orm), axis=0)
        
        pred_albedo = torch.from_numpy(pred_albedo).permute(0, 3, 1, 2).to(self.device)
        pred_orm = torch.from_numpy(pred_orm).permute(0, 3, 1, 2).to(self.device)
        
        pred_albedo = pred_albedo.to(dtype=self.weight_dtype) * mask.to(self.device)
        pred_orm = pred_orm.to(dtype=self.weight_dtype) * mask.to(self.device)

        # 歸一化至 [-1, 1]
        pred_albedo = T.Normalize([0.5], [0.5])(pred_albedo)
        pred_orm = T.Normalize([0.5], [0.5])(pred_orm)

        return pred_albedo, pred_orm

    def forward_process(self, input_image, pred_albedo, pred_orm, envs_darker, envs_brighter, dir_embeds, pluckers, num_inference_steps=35, num_inference_loops=1):
        pipeline = self.forward
        gs = self.config["guidance_scale"]
        h, w = input_image.shape[2:]
        batch_size = input_image.shape[0]

        # 1. Encode Environment
        do_cfg = gs > 1.0
        scene_features = pipeline.encode_env(
            envs_darker, envs_brighter, dir_embeds, do_cfg, self.device, self.weight_dtype
        )

        # 2. Setup Timesteps
        pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = pipeline.scheduler.timesteps

        # 3. Prepare Latents
        latents = pipeline.prepare_latents(
            batch_size, 4, h, w, scene_features.dtype, self.device, None
        )
        condition_latents = pipeline.prepare_condition_latents(
            input_image, pred_albedo, pred_orm, pluckers, batch_size,
            scene_features.dtype, self.device, None, do_cfg
        )

        if do_cfg:
            scene_features_zero, scene_features = scene_features.chunk(2)
        temp_relit_images_ours = []
        # 4. Denoising Loop
        chunk_size = 16  # 🔑 根據顯存大小調整，建議 16 或 8

        for _ in range(num_inference_loops):
            for t in tqdm(timesteps, desc="Forward Relighting", disable=True):
                with torch.no_grad():
                    latents_B = latents.shape[0]
                    perm_indices = torch.randperm(latents_B)
                    latents = latents[perm_indices]
                    scene_features = scene_features[perm_indices]
                    condition_latents = condition_latents[perm_indices]

                    updated_latents_list = []
                    # --- 分塊處理避免 OOM ---
                    for start_idx in range(0, latents_B, chunk_size):
                        end_idx = min(start_idx + chunk_size, latents_B)
                        
                        lat_chunk = latents[start_idx:end_idx]
                        scene_chunk = scene_features[start_idx:end_idx]
                        cond_chunk = condition_latents[start_idx:end_idx]

                        # 若 CFG 啟用且 Pipeline 要求條件張量需預先翻倍，請取消下方註解：
                        if gs > 1.0:
                        #     cond_chunk = torch.cat([torch.zeros_like(cond_chunk), cond_chunk], dim=0)
                            scene_chunk = torch.cat([torch.zeros_like(scene_chunk), scene_chunk], dim=0)

                        out = pipeline.call_1_denoise_permute(
                            latents=lat_chunk,
                            condition_latents=cond_chunk,
                            scene_features=scene_chunk,
                            guidance_scale=gs,
                            t=t
                        ).images[0]  # 提取更新後的 latents

                        # 移至 CPU 釋放顯存，避免累積爆顯存
                        updated_latents_list.append(out.cpu())

                    # 拼接回完整 Batch 並移回 GPU
                    latents = torch.cat(updated_latents_list, dim=0).to(latents.device)

                    # 恢復原始順序
                    latents = reverse_order(latents, [perm_indices])
                    scene_features = reverse_order(scene_features, [perm_indices])
                    condition_latents = reverse_order(condition_latents, [perm_indices])

            relit_images = pipeline.decode_latents(latents, permute=True)
            temp_relit_images_ours.append(relit_images)
        relit_image_mean = np.mean(np.stack(temp_relit_images_ours), axis=0)

        relit_image_mean = torch.from_numpy(relit_image_mean).permute(0, 3, 1, 2)
        return relit_image_mean.to(self.device)

    def __call__(self, batch):
        # Preprocess
        real_frame_size = batch["source_images"].size()[1]
        batch = self.batch_preprocess(batch)

        # Move to device and dtype
        mask = batch["mask"].to(self.device, dtype=self.weight_dtype)
        input_image = batch["image"].to(self.device, dtype=self.weight_dtype)
        dir_embeds = batch["dir_embeds"].to(self.device, dtype=self.weight_dtype)
        pluckers = batch["pluckers"].to(self.device, dtype=self.weight_dtype)
        pose = batch["T"].to(self.device, dtype=self.weight_dtype)
        envs_darker = batch["envs_darker"].to(self.device, dtype=self.weight_dtype)
        envs_brighter = batch["envs_brighter"].to(self.device, dtype=self.weight_dtype)

        # 1. Inverse Process (Material Estimation)
        pred_albedo, pred_orm = self.inverse_process(input_image, pose, mask)

        # 2. Forward Process (Relighting)
        relit_images = self.forward_process(
            input_image, pred_albedo, pred_orm, 
            envs_darker, envs_brighter, dir_embeds, pluckers
        )

        # 3. Post-process Result
        # Apply mask (ensure mask is [BF, 1, H, W] or [BF, 3, H, W])
        relit_images = relit_images * mask

        # 4. Reshape to [B, F, C, H, W]
        # batch_size is the original B, num_views is F (usually 16)
        _, C, H, W = relit_images.shape
        final_output = relit_images.reshape(1, -1, C, H, W)[:,:real_frame_size, ...]

        return final_output

    def batch_preprocess(self, batch):
        """
        Processes the raw batch into a flattened BF (Batch*Frames) format
        ready for the LightSwitch Pipeline.
        """
        batch_size = batch["idx"].shape[0]
        num_views = batch["source_view"].shape[1]
        total_elements = batch_size * num_views
        res = {}
        
        # 1. Standard Tensors [B, V, C, H, W] -> [B*V, C, H, W]
        # Note that mask is not the input!
        masked_image, mask = segment_images(self.sam, batch["source_images"], target_size=512)

        res["mask"] = mask.to(self.device, dtype=self.weight_dtype).flatten(0, 1)
        res["image"] = masked_image.to(self.device, dtype=self.weight_dtype).flatten(0, 1)

        res["image"] = 2.0 * res["image"] - 1.0
        
        # 2. Camera & Pose Extraction (NumPy for trig/math)
        T_c2w_np = blender_to_colmap(batch["source_view"].cpu().float().numpy())
        K_np = batch["source_Ks"].cpu().float().numpy()
        img_shape = batch["source_images"].shape[-2:] # (H, W)
        
        all_pluckers = []
        for b in range(batch_size):
            for v in range(num_views):
                fov = calculate_fov_from_k(K_np[b, v], img_shape)
                p_rays = generate_plucker_rays(T_c2w_np[b, v], img_shape, fov)
                all_pluckers.append(p_rays)
        
        # Convert Pluckers back to Tensor [BF, 6, H/8, W/8]
        res["pluckers"] = torch.from_numpy(np.stack(all_pluckers)).to(self.device, dtype=self.weight_dtype)

        # 3. Spherical Pose Embedding (The "MVP" in your model)
        # T_c2w: (B, V, 4, 4) -> (B*V, 4, 4)
        T_c2w_torch = torch.from_numpy(T_c2w_np).to(self.device, dtype=torch.float32).flatten(0, 1)
        T_w2c = torch.inverse(T_c2w_torch)
        
        # Coordinate system correction: rotate_x(-pi/2)
        rot_x_corr = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        T_w2c_corrected = rot_x_corr @ T_w2c
        
        # Output shape: [BF, 4]
        res["T"] = get_spherical_pose(T_w2c_corrected).to(dtype=self.weight_dtype)

        # 4. Environment Map Processing
        envmap_np = batch["target_lighting"].cpu().float().numpy()
        # Flatten envmap: (B, V, C, H, W) -> (B*V, C, H, W)
        if envmap_np.ndim == 5:
            envmap_np = envmap_np.reshape(-1, *envmap_np.shape[2:])
        elif envmap_np.ndim == 4:
            envmap_np = np.repeat(envmap_np, num_views, axis=0)
                
        max_val = np.maximum(envmap_np.max(), 1e-8)
        darker = (np.log10(envmap_np + 1) / np.log10(max_val + 1)).clip(0, 1)
        brighter = hlg_oetf(darker).clip(0, 1)
        darker = env_transform(np.transpose(darker[0], (1, 2, 0)))
        brighter = env_transform(np.transpose(brighter[0], (1, 2, 0))) # has bbeen shift to [-1, 1] via env_transform

        res["envs_darker"] = darker.unsqueeze(0).repeat(total_elements, 1, 1, 1).to(self.device, dtype=self.weight_dtype)
        res["envs_brighter"] = brighter.unsqueeze(0).repeat(total_elements, 1, 1, 1).to(self.device, dtype=self.weight_dtype)
        # 5. Directional Embeddings
        # Assuming self.dir_embeds is already defined as a tensor [C, H, W]
        dir_embeds =  torch.tensor(generate_directional_embeddings(), dtype=torch.float32).permute(2, 0, 1)
        res["dir_embeds"] = dir_embeds.unsqueeze(0).repeat(total_elements, 1, 1, 1).to(self.device, dtype=self.weight_dtype)

        if total_elements == 1: # Light Switch want input to be 16
            for k, v in res.items():
                res[k] = torch.cat([v]*16, dim=0)
        return res

def expand_batch_to_16(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Expand a tensor with batch_size=1 to batch_size=16 by repeating.
    
    Args:
        tensor: Input tensor, e.g., [1, F, C, H, W] or [1, C, H, W]
        dim: The batch dimension to expand (default: 0)
    
    Returns:
        Tensor with size 16 along the specified dimension
    """
    if tensor.size(dim) == 1:
        # Repeat 16 times along the batch dimension
        repeat_factors = [1] * tensor.ndim
        repeat_factors[dim] = 16
        return tensor.repeat(*repeat_factors)
    elif tensor.size(dim) == 16:
        return tensor  # Already correct size
    else:
        raise ValueError(f"Expected batch size 1 or 16 at dim {dim}, got {tensor.size(dim)}")
    
def calculate_fov_from_k(K, shape):
    """
    K: [3, 3] intrinsic matrix
    shape: (H, W) tuple
    Returns: (fov_x, fov_y) in radians
    """
    K = K[:3, :3]
    H, W = shape
    fx = K[0, 0]
    fy = K[1, 1]
    
    fov_x = 2 * np.arctan(W / (2.0 * fx))
    fov_y = 2 * np.arctan(H / (2.0 * fy))
    
    return fov_x, fov_y

def blender_to_colmap(T_blender):
    """
    Transforms a 4x4 matrix from Blender (RDF: Right-Up-Back) 
    to COLMAP (RDF: Right-Down-Forward).
    T_blender: np.ndarray of shape (..., 4, 4)
    """
    # This matrix flips the Y and Z axes
    flip_yz = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ], dtype=T_blender.dtype)
    
    # If T is Camera-to-World (c2w), we multiply from the right
    # to change the camera's local coordinate system.
    return T_blender @ flip_yz

def get_spherical_pose(T_w2c):
    """
    Converts W2C matrix to [theta, sin(phi), cos(phi), r].
    Expects T_w2c shape (N, 4, 4)
    """
    # Camera position in world space: -R^T @ t
    R = T_w2c[:, :3, :3]
    t = T_w2c[:, :3, 3:4]
    cam_pos = (-R.transpose(-1, -2) @ t).squeeze(-1) # (N, 3)
    
    x, y, z = cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2]
    
    r = torch.linalg.norm(cam_pos, dim=-1)
    # Elevation (theta): angle from Z-axis
    theta = torch.acos(torch.clamp(z / (r + 1e-8), -1.0, 1.0))
    # Azimuth (phi): angle in XY plane
    phi = torch.atan2(y, x)
    
    return torch.stack([theta, torch.sin(phi), torch.cos(phi), r], dim=-1)