import torch
import numpy as np
from PIL import Image
from contextlib import nullcontext
from pipeline import BaselinePipeline
from diffusion_renderer.inference_svd_rgbx import *
from diffusion_renderer.inference_svd_xrgb import *


class DiffusionRendererPipeline(BaselinePipeline):
    def __init__(self, device='cuda', inverse_config=None, forward_config=None):
        # --- Default Inverse Configuration ---
        super(BaselinePipeline).__init__()
        self.inverse_config = {
            "inference_model_weights": './checkpoints/diffusion_renderer-inverse-svd',
            "inference_res": [512, 512],
            "inference_input_dir": "examples/input_video_frames/", # Not used in online pipeline
            "inference_save_dir": "examples/output_delighting/", # Not used in online pipeline
            # "inference_n_frames": 24,
            "overlap_n_frames": 6,
            "inference_n_steps": 20,
            "inference_min_guidance_scale": 1.0,
            "inference_max_guidance_scale": 1.0,
            "model_dtype": "fp16",
            "decode_chunk_size": 8,
            "cond_mode": 'skip',
            "chunk_mode": 'first',
            "image_group_mode": "folder",
            "model_passes": [
                "basecolor", "metallic", "roughness", "normal", "depth"
            ],
            "seed": 0,
            "save_video": True, # Not used in online pipeline
            "save_video_fps": 10, # Not used in online pipeline
            "save_image": True, # Not used in online pipeline
            "autocast": True,
            "use_deterministic_mode": False,
            "subsample_every_n_frames": 1,
        }
        if inverse_config:
            self.inverse_config.update(inverse_config)

        # --- Default Forward Configuration ---
        self.forward_config = {
            "inference_model_weights": './checkpoints/diffusion_renderer-forward-svd',
            "inference_res": [512, 512],
            "inference_input_dir": "examples/output_delighting/", # Not used in online pipeline
            "inference_save_dir": "examples/output_relighting/", # Not used in online pipeline
            # "inference_n_frames": 24,
            "inference_n_steps": 20,
            "inference_min_guidance_scale": 1.2,
            "inference_max_guidance_scale": 1.2,
            "model_dtype": "fp16",
            "decode_chunk_size": 8,
            "cond_mode": 'env',
            "lora_scale": 0.25,
            "model_pipeline": {
                "cond_mode": 'env',
                "target_image": 'rgb',
                "cond_images": {
                    "basecolor": "vae",
                    "normal": "vae",
                    "depth": "vae",
                    "roughness": "vae",
                    "metallic": "vae",
                    "env_ldr": "env",
                    "env_log": "env",
                    "env_nrm": "env",
                },
                "scale_cond_latents": True,
                "motion_bucket_id": 127,
                "fps": 7,
                "cond_aug": None,
                "cond_sigma_mean": -3.0,
                "cond_sigma_std": 0.5,
                "env_resolution": [512, 512],
                "unet_kwargs": {
                    "temporal_cross_attention_dim": None,
                    "cross_attention_dim": [320, 640, 1280, 1280],
                    "multi_res_encoder_hidden_states": True,
                    "in_channels": 24,
                    "conv_in_init": "zero",
                    "reset_cross_attention": False,
                },
            },
            "image_group_mode": "webdataset",
            "envlight": [
                "examples/hdri/sunny_vondelpark_1k.hdr",
                "examples/hdri/pink_sunrise_1k.hdr",
            ],
            "rotate_light": False,
            "cam_elevation": 0,
            "seed": 0,
            "save_video": True, # Not used in online pipeline
            "save_video_fps": 10, # Not used in online pipeline
            "save_image": True, # Not used in online pipeline
            "autocast": True,
            "use_deterministic_mode": False,
            "subsample_every_n_frames": 1,
            "use_fixed_frame_ind": False,
            "fixed_frame_ind": 0,
        }
        if forward_config:
            self.forward_config.update(forward_config)

        # Set Seed (using inverse_config seed as primary, or you can choose one)
        torch.manual_seed(self.inverse_config["seed"])
        
        # Determine Dtype
        self.weight_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.device = device

        # Initialize common components
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            self.inverse_config["inference_model_weights"], subfolder="feature_extractor",
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.inverse_config["inference_model_weights"], subfolder="image_encoder",
        )
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            self.forward_config["inference_model_weights"], subfolder="vae"
        )
        self.unet = UNetCustomSpatioTemporalConditionModel.from_pretrained(
            self.forward_config["inference_model_weights"], subfolder="unet",
            **self.forward_config["model_pipeline"]["unet_kwargs"]
        )
        self.scheduler = EulerDiscreteScheduler.from_pretrained(self.forward_config["inference_model_weights"], subfolder="scheduler")
        self.env_encoder = EnvEncoder.from_pretrained(self.forward_config["inference_model_weights"], subfolder="env_encoder")

        # Forward Pipeline (Relighting)

        self.inverse_pipeline = RGBXVideoDiffusionPipeline.from_pretrained(
            self.inverse_config["inference_model_weights"], 
            torch_dtype=self.weight_dtype, 
            trust_remote_code=True,
            image_encoder=self.image_encoder,
            feature_extractor=self.feature_extractor,
            cond_mode=self.inverse_config["cond_mode"]
        ).to(self.device)


        self.forward_pipeline = RGBXVideoDiffusionPipeline(
            vae=self.vae,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=None, 
            image_encoder=self.image_encoder,
            feature_extractor=self.feature_extractor,
            env_encoder=self.env_encoder,
            scale_cond_latents=self.forward_config["model_pipeline"]["scale_cond_latents"],
            cond_mode=self.forward_config["model_pipeline"]["cond_mode"]
        ).to(device=self.device, dtype=self.weight_dtype)

    def inverse_process(self, input_image, *args, **kwargs):
        """
        Inverse Process (Material Estimation) using Stable Material (rgbx pipeline)
        
        Args:
            input_image: Tensor of shape [BF, 3, H, W]
            pose: Camera poses (not directly used in the provided rgbx inference, but kept for signature)
            mask: Object mask (not directly used in the provided rgbx inference, but kept for signature)
            
        Returns:
            pred_albedo: Estimated albedo (mocked for now)
            pred_orm: Estimated ORM (Occlusion, Roughness, Metallic) (mocked for now)
        """
        B, F, C, H, W = input_image.shape
        g_buf = {}

        cond_images = {"rgb": input_image}
        cond_labels = {"rgb": "vae"}

        if self.inverse_config["cond_mode"] == "image": 
            cond_images["clip_img"] = input_image[:, 0:1, ...]
            cond_labels["clip_img"] = "clip"

        inference_height, inference_width = self.inverse_config["inference_res"]
        for inference_pass in self.inverse_config.model_passes:
            cond_images["input_context"] = inference_pass

        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(self.device.type, enabled=self.inverse_config["autocast"])

        with autocast_ctx:
            inference_image_list = self.inverse_pipeline(
                cond_images, cond_labels,
                height=inference_height, width=inference_width,
                num_frames=F,
                num_inference_steps=self.inverse_config["inference_n_steps"],
                min_guidance_scale=self.inverse_config["inference_min_guidance_scale"],
                max_guidance_scale=self.inverse_config["inference_max_guidance_scale"],
                fps=self.inverse_config["fps"],
                motion_bucket_id=self.inverse_config["motion_bucket_id"],
                noise_aug_strength=self.inverse_config["cond_aug"],
                generator=torch.Generator(device=self.device).manual_seed(self.inverse_config["seed"]),
                decode_chunk_size=self.inverse_config["decode_chunk_size"],
                return_dict=False,
            ).frames
        frames_tensor = torch.stack([
            torch.from_numpy(np.array(img).transpose(2, 0, 1)) 
            for img in inference_image_list
        ])

        # 2. Normalize to [0, 1] if they are uint8 (0-255)
        if frames_tensor.dtype == torch.uint8:
            frames_tensor = frames_tensor.to(dtype=self.weight_dtype).div(255.0)

            # 3. Add batch dimension to match G-buffer requirements [1, F, C, H, W]
        g_buf[inference_pass] = frames_tensor.unsqueeze(0).to(self.device)

        return g_buf

    def forward_process(self, g_buf, envlight_path_list, *args, **kwargs):
        """
        Forward Process (Relighting)
        
        Args:
            input_image: Original input image (not directly used in xrgb pipeline for conditioning, but kept for signature)
            pred_albedo: Estimated albedo from inverse process [BF, C, H, W]
            pred_orm: Estimated ORM from inverse process [BF, C, H, W]
            envlight_path_list: List of environment map paths for relighting
            
        Returns:
            relit_images: The relit images [BF, C, H, W]
        """
        B, F, C, H, W = pred_albedo.shape

        cond_images = {}
        cond_images["basecolor"] = pred_albedo
        cond_images["normal"] = pred_orm 

        env_resolution = tuple(self.forward_config["model_pipeline"]["env_resolution"])
        relit_images_list = []

        if envlight_path_list is None or len(envlight_path_list) == 0:
            envlight_path_list = self.forward_config["envlight"]

        for envlight_path in envlight_path_list:
            envlight_dict = process_environment_map(
                envlight_path,
                resolution=env_resolution,
                num_frames=F,
                fixed_pose=True,
                rotate_envlight=self.forward_config["rotate_light"],
                elevation=self.forward_config["cam_elevation"],
                env_format=["proj", "fixed", "ball"], 
            )
            cond_images["env_ldr"] = envlight_dict["env_ldr"].unsqueeze(0).permute(0, 1, 4, 2, 3).to(self.device, dtype=self.weight_dtype)
            cond_images["env_log"] = envlight_dict["env_log"].unsqueeze(0).permute(0, 1, 4, 2, 3).to(self.device, dtype=self.weight_dtype)
            env_nrm = envmap_vec(env_resolution, device=self.device) * 0.5 + 0.5
            cond_images["env_nrm"] = env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3).expand_as(cond_images["env_ldr"]).to(self.device, dtype=self.weight_dtype)

            cond_labels_forward = self.forward_config["model_pipeline"]["cond_images"]

            inference_height, inference_width = self.forward_config["inference_res"]

            if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
            else:
                autocast_ctx = torch.autocast(self.device.type, enabled=self.forward_config["autocast"])

            with autocast_ctx:
                inference_image_list = self.forward_pipeline(
                    cond_images, cond_labels_forward,
                    height=inference_height, width=inference_width,
                    num_frames=F,
                    num_inference_steps=self.forward_config["inference_n_steps"],
                    min_guidance_scale=self.forward_config["inference_min_guidance_scale"],
                    max_guidance_scale=self.forward_config["inference_max_guidance_scale"],
                    fps=self.forward_config["model_pipeline"]["fps"],
                    motion_bucket_id=self.forward_config["model_pipeline"]["motion_bucket_id"],
                    noise_aug_strength=self.forward_config["model_pipeline"]["cond_aug"] if self.forward_config["model_pipeline"]["cond_aug"] is not None else 0.0,
                    generator=torch.Generator(device=self.device).manual_seed(self.forward_config["seed"]),
                    cross_attention_kwargs={
                        'scale': self.forward_config["lora_scale"]},
                    dynamic_guidance=False,
                    decode_chunk_size=self.forward_config["decode_chunk_size"],
                ).frames[0] 
            
            current_relit_images = []
            for pil_img in inference_image_list:
                current_relit_images.append(torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.weight_dtype) / 255.0)
            relit_images_list.append(torch.cat(current_relit_images, dim=0).reshape(B, F, C, H, W))

        if len(relit_images_list) > 0:
            return relit_images_list[0] 
        else:
            return pred_albedo 

    def __call__(self, batch):
        # Preprocess
        batch = self.batch_preprocess(batch)

        input_image = batch["image"].to(self.device, dtype=self.weight_dtype)
        envs = batch["envs"].to(self.device, dtype=self.weight_dtype)

        # 1. Inverse Process (Material Estimation)
        g_buffer = self.inverse_process(input_image)

        # 2. Forward Process (Relighting)
        relit_images = self.forward_process(
            input_image, g_buffer, 
            envs
        )

        return relit_images

    def batch_preprocess(self, batch):
        """
        Processes the raw batch into a flattened BF (Batch*Frames) format
        ready for the Pipeline.
        """
        res = {}
        res["image"] = batch["source_images"].to(self.device, dtype=self.weight_dtype).flatten(0, 1)
        
        
        return res
