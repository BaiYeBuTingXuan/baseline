import numpy as np
import os
import json
import kiui
import torch
import torch.nn.functional as F
import glob
from .utils import *
import OpenEXR, Imath
import math
from itertools import combinations, product
import random

PI = 3.1415926535
IS_ADDITION_ROTATION_FOR_ARGUMENT = True
IS_CROPPING_FOR_ARGUMENT = True
INFO_BUFFER = "dataset/buffer"
N_LIGHTINGS = 8 # MAX at 16
N_VIEWS = 16
DEPTH_SCALE_PNG = 1
MAX_EXAMPLES = 1_000_000

def resize(x, size, mode='bilinear'):
    """
    Args:
        tensor (torch.Tensor): The input image tensor of shape (C, H, W).
        mask (torch.Tensor): The binary mask tensor of shape (1, H, W).
        size (tuple): The target size as (height, width).
        mode (str): Interpolation mode ('bilinear' for image, 'nearest' for mask).

    Returns:
        torch.Tensor: Resized image tensor.
        torch.Tensor: Resized mask tensor.
    """
    # Ensure the mask is a binary tensor
    # assert torch.all((mask == 0) | (mask == 1)), "Mask must be binary (0s and 1s)."
    assert isinstance(x, torch.Tensor), "the input is not torch.Tensor"

    if x.shape[0] in [1,3]:  # Assuming 1 or 3 channels implies CHW
        is_chw = True
    else:
        is_chw = False # Guess it is HWC

    if x.shape[1:3] == size or x.shape[0:1] == size:
        pass
    else:
        # Check if it is CHW are HWC
        if not is_chw:
            x = x.permute(2, 0, 1)  # Change shape from (H, W, C) to (C, H, W)

        # Resize the image using bilinear interpolation
        if mode in ['linear','bilinear','bicubic','trilinear']:
            x = F.interpolate(x.unsqueeze(0), size=size, mode=mode, align_corners=False).squeeze(0)
        else:
        # Resize the mask using nearest neighbor interpolation
            x = F.interpolate(x.unsqueeze(0), size=size, mode=mode).squeeze(0)
        if not is_chw:
            x = x.permute(1, 2, 0)  # Change shape back to (H, W, C)
    
    return x

def match_path(pattern, id=0):
    # Construct the pattern to match the normal image
    # Use glob to find the matching files
    normal_image_paths = glob.glob(pattern)

    if not normal_image_paths:
        raise FileNotFoundError(f"No normal images found for {pattern}")

    # Return the first matched path (assuming there's only one relevant normal image)
    return normal_image_paths[id]

class EvalDataset():
    def __init__(self, 
                 data_dir='/media/SSD1/hejun/LavalObjaverseDataset', 
                 pair_info='/media/SSD1/hejun/LavalObjaverseDataset/experimental_pair/16_to_16_16mapping_pairs.json', 
                 black_background=True, 
                 resolution=(512, 512), 
                 seed=180,
                 **kwargs):
        self.data_dir = data_dir
        self.pair_info = pair_info
        self.black_background = black_background
        self.resolution = resolution
        self.seed = seed
        self.object_split = kwargs.pop("object_split", "testing")
        # self.lighting_split = kwargs.pop("lighting_split", "testing")
        # self.view_split = kwargs.pop("view_split", "testing")

        self.background = torch.tensor([0.0, 0.0, 0.0]) if black_background else torch.tensor([1.0, 1.0, 1.0])

        with open(os.path.join(pair_info)) as f:
            self.data_pairs = json.load(f)

    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        data_pair = self.data_pairs[idx]
        object_name = data_pair["object"]

        source_lighting = data_pair["source_lighting"]
        target_lighting = data_pair["target_lighting"]

        views = data_pair["view"]
        crop_ratio = data_pair["crop_ratio"]
        try:
            item = self._fetch_one_pair(object_name, 
                                    source_lighting, target_lighting, 
                                    views, crop_ratio)
            item["idx"] = idx # record data pair!
        except Exception as e:
            print(f"Fetch Error:\n Object: {object_name},\
                \n Lighting Mapping: {target_lighting},\
                \n View Mapping: {views}")
            print(e)
            return self.__getitem__(idx+1)
        
        return item

    def _fetch_one_pair(self, object_name, source_lighting, target_lighting, selected_views, crop_ratio):
        rendered_path = os.path.join(self.data_dir, 
            'rendered', 
            self.object_split,
            object_name) # if training, the object_name include subset_x
        
        view_crop_mapping = {}
        for view, crop in zip(selected_views, crop_ratio):
            view_crop_mapping[view] = crop
        
        def _get_crop_ratio(view_name):
            if view_crop_mapping is not None:
                # Sync mode: Check if we've seen this view before in this pair
                if view_name not in view_crop_mapping:
                    view_crop_mapping[view_name] = torch.empty(1).uniform_(0.4, 1.0).item()
                return view_crop_mapping[view_name]
            else:
                # Arbitrary mode: Always generate a new random ratio
                return torch.empty(1).uniform_(0.4, 1.0).item()
    
        with open(os.path.join(rendered_path, 'info.json')) as f:
            info = json.load(f)
            sensor_size = info['basic']['sensor_size']
            # image_size = info['basic']['image_size']
            focal = info['basic']['focal']
            # lightings = info['basic']["lighting"][self.lighting_split]
            # views = info['basic']["view"][self.view_split]
            fov = 2 * math.atan(sensor_size[0] / (2 * focal))
            # V32&Indoor_AG8A0756-50df1786c6_7_image.png
        
        # follow Multi_View_Dataset
        # Dont change the variable name
        source_lighting_name = source_lighting
        target_lighting_name = target_lighting

        source_view_name = target_view_name = selected_views

        def _fetch_lightings(lighting_name, addition_rotations=None):
            # LavalObjaverseDataset/laval/preprocessed
            path = os.path.join(self.data_dir, 'laval/preprocessed', lighting_name)
            ldr, log, rays, raw = self.read_environment(path, addition_rotations)
            lightings = torch.stack([ldr, log], dim=0) # [2, H, W, 3]
            rays = rays.unsqueeze(1)# [6, 1, H, W]
            return lightings, rays, raw

        def _fetch_images(lighting_name, view_name_list):
            images, masks, depths, Ks = [], [], [], []
            for view_name in view_name_list:
                view_name_without_postfix = view_name.split('.')[0]
                lighting_name_without_slash = lighting_name.replace("/","_").split('.')[0]
                image_file_name = f"{view_name_without_postfix}&{lighting_name_without_slash}"
                image, mask = self.read_masked_image(rendered_path, image_file_name)
                depth = self.read_depth(rendered_path, view_name_without_postfix)
                # 1. cropping the image/mask/depth if self.is_train
                # ratio is randomly select from 0.4 ~ 1
                _, H_orig, W_orig = image.shape

                # 1. 居中裁剪 (Center Crop)
                crop_ratio = _get_crop_ratio(view_name)
                
                H_crop = int(H_orig * crop_ratio)
                W_crop = int(W_orig * crop_ratio)
                
                # 計算居中偏移量
                top = (H_orig - H_crop) // 2
                left = (W_orig - W_crop) // 2
                
                image = image[:, top:top+H_crop, left:left+W_crop]
                mask = mask[:, top:top+H_crop, left:left+W_crop]
                depth = depth[:, top:top+H_crop, left:left+W_crop]

                # 2. resize and append
                image = resize(image.contiguous(), self.resolution)
                mask = resize(mask.contiguous(), self.resolution, mode='nearest')
                depth = resize(depth, self.resolution) 

                # 3. calculate intrincs K via resolution and FOV and cropping ratio
                W, _ = self.resolution
                f_effective = (W / 2.0) / torch.tan(torch.tensor(fov / 2.0)) / crop_ratio
                cx = (W - 1.0) / 2.0
                cy = cx
                K = torch.tensor([
                    [f_effective, 0,           cx],
                    [0,           f_effective, cy],
                    [0,           0,           1]
                ], dtype=torch.float32)

                images.append(image)
                masks.append(mask)
                depths.append(depth)
                Ks.append(K)

            images = torch.stack(images, dim=0)
            masks = torch.stack(masks, dim=0)
            depths = torch.stack(depths, dim=0)
            Ks = torch.stack(Ks, dim=0)
            
            depths = depths * masks

            return images, masks, depths, Ks
        
        def _fetch_view(view_name_list):
            views = []
            # Conversion matrix: Blender (-Z forward) to OpenCV (+Z forward)
            # This flips Y and Z axes to match standard depth projection conventions
            blender_to_cv = torch.tensor([
                [1,  0,  0, 0],
                [0, -1,  0, 0],
                [0,  0, -1, 0],
                [0,  0,  0, 1]
            ], dtype=torch.float32)

            for view in view_name_list:
                for item in info["images"]:
                    if item['view'] == view:
                        c2w = torch.tensor(item['transform'], dtype=torch.float32)
                        c2w = c2w @ blender_to_cv
                        views.append(c2w)
                        break
                        
            return torch.stack(views)

        addition_rotation = torch.eye(3, dtype=torch.float32)
        
        # fetch image
        source_images, source_mask, source_depths, source_Ks = \
            _fetch_images(source_lighting_name, source_view_name)
        target_images, target_mask, target_depths, target_Ks = \
            _fetch_images(target_lighting_name, target_view_name)
        
        # fetch environment map
        source_lighting, lighting_rays, raw_source_lighting = _fetch_lightings(source_lighting_name, addition_rotation)
        target_lighting, _, raw_target_lighting = _fetch_lightings(target_lighting_name, addition_rotation)
        
        # fetch pose (view)
        source_view = _fetch_view(source_view_name)
        target_view = _fetch_view(target_view_name)

        # add rotation to the view
        source_view = apply_rotation_to_views(source_view, addition_rotation)
        target_view = apply_rotation_to_views(target_view, addition_rotation)

        source_rays = camera2ray(source_view, source_Ks, source_mask)
        target_rays = camera2ray(target_view, target_Ks, target_mask)
        # lightings_rays = lightings_ray(target_lighting, source_view)

        return_dict = {
            "source_lighting": raw_source_lighting,
            "target_lighting": raw_target_lighting,

            "source_images": source_images,
            "target_images": target_images,

            "source_rays": source_rays,
            "target_rays": target_rays,
            "lighting_rays": lighting_rays,

            "source_view": source_view,
            "target_view": target_view,

            "source_depths": source_depths,
            "target_depths": target_depths,

            "source_mask": source_mask,
            "target_mask": target_mask,

            "source_Ks": source_Ks,
            "target_Ks": target_Ks,

            "addition_rotation": addition_rotation,
        }

        return return_dict
    
    def read_environment(self, path, addition_rotation=None):
        # follow LuxDiT
        M_ldr = 16
        M_log = 10_000

        raw = read_hdr(path, self.resolution)
        ldr = raw / (1.0 + raw) * (1.0 + raw / M_ldr**2)
        log = np.log(1.0 + raw) / np.log(1.0 + M_log)

        # Convert to tensors and resize
        ldr = torch.from_numpy(ldr).float()
        log = torch.from_numpy(log).float()
        raw = torch.from_numpy(raw).float()

        
        ldr = resize(ldr.permute(2,0,1).contiguous(), self.resolution)
        log = resize(log.permute(2,0,1).contiguous(), self.resolution)
        raw = resize(raw.permute(2,0,1).contiguous(), self.resolution)

        rays = mercator2ray(self.resolution[0], self.resolution[1], addition_rotation).permute(2,0,1).contiguous()

        return ldr, log, rays, raw

    def read_masked_image(self, data_path, name):
        try:
            rgba = kiui.read_image(os.path.join(data_path, f"{name}_image.png"), mode='tensor', order='RGBA')
            if rgba is None:
                raise ValueError("Not Found "+ os.path.join(data_path, f"{name}_image.png"))
        except Exception as e:
            # print(e)
            # print("Not found", os.path.join(data_path, f"{name}_image.png"))
            path = os.path.join(data_path, f"{name}_image.png")
            raise FileNotFoundError(f"Not found {path}")
        rgb = rgba[:,:,:3]
        mask = rgba[:,:,3].unsqueeze(-1)
        image = rgb * mask + self.background * (1.0 - mask)

        image = image.permute(2,0,1).contiguous() # CHW
        mask = mask.permute(2,0,1).contiguous() # CHW

        return image, mask

    def read_depth(self, data_path, pose_name):
        """
        Read depth map from disk, preferring .exr over .png.
        
        Args:
            data_path (str): Directory containing depth files
            pose_name (str): Base name (e.g., "001") to match files like "001_depth_0001.exr"
        
        Returns:
            torch.Tensor: Depth map of shape (1, H, W), dtype float32
                        Values are in **meters** if from EXR (Blender convention)
        """
        # Search for EXR files first
        exr_pattern = os.path.join(data_path, f"{pose_name}_depth_*.exr")
        exr_files = glob.glob(exr_pattern)
        # TODO: Why the background should 1 not 0 ??
        if exr_files:
            # Use EXR (preferred)
            depth_file = sorted(exr_files)[0]  # Alphabetical order
            
            try:
                exr = OpenEXR.InputFile(depth_file)
                header = exr.header()
                dw = header['dataWindow']
                w = dw.max.x - dw.min.x + 1
                h = dw.max.y - dw.min.y + 1
                
                # Blender typically writes depth to the 'R' channel
                # (even though it's a scalar, EXR stores it as RGB/RGBA)
                r_channel = exr.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
                depth_array = np.frombuffer(r_channel, dtype=np.float32)
                depth_array = depth_array.reshape((h, w)).copy()  # ← .copy() makes it writable
                depth = torch.from_numpy(depth_array).float().unsqueeze(0)  # (1, H, W)
            except Exception as e:
                raise RuntimeError(f"Failed to read EXR depth file {depth_file}: {e}")
        else:
            # Fallback to PNG
            png_pattern = os.path.join(data_path, f"{pose_name}_depth_*.png")
            png_files = glob.glob(png_pattern)
            
            if not png_files:
                raise FileNotFoundError(
                    f"No depth files found for '{pose_name}' in '{data_path}'.\n"
                    f"Tried EXR pattern: {exr_pattern}\n"
                    f"Tried PNG pattern: {png_pattern}"
                )
            
            depth_file = sorted(png_files)[0]
            
            try:
                # Assume kiui.read_image returns (H, W, C) tensor in [0, 1] or [0, 255]
                img = kiui.read_image(depth_file, mode='tensor', order='RGB')  # (H, W, 3)
                if img is None:
                    raise ValueError(f"kiui.read_image returned None for {depth_file}")
                
                # Take red channel (convention for depth PNGs)
                if img.ndim == 3:
                    depth = img[:, :, 0:1]  # (H, W, 1)
                else:
                    depth = img.unsqueeze(-1)  # (H, W, 1)
                
                depth = depth.permute(2, 0, 1).contiguous().float()  # (1, H, W)

                depth = depth * DEPTH_SCALE_PNG
                
            except Exception as e:
                raise RuntimeError(f"Failed to read PNG depth file {depth_file}: {e}")

        # Replace NaN and Inf with a large finite value (e.g., 1e10 meters)
        LARGE_DEPTH = 1e6
        depth = torch.nan_to_num(depth, nan=LARGE_DEPTH, posinf=LARGE_DEPTH, neginf=LARGE_DEPTH)
        return depth
    

class SingleView_Train_Dataset():
    def __init__(self, 
                 data_dir, 
                 object_split='training', 
                 lighting_split='training', 
                 view_split='training', 
                 source_view_num=1,
                 target_view_num=1, 
                 black_background=True, 
                 resolution=256, 
                 seed=180,
                 is_train=False,
                 **kwargs):
        super().__init__()
        
        self.data_dir = data_dir
        self.object_split = object_split
        self.lighting_split = lighting_split
        self.view_split = view_split
        self.source_view_num = source_view_num
        self.target_view_num = target_view_num
        self.black_background = black_background
        self.resolution = resolution if isinstance(resolution, tuple) else (resolution, resolution)
        self.seed = seed
        self.is_train = is_train

        self.background = torch.tensor([0.0, 0.0, 0.0]) if black_background else torch.tensor([1.0, 1.0, 1.0])
        self.is_ablation = kwargs.pop("ablation", False)
        # ✅ Fixed: Proper assertion messages
        assert source_view_num >= 0, "source_view_num must be >= 0"
        assert target_view_num >= 0, "target_view_num must be >= 0"
        assert object_split in ['training', 'testing', 'validation'], "split should be one of ['training', 'testing', 'validation']"
        assert lighting_split in ['training', 'testing', 'validation'], "split should be one of ['training', 'testing', 'validation']"
        assert view_split in ['training', 'testing', 'validation'], "split should be one of ['training', 'testing', 'validation']"

        if object_split == "training":
            self.objects = []
            # HDD1/hejun/LavalObjaverseDataset/objaverse/info/training_subsets
            json_files = glob.glob(os.path.join(self.data_dir, "objaverse/info", "training_subsets/subset_*.json"))
            for json_file in json_files:
                subset = os.path.splitext(os.path.basename(json_file))[0]
                with open(json_file, 'r') as f:
                    content = json.load(f)
                    # Convert each item from 'original' to 'subset/original'
                for obj_uid in content:
                    self.objects.append(f"{subset}/{obj_uid}")
        else: 
            object_info_file = os.path.join(self.data_dir, "objaverse/info", f"full_{object_split}_objects.json")
            with open(object_info_file, 'r') as f:
                self.objects = json.load(f)
        self.lighting_index_mapping = list(product(range(0, N_LIGHTINGS), repeat=2))
        # [(0, 0), (0, 1), (0, 2), ..., (4, 3), (4, 4)]
        
        self.novel_view = novel_view = False
        self.same_view = same_view = True

        if (not is_train) and same_view:
            valid_view_num = source_view_num
        else:
            valid_view_num = N_VIEWS
        self.view_index_mapping = generate_view_pairs(views=range(0, valid_view_num), 
                            source_view_num=source_view_num, 
                            target_view_num=target_view_num, 
                            novel_view=novel_view, same_view=same_view)
        
        self._rotation_call_count = 0

    @property
    def length_of_objects(self):
        return len(self.objects)
    
    @property
    def length_of_lighting_mapping(self):
        return len(self.lighting_index_mapping)
    
    @property
    def length_of_view_mapping(self):
        return len(self.view_index_mapping)
    
    def __len__(self):
        return MAX_EXAMPLES

    def __getitem__(self, idx):
        object_index = random.randrange(self.length_of_objects)
        lighting_index_mapping_index = random.randrange(self.length_of_lighting_mapping)
        view_index_mapping_index = random.randrange(self.length_of_view_mapping)
        
        object_name = self.objects[object_index]
        lighting_index_mapping = self.lighting_index_mapping[lighting_index_mapping_index]
        view_index_mapping = self.view_index_mapping[view_index_mapping_index]
        
        try:
            item = self._fetch_one_pair(object_name, 
                                        lighting_index_mapping, 
                                        view_index_mapping)
        except Exception as e:
            # print(f"Fetch Error:\n Object: {object_name},\
            #       \n Lighting Mapping: {lighting_index_mapping},\
            #       \n View Mapping: {view_index_mapping}")
            # print(e)
            return self.__getitem__(random.randint(0, self.__len__()))
            
        return item

    def _fetch_one_pair(self, object_name, lighting_index_mapping, view_index_mapping, **kwargs):
        rendered_path = os.path.join(self.data_dir, 
            'rendered', 
            self.object_split,
            object_name) # if training, the object_name include subset_x
        
        # 1. Create a mapping only if same_view is enabled
        # If same_view is False, we keep it as None to trigger independent random crops
        # view_crop_mapping = {} if self.same_view else None
        # ! Important: When training, even we require same_view, we still want to have different crops for different samples to increase data diversity. 
        # So we should not reuse the same crop ratio for the same view across different samples. Therefore, we
        view_crop_mapping = {}
        def _get_crop_ratio(view_name):
            if not IS_CROPPING_FOR_ARGUMENT:
                return 1.0
            
            # --- Conditional Logic ---
            if view_crop_mapping is not None:
                # Sync mode: Check if we've seen this view before in this pair
                if view_name not in view_crop_mapping:
                    view_crop_mapping[view_name] = torch.empty(1).uniform_(0.4, 1.0).item()
                return view_crop_mapping[view_name]
            else:
                # Arbitrary mode: Always generate a new random ratio
                return torch.empty(1).uniform_(0.4, 1.0).item()
    
        with open(os.path.join(rendered_path, 'info.json')) as f:
            info = json.load(f)
            sensor_size = info['basic']['sensor_size']
            # image_size = info['basic']['image_size']
            focal = info['basic']['focal']
            lightings = info['basic']["lighting"][self.lighting_split]
            views = info['basic']["view"][self.view_split]
            fov = 2 * math.atan(sensor_size[0] / (2 * focal))
            # V32&Indoor_AG8A0756-50df1786c6_7_image.png
        
        source_lighting_index, \
        target_lighting_index = lighting_index_mapping

        source_lighting_name = lightings[source_lighting_index]
        target_lighting_name = lightings[target_lighting_index]

        source_view_index, \
        target_view_index = view_index_mapping

        source_view_name = [views[i] for i in source_view_index]
        target_view_name = [views[i] for i in target_view_index]

        def _fetch_lightings(lighting_name, addition_rotations=None):
            # LavalObjaverseDataset/laval/preprocessed
            path = os.path.join(self.data_dir, 'laval/preprocessed', lighting_name)
            ldr, log, rays = self.read_environment(path, addition_rotations)
            lightings = torch.stack([ldr, log], dim=0) # [2, H, W, 3]
            rays = rays.unsqueeze(1) # [6, 1, H, W]
            return lightings, rays

        def _fetch_images(lighting_name, view_name_list):
            images, masks, depths, Ks = [], [], [], []
            for view_name in view_name_list:
                view_name_without_postfix = view_name.split('.')[0]
                lighting_name_without_slash = lighting_name.replace("/","_").split('.')[0]
                image_file_name = f"{view_name_without_postfix}&{lighting_name_without_slash}"
                image, mask = self.read_masked_image(rendered_path, image_file_name)
                depth = self.read_depth(rendered_path, view_name_without_postfix)
                # 1. cropping the image/mask/depth if self.is_train
                # ratio is randomly select from 0.4 ~ 1
                _, H_orig, W_orig = image.shape

                # 1. 居中裁剪 (Center Crop)
                if IS_CROPPING_FOR_ARGUMENT:
                    # 隨機選擇裁剪比例 0.4 ~ 1.0
                    crop_ratio = _get_crop_ratio(view_name)
                    
                    H_crop = int(H_orig * crop_ratio)
                    W_crop = int(W_orig * crop_ratio)
                    
                    # 計算居中偏移量
                    top = (H_orig - H_crop) // 2
                    left = (W_orig - W_crop) // 2
                    
                    image = image[:, top:top+H_crop, left:left+W_crop]
                    mask = mask[:, top:top+H_crop, left:left+W_crop]
                    depth = depth[:, top:top+H_crop, left:left+W_crop]
                else:
                    crop_ratio = 1.0

                # 2. resize and append
                image = resize(image.contiguous(), self.resolution)
                mask = resize(mask.contiguous(), self.resolution, mode='nearest')
                depth = resize(depth, self.resolution) 

                # 3. calculate intrincs K via resolution and FOV and cropping ratio
                W, _ = self.resolution
                f_effective = (W / 2.0) / torch.tan(torch.tensor(fov / 2.0)) / crop_ratio
                cx = (W - 1.0) / 2.0
                cy = cx
                K = torch.tensor([
                    [f_effective, 0,           cx],
                    [0,           f_effective, cy],
                    [0,           0,           1]
                ], dtype=torch.float32)

                images.append(image)
                masks.append(mask)
                depths.append(depth)
                Ks.append(K)

            images = torch.stack(images, dim=0)
            masks = torch.stack(masks, dim=0)
            depths = torch.stack(depths, dim=0)
            Ks = torch.stack(Ks, dim=0)
            
            depths = depths * masks

            return images, masks, depths, Ks

        def _fetch_view(view_name_list):
            views = []
            # Conversion matrix: Blender (-Z forward) to OpenCV (+Z forward)
            # This flips Y and Z axes to match standard depth projection conventions
            blender_to_cv = torch.tensor([
                [1,  0,  0, 0],
                [0, -1,  0, 0],
                [0,  0, -1, 0],
                [0,  0,  0, 1]
            ], dtype=torch.float32)

            for view in view_name_list:
                for item in info["images"]:
                    if item['view'] == view:
                        c2w = torch.tensor(item['transform'], dtype=torch.float32)
                        c2w = c2w @ blender_to_cv
                        views.append(c2w)
                        break
                        
            return torch.stack(views)

        if IS_ADDITION_ROTATION_FOR_ARGUMENT and self.is_train:
            # addition_rotation_scipy =  # Returns scipy Rotation object
            # Convert to rotation matrix
            addition_rotation = torch.tensor(self.get_random_rotation()[0].as_matrix(), dtype=torch.float32)
        else:
            addition_rotation = torch.eye(3, dtype=torch.float32)
        
        # fetch image
        source_images, source_mask, source_depths, source_Ks = \
            _fetch_images(source_lighting_name, source_view_name)
        target_images, target_mask, target_depths, target_Ks = \
            _fetch_images(target_lighting_name, target_view_name)
        
        # fetch environment map
        source_lighting, lighting_rays = _fetch_lightings(source_lighting_name, addition_rotation)
        target_lighting, _ = _fetch_lightings(target_lighting_name, addition_rotation)
        
        # fetch pose (view)
        source_view = _fetch_view(source_view_name)
        target_view = _fetch_view(target_view_name)

        # add rotation to the view
        source_view = apply_rotation_to_views(source_view, addition_rotation)
        target_view = apply_rotation_to_views(target_view, addition_rotation)

        source_rays = camera2ray(source_view, source_Ks, H=self.resolution[1], W=self.resolution[0])
        target_rays = camera2ray(target_view, target_Ks, H=self.resolution[1], W=self.resolution[0])
        # lightings_rays = lightings_ray(target_lighting, source_view)

        return_dict = {
            "source_lighting": source_lighting,
            "target_lighting": target_lighting,

            "source_images": source_images,
            "target_images": target_images,

            "source_rays": source_rays,
            "target_rays": target_rays,
            "lighting_rays": lighting_rays,

            "source_view": source_view,
            "target_view": target_view,

            "source_depths": source_depths,
            "target_depths": target_depths,

            "source_mask": source_mask,
            "target_mask": target_mask,

            "source_Ks": source_Ks,
            "target_Ks": target_Ks,

            "addition_rotation": addition_rotation,

        }
        return return_dict

    def read_environment(self, path, addition_rotation=None):
        # follow LuxDiT
        M_ldr = 16
        M_log = 10_000

        raw = read_hdr(path, self.resolution)
        ldr = raw / (1.0 + raw) * (1.0 + raw / M_ldr**2)
        log = np.log(1.0 + raw) / np.log(1.0 + M_log)

        # Convert to tensors and resize
        ldr = torch.from_numpy(ldr).float()
        log = torch.from_numpy(log).float()
        
        ldr = resize(ldr.permute(2,0,1).contiguous(), self.resolution)
        log = resize(log.permute(2,0,1).contiguous(), self.resolution)

        rays = mercator2ray(self.resolution[0], self.resolution[1], addition_rotation).permute(2,0,1).contiguous()

        return ldr, log, rays

    def read_masked_image(self, data_path, name):
        try:
            rgba = kiui.read_image(os.path.join(data_path, f"{name}_image.png"), mode='tensor', order='RGBA')
            if rgba is None:
                raise ValueError("Not Found "+ os.path.join(data_path, f"{name}_image.png"))
        except Exception as e:
            # print(e)
            # print("Not found", os.path.join(data_path, f"{name}_image.png"))
            path = os.path.join(data_path, f"{name}_image.png")
            raise FileNotFoundError(f"Not found {path}")
        rgb = rgba[:,:,:3]
        mask = rgba[:,:,3].unsqueeze(-1)
        image = rgb * mask + self.background * (1.0 - mask)

        image = image.permute(2,0,1).contiguous() # CHW
        mask = mask.permute(2,0,1).contiguous() # CHW

        return image, mask

    def read_depth(self, data_path, pose_name):
        """
        Read depth map from disk, preferring .exr over .png.
        
        Args:
            data_path (str): Directory containing depth files
            pose_name (str): Base name (e.g., "001") to match files like "001_depth_0001.exr"
        
        Returns:
            torch.Tensor: Depth map of shape (1, H, W), dtype float32
                        Values are in **meters** if from EXR (Blender convention)
        """
        # Search for EXR files first
        exr_pattern = os.path.join(data_path, f"{pose_name}_depth_*.exr")
        exr_files = glob.glob(exr_pattern)
        # TODO: Why the background should 1 not 0 ??
        if exr_files:
            # Use EXR (preferred)
            depth_file = sorted(exr_files)[0]  # Alphabetical order
            
            try:
                exr = OpenEXR.InputFile(depth_file)
                header = exr.header()
                dw = header['dataWindow']
                w = dw.max.x - dw.min.x + 1
                h = dw.max.y - dw.min.y + 1
                
                # Blender typically writes depth to the 'R' channel
                # (even though it's a scalar, EXR stores it as RGB/RGBA)
                r_channel = exr.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
                depth_array = np.frombuffer(r_channel, dtype=np.float32)
                depth_array = depth_array.reshape((h, w)).copy()  # ← .copy() makes it writable
                depth = torch.from_numpy(depth_array).float().unsqueeze(0)  # (1, H, W)
            except Exception as e:
                raise RuntimeError(f"Failed to read EXR depth file {depth_file}: {e}")
        else:
            # Fallback to PNG
            png_pattern = os.path.join(data_path, f"{pose_name}_depth_*.png")
            png_files = glob.glob(png_pattern)
            
            if not png_files:
                raise FileNotFoundError(
                    f"No depth files found for '{pose_name}' in '{data_path}'.\n"
                    f"Tried EXR pattern: {exr_pattern}\n"
                    f"Tried PNG pattern: {png_pattern}"
                )
            
            depth_file = sorted(png_files)[0]
            
            try:
                # Assume kiui.read_image returns (H, W, C) tensor in [0, 1] or [0, 255]
                img = kiui.read_image(depth_file, mode='tensor', order='RGB')  # (H, W, 3)
                if img is None:
                    raise ValueError(f"kiui.read_image returned None for {depth_file}")
                
                # Take red channel (convention for depth PNGs)
                if img.ndim == 3:
                    depth = img[:, :, 0:1]  # (H, W, 1)
                else:
                    depth = img.unsqueeze(-1)  # (H, W, 1)
                
                depth = depth.permute(2, 0, 1).contiguous().float()  # (1, H, W)

                depth = depth * DEPTH_SCALE_PNG
                
            except Exception as e:
                raise RuntimeError(f"Failed to read PNG depth file {depth_file}: {e}")

        # Replace NaN and Inf with a large finite value (e.g., 1e10 meters)
        LARGE_DEPTH = 1e6
        depth = torch.nan_to_num(depth, nan=LARGE_DEPTH, posinf=LARGE_DEPTH, neginf=LARGE_DEPTH)
        return depth

def generate_view_pairs(views, source_view_num, target_view_num, novel_view=False, same_view=False):
    """Generate source-target view pairs with overlap constraints"""
    from itertools import combinations, product
    
    if novel_view and same_view:
        raise ValueError("Cannot set both novel_view=True and same_view=True")
    
    pairs = []
    
    if novel_view:
        # Disjoint: source and target must not overlap
        for source_combo in combinations(views, source_view_num):
            remaining = [v for v in views if v not in source_combo]
            if len(remaining) >= target_view_num:
                for target_combo in combinations(remaining, target_view_num):
                    pairs.append((source_combo, target_combo))
    elif same_view:
        # Identical: source and target must be the same
        if source_view_num != target_view_num:
            raise ValueError("same_view=True requires source_view_num == target_view_num")
        for combo in combinations(views, source_view_num):
            pairs.append((combo, combo))
    else:
        # Default: no restriction
        for pair in product(combinations(views, source_view_num), 
                           combinations(views, target_view_num)):
            pairs.append(pair)
    
    return pairs