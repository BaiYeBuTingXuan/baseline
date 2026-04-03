from pathlib import Path
import torch
import cv2
import numpy as np
import imageio
import os


def sub_files_path(base):
    files = []
    base = Path(base)
    if not base.exists() or not base.is_dir():
        print("Given Path Does NOT Exist or NOT a Directory")
        print(str(base))
    else:
        files = [file for file in base.rglob('*') if file.is_file()]

    return files

def camera2ray(Ts, Ks, masks, device='cpu'):
    """
    T K -> Rays (Plücker Coordinates: m, d)。
    
    Args:
        Ts: [N, 4, 4] 
        Ks: [N, 3, 3]
        masks: [N, 1, H, W]
    """
    N, _, H, W = masks.shape
    device = Ts.device

    # 1. 創建像素坐標網格 (Pixel Grid)
    # y 對應 H, x 對應 W
    y_range = torch.arange(H, dtype=torch.float32, device=device)
    x_range = torch.arange(W, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij') # [H, W]

    # 2. 提取內參參數
    # K 矩陣格式: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    fx = Ks[:, 0, 0].view(N, 1, 1)
    fy = Ks[:, 1, 1].view(N, 1, 1)
    cx = Ks[:, 0, 2].view(N, 1, 1)
    cy = Ks[:, 1, 2].view(N, 1, 1)

    # 3. 將像素坐標轉換為相機坐標系下的方向 (Camera Space)
    # 根據公式: x_cam = (x_pixel - cx) / fx, y_cam = (y_pixel - cy) / fy
    # z 方向在相機空間通常定義為 1
    x_cam = (x_grid.unsqueeze(0) - cx) / fx
    y_cam = (y_grid.unsqueeze(0) - cy) / fy
    z_cam = torch.ones_like(x_cam)

    directions_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1) # [N, H, W, 3]

    # 4. 歸一化方向向量 (Normalize)
    directions_cam = directions_cam / torch.norm(directions_cam, dim=-1, keepdim=True)

    # 5. 轉換到世界坐標系 (World Space)
    R = Ts[:, :3, :3]  # [N, 3, 3]
    t = Ts[:, :3, 3]   # [N, 3] (Camera Origin)

    # 使用矩陣乘法旋轉方向向量: [N, H*W, 3] @ [N, 3, 3]^T
    directions_world = torch.matmul(directions_cam.view(N, -1, 3), R.transpose(-2, -1))
    directions_world = directions_world.view(N, H, W, 3)

    # 6. 計算普呂克坐標之矩 (Plücker Momentum: m = o x d)
    camera_pos = t.view(N, 1, 1, 3).expand(-1, H, W, -1)
    ray_momentum = torch.cross(camera_pos, directions_world, dim=-1)

    # 7. 拼接並調整維度
    # 返回 [N, 6, H, W] -> (mx, my, mz, dx, dy, dz)
    rays = torch.cat([ray_momentum, directions_world], dim=-1)
    return rays.permute(0, 3, 1, 2)


def mercator2ray(H=256, W=256, addition_rotation=None, device='cpu'):
    """
    Generate rays for mercator projection, where rays go from -pi to pi horizontally
    and +pi/2 to -pi/2 vertically.
    - H: height of the image
    - W: width of the image
    """
    # Create coordinate grids for mercator projection
    # Horizontal: -π to π
    x_coords = torch.linspace(-torch.pi, torch.pi, W, device=device)
    # Vertical: π/2 to -π/2 (top to bottom)
    y_coords = torch.linspace(torch.pi/2, -torch.pi/2, H, device=device)
    
    # Create meshgrid
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H, W]
    
    # Convert spherical coordinates to Cartesian coordinates
    # x = cos(lat) * cos(lon)
    # y = cos(lat) * sin(lon) 
    # z = sin(lat)
    cos_lat = torch.cos(y_grid)
    sin_lat = torch.sin(y_grid)
    cos_lon = torch.cos(x_grid)
    sin_lon = torch.sin(x_grid)
    
    # Calculate ray directions
    x_dir = cos_lat * cos_lon  # [H, W]
    y_dir = cos_lat * sin_lon  # [H, W]
    z_dir = sin_lat           # [H, W]
    
    # Stack to get ray directions
    ray_dirs = torch.stack([x_dir, y_dir, z_dir], dim=-1)  # [H, W, 3]
    
    # Apply additional rotation if provided
    if addition_rotation is not None:
        # ray_dirs: [H, W, 3] -> [H*W, 3]
        H_orig, W_orig, _ = ray_dirs.shape
        ray_dirs_flat = ray_dirs.view(-1, 3)  # [H*W, 3]
        
        # Apply rotation: [H*W, 3] @ [3, 3] -> [H*W, 3]
        ray_dirs_rotated_flat = torch.matmul(ray_dirs_flat, addition_rotation.T)  # [H*W, 3]
        
        # Reshape back to original shape
        ray_dirs = ray_dirs_rotated_flat.view(H_orig, W_orig, 3)  # [H, W, 3]
    
        # All Ray coming from the finite to the center
    ray_pos = torch.zeros_like(ray_dirs)
    rays = torch.cat([ray_dirs, ray_pos], dim=-1)
    return rays  # [H, W, 6]

def apply_rotation_to_views(views, rotation_matrix):
    """
    Apply a rotation matrix to a batch of camera view matrices.
    Rotates both the rotation (R) and translation (T) components.

    Args:
        views: torch.Tensor of shape (N, 4, 4) - batch of camera pose matrices
        rotation_matrix: torch.Tensor of shape (3, 3) - rotation matrix to apply

    Returns:
        torch.Tensor of shape (N, 4, 4) - rotated view matrices
        [R_add | 0 ][R_original | t_original] = [R_add*R_original | R_add*t_original]
        [   0  | 1 ][   0       |     1     ] = [   0             |           1     ]
        
    """
    if rotation_matrix is None:
        return views
    
    # Extract rotation part (N, 3, 3) and translation part (N, 3, 1)
    R_original = views[:, :3, :3]  # (N, 3, 3)
    t_original = views[:, :3, 3:4]  # (N, 3, 1)
    
    # Apply the rotation to the original rotation: R_new = R_rotation @ R_original
    R_rotated = torch.matmul(rotation_matrix.unsqueeze(0), R_original)  # (1, 3, 3) @ (N, 3, 3) -> (N, 3, 3)
    
    # Apply the rotation to the original translation: t_new = R_rotation @ t_original
    t_rotated = torch.matmul(rotation_matrix.unsqueeze(0), t_original)  # (1, 3, 3) @ (N, 3, 1) -> (N, 3, 1)
    
    # Reconstruct the rotated pose matrix
    views_rotated = torch.cat([
        torch.cat([R_rotated, t_rotated], dim=2),  # (N, 3, 4)
        views[:, 3:, :]  # (N, 1, 4) - keep the last row [0, 0, 0, 1]
    ], dim=1)  # (N, 4, 4)
    
    return views_rotated

def view_normalize(views, base_view):
    """
    Normalize views relative to a base view by multiplying with the transpose of the base view.
    - views: tensor of shape [N, 4, 4] representing camera extrinsic matrices
    - base_view: tensor of shape [4, 4] representing the base camera extrinsic matrix
    """
    # TODO: this one do not concern the translation
    # Get the rotation part of base_view (3x3) and transpose it
    base_rotation = base_view[:3, :3]  # [3, 3]
    base_rotation_T = base_rotation.transpose(-2, -1)  # [3, 3]
    
    # Get the rotation part of views (3x3)
    views_rotation = views[:, :3, :3]  # [N, 3, 3]
    
    # Multiply views by base_view^T
    normalized_rotation = torch.matmul(base_rotation_T, views_rotation)  # [N, 3, 3]
    
    # Create normalized views by replacing the rotation part
    normalized_views = views.clone()  # [N, 4, 4]
    normalized_views[:, :3, :3] = normalized_rotation
    
    return normalized_views

def read_hdr(path, size):
    """
    Reads an HDR map from disk (.hdr or .exr).
    Prioritizes OpenCV/imageio for stability, falling back to OpenEXR only if necessary.
    """
    path = str(path) # Ensure it's a string for library compatibility
    if not os.path.exists(path):
        raise FileNotFoundError(f"HDR file not found: {path}")

    img = None
    
    # --- Method 1: OpenCV (Fastest & Most Stable for DataLoaders) ---
    # Note: Ensure 'OPENCV_IO_ENABLE_OPENEXR=1' is set in your bashrc or env
    try:
        # IMREAD_ANYCOLOR | IMREAD_ANYDEPTH is vital for HDR/EXR float values
        img_cv2 = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img_cv2 is not None:
            # OpenCV loads as BGR; convert to RGB
            if len(img_cv2.shape) == 3 and img_cv2.shape[2] >= 3:
                img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            else:
                img = img_cv2
    except Exception:
        pass

    # --- Method 2: imageio (Excellent fallback for EXR/HDR) ---
    if img is None:
        try:
            import imageio
            # imageio v3+ handles EXR well via the freeimage or pyexr plugin
            img = imageio.imread(path)
        except Exception:
            pass

    # --- Method 3: OpenEXR (Last resort, handled carefully) ---
    if img is None and path.lower().endswith('.exr'):
        try:
            import OpenEXR
            import Imath
            exr_file = OpenEXR.InputFile(path)
            header = exr_file.header()
            dw = header['dataWindow']
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1
            
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = ['R', 'G', 'B']
            # Only read channels that actually exist in the file
            available = list(header['channels'].keys())
            to_read = [c for c in channels if c in available]
            
            channel_data = exr_file.channels(to_read, pt)
            exr_file.close() # CRITICAL: Close immediately to prevent segfaults
            
            decoded = [np.frombuffer(c, dtype=np.float32).reshape(h, w) for c in channel_data]
            img = np.stack(decoded, axis=-1)
        except Exception as e:
            raise RuntimeError(f"All HDR load methods failed for {path}. Error: {e}")

    if img is None:
        raise RuntimeError(f"Failed to load HDR image at {path}")

    # --- Post-processing ---
    img = img.astype(np.float32)

    # Standardize to 3 channels (RGB)
    if img.ndim == 2: # Gray
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1: # Gray with channel dim
        img = np.tile(img, (1, 1, 3))
    elif img.shape[-1] > 3: # Remove Alpha
        img = img[:, :, :3]

    # Resize
    if size is not None:
        # cv2.resize expects (width, height)
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

    return img

# def read_hdr(path, size):
#     """
#     Reads an HDR map from disk (.hdr or .exr).
#     Prioritizes robust EXR reading with OpenEXR.Imath, then imageio, then OpenCV.
#     """
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"HDR file not found: {path}")

#     ext = os.path.splitext(path)[1].lower()
#     img = None
#     method_used = ""

#     # --- Attempt 1: Use Official OpenEXR Library for .exr files ---
#     if ext == '.exr':
#         try:
#             import OpenEXR
#             import Imath
            
#             exr_file = OpenEXR.InputFile(path)
#             header = exr_file.header()
#             dw = header['dataWindow']
#             sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

#             # Determine number of channels
#             ch_list = list(header['channels'].keys())
            
#             # Read channels based on what's available
#             if 'R' in ch_list and 'G' in ch_list and 'B' in ch_list:
#                 # Standard RGB
#                 pt = Imath.PixelType(Imath.PixelType.FLOAT)
#                 rgb_data = exr_file.channels(['R', 'G', 'B'], pt)
#                 r = np.frombuffer(rgb_data[0], dtype=np.float32).reshape((sz[1], sz[0]))
#                 g = np.frombuffer(rgb_data[1], dtype=np.float32).reshape((sz[1], sz[0]))
#                 b = np.frombuffer(rgb_data[2], dtype=np.float32).reshape((sz[1], sz[0]))
#                 img = np.stack([r, g, b], axis=2) # Shape: (H, W, 3)
#             elif len(ch_list) >= 3:
#                  # Assume first 3 channels are R, G, B if standard names aren't found
#                  pt = Imath.PixelType(Imath.PixelType.FLOAT)
#                  first_three_ch = ch_list[:3]
#                  channel_data = exr_file.channels(first_three_ch, pt)
#                  ch_arrays = [
#                      np.frombuffer(data, dtype=np.float32).reshape((sz[1], sz[0]))
#                      for data in channel_data
#                  ]
#                  img = np.stack(ch_arrays, axis=2) # Shape: (H, W, 3 elif len(ch_list) == 1:
#                  # Grayscale
#                  pt = Imath.PixelType(Imath.PixelType.FLOAT)
#                  gray_data = exr_file.channel(ch_list[0], pt)
#                  gray = np.frombuffer(gray_data, dtype=np.float32).reshape((sz[1], sz[0]))
#                  img = np.stack([gray, gray, gray], axis=2) # Shape: (H, W, 3)
#             else:
#                  raise RuntimeError(f"Unsupported channel configuration in {path}: {ch_list}")

#             method_used = "OpenEXR"

#         except ImportError:
#             print(f"OpenEXR library not found. Cannot read {path} with the primary method. Attempting fallbacks...")
#             pass # Continue to fallbacks
#         except Exception as e_openexr:
#              print(f"OpenEXR failed to read {path}: {e_openexr}. Attempting fallbacks...")
#              pass # Continue to fallbacks

#     # --- Fallback 2: Use imageio for .hdr, .exr, or any other supported format ---
#     if img is None:
#         try:
#             import imageio
#             img = imageio.imread(path)
#             img = img.astype(np.float32)
#             method_used = "imageio"
#         except ImportError:
#             print(f"imageio library not found. Cannot read {path} with the imageio fallback. Attempting OpenCV...")
#             pass # Continue to fallbacks
#         except Exception as e_imageio:
#              print(f"imageio failed to read {path}: {e_imageio}. Attempting OpenCV fallback...")
#              pass # Continue to fallbacks

#     # --- Fallback 3: Use OpenCV (requires OPENCV_IO_ENABLE_OPENEXR to be set externally) ---
#     if img is None:
#         try:
#             # The crucial check: the environment variable MUST be set *before* cv2 is imported.
#             # This check is just for info; the actual enabling happens at the system/env level.
#             if not os.getenv("OPENCV_IO_ENABLE_OPENEXR"):
#                  print("WARNING: OPENCV_IO_ENABLE_OPENEXR environment variable is not set. OpenCV EX fail.")
#                  raise RuntimeError("OPENCV_IO_ENABLE_OPENEXR not set.")

#             # Attempt OpenCV read
#             img_cv2 = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

#             if img_cv2 is None:
#                 raise RuntimeError(f"OpenCV failed to load the image: {path}")

#             # OpenCV loads as BGR by default for multi-channel
#             if img_cv2.ndim == 3 and img_cv2.shape[2] >= 3:
#                 img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

#             img = img_cv2.astype(np.float32) # Ensure float32
#             method_used = "OpenCV"

#         except Exception as e_opencv:
#              error_msg = (
#                  f"All methods failed to read HDR/EXR file {path}:\n"
#                  f"- OpenEXR (primary): {'Used' if 'OpenEXR' == method_used else 'Skipped/Tried'}\n"
#                  f"fallback 1): {'Used' if 'imageio' == method_used else 'Skipped/Tried'}\n"
#                  f"- OpenCV (fallback 2): Failed ({e_opencv})\n"
#                  f"Check file integrity and ensure 'OPENCV_IO_ENABLE_OPENEXR=1' is set in the environment *before* running the script if relying on OpenCV."
#              )
#              print(error_msg)
#              raise RuntimeError(error_msg)


#     # --- Post-processing (same for all methods) ---
#     assert img is not None # Should be guaranteed by now

#     # Handle grayscale or >3 channels if needed after loading
#     if img.ndim == 2:
#         img = np.stack([img, img, img], axis=-1)
#     elif img.shape[-1] == 1:
#         img = np.concatenate([img, img, img], axis=-1)
#     elif img.shape[-1] > 3:
#         img = img[:, :, :3]

#     # Resize using OpenCV
#     img_resized = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

#     return img_resized
