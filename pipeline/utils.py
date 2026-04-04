import torch
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
    return torch.nn.functional.grid_sample(lighting, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
