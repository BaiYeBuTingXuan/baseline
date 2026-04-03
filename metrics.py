import torch
import numpy as np
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def resize_5d(tensor, size=(256, 256), mode='bilinear', align_corners=False):
    """
    對 5D Tensor (B, N, C, H, W) 進行 2D 插值縮放
    """
    B, N, C, H, W = tensor.shape
    # 1. Flatten (B, N) -> single batch dim for 2D interpolation
    tensor = tensor.reshape(B * N, C, H, W)
    # 2. Interpolate
    tensor = F.interpolate(tensor, size=size, mode=mode, align_corners=align_corners)
    # 3. Restore original 5D shape
    return tensor.reshape(B, N, C, size[0], size[1])

class MetricCalculator:
    def __init__(self, device, depth_tolerance=0.25):
        self.device = device
        self.depth_tolerance = depth_tolerance  # δ < 1.25 (工业标准)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).to(device).eval()

    @staticmethod
    def _align_scale(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """严格线性缩放 (Scale-Only, 无 Shift, 无 Clamp)
        闭式最优解: s = <pred, gt> / <pred, pred>"""
        scale = (pred * gt).sum() / ((pred * pred).sum() + 1e-8)
        return pred * scale

    def _resize_if_spatial_mismatch(self, pred: torch.Tensor, target: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
        """自动对齐空间尺寸，安全处理 4D/5D、bool/int 类型，严格管控 align_corners"""
        if pred.shape[-2:] == target.shape[-2:]:
            return pred

        target_h, target_w = target.shape[-2], target.shape[-1]
        original_dtype = pred.dtype
        is_5d = pred.ndim == 5

        if is_5d:
            B, S, C, H, W = pred.shape
            pred = pred.reshape(B * S, C, H, W)

        needs_cast = not pred.is_floating_point()
        if needs_cast:
            pred = pred.float()

        # 🔑 nearest 模式绝不传 align_corners
        pred = F.interpolate(pred, size=(target_h, target_w), mode=mode)

        if original_dtype == torch.bool:
            pred = (pred > 0.5).bool()
        elif needs_cast:
            pred = pred.to(original_dtype)

        if is_5d:
            pred = pred.view(B, S, C, target_h, target_w)
        return pred

    def compute_iou(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_bin = (pred > 0.5).float()
        gt_bin = (gt > 0.5).float()
        intersection = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
        union = pred_bin.sum(dim=(1, 2, 3)) + gt_bin.sum(dim=(1, 2, 3)) - intersection
        return (intersection + 1e-8) / (union + 1e-8)

    @torch.no_grad()
    def __call__(self, outputs, labels, mask_pred=None, mask_gt=None, 
                 depth_pred=None, depth_gt=None, average=False):
        outputs = outputs.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).to(self.device).clamp(0.0, 1.0)
        labels = labels.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).to(self.device).clamp(0.0, 1.0)
        if outputs.ndim == 4: outputs = outputs.unsqueeze(1)
        if labels.ndim == 4: labels = labels.unsqueeze(1)

        def _prep(t):
            if t is None: return None
            t = t.to(self.device)
            return t.unsqueeze(1) if t.ndim == 4 else t

        mask_pred = _prep(mask_pred)
        mask_gt = _prep(mask_gt)
        depth_pred = _prep(depth_pred)
        depth_gt = _prep(depth_gt)

        if outputs.shape[-2:] != labels.shape[-2:]:
            outputs = self._resize_if_spatial_mismatch(outputs, labels, mode='bilinear')
        if mask_pred is not None and mask_gt is not None:
            if mask_pred.shape[-2:] != mask_gt.shape[-2:]:
                mask_pred = self._resize_if_spatial_mismatch(mask_pred, mask_gt, mode='nearest')
        if depth_pred is not None and depth_gt is not None:
            if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
                depth_pred = self._resize_if_spatial_mismatch(depth_pred, depth_gt, mode='bilinear')
                if mask_gt is not None and mask_gt.shape[-2:] != depth_gt.shape[-2:]:
                    mask_gt = self._resize_if_spatial_mismatch(mask_gt, depth_gt, mode='nearest')

        B, S, C, H, W = outputs.shape
        outputs_lpips = outputs * 2.0 - 1.0
        labels_lpips = labels * 2.0 - 1.0

        res_data = {k: [] for k in ["psnr", "spsnr", "ssim", "lpips", "depth_acc", "depth_mse", "mask_iou"]}

        for b in range(B):
            batch_metrics = {k: [] for k in res_data.keys()}
            for s in range(S):
                pred_f = outputs[b, s]
                gt_f = labels[b, s]

                # PSNR
                mse = F.mse_loss(pred_f, gt_f)
                batch_metrics["psnr"].append(10 * torch.log10(1.0 / (mse + 1e-8)).item())

                pred_scaled = self._align_scale(pred_f, gt_f)
                mse_s = F.mse_loss(pred_scaled, gt_f)
                batch_metrics["spsnr"].append(10 * torch.log10(1.0 / (mse_s + 1e-8)).item())

                # SSIM & LPIPS
                batch_metrics["ssim"].append(self.ssim(outputs[b, s:s+1], labels[b, s:s+1]).item())
                batch_metrics["lpips"].append(self.lpips(outputs_lpips[b, s:s+1], labels_lpips[b, s:s+1]).item())

                # Mask IoU
                if mask_pred is not None and mask_gt is not None:
                    batch_metrics["mask_iou"].append(
                        self.compute_iou(mask_pred[b, s:s+1], mask_gt[b, s:s+1]).mean().item()
                    )

                # Depth Metrics
                if depth_pred is not None and depth_gt is not None:
                    d_p = depth_pred[b, s]  # (C_d, H, W)
                    d_g = depth_gt[b, s]    # (C_d, H, W)
                    
                    # 🔑 修复：移除 .squeeze()，使用 expand_as 保证形状严格匹配
                    if mask_gt is not None:
                        valid_mask = mask_gt[b, s] > 0.5
                        if valid_mask.shape != d_p.shape:
                            valid_mask = valid_mask.expand_as(d_p)
                    else:
                        valid_mask = torch.ones_like(d_p, dtype=torch.bool, device=d_p.device)
                        
                    d_p_v, d_g_v = d_p[valid_mask], d_g[valid_mask]

                    if d_p_v.numel() > 0:
                        d_p_aligned = self._align_scale(d_p_v, d_g_v)
                        batch_metrics["depth_mse"].append(F.mse_loss(d_p_aligned, d_g_v).item())
                        
                        eps = 1e-8
                        ratio = torch.maximum(d_p_aligned / (d_g_v + eps), d_g_v / (d_p_aligned + eps))
                        acc = (ratio < (1.0 + self.depth_tolerance)).float().mean().item()
                        batch_metrics["depth_acc"].append(acc)
                    else:
                        batch_metrics["depth_mse"].append(None)
                        batch_metrics["depth_acc"].append(None)

            # 单 Batch 内聚合
            for k in res_data.keys():
                if batch_metrics[k]:
                    res_data[k].append(float(np.mean(batch_metrics[k])))
                else:
                    res_data[k].append(None)

        # 全局聚合
        final_results = []
        for k in ["psnr", "spsnr", "ssim", "lpips", "depth_acc", "depth_mse", "mask_iou"]:
            valid_vals = [v for v in res_data[k] if v is not None]
            final_results.append(float(np.mean(valid_vals)) if average and valid_vals else res_data[k])

        return tuple(final_results)
            
def resize_5d(tensor, size=(256, 256), mode='bilinear', align_corners=False):
    B, N, C, H, W = tensor.shape
    # 1. Flatten (B, N) -> single batch dim for 2D interpolation
    tensor = tensor.reshape(B * N, C, H, W)
    # 2. Interpolate
    tensor = F.interpolate(tensor, size=size, mode=mode, align_corners=align_corners)
    # 3. Restore 5D shape
    return tensor.reshape(B, N, C, size[0], size[1])