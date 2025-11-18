"""
Training script for ConvNeXt Reference Detection Model

Includes:
- Dataset loader for reference detection
- Loss functions (focal loss for objectness, IoU loss for bbox)
- Training loop with validation
- Metrics computation
- Checkpointing and logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import math
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from model import build_convnext_refdet


# ============================================================================
# Checkpoint Management
# ============================================================================

def load_checkpoint(checkpoint_path, model):
    """
    Load model weights from checkpoint.
    
    Args:
        checkpoint_path: path to checkpoint file
        model: model to load state into
    
    Returns:
        dict: checkpoint info (epoch, stats, config)
    """
    print(f"Loading model weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Detect checkpoint precision by checking first parameter
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # Handle weights-only checkpoints
        state_dict = checkpoint
    
    # Get first parameter to check precision
    first_param_key = next(iter(state_dict))
    first_param = state_dict[first_param_key]
    checkpoint_precision = "FP16" if first_param.dtype == torch.float16 else "FP32"
    print(f"Checkpoint precision: {checkpoint_precision}")
    
    # Load weights (PyTorch handles precision conversion automatically)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Handle weights-only checkpoints
        model.load_state_dict(checkpoint)
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'train_stats': checkpoint.get('train_stats', {}),
        'val_stats': checkpoint.get('val_stats', {}),
        'config': checkpoint.get('config', {})
    }
    
    print(f"Loaded weights from epoch {info['epoch']}")
    if 'val_stats' in checkpoint and 'loss' in checkpoint['val_stats']:
        print(f"Previous val loss: {checkpoint['val_stats']['loss']:.4f}")
    
    return info


def save_weights_with_cleanup(model, checkpoint_dir, epoch, is_best=False, save_interval=1):
    """
    Save weights only with automatic cleanup and save_interval support.
    
    Args:
        model: model instance for weights saving
        checkpoint_dir: directory to save checkpoints
        epoch: current epoch number
        is_best: whether this is the best model so far
        save_interval: save checkpoint every N epochs
    
    Returns:
        tuple: (weights_path if saved else None, best_path if is_best else None)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    weights_path = None
    best_path = None
    
    # Save regular checkpoint only every save_interval epochs
    if epoch % save_interval == 0:
        weights_name = f'weights_{epoch}e.pth'
        weights_path = checkpoint_dir / weights_name
        
        # Remove if exists before saving
        if weights_path.exists():
            weights_path.unlink()
            print(f"Removed existing weights: {weights_path}")
        
        # Convert to FP16 before saving for optimized inference
        model_fp16 = model.half()
        torch.save(model_fp16.state_dict(), weights_path)
        print(f"Saved FP16 weights: {weights_path}")
        
        # Convert back to original precision for continued training
        model = model.float()
        
        # Clean up previous checkpoint (epoch - save_interval)
        if epoch > save_interval:
            prev_weights_name = f'weights_{epoch - save_interval}e.pth'
            prev_weights_path = checkpoint_dir / prev_weights_name
            if prev_weights_path.exists():
                prev_weights_path.unlink()
                print(f"Removed previous weights: {prev_weights_path}")
    
    # Always save best model (regardless of save_interval)
    if is_best:
        best_path = checkpoint_dir / 'best_weights.pth'
        
        # Remove if exists before saving
        if best_path.exists():
            best_path.unlink()
            print(f"Removed existing best weights: {best_path}")
        
        # Convert to FP16 before saving for optimized inference
        model_fp16 = model.half()
        torch.save(model_fp16.state_dict(), best_path)
        print(f"Saved best FP16 weights: {best_path}")
        
        # Convert back to original precision for continued training
        model = model.float()
    
    return weights_path, best_path


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the directory.
    
    Args:
        checkpoint_dir: directory to search
    
    Returns:
        str or None: path to latest checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    # Look for last_Ne.pth files
    last_checkpoints = list(checkpoint_dir.glob('last_*e.pth'))
    if not last_checkpoints:
        return None
    
    # Extract epoch numbers and find the latest
    latest_epoch = 0
    latest_checkpoint = None
    
    for ckpt_path in last_checkpoints:
        # Extract epoch number from filename like 'last_15e.pth'
        try:
            epoch_str = ckpt_path.stem.split('_')[1].rstrip('e')
            epoch_num = int(epoch_str)
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_checkpoint = str(ckpt_path)
        except (ValueError, IndexError):
            continue
    
    return latest_checkpoint


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred_boxes, target_boxes):
    """
    Compute IoU between predicted and target boxes.
    
    Args:
        pred_boxes: [N, 4] tensor in (cx, cy, w, h) format
        target_boxes: [N, 4] tensor in (cx, cy, w, h) format
    
    Returns:
        ious: [N] tensor of IoU values
    """
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return torch.tensor([], device=pred_boxes.device if len(pred_boxes) > 0 else target_boxes.device)
    
    # Convert to (x1, y1, x2, y2) format
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    
    target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
    
    # Intersection area
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union area
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area
    
    # IoU
    ious = inter_area / (union_area + 1e-7)
    return ious


# ============================================================================
# Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    where p_t is the model's estimated probability for the class.
    FP16-safe version: computes in float32 and clamps pt to avoid underflow.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps  # FP16-safe epsilon
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N] - logits
            targets: [N] - binary labels {0, 1}
        Returns:
            loss: scalar
        """
        # Always compute loss in float32 to avoid FP16 underflow
        dtype_orig = inputs.dtype
        inputs = inputs.float()
        targets = targets.float()
        
        # Compute probability
        p = torch.sigmoid(inputs)
        
        # Compute p_t: p for positive class, (1-p) for negative class
        pt = p * targets + (1 - p) * (1 - targets)
        
        # Clamp pt to avoid log(0) which causes -inf in FP16
        pt = pt.clamp(min=self.eps, max=1.0 - self.eps)
        
        # Compute alpha_t: alpha for positive, (1-alpha) for negative
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal loss formula: -alpha_t * (1-pt)^gamma * log(pt)
        focal_term = (1 - pt) ** self.gamma
        loss = -alpha_t * focal_term * torch.log(pt)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        # Return in original dtype (usually float16) but with stable values
        return loss.to(dtype_orig)


class IoULoss(nn.Module):
    """IoU-based loss for bounding box regression.
    
    Computes 1 - IoU between predicted and target boxes.
    Can use IoU, GIoU, DIoU, or CIoU variants.
    
    - IoU: Standard Intersection over Union
    - GIoU: Generalized IoU (adds penalty for non-overlapping area)
    - DIoU: Distance IoU (considers center point distance)
    - CIoU: Complete IoU (adds aspect ratio consistency) - best for small objects
    """
    def __init__(self, loss_type='giou', reduction='mean'):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] - (cx_offset, cy_offset, w, h) format in pixels
            target_boxes: [N, 4] - (cx_offset, cy_offset, w, h) format in pixels
        Returns:
            loss: scalar
            
        Note: Uses offset coordinates (cx_offset, cy_offset) instead of absolute (cx, cy).
        IoU is translation invariant, so this works correctly.
        """
        # Convert to (x1, y1, x2, y2) format
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        
        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
        
        # Intersection area
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union area
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-7)
        
        if self.loss_type == 'iou':
            loss = 1 - iou
        elif self.loss_type == 'giou':
            # GIoU: add penalty for non-overlapping bounding box area
            enclosing_x1 = torch.min(pred_x1, target_x1)
            enclosing_y1 = torch.min(pred_y1, target_y1)
            enclosing_x2 = torch.max(pred_x2, target_x2)
            enclosing_y2 = torch.max(pred_y2, target_y2)
            enclosing_area = (enclosing_x2 - enclosing_x1) * (enclosing_y2 - enclosing_y1)
            
            giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-7)
            loss = 1 - giou
        elif self.loss_type == 'diou':
            # DIoU: Distance IoU - considers center point distance
            # Better for small objects than GIoU
            # Get enclosing box
            enclosing_x1 = torch.min(pred_x1, target_x1)
            enclosing_y1 = torch.min(pred_y1, target_y1)
            enclosing_x2 = torch.max(pred_x2, target_x2)
            enclosing_y2 = torch.max(pred_y2, target_y2)
            
            # Diagonal length of enclosing box
            c_squared = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2
            
            # Center point distance
            center_dist_squared = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
                                  (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2
            
            # DIoU = IoU - (center_distance^2 / diagonal^2)
            diou = iou - center_dist_squared / (c_squared + 1e-7)
            loss = 1 - diou
        elif self.loss_type == 'ciou':
            # CIoU: Complete IoU - adds aspect ratio consistency
            # Best for small objects and tight bbox regression
            # Get enclosing box
            enclosing_x1 = torch.min(pred_x1, target_x1)
            enclosing_y1 = torch.min(pred_y1, target_y1)
            enclosing_x2 = torch.max(pred_x2, target_x2)
            enclosing_y2 = torch.max(pred_y2, target_y2)
            
            # Diagonal length of enclosing box
            c_squared = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2
            
            # Center point distance
            center_dist_squared = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
                                  (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2
            
            # Aspect ratio consistency
            with torch.no_grad():
                arctan = torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + 1e-7)) - \
                         torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + 1e-7))
                v = (4 / (math.pi ** 2)) * (arctan ** 2)
                alpha = v / (1 - iou + v + 1e-7)
            
            # CIoU = IoU - (center_distance^2 / diagonal^2) - alpha * v
            ciou = iou - center_dist_squared / (c_squared + 1e-7) - alpha * v
            loss = 1 - ciou
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class RefDetLoss(nn.Module):
    """Combined loss for reference detection.
    
    Total loss = lambda_obj * focal_loss + lambda_bbox * iou_loss
    """
    def __init__(self, lambda_obj=1.0, lambda_bbox=5.0, 
                 focal_alpha=0.25, focal_gamma=2.0,
                 iou_type='giou', pos_radius=0):
        super().__init__()
        self.lambda_obj = lambda_obj
        self.lambda_bbox = lambda_bbox
        self.pos_radius = pos_radius  # Multi-cell positive assignment
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.iou_loss = IoULoss(loss_type=iou_type)
    
    def forward(self, obj_map, bbox_map, targets, stride=4):
        """
        Args:
            obj_map: [B, 1, H, W] - predicted objectness logits
            bbox_map: [B, 4, H, W] - predicted bboxes (tx, ty, tw, th logits)
            targets: List of dicts, each with 'bbox' [x1, y1, x2, y2] and 'label'
            stride: feature map stride
        
        Returns:
            loss: scalar
            loss_dict: dict with individual loss components
        """
        device = obj_map.device
        B, _, H, W = obj_map.shape
        
        # Prepare target maps
        obj_target = torch.zeros_like(obj_map)  # [B, 1, H, W]
        bbox_target = torch.zeros_like(bbox_map)  # [B, 4, H, W]
        pos_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
        
        for b, target in enumerate(targets):
            # Check for None or missing bbox
            if target is None or 'bbox' not in target or target['bbox'] is None:
                continue
            
            bbox = target['bbox']  # [x1, y1, x2, y2]
            
            # Convert to center format
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            # Find corresponding grid cell
            grid_x = int(cx / stride)
            grid_y = int(cy / stride)
            
            # Clip to valid range
            grid_x = max(0, min(W - 1, grid_x))
            grid_y = max(0, min(H - 1, grid_y))
            
            # Multi-cell positive assignment for denser supervision
            # Each cell learns its own offset for better consistency with inference decode
            for dy in range(-self.pos_radius, self.pos_radius + 1):
                for dx in range(-self.pos_radius, self.pos_radius + 1):
                    gy = grid_y + dy
                    gx = grid_x + dx
                    
                    # Check bounds
                    if 0 <= gy < H and 0 <= gx < W:
                        # Local grid center for THIS cell
                        grid_center_x = (gx + 0.5) * stride
                        grid_center_y = (gy + 0.5) * stride

                        # Offsets from THIS cell center (in pixels)
                        cx_offset = cx - grid_center_x
                        cy_offset = cy - grid_center_y

                        # Mark as positive
                        obj_target[b, 0, gy, gx] = 1.0
                        pos_mask[b, gy, gx] = True

                        # Store bbox targets in pixel space (cx_offset, cy_offset, w, h)
                        # Each cell gets its own correct offset
                        bbox_target[b, 0, gy, gx] = cx_offset
                        bbox_target[b, 1, gy, gx] = cy_offset
                        bbox_target[b, 2, gy, gx] = w
                        bbox_target[b, 3, gy, gx] = h
        
        # Objectness loss (focal loss on all locations)
        obj_map_flat = obj_map.view(-1)
        obj_target_flat = obj_target.view(-1)
        loss_obj = self.focal_loss(obj_map_flat, obj_target_flat)
        
        # BBox loss (only on positive locations)
        if pos_mask.sum() > 0:
            # Extract predictions and targets at positive locations
            pos_indices = pos_mask.view(B, -1)  # [B, H*W]
            bbox_map_flat = bbox_map.view(B, 4, -1)  # [B, 4, H*W]
            bbox_target_flat = bbox_target.view(B, 4, -1)  # [B, 4, H*W]
            
            # Gather positive predictions and targets
            pred_bboxes = []
            target_bboxes = []
            for b in range(B):
                if pos_indices[b].sum() > 0:
                    pred_bboxes.append(bbox_map_flat[b, :, pos_indices[b]].T)  # [N_pos, 4]
                    target_bboxes.append(bbox_target_flat[b, :, pos_indices[b]].T)  # [N_pos, 4]
            
            if len(pred_bboxes) > 0:
                pred_bboxes = torch.cat(pred_bboxes, dim=0)  # [total_pos, 4]
                target_bboxes = torch.cat(target_bboxes, dim=0)  # [total_pos, 4]
                
                # Decode predictions from logit space (tx, ty, tw, th) to pixel space
                # Using model's decode_bbox_logits helper for consistency
                # Note: We can't access model here, so inline the same logic
                pred_tx, pred_ty, pred_tw, pred_th = pred_bboxes[:, 0], pred_bboxes[:, 1], pred_bboxes[:, 2], pred_bboxes[:, 3]
                pred_cx_offset = torch.tanh(pred_tx) * stride
                pred_cy_offset = torch.tanh(pred_ty) * stride
                pred_w = F.softplus(pred_tw)
                pred_h = F.softplus(pred_th)
                
                # Stack predictions: (cx_offset, cy_offset, w, h) in pixels
                pred_decoded = torch.stack([pred_cx_offset, pred_cy_offset, pred_w, pred_h], dim=1)
                
                # Targets are already in pixel space (cx_offset, cy_offset, w, h)
                # No decoding needed!
                
                # Use IoU loss: compare predicted vs target in pixel space
                loss_bbox = self.iou_loss(pred_decoded, target_bboxes)
            else:
                loss_bbox = torch.tensor(0.0, device=device)
        else:
            loss_bbox = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = self.lambda_obj * loss_obj + self.lambda_bbox * loss_bbox
        
        loss_dict = {
            'loss': total_loss.item(),
            'loss_obj': loss_obj.item(),
            'loss_bbox': loss_bbox.item(),
            'num_pos': pos_mask.sum().item()
        }
        
        return total_loss, loss_dict


# ============================================================================
# Dataset
# ============================================================================

class ReferenceDetectionDataset(Dataset):
    """Dataset for reference-based detection.
    
    Expected data structure:
    data_root/
        templates/
            {object_id}/
                template_0.jpg
                template_1.jpg
                template_2.jpg
        search/
            images/
                {object_id}_frame_{frame_id}.jpg
            labels/
                {object_id}_frame_{frame_id}.txt  # Contains: x1 y1 x2 y2
    """
    def __init__(self, 
                 data_root,
                 split='train',
                 search_size=(1024, 576),
                 template_size=(256, 256),
                 augmentation=True):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.search_size = search_size
        self.template_size = template_size
        self.augmentation = augmentation
        
        # Build sample list
        self.samples = self._build_samples()
        print("[Dataset] Using pixel bbox mode")
        
        # Augmentations - SOTA tracking augmentation (SiamRPN++ + STARK)
        if augmentation and split == 'train':
            # Search transform: Strong augmentation for robustness
            self.search_transform = A.Compose([
                # 1. Shift + Scale jitter (most critical for tracking)
                A.Affine(
                    translate_percent={'x': (-0.10, 0.10), 'y': (-0.10, 0.10)},  # ±10% image shift
                    scale=(0.8, 1.2),      # ±20% scale jitter (SiamRPN++ standard)
                    rotate=0,              # No rotation (tracking doesn't need it)
                    p=0.7,
                ),
                
                # 2. Color jitter (SiamRPN++ / STARK standard)
                A.ColorJitter(
                    brightness=0.3,        # Camera exposure variation
                    contrast=0.3,          # Lighting condition changes
                    saturation=0.3,        # Color saturation shifts
                    hue=0.05,              # Light hue shifts (conservative)
                    p=0.5
                ),
                
                # 3. Blur + Noise (real-world camera effects)
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5)),   # Out-of-focus
                    A.MotionBlur(blur_limit=5),          # Camera/object motion
                ], p=0.3),
                
                # 4. Compression artifacts (real camera compression)
                A.ImageCompression(quality_range=(60, 100), p=0.2),
                
                # 5. Resize to fixed input size
                A.Resize(height=search_size[1], width=search_size[0]),
                
                # 6. Normalize (ImageNet pretrained)
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc', 
                label_fields=['labels'], 
                min_visibility=0.3  # Keep bbox if 30% visible after augmentation
            ))
            
            # Template transform: NO augmentation for safety and accuracy
            # Templates should be clean references for matching
            self.template_transform = A.Compose([
                A.Resize(height=template_size[1], width=template_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # Validation: No augmentation for both search and template
            self.search_transform = A.Compose([
                A.Resize(height=search_size[1], width=search_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))
            
            self.template_transform = A.Compose([
                A.Resize(height=template_size[1], width=template_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _build_samples(self):
        """Build list of samples from data directory."""
        samples = []
        
        search_dir = self.data_root / self.split / 'search'
        label_dir = search_dir / 'labels'
        image_dir = search_dir / 'images'
        
        if not label_dir.exists():
            raise ValueError(f"Label directory not found: {label_dir}")
        
        # Iterate through label files
        for label_file in sorted(label_dir.glob('*.txt')):
            image_file = image_dir / label_file.name.replace('.txt', '.jpg')
            if not image_file.exists():
                image_file = image_dir / label_file.name.replace('.txt', '.png')
            
            if not image_file.exists():
                continue
            
            # Extract object_id from filename
            # Assuming format: {object_id}_frame_{frame_id}.txt
            parts = label_file.stem.split('_frame_')
            if len(parts) == 2:
                object_id = parts[0]
            else:
                object_id = label_file.stem
            
            samples.append({
                'object_id': object_id,
                'search_image': str(image_file),
                'label_file': str(label_file)
            })
        
        print(f"Found {len(samples)} samples for {self.split} split")
        return samples
    
    def _load_templates(self, object_id):
        """Load 3 template images for the object.
        
        Supports two directory structures:
        1. Flat: templates/{object_id}_ref_001.jpg, {object_id}_ref_002.jpg, ...
        2. Nested: templates/{object_id}/template_0.jpg, template_1.jpg, ...
        """
        templates_root = self.data_root / self.split / 'templates'
        
        # Try flat structure first (pattern: {object_id}_ref_*.jpg)
        template_files = sorted(templates_root.glob(f'{object_id}_ref_*.jpg')) + \
                        sorted(templates_root.glob(f'{object_id}_ref_*.png'))
        
        # If no files found, try nested directory structure
        if len(template_files) == 0:
            template_dir = templates_root / object_id
            if template_dir.exists():
                template_files = sorted(template_dir.glob('*.jpg')) + \
                               sorted(template_dir.glob('*.png'))
        
        # If still no templates found, raise error
        if len(template_files) == 0:
            raise ValueError(f"No template files found for {object_id} in {templates_root}")
        
        # Load up to 3 templates
        templates = []
        for i in range(min(3, len(template_files))):
            img = Image.open(template_files[i]).convert('RGB')
            img = np.array(img)
            transformed = self.template_transform(image=img)
            templates.append(transformed['image'])
        
        # If less than 3 templates, repeat the last one
        while len(templates) < 3:
            templates.append(templates[-1].clone())
        
        return templates
    
    def _load_bbox(self, label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            return None

        parts = lines[0].strip().split()
        if len(parts) < 5:
            return None

        cls, cx, cy, w, h = map(float, parts[:5])

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        x1 = max(0, min(1, x1))
        y1 = max(0, min(1, y1))
        x2 = max(0, min(1, x2))
        y2 = max(0, min(1, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return [x1, y1, x2, y2]    # normalized xyxy
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load templates
        templates = self._load_templates(sample['object_id'])

        # Load search image
        img = Image.open(sample['search_image']).convert('RGB')
        img = np.array(img)
        orig_h, orig_w = img.shape[:2]

        # Load normalized bbox
        bbox_norm = self._load_bbox(sample['label_file'])

        if bbox_norm is not None:
            # Convert normalized bbox → pixel on ORIGINAL image
            x1 = bbox_norm[0] * orig_w
            y1 = bbox_norm[1] * orig_h
            x2 = bbox_norm[2] * orig_w
            y2 = bbox_norm[3] * orig_h
            bbox_px = [x1, y1, x2, y2]

            transformed = self.search_transform(
                image=img,
                bboxes=[bbox_px],
                labels=[1]
            )

            img = transformed['image']

            # Albumentations output is pixel xyxy at resized size
            if len(transformed["bboxes"]) > 0:
                bbox = list(transformed["bboxes"][0])  # pixel xyxy
            else:
                bbox = None

        else:
            transformed = self.search_transform(
                image=img,
                bboxes=[],
                labels=[]
            )
            img = transformed['image']
            bbox = None

        target = {
            'bbox': bbox,                # pixel xyxy (absolute!)
            'object_id': sample['object_id']
        }

        return {
            'templates': templates,
            'search': img,
            'target': target
        }


def collate_fn(batch):
    """Custom collate function for dataloader."""
    templates = []
    searches = []
    targets = []
    
    for item in batch:
        templates.append(item['templates'])
        searches.append(item['search'])
        targets.append(item['target'])
    
    # Stack templates: each is list of 3 tensors
    # We want 3 batches, one for each template position
    templates_batched = [
        torch.stack([t[i] for t in templates]) for i in range(3)
    ]
    
    searches_batched = torch.stack(searches)
    
    return {
        'templates': templates_batched,
        'search': searches_batched,
        'targets': targets
    }


# ============================================================================
# Training
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, config, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    
    total_loss = 0
    total_loss_obj = 0
    total_loss_bbox = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        templates = [t.to(device) for t in batch['templates']]
        search = batch['search'].to(device)
        targets = batch['targets']
        
        optimizer.zero_grad()
        
        # Forward with mixed precision
        if scaler is not None:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                obj_map, bbox_map = model(templates, search)
                loss, loss_dict = criterion(obj_map, bbox_map, targets, stride=model.stride)
        else:
            obj_map, bbox_map = model(templates, search)
            loss, loss_dict = criterion(obj_map, bbox_map, targets, stride=model.stride)
        
        # Check for NaN/inf loss (FP16 safety guard)
        if not torch.isfinite(loss):
            # Use safe values for logging to avoid -inf display
            safe_loss = float(torch.nan_to_num(loss.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4).item())
            safe_loss_obj = float(torch.nan_to_num(torch.tensor(loss_dict['loss_obj']), nan=0.0, posinf=1e4, neginf=-1e4).item())
            safe_loss_bbox = float(torch.nan_to_num(torch.tensor(loss_dict['loss_bbox']), nan=0.0, posinf=1e4, neginf=-1e4).item())
            
            print(f"[WARN] Non-finite loss at epoch {epoch}, batch {batch_idx}: "
                  f"loss={safe_loss:.4f}, obj={safe_loss_obj:.4f}, "
                  f"bbox={safe_loss_bbox:.4f}, num_pos={loss_dict['num_pos']}")
            optimizer.zero_grad(set_to_none=True)
            # Do NOT call scaler.update() here - no inf checks were recorded
            continue
        
        # Backward with mixed precision
        if scaler is not None:
            scaler.scale(loss).backward()
            
            # Gradient clipping with scaler
            if config.get('grad_clip', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
        
        # Update stats
        batch_size = search.size(0)
        total_loss += loss.item() * batch_size
        total_loss_obj += loss_dict['loss_obj'] * batch_size
        total_loss_bbox += loss_dict['loss_bbox'] * batch_size
        total_samples += batch_size
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'obj': f"{loss_dict['loss_obj']:.4f}",
            'bbox': f"{loss_dict['loss_bbox']:.4f}",
            'pos': loss_dict['num_pos']
        })
        
        # Log to wandb
        if config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.log({
                'train/loss_step': loss.item(),
                'train/loss_obj_step': loss_dict['loss_obj'],
                'train/loss_bbox_step': loss_dict['loss_bbox'],
                'train/num_pos_step': loss_dict['num_pos']
            })
    
    # Epoch stats
    avg_loss = total_loss / total_samples
    avg_loss_obj = total_loss_obj / total_samples
    avg_loss_bbox = total_loss_bbox / total_samples
    
    return {
        'loss': avg_loss,
        'loss_obj': avg_loss_obj,
        'loss_bbox': avg_loss_bbox
    }


def validate(model, dataloader, criterion, device, config):
    """Validate the model with mixed precision."""
    model.eval()
    
    total_loss = 0
    total_loss_obj = 0
    total_loss_bbox = 0
    total_samples = 0
    
    # IoU metrics
    all_ious = []
    num_detections = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        
        for batch in pbar:
            # Move to device
            templates = [t.to(device) for t in batch['templates']]
            search = batch['search'].to(device)
            targets = batch['targets']
            
            # Forward with mixed precision (faster validation, no scaler needed)
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                obj_map, bbox_map = model(templates, search)
                loss, loss_dict = criterion(obj_map, bbox_map, targets, stride=model.stride)
            
            # Compute IoU metrics
            B, _, H, W = obj_map.shape
            stride = model.stride
            
            # Get predictions (use simple max detection for validation)
            obj_scores = torch.sigmoid(obj_map)  # [B, 1, H, W]
            
            for b in range(B):
                target = targets[b]
                if target is None or 'bbox' not in target or target['bbox'] is None:
                    continue
                
                # Get ground truth bbox
                gt_bbox = target['bbox']  # [x1, y1, x2, y2] in pixels
                gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2
                gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2
                gt_w = gt_bbox[2] - gt_bbox[0]
                gt_h = gt_bbox[3] - gt_bbox[1]
                
                # Find best prediction (highest objectness score)
                obj_score_b = obj_scores[b, 0]  # [H, W]
                max_idx = torch.argmax(obj_score_b.flatten())
                max_y, max_x = divmod(max_idx.item(), W)
                
                # Decode predicted bbox at max location
                bbox_pred = bbox_map[b, :, max_y, max_x]  # [4] - (tx, ty, tw, th)
                
                # Decode to pixel coordinates (same as model's decode logic)
                grid_center_x = (max_x + 0.5) * stride
                grid_center_y = (max_y + 0.5) * stride
                
                pred_cx = grid_center_x + torch.tanh(bbox_pred[0]) * stride
                pred_cy = grid_center_y + torch.tanh(bbox_pred[1]) * stride
                pred_w = F.softplus(bbox_pred[2])
                pred_h = F.softplus(bbox_pred[3])
                
                # Compute IoU
                pred_box = torch.tensor([pred_cx, pred_cy, pred_w, pred_h], device=device).unsqueeze(0)
                gt_box = torch.tensor([gt_cx, gt_cy, gt_w, gt_h], device=device).unsqueeze(0)
                
                iou = compute_iou(pred_box, gt_box)
                if len(iou) > 0:
                    all_ious.append(iou[0].item())
                    num_detections += 1
            
            # Update stats
            batch_size = search.size(0)
            total_loss += loss.item() * batch_size
            total_loss_obj += loss_dict['loss_obj'] * batch_size
            total_loss_bbox += loss_dict['loss_bbox'] * batch_size
            total_samples += batch_size
            
            # Update progress bar with IoU info
            current_iou = np.mean(all_ious) if all_ious else 0.0
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'obj': f"{loss_dict['loss_obj']:.4f}",
                'bbox': f"{loss_dict['loss_bbox']:.4f}",
                'iou': f"{current_iou:.4f}"
            })
    
    # Epoch stats
    avg_loss = total_loss / total_samples
    avg_loss_obj = total_loss_obj / total_samples
    avg_loss_bbox = total_loss_bbox / total_samples
    
    # IoU stats
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    iou_50 = np.mean([iou >= 0.5 for iou in all_ious]) if all_ious else 0.0  # IoU@0.5
    iou_75 = np.mean([iou >= 0.75 for iou in all_ious]) if all_ious else 0.0  # IoU@0.75
    
    return {
        'loss': avg_loss,
        'loss_obj': avg_loss_obj,
        'loss_bbox': avg_loss_bbox,
        'mean_iou': mean_iou,
        'iou_50': iou_50,
        'iou_75': iou_75,
        'num_detections': num_detections
    }


def train(config):
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if config.get('use_wandb', False):
        if WANDB_AVAILABLE:
            wandb.init(project=config['project_name'], config=config)
        else:
            print("Warning: wandb not installed. Logging disabled.")
            config['use_wandb'] = False
    
    # Check for checkpoint to load
    checkpoint_path = config.get('checkpoint_path', None)
    if checkpoint_path:
        print(f"Will load checkpoint from: {checkpoint_path}")
    else:
        print("Training from scratch")
    
    # Build model
    print("Building model...")
    model = build_convnext_refdet(
        fpn_channels=config['fpn_channels'],
        corr_channels=config['corr_channels'],
        det_channels=config['det_channels'],
        template_fusion=config['template_fusion']
    )
    model = model.to(device)
    
    print(f"Model parameters: {model.count_parameters() / 1e6:.2f}M")
    
    # Build dataset
    print("Building datasets...")
    use_aug = config.get('use_augmentation', True)
    print(f"Data augmentation: {'ENABLED' if use_aug else 'DISABLED'}")
    
    train_dataset = ReferenceDetectionDataset(
        data_root=config['data_root'],
        split='train',
        search_size=config['search_size'],
        template_size=config['template_size'],
        augmentation=use_aug
    )
    
    # Debug pixel bbox
    if len(train_dataset) > 10:
        s = train_dataset[10]
        print("bbox pixel:", s["target"]["bbox"])
    
    val_dataset = ReferenceDetectionDataset(
        data_root=config['data_root'],
        split='val',
        search_size=config['search_size'],
        template_size=config['template_size'],
        augmentation=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Build criterion
    criterion = RefDetLoss(
        lambda_obj=config['lambda_obj'],
        lambda_bbox=config['lambda_bbox'],
        focal_alpha=config['focal_alpha'],
        focal_gamma=config['focal_gamma'],
        iou_type=config['iou_type'],
        pos_radius=config.get('pos_radius', 0)
    )
    
    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Build scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['lr'] * 0.1
    )
    
    # Initialize mixed precision scaler
    use_amp = config.get('use_amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    
    if use_amp:
        print("Using mixed precision (FP16) training")
    else:
        print("Using FP32 training")
    
    # Load checkpoint if specified (backward compatible with FP32 checkpoints)
    start_epoch = 1
    best_val_loss = float('inf')
    
    if checkpoint_path:
        try:
            checkpoint_info = load_checkpoint(checkpoint_path, model)
            if 'val_stats' in checkpoint_info and 'loss' in checkpoint_info['val_stats']:
                best_val_loss = checkpoint_info['val_stats']['loss']
                print(f"Previous best val loss: {best_val_loss:.4f}")
            print("Model weights loaded, optimizer/scheduler will be fresh")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting training from scratch")
    
    # Training loop
    
    for epoch in range(start_epoch, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        
        # Train
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, scaler=scaler
        )
        
        print(f"Train - Loss: {train_stats['loss']:.4f}, "
              f"Obj: {train_stats['loss_obj']:.4f}, "
              f"BBox: {train_stats['loss_bbox']:.4f}")
        
        # Validate
        val_stats = validate(model, val_loader, criterion, device, config)
        
        print(f"Val - Loss: {val_stats['loss']:.4f}, "
              f"Obj: {val_stats['loss_obj']:.4f}, "
              f"BBox: {val_stats['loss_bbox']:.4f}")
        print(f"Val - IoU: {val_stats['mean_iou']:.4f}, "
              f"IoU@0.5: {val_stats['iou_50']:.4f}, "
              f"IoU@0.75: {val_stats['iou_75']:.4f}, "
              f"Dets: {val_stats['num_detections']}")
        
        # Update scheduler
        scheduler.step()
        
        # Log to wandb
        if config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_stats['loss'],
                'train/loss_obj': train_stats['loss_obj'],
                'train/loss_bbox': train_stats['loss_bbox'],
                'val/loss': val_stats['loss'],
                'val/loss_obj': val_stats['loss_obj'],
                'val/loss_bbox': val_stats['loss_bbox'],
                'val/mean_iou': val_stats['mean_iou'],
                'val/iou_50': val_stats['iou_50'],
                'val/iou_75': val_stats['iou_75'],
                'val/num_detections': val_stats['num_detections'],
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Check if this is the best model  
        is_best = val_stats['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_stats['loss']
        
        # Save weights only with cleanup and save_interval
        save_weights_with_cleanup(
            model,
            config['checkpoint_dir'], 
            epoch, 
            is_best=is_best,
            save_interval=config.get('save_interval', 1)
        )
    
    print("\nTraining completed!")
    
    if config.get('use_wandb', False) and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    # Training configuration
    config = {
        # Data
        'data_root': 'refdet/retrieval_dataset_flat_zoomed',
        # 'data_root': '/home/ta-thai-24/Workspace/hienhq/refdet/retrieval_dataset_flat_zoomed',
        'search_size': (1024, 576),
        'template_size': (256, 256),
        
        # Model
        'fpn_channels': 256,
        'corr_channels': 64,
        'det_channels': 256,
        'template_fusion': 'attention',  # 'attention' or 'average'
        
        # Loss
        'lambda_obj': 1.0,
        'lambda_bbox': 5.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'iou_type': 'giou',  # 'iou' or 'giou'
        'pos_radius': 1,  # Multi-cell positive: 0=single cell, 1=3x3 patch, 2=5x5 patch
        
        # Training
        'batch_size': 16,
        'epochs': 40,
        'lr': 1e-5,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'num_workers': 8,
        'use_amp': True,  # Enable mixed precision (FP16) training for speed + memory efficiency
        'use_augmentation': True,  # Enable/disable data augmentation (True=augment, False=no augment)
        
        # Logging
        'use_wandb': False,
        'project_name': 'convnext-refdet',
        'checkpoint_dir': 'drive/MyDrive/ZALO2025',
        # 'checkpoint_dir': 'checkpoints_refdet',
        'save_interval': 10,
        
        # Checkpoint
        'checkpoint_path': 'drive/MyDrive/ZALO2025/best_32.pth',  # path to checkpoint to load, or None for training from scratch
    }
    
    train(config)
