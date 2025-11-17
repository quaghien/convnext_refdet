"""
Utility functions for ConvNeXt Reference Detection

Includes:
- Model weight loading/saving utilities
- Pretrained weight conversion from timm
- Data augmentation utilities
- Bbox processing utilities
- Evaluation helpers
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


# ============================================================================
# Model Weight Utilities
# ============================================================================

def load_pretrained_convnext(model, pretrained_path=None, strict=False):
    """Load pretrained ConvNeXt weights into the model.
    
    Args:
        model: ConvNeXtRefDet model instance
        pretrained_path: Path to pretrained weights or None to download from timm
        strict: Whether to strictly enforce key matching
    
    Returns:
        model: Model with loaded weights
    """
    if pretrained_path is None:
        # Try to download from timm
        try:
            import timm
            print("Downloading ConvNeXt-Tiny pretrained weights from timm...")
            pretrained = timm.create_model('convnext_tiny', pretrained=True)
            pretrained_dict = pretrained.state_dict()
        except ImportError:
            print("Warning: timm not installed. Cannot download pretrained weights.")
            print("Install with: pip install timm")
            return model
    else:
        print(f"Loading pretrained weights from {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    
    # Get model dict
    model_dict = model.state_dict()
    
    # Filter out unnecessary keys and rename if needed
    pretrained_dict_filtered = {}
    for k, v in pretrained_dict.items():
        # Only load backbone weights
        if k.startswith('backbone.'):
            pretrained_dict_filtered[k] = v
        # Handle different naming conventions
        elif 'stem' in k or 'stages' in k or 'downsample' in k:
            new_k = f'backbone.{k}'
            if new_k in model_dict and model_dict[new_k].shape == v.shape:
                pretrained_dict_filtered[new_k] = v
    
    # Update model dict
    model_dict.update(pretrained_dict_filtered)
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
    
    print(f"Loaded {len(pretrained_dict_filtered)} pretrained weights")
    if len(missing_keys) > 0 and not strict:
        print(f"Missing keys (will be randomly initialized): {len(missing_keys)}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {len(unexpected_keys)}")
    
    return model


def save_checkpoint(model, optimizer, scheduler, epoch, save_path, **kwargs):
    """Save training checkpoint.
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Scheduler instance
        epoch: Current epoch
        save_path: Path to save checkpoint
        **kwargs: Additional items to save (e.g., metrics, config)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    checkpoint.update(kwargs)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model instance
        optimizer: Optional optimizer instance
        scheduler: Optional scheduler instance
    
    Returns:
        epoch: Loaded epoch number
        checkpoint: Full checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    
    return epoch, checkpoint


# ============================================================================
# BBox Utilities
# ============================================================================

def bbox_iou(box1, box2, format='xyxy'):
    """Compute IoU between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] or [cx, cy, w, h]
        box2: [x1, y1, x2, y2] or [cx, cy, w, h]
        format: 'xyxy' or 'cxcywh'
    
    Returns:
        iou: float
    """
    if format == 'cxcywh':
        # Convert to xyxy
        box1 = [box1[0] - box1[2]/2, box1[1] - box1[3]/2, 
                box1[0] + box1[2]/2, box1[1] + box1[3]/2]
        box2 = [box2[0] - box2[2]/2, box2[1] - box2[3]/2,
                box2[0] + box2[2]/2, box2[1] + box2[3]/2]
    
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / (union_area + 1e-7)
    return iou


def bbox_xywh_to_xyxy(bbox):
    """Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]."""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def bbox_xyxy_to_xywh(bbox):
    """Convert bbox from [x1, y1, x2, y2] to [x, y, w, h]."""
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox from [cx, cy, w, h] to [x1, y1, x2, y2]."""
    return [bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2,
            bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]


def bbox_xyxy_to_cxcywh(bbox):
    """Convert bbox from [x1, y1, x2, y2] to [cx, cy, w, h]."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return [cx, cy, w, h]


def clip_bbox(bbox, img_width, img_height, format='xyxy'):
    """Clip bbox to image boundaries.
    
    Args:
        bbox: Bounding box
        img_width: Image width
        img_height: Image height
        format: 'xyxy' or 'cxcywh'
    
    Returns:
        clipped_bbox: Clipped bounding box
    """
    if format == 'xyxy':
        x1 = max(0, min(img_width, bbox[0]))
        y1 = max(0, min(img_height, bbox[1]))
        x2 = max(0, min(img_width, bbox[2]))
        y2 = max(0, min(img_height, bbox[3]))
        return [x1, y1, x2, y2]
    elif format == 'cxcywh':
        # Convert to xyxy, clip, convert back
        xyxy = bbox_cxcywh_to_xyxy(bbox)
        clipped_xyxy = clip_bbox(xyxy, img_width, img_height, format='xyxy')
        return bbox_xyxy_to_cxcywh(clipped_xyxy)
    else:
        raise ValueError(f"Unknown format: {format}")


def nms(detections, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to detections.
    
    Args:
        detections: List of detection dicts with 'bbox' and 'score'
        iou_threshold: IoU threshold for suppression
    
    Returns:
        filtered_detections: List of detections after NMS
    """
    if len(detections) == 0:
        return []
    
    # Sort by score
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    keep = []
    while len(detections) > 0:
        # Keep the highest scoring detection
        best = detections.pop(0)
        keep.append(best)
        
        # Remove detections with high IoU
        detections = [
            det for det in detections
            if bbox_iou(best['bbox'], det['bbox'], format='xyxy') < iou_threshold
        ]
    
    return keep


# ============================================================================
# Data Processing Utilities
# ============================================================================

def parse_label_file(label_path, format='xyxy'):
    """Parse label file to extract bounding boxes.
    
    Args:
        label_path: Path to label file
        format: Expected format ('xyxy', 'xywh', 'cxcywh_norm', etc.)
    
    Returns:
        bboxes: List of bboxes
    """
    bboxes = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                bbox = [float(x) for x in parts[:4]]
                bboxes.append(bbox)
    
    return bboxes


def save_label_file(bboxes, save_path, format='xyxy'):
    """Save bboxes to label file.
    
    Args:
        bboxes: List of bboxes
        save_path: Path to save label file
        format: Format to save ('xyxy', 'xywh', etc.)
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        for bbox in bboxes:
            f.write(' '.join([f"{x:.6f}" for x in bbox]) + '\n')


# ============================================================================
# Evaluation Utilities
# ============================================================================

def compute_ap(precisions, recalls):
    """Compute Average Precision (AP) from precision-recall curve.
    
    Args:
        precisions: List of precision values
        recalls: List of recall values
    
    Returns:
        ap: Average Precision
    """
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = np.array(recalls)[sorted_indices]
    precisions = np.array(precisions)[sorted_indices]
    
    # Compute AP using 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def compute_metrics_at_thresholds(predictions, ground_truths, score_thresholds, iou_threshold=0.5):
    """Compute metrics at different score thresholds.
    
    Args:
        predictions: List of predictions per image
        ground_truths: List of ground truths per image
        score_thresholds: List of score thresholds to evaluate
        iou_threshold: IoU threshold for matching
    
    Returns:
        metrics: Dict with metrics at each threshold
    """
    results = []
    
    for score_thresh in score_thresholds:
        # Filter predictions by score threshold
        filtered_preds = []
        for preds in predictions:
            filtered = [p for p in preds if p['score'] >= score_thresh]
            filtered_preds.append(filtered)
        
        # Compute metrics
        tp = 0
        fp = 0
        fn = 0
        
        for preds, gts in zip(filtered_preds, ground_truths):
            if len(gts) == 0:
                fp += len(preds)
                continue
            
            if len(preds) == 0:
                fn += len(gts)
                continue
            
            gt_matched = [False] * len(gts)
            
            for pred in preds:
                matched = False
                for gt_idx, gt in enumerate(gts):
                    if gt_matched[gt_idx]:
                        continue
                    
                    iou = bbox_iou(pred['bbox'], gt, format='xyxy')
                    if iou >= iou_threshold:
                        tp += 1
                        gt_matched[gt_idx] = True
                        matched = True
                        break
                
                if not matched:
                    fp += 1
            
            fn += sum(1 for m in gt_matched if not m)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'score_threshold': score_thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    return results


# ============================================================================
# Miscellaneous Utilities
# ============================================================================

def count_parameters(model):
    """Count trainable and total parameters in model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        'trainable': trainable,
        'total': total,
        'trainable_M': trainable / 1e6,
        'total_M': total / 1e6
    }


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_bn_eval(module):
    """Set BatchNorm layers to eval mode (useful for fine-tuning)."""
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        module.eval()


def freeze_backbone(model):
    """Freeze backbone weights for fine-tuning."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("Backbone frozen")


def unfreeze_backbone(model):
    """Unfreeze backbone weights."""
    for param in model.backbone.parameters():
        param.requires_grad = True
    print("Backbone unfrozen")


def print_model_summary(model, input_size=(1, 3, 1024, 576), template_size=(1, 3, 256, 256)):
    """Print model summary with layer shapes.
    
    Args:
        model: Model instance
        input_size: Input size (B, C, H, W) for search image
        template_size: Input size (B, C, H, W) for templates
    """
    try:
        from torchinfo import summary
        
        # Create dummy inputs
        templates = [torch.randn(template_size) for _ in range(3)]
        search = torch.randn(input_size)
        
        print("\n" + "="*80)
        print("MODEL SUMMARY")
        print("="*80)
        
        # Try to print summary (might fail for complex models)
        try:
            summary(model, input_data=(templates, search), 
                   col_names=["output_size", "num_params"],
                   depth=3)
        except:
            print(f"Total parameters: {count_parameters(model)['total_M']:.2f}M")
            print(f"Trainable parameters: {count_parameters(model)['trainable_M']:.2f}M")
        
        print("="*80 + "\n")
        
    except ImportError:
        print("torchinfo not installed. Install with: pip install torchinfo")
        print(f"Total parameters: {count_parameters(model)['total_M']:.2f}M")


if __name__ == "__main__":
    # Test utilities
    print("Testing bbox utilities...")
    
    # Test IoU
    box1 = [10, 10, 50, 50]
    box2 = [30, 30, 70, 70]
    iou = bbox_iou(box1, box2, format='xyxy')
    print(f"IoU between {box1} and {box2}: {iou:.4f}")
    
    # Test conversions
    xyxy = [10, 20, 50, 60]
    cxcywh = bbox_xyxy_to_cxcywh(xyxy)
    xyxy_back = bbox_cxcywh_to_xyxy(cxcywh)
    print(f"XYXY: {xyxy} -> CXCYWH: {cxcywh} -> XYXY: {xyxy_back}")
    
    # Test NMS
    detections = [
        {'bbox': [10, 10, 50, 50], 'score': 0.9},
        {'bbox': [15, 15, 55, 55], 'score': 0.8},
        {'bbox': [100, 100, 150, 150], 'score': 0.95},
    ]
    filtered = nms(detections, iou_threshold=0.5)
    print(f"\nBefore NMS: {len(detections)} detections")
    print(f"After NMS: {len(filtered)} detections")
    for det in filtered:
        print(f"  BBox: {det['bbox']}, Score: {det['score']}")
    
    print("\n✓ Utility tests completed!")
