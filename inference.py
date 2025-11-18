"""
Inference and evaluation script for ConvNeXt Reference Detection Model

Includes:
- Single image inference
- Batch inference on test set
- Evaluation metrics (mAP, precision, recall)
- Visualization utilities
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from model import build_convnext_refdet


# ============================================================================
# Inference
# ============================================================================

class RefDetInference:
    """Inference wrapper for reference detection model."""
    
    def __init__(self, 
                 checkpoint_path,
                 device='cuda',
                 score_threshold=0.3,
                 search_size=(1024, 576),
                 template_size=(256, 256)):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.score_threshold = score_threshold
        self.search_size = search_size
        self.template_size = template_size
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Build model
        config = checkpoint.get('config', {})
        self.model = build_convnext_refdet(
            fpn_channels=config.get('fpn_channels', 256),
            corr_channels=config.get('corr_channels', 64),
            det_channels=config.get('det_channels', 256),
            template_fusion=config.get('template_fusion', 'attention')
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        
        # Transforms
        self.search_transform = A.Compose([
            A.Resize(height=search_size[1], width=search_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        self.template_transform = A.Compose([
            A.Resize(height=template_size[1], width=template_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image_path, is_template=False):
        """Load and preprocess image."""
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        original_size = img_np.shape[:2]  # (H, W)
        
        if is_template:
            transformed = self.template_transform(image=img_np)
        else:
            transformed = self.search_transform(image=img_np)
        
        img_tensor = transformed['image']
        
        return img_tensor, original_size
    
    def load_templates(self, template_paths):
        """Load 3 template images.
        
        Args:
            template_paths: List of 3 image paths or directory path
        """
        if isinstance(template_paths, (str, Path)):
            # Directory path
            template_dir = Path(template_paths)
            template_files = sorted(template_dir.glob('*.jpg')) + sorted(template_dir.glob('*.png'))
            template_paths = [str(f) for f in template_files[:3]]
        
        assert len(template_paths) >= 3, "Need at least 3 template images"
        
        templates = []
        for i in range(3):
            img_tensor, _ = self.preprocess_image(template_paths[i], is_template=True)
            templates.append(img_tensor)
        
        # Stack as batch
        templates = [t.unsqueeze(0).to(self.device) for t in templates]
        
        return templates
    
    @torch.no_grad()
    def predict(self, template_paths, search_image_path, top_k=5):
        """Run inference on single search image.
        
        Args:
            template_paths: List of 3 template paths or template directory
            search_image_path: Path to search image
            top_k: Return top k detections
        
        Returns:
            detections: List of detection dicts
            search_size: Original search image size
        """
        # Load templates
        templates = self.load_templates(template_paths)
        
        # Load search image
        search_tensor, original_size = self.preprocess_image(search_image_path, is_template=False)
        search_tensor = search_tensor.unsqueeze(0).to(self.device)
        
        # Forward pass
        obj_map, bbox_map = self.model(templates, search_tensor)
        
        # Decode predictions
        detections = self.model.decode_predictions(
            obj_map, bbox_map,
            score_threshold=self.score_threshold,
            top_k=top_k
        )[0]  # Get first (and only) batch item
        
        # Scale bboxes back to original image size
        scale_x = original_size[1] / self.search_size[0]
        scale_y = original_size[0] / self.search_size[1]
        
        for det in detections:
            bbox = det['bbox']
            det['bbox'] = [
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y
            ]
            det['center_coords'] = [
                det['center_coords'][0] * scale_x,
                det['center_coords'][1] * scale_y
            ]
        
        return detections, original_size
    
    def predict_batch(self, template_paths, search_image_paths, top_k=5):
        """Run inference on batch of search images.
        
        Args:
            template_paths: List of 3 template paths or template directory
            search_image_paths: List of search image paths
            top_k: Return top k detections per image
        
        Returns:
            results: List of (detections, original_size) per image
        """
        results = []
        
        for search_path in tqdm(search_image_paths, desc="Inference"):
            detections, original_size = self.predict(template_paths, search_path, top_k)
            results.append({
                'search_path': search_path,
                'detections': detections,
                'original_size': original_size
            })
        
        return results


# ============================================================================
# Visualization
# ============================================================================

def visualize_detection(image_path, detections, templates_path=None, save_path=None, show=True):
    """Visualize detection results on image.
    
    Args:
        image_path: Path to search image
        detections: List of detection dicts from model
        templates_path: Optional path to template directory
        save_path: Optional path to save visualization
        show: Whether to display the image
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Try to load a better font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw detections
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    
    for i, det in enumerate(detections):
        bbox = det['bbox']
        score = det['score']
        color = colors[i % len(colors)]
        
        # Draw bbox
        draw.rectangle(bbox, outline=color, width=3)
        
        # Draw score
        text = f"Score: {score:.3f}"
        text_bbox = draw.textbbox((bbox[0], bbox[1] - 20), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((bbox[0], bbox[1] - 20), text, fill='white', font=font)
        
        # Draw center point
        cx, cy = det['center_coords']
        r = 5
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color, outline='white')
    
    if save_path:
        img.save(save_path)
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Detections: {len(detections)}")
        plt.tight_layout()
        plt.show()
    
    return img


def visualize_with_templates(search_path, templates_path, detections, save_path=None):
    """Visualize detection with template images side by side.
    
    Args:
        search_path: Path to search image
        templates_path: Path to template directory
        detections: List of detection dicts
        save_path: Optional path to save visualization
    """
    # Load images
    search_img = Image.open(search_path).convert('RGB')
    
    # Load templates
    template_dir = Path(templates_path)
    template_files = sorted(template_dir.glob('*.jpg')) + sorted(template_dir.glob('*.png'))
    template_imgs = [Image.open(f).convert('RGB') for f in template_files[:3]]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot templates
    for i, (ax, template_img) in enumerate(zip(axes[0], template_imgs)):
        ax.imshow(template_img)
        ax.set_title(f"Template {i+1}")
        ax.axis('off')
    
    # Plot search with detections
    ax_search = axes[1, 0]
    ax_search.imshow(search_img)
    
    # Draw bboxes
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    for i, det in enumerate(detections):
        bbox = det['bbox']
        score = det['score']
        color = colors[i % len(colors)]
        
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                            fill=False, edgecolor=color, linewidth=2)
        ax_search.add_patch(rect)
        ax_search.text(bbox[0], bbox[1]-5, f"{score:.3f}", 
                      bbox=dict(facecolor=color, alpha=0.5),
                      fontsize=10, color='white')
        
        # Draw center
        cx, cy = det['center_coords']
        ax_search.plot(cx, cy, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=2)
    
    ax_search.set_title(f"Search Image ({len(detections)} detections)")
    ax_search.axis('off')
    
    # Hide last subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
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


def evaluate_predictions(predictions, ground_truths, iou_threshold=0.5):
    """Evaluate predictions against ground truths.
    
    Args:
        predictions: List of detection results per image
        ground_truths: List of ground truth bboxes per image
        iou_threshold: IoU threshold for positive match
    
    Returns:
        metrics: Dict with precision, recall, f1, etc.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    ious = []
    
    for pred, gt in zip(predictions, ground_truths):
        if len(pred) == 0 and len(gt) == 0:
            continue
        
        if len(gt) == 0:
            total_fp += len(pred)
            continue
        
        if len(pred) == 0:
            total_fn += len(gt)
            continue
        
        # Match predictions to ground truths
        gt_matched = [False] * len(gt)
        
        for pred_det in pred:
            pred_bbox = pred_det['bbox']
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_bbox in enumerate(gt):
                if gt_matched[gt_idx]:
                    continue
                
                iou = compute_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                total_tp += 1
                gt_matched[best_gt_idx] = True
                ious.append(best_iou)
            else:
                total_fp += 1
        
        # Unmatched ground truths are false negatives
        total_fn += sum(1 for matched in gt_matched if not matched)
    
    # Compute metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = np.mean(ious) if len(ious) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_iou': avg_iou,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }
    
    return metrics


# ============================================================================
# Main inference script
# ============================================================================

def main_inference():
    """Example inference script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with ConvNeXt RefDet")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--templates', type=str, required=True, help='Path to template directory')
    parser.add_argument('--search', type=str, required=True, help='Path to search image or directory')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--score-threshold', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--top-k', type=int, default=5, help='Top k detections')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference
    inferencer = RefDetInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold
    )
    
    # Check if search is directory or single image
    search_path = Path(args.search)
    if search_path.is_dir():
        search_images = list(search_path.glob('*.jpg')) + list(search_path.glob('*.png'))
    else:
        search_images = [search_path]
    
    print(f"Processing {len(search_images)} images...")
    
    # Run inference
    results = []
    for img_path in tqdm(search_images):
        detections, original_size = inferencer.predict(
            args.templates, 
            img_path, 
            top_k=args.top_k
        )
        
        results.append({
            'image': str(img_path),
            'detections': detections,
            'size': original_size
        })
        
        # Visualize if requested
        if args.visualize:
            vis_path = output_dir / f"vis_{img_path.stem}.jpg"
            visualize_with_templates(
                img_path, 
                args.templates, 
                detections,
                save_path=vis_path
            )
    
    # Save results to JSON
    results_json = []
    for r in results:
        results_json.append({
            'image': r['image'],
            'size': r['size'],
            'detections': [{
                'bbox': det['bbox'],
                'score': det['score'],
                'center': det['center']
            } for det in r['detections']]
        })
    
    json_path = output_dir / 'detections.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to {json_path}")
    print(f"Average detections per image: {np.mean([len(r['detections']) for r in results]):.2f}")


if __name__ == "__main__":
    main_inference()
