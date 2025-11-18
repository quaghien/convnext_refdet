"""
Single-Frame GPU Inference for Public Test

Processes each video frame-by-frame on GPU with proper numpy array handling.
Uses tqdm for progress tracking.
"""

import os
import sys
import torch
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tempfile
import shutil
import gc
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add path
sys.path.append('.')
from model import build_convnext_refdet


class SingleFrameGPUInference:
    """Single-frame GPU inference with proper numpy handling."""
    
    def __init__(self, checkpoint_path, device='cuda', confidence_threshold=0.4, use_fp16=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.use_fp16 = use_fp16
        
        print("Loading model...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle both full checkpoints and weights-only files
        if 'model_state_dict' in checkpoint:
            # Full checkpoint format
            state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
        else:
            # Weights-only format
            state_dict = checkpoint
            config = {}  # Use default config
        
        # Detect checkpoint precision
        first_param_key = next(iter(state_dict))
        first_param = state_dict[first_param_key]
        checkpoint_precision = "FP16" if first_param.dtype == torch.float16 else "FP32"
        print(f"Checkpoint precision: {checkpoint_precision}")
        
        # Build model
        self.model = build_convnext_refdet(
            fpn_channels=config.get('fpn_channels', 256),
            corr_channels=config.get('corr_channels', 64),
            det_channels=config.get('det_channels', 256),
            template_fusion=config.get('template_fusion', 'attention')
        )
        
        # Load weights
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        
        # Convert to FP16 if enabled
        if self.use_fp16 and self.device.type == 'cuda':
            self.model = self.model.half()
            print(f"Model loaded on {self.device} with FP16")
        else:
            print(f"Model loaded on {self.device} with FP32")
            
        self.model.eval()
        
        # Transforms
        self.search_transform = A.Compose([
            A.Resize(height=576, width=1024),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        self.template_transform = A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def preprocess_numpy_frame(self, frame_bgr):
        """Preprocess numpy frame (BGR from OpenCV) to tensor."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Get original size
        orig_h, orig_w = frame_rgb.shape[:2]
        
        # Apply transforms
        transformed = self.search_transform(image=frame_rgb)
        tensor = transformed['image']  # Already torch tensor
        
        return tensor, (orig_h, orig_w)
    
    def load_templates(self, template_paths):
        """Load templates from paths."""
        templates = []
        
        for i in range(3):
            if i < len(template_paths):
                path = template_paths[i]
            else:
                path = template_paths[-1]  # Repeat last
            
            # Load image
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Cannot load template: {path}")
            
            # Convert BGR to RGB  
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Transform
            transformed = self.template_transform(image=img_rgb)
            templates.append(transformed['image'])
        
        # Create batch structure for model
        templates_batched = []
        for template_tensor in templates:
            template_batch = template_tensor.unsqueeze(0).to(self.device)
            if self.use_fp16 and self.device.type == 'cuda':
                template_batch = template_batch.half()
            templates_batched.append(template_batch)
        
        return templates_batched
    
    def infer_single_frame(self, frame_bgr, templates):
        """Run inference on single numpy frame."""
        # Preprocess frame
        tensor, orig_size = self.preprocess_numpy_frame(frame_bgr)
        tensor = tensor.unsqueeze(0).to(self.device)  # Add batch dim
        
        if self.use_fp16 and self.device.type == 'cuda':
            tensor = tensor.half()
        
        with torch.no_grad():
            if self.use_fp16 and self.device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # Forward pass
                    obj_map, bbox_map = self.model(templates, tensor)
            else:
                # Forward pass
                obj_map, bbox_map = self.model(templates, tensor)
            
            # Decode predictions
            predictions = self.model.decode_predictions(
                obj_map, bbox_map,
                score_threshold=0.05,
                top_k=5
            )[0]  # Get first (only) batch item
        
        # Find best detection
        best_det = None
        best_score = 0.0
        
        for det in predictions:
            if det['score'] > best_score and det['score'] >= self.confidence_threshold:
                best_score = det['score']
                best_det = det
        
        if best_det is None:
            return None
        
        # Scale bbox to original size
        bbox = best_det['bbox']
        scale_x = orig_size[1] / 1024  # orig_w / model_w
        scale_y = orig_size[0] / 576   # orig_h / model_h
        
        scaled_bbox = [
            bbox[0] * scale_x,
            bbox[1] * scale_y, 
            bbox[2] * scale_x,
            bbox[3] * scale_y
        ]
        
        # Clip to image bounds
        h, w = orig_size
        x1 = max(0, min(w-1, int(scaled_bbox[0])))
        y1 = max(0, min(h-1, int(scaled_bbox[1])))
        x2 = max(x1+1, min(w, int(scaled_bbox[2])))
        y2 = max(y1+1, min(h, int(scaled_bbox[3])))
        
        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "score": float(best_score)
        }
    
    def extract_video_frames(self, video_path, max_frames=2000):
        """Extract frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return [], []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine frame indices
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int).tolist()
        
        frames = []
        valid_indices = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
                valid_indices.append(frame_idx)
        
        cap.release()
        return frames, valid_indices
    
    def process_object(self, object_dir):
        """Process one object directory."""
        object_name = object_dir.name
        
        # Check paths
        video_path = object_dir / "drone_video.mp4"
        template_dir = object_dir / "object_images"
        
        if not video_path.exists() or not template_dir.exists():
            return {"video_id": object_name, "detections": []}
        
        # Get templates
        template_files = sorted(list(template_dir.glob("*.jpg")) + list(template_dir.glob("*.png")))
        if len(template_files) < 3:
            return {"video_id": object_name, "detections": []}
        
        template_paths = template_files[:3]
        
        # Load templates
        try:
            templates = self.load_templates(template_paths)
        except Exception as e:
            print(f"Error loading templates: {e}")
            return {"video_id": object_name, "detections": []}
        
        # Extract frames
        frames, frame_indices = self.extract_video_frames(video_path, max_frames=2000)
        
        if not frames:
            return {"video_id": object_name, "detections": []}
        
        print(f"Processing {len(frames)} frames...")
        
        all_detections = []
        
        # Process each frame individually with tqdm
        for frame, frame_idx in tqdm(zip(frames, frame_indices), 
                                   desc=f"{object_name}", 
                                   total=len(frames),
                                   leave=False):
            try:
                detection = self.infer_single_frame(frame, templates)
                
                if detection is not None:
                    all_detections.append({
                        "frame": int(frame_idx),
                        "x1": detection["x1"],
                        "y1": detection["y1"],
                        "x2": detection["x2"],
                        "y2": detection["y2"]
                    })
                
            except Exception as e:
                # Silent skip on error
                continue
        
        # Format result
        result = {"video_id": object_name, "detections": []}
        if all_detections:
            result["detections"] = [{"bboxes": all_detections}]
        
        return result


def main():
    """Main function."""
    
    # Configuration
    CHECKPOINT_PATH = "drive/MyDrive/ZALO2025/best_weights_fp32.pth"
    SAMPLES_DIR = "public_test/samples"
    OUTPUT_PATH = "submission_output.json"
    CONFIDENCE_THRESHOLD = 0.4
    USE_FP16 = False 
    
    print("=== Single-Frame GPU Inference ===")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Samples: {SAMPLES_DIR}")
    print(f"Confidence: {CONFIDENCE_THRESHOLD}")
    print(f"Mixed Precision: {'FP16' if USE_FP16 else 'FP32'}")
    
    # Initialize inference
    inference = SingleFrameGPUInference(
        checkpoint_path=CHECKPOINT_PATH,
        device='cuda',
        confidence_threshold=CONFIDENCE_THRESHOLD,
        use_fp16=USE_FP16
    )
    
    # Get objects
    samples_dir = Path(SAMPLES_DIR)
    object_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    
    print(f"\nFound {len(object_dirs)} objects: {[d.name for d in object_dirs]}")
    
    # Process each object
    submission = []
    
    for i, obj_dir in enumerate(object_dirs, 1):
        print(f"\n[{i}/{len(object_dirs)}] Processing {obj_dir.name}...")
        
        result = inference.process_object(obj_dir)
        submission.append(result)
        
        # Summary
        det_count = sum(len(det.get('bboxes', [])) for det in result['detections'])
        print(f"  → {det_count} detections found")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Save results
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(submission, f, indent=2)
    
    # Summary
    total_detections = sum(
        sum(len(det.get('bboxes', [])) for det in r['detections']) 
        for r in submission
    )
    
    print(f"\n=== SUMMARY ===")
    for result in submission:
        count = sum(len(det.get('bboxes', [])) for det in result['detections'])
        print(f"{result['video_id']}: {count} detections")
    
    print(f"\nTotal detections: {total_detections}")
    print(f"Output saved: {OUTPUT_PATH}")
    print("✅ Done!")


if __name__ == "__main__":
    main()