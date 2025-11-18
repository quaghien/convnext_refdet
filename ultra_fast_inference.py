"""
Ultra-Fast Multi-Threaded GPU Inference

Features:
- True batch processing (process multiple frames simultaneously)
- Multi-threading for video extraction and preprocessing  
- Memory-efficient streaming
- Maximum GPU utilization
- Only highest confidence bbox per frame
"""

import os
import sys
import torch
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gc

# Add path
sys.path.append('.')
from model import build_convnext_refdet


class UltraFastInference:
    """Ultra-fast inference with true batching and multi-threading."""
    
    def __init__(self, checkpoint_path, device='cuda', batch_size=32):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        print(f"Loading model on {self.device}...")
        
        # Load checkpoint
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
        
        print(f"Model loaded, batch size: {batch_size}")
    
    def extract_frames_parallel(self, video_path, max_frames=2000, num_threads=4):
        """Extract frames using parallel processing."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine frame indices
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int).tolist()
        
        frames = [None] * len(frame_indices)
        valid_indices = []
        
        def extract_frame_range(start_idx, end_idx):
            """Extract frames in a range."""
            local_cap = cv2.VideoCapture(video_path)
            for i in range(start_idx, end_idx):
                if i >= len(frame_indices):
                    break
                frame_idx = frame_indices[i]
                local_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = local_cap.read()
                if ret:
                    frames[i] = frame
            local_cap.release()
        
        # Split work among threads
        chunk_size = len(frame_indices) // num_threads + 1
        threads = []
        
        for i in range(0, len(frame_indices), chunk_size):
            thread = threading.Thread(target=extract_frame_range, args=(i, i + chunk_size))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        cap.release()
        
        # Filter valid frames
        valid_frames = []
        for i, frame in enumerate(frames):
            if frame is not None:
                valid_frames.append(frame)
                valid_indices.append(frame_indices[i])
        
        return valid_frames, valid_indices
    
    def preprocess_batch(self, frames):
        """Preprocess batch of frames efficiently."""
        batch_tensors = []
        original_sizes = []
        
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_sizes.append(frame_rgb.shape[:2])
            
            # Transform
            transformed = self.search_transform(image=frame_rgb)
            batch_tensors.append(transformed['image'])
        
        # Stack into batch tensor
        if batch_tensors:
            batch_tensor = torch.stack(batch_tensors).to(self.device)
        else:
            batch_tensor = torch.empty(0, 3, 576, 1024).to(self.device)
        
        return batch_tensor, original_sizes
    
    def load_templates_fast(self, template_paths):
        """Load templates efficiently."""
        templates = []
        
        for i in range(3):
            if i < len(template_paths):
                path = template_paths[i]
            else:
                path = template_paths[-1]  # Repeat last template
            
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transformed = self.template_transform(image=img_rgb)
            templates.append(transformed['image'])
        
        # Stack as batch [3, 3, 256, 256] -> 3 batches of size 1
        templates = [t.unsqueeze(0).to(self.device) for t in templates]
        
        return templates
    
    def inference_batch(self, batch_tensor, templates, confidence_threshold=0.4):
        """Run inference on batch of frames."""
        if batch_tensor.size(0) == 0:
            return []
        
        with torch.no_grad():
            # Forward pass
            obj_map, bbox_map = self.model(templates, batch_tensor)
            
            # Decode all predictions
            batch_detections = self.model.decode_predictions(
                obj_map, bbox_map,
                score_threshold=0.05,
                top_k=5
            )
        
        # Process each frame in batch
        results = []
        for frame_dets in batch_detections:
            # Find best detection for this frame
            best_det = None
            best_score = 0.0
            
            for det in frame_dets:
                if det['score'] > best_score and det['score'] >= confidence_threshold:
                    best_score = det['score']
                    best_det = det
            
            results.append(best_det)
        
        return results
    
    def process_object_ultra_fast(self, object_dir, confidence_threshold=0.4):
        """Process object with maximum speed."""
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
        
        template_paths = [str(f) for f in template_files[:3]]
        
        # Load templates once
        templates = self.load_templates_fast(template_paths)
        
        # Extract frames in parallel
        frames, frame_indices = self.extract_frames_parallel(str(video_path), max_frames=3000, num_threads=6)
        
        if not frames:
            return {"video_id": object_name, "detections": []}
        
        print(f"  Processing {len(frames)} frames in batches of {self.batch_size}...")
        
        all_detections = []
        
        # Process in true batches
        for i in tqdm(range(0, len(frames), self.batch_size), desc=f"{object_name}", leave=False):
            batch_frames = frames[i:i+self.batch_size]
            batch_indices = frame_indices[i:i+self.batch_size]
            
            # Preprocess batch
            batch_tensor, original_sizes = self.preprocess_batch(batch_frames)
            
            # Run inference on batch
            batch_results = self.inference_batch(batch_tensor, templates, confidence_threshold)
            
            # Process results
            for j, (result, frame_idx, orig_size) in enumerate(zip(batch_results, batch_indices, original_sizes)):
                if result is not None:
                    # Scale bbox to original size
                    bbox = result['bbox']
                    scale_x = orig_size[1] / 1024
                    scale_y = orig_size[0] / 576
                    
                    scaled_bbox = [
                        bbox[0] * scale_x,
                        bbox[1] * scale_y,
                        bbox[2] * scale_x,
                        bbox[3] * scale_y
                    ]
                    
                    # Clip to bounds
                    h, w = orig_size
                    x1 = max(0, min(w-1, int(scaled_bbox[0])))
                    y1 = max(0, min(h-1, int(scaled_bbox[1])))
                    x2 = max(x1+1, min(w, int(scaled_bbox[2])))
                    y2 = max(y1+1, min(h, int(scaled_bbox[3])))
                    
                    all_detections.append({
                        "frame": int(frame_idx),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })
        
        # Format result
        result = {"video_id": object_name, "detections": []}
        if all_detections:
            result["detections"] = [{"bboxes": all_detections}]
        
        return result


def main():
    """Ultra-fast main function."""
    
    # Configuration
    CHECKPOINT_PATH = "drive/MyDrive/ZALO2025/last_10e.pth"
    SAMPLES_DIR = "public_test/samples"  
    OUTPUT_PATH = "submission_output.json"
    CONFIDENCE_THRESHOLD = 0.2
    BATCH_SIZE = 64  # Large batch for maximum GPU usage
    
    print("=== ULTRA-FAST GPU INFERENCE ===")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Confidence: {CONFIDENCE_THRESHOLD}")
    
    # Initialize ultra-fast inference
    inference = UltraFastInference(
        checkpoint_path=CHECKPOINT_PATH,
        device='cuda',
        batch_size=BATCH_SIZE
    )
    
    # Get objects
    samples_dir = Path(SAMPLES_DIR)
    object_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    
    print(f"\nProcessing {len(object_dirs)} objects...")
    
    # Process all objects
    submission = []
    
    for i, obj_dir in enumerate(object_dirs, 1):
        print(f"\n[{i}/{len(object_dirs)}] {obj_dir.name}")
        
        result = inference.process_object_ultra_fast(obj_dir, CONFIDENCE_THRESHOLD)
        submission.append(result)
        
        # Summary
        det_count = sum(len(det.get('bboxes', [])) for det in result['detections'])
        print(f"  → {det_count} detections")
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(submission, f, indent=2)
    
    # Final summary
    total = sum(sum(len(det.get('bboxes', [])) for det in r['detections']) for r in submission)
    print(f"\n🚀 DONE! Total detections: {total}")
    print(f"📁 Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()