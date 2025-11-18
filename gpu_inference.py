"""
Optimized GPU Inference for Public Test

Features:
- Maximum GPU utilization with batch processing
- Silent detection (no prints during inference)
- Only highest confidence bbox per frame (0 or 1 bbox per frame)
- Process all videos in one run
- Efficient memory management
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

# Add convnext_refdet to path
sys.path.append('convnext_refdet')
from inference import RefDetInference


class OptimizedInference:
    """Optimized inference with batch processing and GPU maximization."""
    
    def __init__(self, checkpoint_path, device='cuda', confidence_threshold=0.3):
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Load model
        print("Loading model...")
        self.model = RefDetInference(
            checkpoint_path=checkpoint_path,
            device=device,
            score_threshold=0.1,  # Lower threshold, we'll filter later
            search_size=(1024, 576),
            template_size=(256, 256)
        )
        print(f"Model loaded on {device}")
    
    def extract_all_frames_batch(self, video_path, max_frames=2000):
        """Extract frames efficiently with OpenCV."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return [], []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine frames to extract
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
    
    def process_frames_batch(self, frames, frame_indices, template_paths, batch_size=8):
        """Process frames in batches for GPU efficiency."""
        
        # Load templates once
        templates = self.model.load_templates(template_paths)
        
        all_detections = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Process in batches
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i+batch_size]
                batch_indices = frame_indices[i:i+batch_size]
                
                # Save batch frames
                frame_paths = []
                for j, frame in enumerate(batch_frames):
                    frame_path = os.path.join(temp_dir, f"batch_{i}_frame_{j}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                
                # Process batch
                batch_detections = self.process_batch_silent(
                    frame_paths, batch_indices, templates
                )
                
                all_detections.extend(batch_detections)
                
                # Cleanup batch files
                for fp in frame_paths:
                    os.unlink(fp)
                
                # Clear GPU cache
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return all_detections
    
    def process_batch_silent(self, frame_paths, frame_indices, templates):
        """Process batch of frames silently (no prints)."""
        detections = []
        
        for frame_path, frame_idx in zip(frame_paths, frame_indices):
            try:
                # Load and preprocess frame
                search_tensor, original_size = self.model.preprocess_image(frame_path, is_template=False)
                search_tensor = search_tensor.unsqueeze(0).to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    obj_map, bbox_map = self.model.model(templates, search_tensor)
                    
                    # Decode predictions
                    frame_detections = self.model.model.decode_predictions(
                        obj_map, bbox_map,
                        score_threshold=0.05,  # Very low threshold
                        top_k=10  # Get multiple candidates
                    )[0]
                
                # Filter and select best detection
                best_detection = None
                best_score = 0.0
                
                for det in frame_detections:
                    if det['score'] > best_score and det['score'] >= self.confidence_threshold:
                        best_score = det['score']
                        best_detection = det
                
                # If we have a valid detection, scale and add
                if best_detection is not None:
                    # Scale bbox back to original size
                    scale_x = original_size[1] / self.model.search_size[0]
                    scale_y = original_size[0] / self.model.search_size[1]
                    
                    bbox = best_detection['bbox']
                    scaled_bbox = [
                        bbox[0] * scale_x,
                        bbox[1] * scale_y,
                        bbox[2] * scale_x,
                        bbox[3] * scale_y
                    ]
                    
                    # Clip to image bounds
                    h, w = original_size
                    x1 = max(0, min(w-1, int(scaled_bbox[0])))
                    y1 = max(0, min(h-1, int(scaled_bbox[1])))
                    x2 = max(x1+1, min(w, int(scaled_bbox[2])))
                    y2 = max(y1+1, min(h, int(scaled_bbox[3])))
                    
                    detections.append({
                        "frame": int(frame_idx),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": float(best_score)
                    })
                
            except Exception:
                # Silent failure - skip frame
                continue
        
        return detections
    
    def process_object(self, object_dir, batch_size=8):
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
        
        template_paths = [str(f) for f in template_files[:3]]
        
        # Extract frames
        frames, frame_indices = self.extract_all_frames_batch(str(video_path), max_frames=3000)
        
        if not frames:
            return {"video_id": object_name, "detections": []}
        
        # Process frames in batches
        all_detections = self.process_frames_batch(
            frames, frame_indices, template_paths, batch_size
        )
        
        # Format result (remove confidence from output)
        formatted_detections = []
        for det in all_detections:
            formatted_detections.append({
                "frame": det["frame"],
                "x1": det["x1"],
                "y1": det["y1"],
                "x2": det["x2"],
                "y2": det["y2"]
            })
        
        result = {
            "video_id": object_name,
            "detections": []
        }
        
        if formatted_detections:
            result["detections"] = [{"bboxes": formatted_detections}]
        
        return result


def main():
    """Main optimized inference."""
    
    # Configuration
    CHECKPOINT_PATH = "drive/MyDrive/ZALO2025/last_10e.pth"
    SAMPLES_DIR = "public_test/samples"  
    OUTPUT_PATH = "submission_output.json"
    CONFIDENCE_THRESHOLD = 0.1
    BATCH_SIZE = 32  # Process multiple frames at once
    
    print("=== Optimized GPU Inference ===")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Samples: {SAMPLES_DIR}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Validate paths
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        return
    
    if not os.path.exists(SAMPLES_DIR):
        print(f"ERROR: Samples directory not found: {SAMPLES_DIR}")
        return
    
    # Initialize optimized inference
    inference = OptimizedInference(
        checkpoint_path=CHECKPOINT_PATH,
        device='cuda',
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    # Get all object directories
    samples_dir = Path(SAMPLES_DIR)
    object_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    
    print(f"\nProcessing {len(object_dirs)} objects: {[d.name for d in object_dirs]}")
    
    # Process all objects
    submission = []
    
    for i, obj_dir in enumerate(object_dirs, 1):
        print(f"\n[{i}/{len(object_dirs)}] Processing {obj_dir.name}...")
        
        result = inference.process_object(obj_dir, batch_size=BATCH_SIZE)
        submission.append(result)
        
        # Print summary for this object
        det_count = sum(len(det.get('bboxes', [])) for det in result['detections'])
        print(f"  → {det_count} detections found")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Save final results
    print(f"\nSaving results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(submission, f, indent=2)
    
    # Final summary
    print("\n=== FINAL SUMMARY ===")
    total_detections = 0
    for result in submission:
        count = sum(len(det.get('bboxes', [])) for det in result['detections'])
        total_detections += count
        print(f"{result['video_id']}: {count} detections")
    
    print(f"\nTotal detections: {total_detections}")
    print(f"Output saved: {OUTPUT_PATH}")
    print("✅ Inference completed!")


if __name__ == "__main__":
    main()