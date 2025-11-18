"""
Simple Public Test Inference Runner

Usage:
    python run_public_test.py

This script will:
1. Load the trained model from checkpoint
2. Process each object directory in public_test/samples/
3. Extract frames from drone_video.mp4
4. Run inference with 3 template images
5. Generate submission.json in required format
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

# Add convnext_refdet to path
sys.path.append('convnext_refdet')
from inference import RefDetInference


def extract_frames_fast(video_path, max_frames=1000):
    """Extract frames from video efficiently."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return [], []
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {total_frames} frames, {fps:.1f} FPS")
    
    # Decide which frames to extract
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        # Sample uniformly
        frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int).tolist()
    
    frames = []
    valid_indices = []
    
    for frame_idx in tqdm(frame_indices, desc="Extracting frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frames.append(frame)
            valid_indices.append(frame_idx)
        else:
            print(f"Failed to read frame {frame_idx}")
    
    cap.release()
    return frames, valid_indices


def run_inference_on_object(object_dir, model, confidence_threshold=0.3):
    """Process one object directory and return detections."""
    
    object_name = object_dir.name
    print(f"\n=== Processing {object_name} ===")
    
    # Check paths
    video_path = object_dir / "drone_video.mp4"
    template_dir = object_dir / "object_images"
    
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return {"video_id": object_name, "detections": []}
    
    if not template_dir.exists():
        print(f"Templates not found: {template_dir}")
        return {"video_id": object_name, "detections": []}
    
    # Get templates
    template_files = sorted(list(template_dir.glob("*.jpg")) + list(template_dir.glob("*.png")))
    
    if len(template_files) < 3:
        print(f"Need 3 templates, found {len(template_files)}")
        return {"video_id": object_name, "detections": []}
    
    template_paths = [str(f) for f in template_files[:3]]
    print(f"Using templates: {[f.name for f in template_files[:3]]}")
    
    # Extract frames
    print("Extracting video frames...")
    frames, frame_indices = extract_frames_fast(str(video_path), max_frames=2000)
    
    if not frames:
        print("No frames extracted")
        return {"video_id": object_name, "detections": []}
    
    print(f"Processing {len(frames)} frames...")
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    all_detections = []
    
    try:
        # Save frames and run inference
        for i, (frame, frame_idx) in enumerate(tqdm(zip(frames, frame_indices), 
                                                   desc="Inference", 
                                                   total=len(frames))):
            
            # Save frame temporarily
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            try:
                # Run inference
                detections, original_size = model.predict(
                    template_paths=template_paths,
                    search_image_path=frame_path,
                    top_k=1
                )
                
                # Process detections
                for det in detections:
                    if det['score'] >= confidence_threshold:
                        bbox = det['bbox']
                        
                        # Convert to integers and clip to image bounds
                        h, w = original_size
                        x1 = max(0, min(w-1, int(bbox[0])))
                        y1 = max(0, min(h-1, int(bbox[1])))
                        x2 = max(x1+1, min(w, int(bbox[2])))
                        y2 = max(y1+1, min(h, int(bbox[3])))
                        
                        all_detections.append({
                            "frame": int(frame_idx),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        })
                        
                        print(f"Frame {frame_idx}: detected at ({x1},{y1},{x2},{y2}), score={det['score']:.3f}")
                        
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue
            
            # Clean up frame file
            os.unlink(frame_path)
    
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"Found {len(all_detections)} detections")
    
    # Format result
    result = {
        "video_id": object_name,
        "detections": []
    }
    
    if all_detections:
        result["detections"] = [{"bboxes": all_detections}]
    
    return result


def main():
    """Main function."""
    
    # Configuration
    CHECKPOINT_PATH = "/home/ta-thai-24/Workspace/hienhq/refdet/last_9e.pth"
    SAMPLES_DIR = "/home/ta-thai-24/Workspace/hienhq/refdet/public_test/samples"  
    OUTPUT_PATH = "/home/ta-thai-24/Workspace/hienhq/refdet/submission_output.json"
    CONFIDENCE_THRESHOLD = 0.3
    
    print("=== Public Test Inference ===")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Samples: {SAMPLES_DIR}")
    print(f"Output: {OUTPUT_PATH}")
    
    # Validate paths
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        return
    
    if not os.path.exists(SAMPLES_DIR):
        print(f"ERROR: Samples directory not found: {SAMPLES_DIR}")
        return
    
    # Load model
    print("\nLoading model...")
    model = RefDetInference(
        checkpoint_path=CHECKPOINT_PATH,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        score_threshold=CONFIDENCE_THRESHOLD,
        search_size=(1024, 576),
        template_size=(256, 256)
    )
    
    # Get object directories
    samples_dir = Path(SAMPLES_DIR)
    object_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    
    print(f"\nFound objects: {[d.name for d in object_dirs]}")
    
    # Process each object
    submission = []
    
    for obj_dir in object_dirs:
        result = run_inference_on_object(obj_dir, model, CONFIDENCE_THRESHOLD)
        submission.append(result)
    
    # Save results
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(submission, f, indent=2)
    
    # Summary
    print("\n=== SUMMARY ===")
    total_detections = 0
    for result in submission:
        count = sum(len(det.get('bboxes', [])) for det in result['detections'])
        total_detections += count
        print(f"{result['video_id']}: {count} detections")
    
    print(f"Total detections: {total_detections}")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()