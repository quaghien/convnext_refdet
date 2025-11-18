"""
GPU Inference — Single Frame Version
- Model xử lý từng frame 1 (batch_size = 1 thật)
- Không ghép batch vào GPU
- tqdm theo từng object
- Cực kỳ ổn định
"""

import os
import sys
import cv2
import gc
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add model path
sys.path.append("convnext_refdet")
from inference import RefDetInference


# ===================================================================
# Inference Engine
# ===================================================================

class SingleFrameRunner:
    def __init__(self, ckpt_path, device="cuda", conf_thresh=0.2):
        self.device = device
        self.conf_thresh = conf_thresh

        print("Loading model...")
        self.model = RefDetInference(
            checkpoint_path=ckpt_path,
            device=device,
            score_threshold=0.05,
            search_size=(1024, 576),
            template_size=(256, 256)
        )
        print("Model loaded on", device)

    # ---------------------------------------------------------------
    def load_video_frames(self, video_path, max_frames=3000):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ids = (
            list(range(total))
            if total <= max_frames else
            np.linspace(0, total - 1, max_frames, dtype=int).tolist()
        )

        frames, valid_ids = [], []
        for fid in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
                valid_ids.append(fid)

        cap.release()
        return frames, valid_ids

    # ---------------------------------------------------------------
    def infer_single_frame(self, frame, frame_id, templates):
        """
        Inference chỉ 1 frame (batch size = 1)
        """
        # Preprocess → tensor shape [3, H', W']
        tensor, (orig_h, orig_w) = self.model.preprocess_image(frame, is_template=False)
        tensor = tensor.unsqueeze(0).to(self.device)        # thành [1, 3, H', W']

        with torch.no_grad():
            obj, bbox = self.model.model(templates, tensor)
            preds = self.model.model.decode_predictions(
                obj, bbox, score_threshold=0.05, top_k=10
            )[0]  # 1 frame → lấy index 0

        # chọn detection tốt nhất
        best = None
        best_score = 0

        for det in preds:
            if det["score"] >= self.conf_thresh and det["score"] > best_score:
                best = det
                best_score = det["score"]

        if best is None:
            return None

        # scale về kích thước gốc
        sx = orig_w / self.model.search_size[0]
        sy = orig_h / self.model.search_size[1]
        x1, y1, x2, y2 = best["bbox"]

        x1 = int(max(0, min(orig_w - 1, x1 * sx)))
        y1 = int(max(0, min(orig_h - 1, y1 * sy)))
        x2 = int(max(x1 + 1, min(orig_w, x2 * sx)))
        y2 = int(max(y1 + 1, min(orig_h, y2 * sy)))

        return {
            "frame": frame_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }

    # ---------------------------------------------------------------
    def process_object(self, folder):
        name = folder.name
        video_path = folder / "drone_video.mp4"
        templ_dir = folder / "object_images"

        if not video_path.exists() or not templ_dir.exists():
            return {"video_id": name, "detections": []}

        temp_files = list(templ_dir.glob("*.jpg")) + list(templ_dir.glob("*.png"))
        temp_files = sorted(temp_files)

        if len(temp_files) < 3:
            return {"video_id": name, "detections": []}

        templates = self.model.load_templates([str(f) for f in temp_files[:3]])

        frames, ids = self.load_video_frames(str(video_path))
        if len(frames) == 0:
            return {"video_id": name, "detections": []}

        detections = []

        pbar = tqdm(total=len(frames), desc=name, leave=True)

        # === xử lý từng frame 1 ===
        for frame, fid in zip(frames, ids):
            det = self.infer_single_frame(frame, fid, templates)
            if det is not None:
                detections.append(det)

            pbar.update(1)

        pbar.close()

        # Clean GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return {
            "video_id": name,
            "detections": [{"bboxes": detections}] if detections else []
        }


# ===================================================================
# MAIN
# ===================================================================

def main():
    CKPT = "drive/MyDrive/ZALO2025/best.pth"
    ROOT = "public_test/samples"
    OUT = "submission_output.json"
    CONF = 0.2

    print("=== Single-Frame Inference ===")
    print("Checkpoint:", CKPT)
    print("Samples:", ROOT)

    runner = SingleFrameRunner(CKPT, device="cuda", conf_thresh=CONF)

    folders = sorted([d for d in Path(ROOT).iterdir() if d.is_dir()])
    print(f"\nFound {len(folders)} objects:", [f.name for f in folders])

    results = []

    for i, folder in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] Processing {folder.name}...")
        res = runner.process_object(folder)
        results.append(res)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", OUT)
    print("Inference done ✔")


if __name__ == "__main__":
    main()
