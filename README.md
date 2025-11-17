# ConvNeXt Reference Detection

Complete implementation of reference-based object detection using ConvNeXt-Tiny backbone.

## Architecture Overview

**Pipeline:**
1. **Template Encoder**: 3 template images → ConvNeXt-Tiny (shared) → FPN → {P2, P3}
2. **Template Fusion**: Attention-based aggregation of 3 templates at each scale
3. **Search Encoder**: Search image → ConvNeXt-Tiny (shared) → FPN → {P2, P3}
4. **Multi-scale Correlation**: Depthwise cross-correlation at P2 (stride 4) and P3 (stride 8)
5. **Feature Fusion**: Concatenate search features + correlations
6. **Detection Head**: Anchor-free detection (objectness + bbox regression)

**Model Stats:**
- Backbone: ConvNeXt-Tiny (~28M params)
- Total model: ~35-40M params
- Main detection stride: 4 (high resolution for small objects)

## Installation

```bash
# Install from requirements.txt in project root
cd /path/to/refdet
pip install -r requirements.txt

# Or install manually:
pip install torch>=2.0.0 torchvision>=0.15.0
pip install albumentations>=1.3.0
pip install opencv-python>=4.7.0 Pillow>=9.5.0
pip install matplotlib>=3.7.0 tqdm>=4.65.0
pip install timm>=0.9.0  # Optional: for pretrained weights
pip install wandb>=0.15.0  # Optional: for experiment tracking
```

## Quick Start

### 1. Test Model Architecture

```bash
cd convnext_refdet
python model.py
```

This will instantiate the model and print architecture details.

### 2. Training

**Basic Training:**
```bash
python train.py
```

**Configure Training Parameters:**
Edit `train.py` configuration dict (lines ~848+):

```python
config = {
    # Data paths and preprocessing
    'data_root': '/path/to/retrieval_dataset_flat_zoomed',
    'search_size': (1024, 576),      # Search image resolution
    'template_size': (256, 256),     # Template resolution
    
    # Model architecture
    'fpn_channels': 256,             # FPN feature channels (128/256/512)
    'corr_channels': 64,             # Correlation channels (32/64/128)
    'det_channels': 256,             # Detection head channels (128/256/512)
    'template_fusion': 'attention',  # 'attention' or 'average'
    
    # Loss functions
    'lambda_obj': 1.0,               # Objectness loss weight
    'lambda_bbox': 5.0,              # BBox regression loss weight
    'focal_alpha': 0.25,             # Focal loss alpha (0.25 recommended)
    'focal_gamma': 2.0,              # Focal loss gamma (2.0 recommended)
    'iou_type': 'giou',              # 'iou', 'giou', 'diou', or 'ciou'
    'pos_radius': 1,                 # Positive sample radius (0/1/2)
    
    # Training hyperparameters
    'batch_size': 4,                 # Batch size (adjust for GPU memory)
    'epochs': 100,                   # Training epochs
    'lr': 1e-4,                      # Initial learning rate
    'weight_decay': 1e-4,            # AdamW weight decay
    'grad_clip': 1.0,                # Gradient clipping threshold
    'num_workers': 4,                # DataLoader workers
    
    # Logging and checkpointing
    'use_wandb': False,              # Enable Weights & Biases logging
    'project_name': 'convnext-refdet',
    'checkpoint_dir': './checkpoints',
    'save_interval': 10              # Save checkpoint every N epochs
}
```

**Key Configuration Tips:**
- **IoU Loss Types:**
  - `'iou'`: Standard IoU loss
  - `'giou'`: Generalized IoU (default, good for general cases)
  - `'diou'`: Distance IoU (better for small objects)
  - `'ciou'`: Complete IoU (best for small objects & tight regression)
  
- **For Small Objects:** Use `'ciou'` or `'diou'` with higher `lambda_bbox` (5.0-10.0)
- **For Speed:** Reduce `fpn_channels` to 128, `corr_channels` to 32
- **For Accuracy:** Increase `fpn_channels` to 512, `det_channels` to 512

**Enable WandB Logging:**
```python
config['use_wandb'] = True
config['project_name'] = 'my-refdet-experiment'
```

### 3. Inference

**Command Line Interface:**
```bash
python inference.py \
    --checkpoint checkpoints/best.pth \
    --templates /path/to/templates/object_id \
    --search /path/to/search/image.jpg \
    --output ./output \
    --score-threshold 0.3 \
    --visualize
```

**Python API - Single Image:**
```python
from inference import RefDetInference

# Initialize inference engine
inferencer = RefDetInference(
    checkpoint_path='checkpoints/best.pth',
    device='cuda',                    # 'cuda' or 'cpu'
    score_threshold=0.3,              # Detection confidence threshold
    search_size=(1024, 576),          # Must match training size
    template_size=(256, 256)          # Must match training size
)

# Run inference
detections, original_size = inferencer.predict(
    template_paths=['t1.jpg', 't2.jpg', 't3.jpg'],  # 3 template images
    search_image_path='search.jpg'
)

# Process results
for det in detections:
    bbox = det['bbox']  # [x1, y1, x2, y2]
    score = det['score']
    print(f"Detection: {bbox} with confidence {score:.3f}")
```

**Batch Inference on Test Set:**
```python
# Predict on multiple images with same templates
results = inferencer.predict_batch(
    template_paths='templates/object_id/',  # Directory with 3 templates
    search_image_paths=['img1.jpg', 'img2.jpg', 'img3.jpg']
)

# Evaluate on test set with ground truth labels
metrics = inferencer.evaluate_on_testset(
    test_data_root='/path/to/test/data',
    save_results=True,
    output_dir='./eval_results'
)
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"mAP: {metrics['mAP']:.3f}")
```

### 4. Model Export (Optional)

**Export to TorchScript for Deployment:**
```python
import torch
from model import build_convnext_refdet

# Load trained model
model = build_convnext_refdet()
checkpoint = torch.load('checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create example inputs
templates = torch.randn(1, 3, 3, 256, 256)  # [B, 3, C, H, W]
search = torch.randn(1, 3, 1024, 576)       # [B, C, H, W]

# Export to TorchScript
traced_model = torch.jit.trace(model, (templates, search))
traced_model.save('model_traced.pt')
print("Model exported to model_traced.pt")
```

**Export to ONNX:**
```python
torch.onnx.export(
    model,
    (templates, search),
    'model.onnx',
    input_names=['templates', 'search'],
    output_names=['objectness', 'bbox'],
    dynamic_axes={
        'templates': {0: 'batch'},
        'search': {0: 'batch'},
        'objectness': {0: 'batch'},
        'bbox': {0: 'batch'}
    },
    opset_version=14
)
print("Model exported to model.onnx")
```

## File Structure

```
convnext_refdet/
├── model.py           # Complete model implementation
│   ├── ConvNeXtTiny   # Backbone (~28M params)
│   ├── FPN            # Feature Pyramid Network
│   ├── TemplateFusion # Multi-template aggregation
│   ├── DepthwiseCorrelation  # Template-search matching
│   ├── DetectionHead  # Anchor-free detection
│   └── ConvNeXtRefDet # Complete model
│
├── train.py           # Training script
│   ├── Loss functions (Focal + IoU)
│   ├── Dataset loader
│   ├── Training loop
│   └── Validation
│
├── inference.py       # Inference & evaluation
│   ├── Single/batch inference
│   ├── Visualization
│   └── Metrics computation
│
├── utils.py           # Utilities
│   ├── Checkpoint loading/saving
│   ├── BBox processing
│   ├── NMS
│   └── Evaluation helpers
│
└── README.md          # This file
```

## Data Format

Expected directory structure:

```
data_root/
├── templates/
│   ├── object_1/
│   │   ├── template_0.jpg
│   │   ├── template_1.jpg
│   │   └── template_2.jpg
│   └── object_2/
│       └── ...
│
├── train/
│   └── search/
│       ├── images/
│       │   ├── object_1_frame_0001.jpg
│       │   └── ...
│       └── labels/
│           ├── object_1_frame_0001.txt  # Format: x1 y1 x2 y2
│           └── ...
│
└── val/
    └── search/
        ├── images/
        └── labels/
```

## Key Features

### Recent Optimizations

**1. Improved Focal Loss Initialization:**
- Detection head objectness bias initialized to -4.6
- Initial prediction ~1% instead of 50% (sigmoid(-4.6) ≈ 0.01)
- Prevents model from predicting all positives at start
- Standard practice in RetinaNet, FCOS, YOLOX
- Improves training stability and convergence

**2. Advanced IoU Loss Options:**
- **IoU**: Standard Intersection over Union
- **GIoU**: Generalized IoU - adds penalty for non-overlapping area (default)
- **DIoU**: Distance IoU - considers center point distance
- **CIoU**: Complete IoU - adds aspect ratio consistency (best for small objects)
- Use `config['iou_type'] = 'ciou'` for improved small object detection

**3. Bbox Coordinate Clamping:**
- Predicted bboxes automatically clamped to image boundaries
- Prevents visualization crashes and invalid coordinates
- Applied during inference in `decode_predictions()`
- Ensures x1, y1, x2, y2 ∈ [0, image_size-1]

### Multi-Scale Correlation
- P2 (stride 4): High resolution for small objects
- P3 (stride 8): Larger context
- Both correlations fused for robust detection

### Anchor-Free Detection
- No anchor design needed
- Direct prediction of object center and size
- Suitable for varying object scales

### Template Fusion
- Attention-based aggregation of 3 templates
- Learns to weight different views
- Robust to viewpoint changes

### Loss Function
- Focal Loss for objectness (handles class imbalance)
- GIoU Loss for bbox regression (geometry-aware)
- Balanced with configurable weights

## Training Tips

1. **Start with pretrained backbone** (if available):
   ```python
   from utils import load_pretrained_convnext
   model = load_pretrained_convnext(model)
   ```

2. **Two-stage training**:
   - Stage 1: Freeze backbone, train detection heads (10-20 epochs)
   - Stage 2: Unfreeze all, fine-tune end-to-end (50-100 epochs)

3. **Learning rate**:
   - Initial: 1e-4
   - Cosine annealing to 1e-6
   - Lower LR for backbone if using pretrained weights

4. **Data augmentation**:
   - Horizontal flip, brightness/contrast
   - Avoid heavy augmentation on small objects

5. **Loss weights**:
   - Start with lambda_obj=1.0, lambda_bbox=5.0
   - Adjust based on validation performance

## Evaluation

Metrics computed:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall
- **Average IoU**: Mean IoU of matched detections
- **AP**: Average Precision (precision-recall curve)

IoU threshold: 0.5 (configurable)

## Inference Examples

### Single Image
```python
from inference import RefDetInference

inferencer = RefDetInference(
    checkpoint_path='checkpoints/best.pth',
    score_threshold=0.3
)

detections, size = inferencer.predict(
    template_paths=['t1.jpg', 't2.jpg', 't3.jpg'],
    search_image_path='search.jpg'
)

for det in detections:
    print(f"BBox: {det['bbox']}, Score: {det['score']:.3f}")
```

### Batch Processing
```python
results = inferencer.predict_batch(
    template_paths='templates/object_id/',
    search_image_paths=['img1.jpg', 'img2.jpg', ...]
)
```

## Customization

### Change Backbone
Edit `model.py` to use different ConvNeXt variants:
- Tiny: [3, 3, 9, 3], [96, 192, 384, 768] (~28M)
- Small: [3, 3, 27, 3], [96, 192, 384, 768] (~50M)
- Base: [3, 3, 27, 3], [128, 256, 512, 1024] (~88M)

### Adjust FPN Channels
```python
model = build_convnext_refdet(
    fpn_channels=128,  # Lower = faster, less accurate
    corr_channels=32,
    det_channels=128
)
```

### Change Detection Stride
To use P3 (stride 8) as main detection:
1. Modify `forward()` in `ConvNeXtRefDet`
2. Change `self.stride = 8`
3. Adjust fusion to use P3 features

## Performance Tips

### Speed Optimization
- Use smaller `fpn_channels` (128 instead of 256)
- Reduce `corr_channels` (32 instead of 64)
- Use FP16 mixed precision training
- Reduce search image resolution

### Accuracy Optimization
- Use larger `fpn_channels` (256 or 512)
- Add more detection head convolutions
- Use ensemble of multiple checkpoints
- Test-time augmentation (flip, multi-scale)

## Troubleshooting

**No detections:**
- Lower `score_threshold`
- Check if templates match search objects
- Verify data format and labels

**Poor localization:**
- Increase `lambda_bbox` weight
- Use DIoU or CIoU instead of GIoU
- Add more training data

**Overfitting:**
- Increase data augmentation
- Add dropout in detection head
- Reduce model capacity

**Out of memory:**
- Reduce batch size
- Reduce input resolution
- Use gradient accumulation
- Enable FP16 training

## Citation

If you use this code, please cite:
```
ConvNeXt Reference Detection
Built with ConvNeXt-Tiny backbone
```

## License

MIT License
