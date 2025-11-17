"""
Complete ConvNeXt-Tiny based Reference Detection Model
- Siamese encoder with shared ConvNeXt-Tiny backbone (~28M params)
- Feature Pyramid Network (FPN) extracting P2 (stride 4) and P3 (stride 8)
- Multi-scale correlation between templates and search
- Anchor-free detection head on stride-4 features
- Total params: ~35-40M (backbone ~28M + FPN + heads)

Architecture:
1. Template Encoder: 3 templates → ConvNeXt-Tiny → FPN → {P2, P3} each
2. Template Fusion: Aggregate 3 templates at each scale
3. Search Encoder: Search image → ConvNeXt-Tiny (shared) → FPN → {P2, P3}
4. Correlation: Depthwise cross-correlation at P2 and P3
5. Feature Fusion: Concat search + corr2 + upsampled_corr3
6. Detection Head: Conv layers → objectness + bbox regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


# ============================================================================
# ConvNeXt Building Blocks
# ============================================================================

class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The channels_first format is needed for Conv2D inputs.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block.
    
    There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    
    We use (1) as it's more efficient.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma[:, None, None] * x
        
        x = input_x + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny backbone
    
    Architecture: [3, 3, 9, 3] blocks with [96, 192, 384, 768] channels
    Total params: ~28M
    
    We extract features at multiple stages for FPN:
    - Stage 1 (C1): stride 4, 96 channels
    - Stage 2 (C2): stride 8, 192 channels
    - Stage 3 (C3): stride 16, 384 channels
    - Stage 4 (C4): stride 32, 768 channels
    """
    def __init__(self, 
                 in_chans=3,
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6):
        super().__init__()
        
        self.depths = depths
        self.dims = dims
        
        # Stem: conv 4x4 with stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # Stochastic depth decay rule
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # 4 stages
        self.stages = nn.ModuleList()
        cur_dp = 0
        for i in range(4):
            # Downsample layer (except first stage)
            if i > 0:
                downsample = nn.Sequential(
                    LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2)
                )
            else:
                downsample = nn.Identity()
            
            # Stack of ConvNeXt blocks
            blocks = []
            for j in range(depths[i]):
                blocks.append(ConvNeXtBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur_dp + j],
                    layer_scale_init_value=layer_scale_init_value
                ))
            
            stage = nn.Sequential(downsample, *blocks)
            self.stages.append(stage)
            cur_dp += depths[i]
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            List of features [C1, C2, C3, C4]
            C1: [B, 96, H/4, W/4]
            C2: [B, 192, H/8, W/8]
            C3: [B, 384, H/16, W/16]
            C4: [B, 768, H/32, W/32]
        """
        x = self.stem(x)  # stride 4
        
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
        
        return features


# ============================================================================
# Feature Pyramid Network (FPN)
# ============================================================================

class FPN(nn.Module):
    """Feature Pyramid Network
    
    Takes backbone features [C1, C2, C3, C4] and produces [P2, P3, P4, P5]
    For this task, we mainly use P2 (stride 4) and P3 (stride 8)
    
    Args:
        in_channels_list: list of input channels [96, 192, 384, 768] for ConvNeXt-Tiny
        out_channels: output channels for all pyramid levels (default: 256)
    """
    def __init__(self, in_channels_list=[96, 192, 384, 768], out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        
        # Lateral convs (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        
        # Output convs (3x3 conv to reduce aliasing)
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
    
    def forward(self, features):
        """
        Args:
            features: List of [C1, C2, C3, C4]
        Returns:
            List of [P2, P3, P4, P5] with same spatial sizes as inputs
            All with out_channels channels
        """
        # Build laterals
        laterals = []
        for i, (feat, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            laterals.append(lateral_conv(feat))
        
        # Top-down pathway with lateral connections
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample and add
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], 
                size=laterals[i-1].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Apply output convolutions
        outputs = []
        for lateral, fpn_conv in zip(laterals, self.fpn_convs):
            outputs.append(fpn_conv(lateral))
        
        return outputs


# ============================================================================
# Template Fusion
# ============================================================================

class TemplateFusion(nn.Module):
    """Fuse multiple template features at the same scale.
    
    Supports two fusion strategies:
    1. Average: Simple averaging
    2. Attention: Learnable attention weights per template
    """
    def __init__(self, channels, num_templates=3, fusion_type='attention'):
        super().__init__()
        self.channels = channels
        self.num_templates = num_templates
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            # Compute attention weights from concatenated features
            self.attention_conv = nn.Sequential(
                nn.Conv2d(channels * num_templates, channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, num_templates, 1)
            )
        elif fusion_type == 'average':
            pass
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def forward(self, template_features):
        """
        Args:
            template_features: List of template features [B, C, H, W] * num_templates
        Returns:
            fused: [B, C, H, W]
        """
        if self.fusion_type == 'average':
            # Simple average
            stacked = torch.stack(template_features, dim=0)  # [N, B, C, H, W]
            fused = stacked.mean(dim=0)  # [B, C, H, W]
        
        elif self.fusion_type == 'attention':
            # Attention-weighted fusion
            concat = torch.cat(template_features, dim=1)  # [B, N*C, H, W]
            attn_logits = self.attention_conv(concat)  # [B, N, H, W]
            attn_weights = F.softmax(attn_logits, dim=1)  # [B, N, H, W]
            
            # Weighted sum
            stacked = torch.stack(template_features, dim=1)  # [B, N, C, H, W]
            attn_weights = attn_weights.unsqueeze(2)  # [B, N, 1, H, W]
            fused = (stacked * attn_weights).sum(dim=1)  # [B, C, H, W]
        
        return fused


# ============================================================================
# Correlation Module
# ============================================================================

class DepthwiseCorrelation(nn.Module):
    """Depthwise cross-correlation between template and search features.
    
    Implements true spatial cross-correlation by using template features as
    depthwise convolution kernels on search features.
    More accurate pattern matching compared to global pooling.
    """
    def __init__(self, channels, corr_channels=64):
        super().__init__()
        self.channels = channels
        self.corr_channels = corr_channels
        
        # Project to lower dimension for efficiency
        self.template_proj = nn.Sequential(
            nn.Conv2d(channels, corr_channels, 1, bias=False),
            nn.GroupNorm(num_groups=min(32, corr_channels), num_channels=corr_channels),
            nn.ReLU(inplace=True)
        )
        
        self.search_proj = nn.Sequential(
            nn.Conv2d(channels, corr_channels, 1, bias=False),
            nn.GroupNorm(num_groups=min(32, corr_channels), num_channels=corr_channels),
            nn.ReLU(inplace=True)
        )
        
        # Post-correlation processing
        self.post_corr = nn.Sequential(
            nn.Conv2d(corr_channels, corr_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, corr_channels), num_channels=corr_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(corr_channels, 1, 1)  # Output single channel similarity map
        )
    
    def forward(self, template_feat, search_feat):
        """
        Args:
            template_feat: [B, C, H_t, W_t] - fused template features
            search_feat: [B, C, H_s, W_s] - search features
        Returns:
            correlation: [B, 1, H_s, W_s] - similarity map
        """
        B = search_feat.size(0)
        
        # Project features
        template_proj = self.template_proj(template_feat)  # [B, corr_C, H_t, W_t]
        search_proj = self.search_proj(search_feat)        # [B, corr_C, H_s, W_s]
        
        # Method 1: Global + Local correlation (hybrid approach)
        # Global pooling on template to get representative vector
        template_global = F.adaptive_avg_pool2d(template_proj, (1, 1))  # [B, corr_C, 1, 1]
        global_corr = search_proj * template_global  # [B, corr_C, H_s, W_s]
        
        # Local correlation: downsample template and compute patch-wise correlation
        # Adaptively resize template to reasonable kernel size (e.g., 3x3 or 5x5)
        kernel_size = min(7, min(template_proj.shape[2], template_proj.shape[3]))
        if kernel_size >= 3:
            template_kernel = F.adaptive_avg_pool2d(template_proj, (kernel_size, kernel_size))
            # Apply depthwise correlation for each sample in batch
            local_corr_list = []
            for i in range(B):
                # Use template as depthwise conv kernel: [corr_C, 1, K, K]
                kernel = template_kernel[i:i+1].view(self.corr_channels, 1, kernel_size, kernel_size)
                search_single = search_proj[i:i+1]  # [1, corr_C, H_s, W_s]
                # Depthwise convolution
                corr = F.conv2d(search_single, kernel, padding=kernel_size//2, groups=self.corr_channels)
                local_corr_list.append(corr)
            local_corr = torch.cat(local_corr_list, dim=0)  # [B, corr_C, H_s, W_s]
            
            # Combine global and local correlations
            correlation = global_corr + local_corr
        else:
            # Fallback to global only if template too small
            correlation = global_corr
        
        # Post-process to get single channel similarity
        correlation = self.post_corr(correlation)  # [B, 1, H_s, W_s]
        
        return correlation


# ============================================================================
# Detection Head
# ============================================================================

class DetectionHead(nn.Module):
    """Anchor-free detection head.
    
    Predicts:
    - Objectness map: [B, 1, H, W] - probability of object center at each location
    - BBox regression: [B, 4, H, W] - (tx, ty, tw, th) at each location
    
    The bbox values are:
    - tx, ty: offset logits (will be passed through tanh during decode)
    - tw, th: size logits (will be passed through softplus during decode)
    
    Constraints applied during decode:
    - cx_offset = tanh(tx) * stride  → bounded offset within ~1 cell
    - cy_offset = tanh(ty) * stride  → bounded offset within ~1 cell  
    - w = softplus(tw)  → ensures w > 0
    - h = softplus(th)  → ensures h > 0
    """
    def __init__(self, in_channels, feat_channels=256, num_convs=4):
        super().__init__()
        
        # Shared feature extraction
        shared_layers = []
        for i in range(num_convs):
            shared_layers.extend([
                nn.Conv2d(in_channels if i == 0 else feat_channels,
                         feat_channels, 3, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=feat_channels),
                nn.ReLU(inplace=True)
            ])
        self.shared_conv = nn.Sequential(*shared_layers)
        
        # Objectness head
        self.obj_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 1, 1)
        )
        
        # BBox regression head
        self.bbox_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Set objectness head bias to -4.6 for focal loss stability
        # This ensures initial objectness prediction ~1% (sigmoid(-4.6) ≈ 0.01)
        # Prevents model from predicting all 1s at initialization
        # Standard practice in RetinaNet, FCOS, YOLOX
        if hasattr(self.obj_head[-1], 'bias') and self.obj_head[-1].bias is not None:
            nn.init.constant_(self.obj_head[-1].bias, -4.6)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            obj_map: [B, 1, H, W] - objectness scores (logits)
            bbox_map: [B, 4, H, W] - bbox predictions (tx, ty, tw, th logits)
        """
        shared_feat = self.shared_conv(x)
        
        obj_map = self.obj_head(shared_feat)  # [B, 1, H, W]
        bbox_map = self.bbox_head(shared_feat)  # [B, 4, H, W]
        
        return obj_map, bbox_map


# ============================================================================
# Complete Model
# ============================================================================

class ConvNeXtRefDet(nn.Module):
    """Complete Reference Detection Model with ConvNeXt-Tiny backbone.
    
    Pipeline:
    1. Extract features from 3 templates using shared ConvNeXt + FPN
    2. Fuse template features at each scale
    3. Extract features from search image using shared ConvNeXt + FPN
    4. Compute correlations at P2 (stride 4) and P3 (stride 8)
    5. Fuse correlations with search features
    6. Predict objectness and bbox on stride-4 features
    """
    def __init__(self,
                 fpn_channels=256,
                 corr_channels=64,
                 det_channels=256,
                 template_fusion='attention'):
        super().__init__()
        
        # Shared backbone
        self.backbone = ConvNeXtTiny(
            in_chans=3,
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            drop_path_rate=0.1
        )
        
        # FPN
        self.fpn = FPN(
            in_channels_list=[96, 192, 384, 768],
            out_channels=fpn_channels
        )
        
        # Template fusion modules (one for each scale we use)
        self.template_fusion_p2 = TemplateFusion(
            channels=fpn_channels,
            num_templates=3,
            fusion_type=template_fusion
        )
        self.template_fusion_p3 = TemplateFusion(
            channels=fpn_channels,
            num_templates=3,
            fusion_type=template_fusion
        )
        
        # Correlation modules
        self.correlation_p2 = DepthwiseCorrelation(
            channels=fpn_channels,
            corr_channels=corr_channels
        )
        self.correlation_p3 = DepthwiseCorrelation(
            channels=fpn_channels,
            corr_channels=corr_channels
        )
        
        # Feature fusion conv (concat search_P2 + corr2 + corr3_upsampled)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fpn_channels + 2, det_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=det_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(det_channels, det_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=det_channels),
            nn.ReLU(inplace=True)
        )
        
        # Detection head
        self.detection_head = DetectionHead(
            in_channels=det_channels,
            feat_channels=256,
            num_convs=4
        )
        
        self.fpn_channels = fpn_channels
        self.stride = 4  # Main detection stride
    
    def extract_features(self, x):
        """Extract backbone + FPN features.
        
        Args:
            x: [B, 3, H, W]
        Returns:
            fpn_features: List of [P2, P3, P4, P5]
        """
        backbone_features = self.backbone(x)  # [C1, C2, C3, C4]
        fpn_features = self.fpn(backbone_features)  # [P2, P3, P4, P5]
        return fpn_features
    
    def forward(self, templates, search):
        """
        Args:
            templates: List of 3 template images, each [B, 3, H_t, W_t]
            search: Search image [B, 3, H_s, W_s] (typically 1024x576)
        Returns:
            obj_map: [B, 1, H_s/4, W_s/4] - objectness scores
            bbox_map: [B, 4, H_s/4, W_s/4] - bbox predictions
        """
        assert len(templates) == 3, "Expecting 3 templates"
        
        # -------------------------
        # 1. Template Encoder
        # -------------------------
        template_features_list = []
        for template in templates:
            fpn_feats = self.extract_features(template)  # [P2, P3, P4, P5]
            template_features_list.append(fpn_feats)
        
        # Extract P2 and P3 from each template
        p2_templates = [feats[0] for feats in template_features_list]  # 3 x [B, C, H/4, W/4]
        p3_templates = [feats[1] for feats in template_features_list]  # 3 x [B, C, H/8, W/8]
        
        # Fuse templates at each scale
        template_p2 = self.template_fusion_p2(p2_templates)  # [B, C, H/4, W/4]
        template_p3 = self.template_fusion_p3(p3_templates)  # [B, C, H/8, W/8]
        
        # -------------------------
        # 2. Search Encoder
        # -------------------------
        search_features = self.extract_features(search)  # [P2, P3, P4, P5]
        search_p2 = search_features[0]  # [B, C, H_s/4, W_s/4]
        search_p3 = search_features[1]  # [B, C, H_s/8, W_s/8]
        
        # -------------------------
        # 3. Multi-scale Correlation
        # -------------------------
        # Correlation at P2 (stride 4)
        corr2 = self.correlation_p2(template_p2, search_p2)  # [B, 1, H_s/4, W_s/4]
        
        # Correlation at P3 (stride 8)
        corr3 = self.correlation_p3(template_p3, search_p3)  # [B, 1, H_s/8, W_s/8]
        
        # Upsample corr3 to match P2 size
        corr3_up = F.interpolate(corr3, size=search_p2.shape[-2:],
                                mode='bilinear', align_corners=False)  # [B, 1, H_s/4, W_s/4]
        
        # -------------------------
        # 4. Feature Fusion
        # -------------------------
        # Concatenate search features + correlations
        fused_features = torch.cat([search_p2, corr2, corr3_up], dim=1)  # [B, C+2, H_s/4, W_s/4]
        fused_features = self.fusion_conv(fused_features)  # [B, det_channels, H_s/4, W_s/4]
        
        # -------------------------
        # 5. Detection Head
        # -------------------------
        obj_map, bbox_map = self.detection_head(fused_features)
        
        return obj_map, bbox_map
    
    def decode_bbox_logits(self, bbox_logits, stride=None):
        """Decode bbox logits to pixel space.
        
        This is a shared helper to ensure consistent decoding between
        training (loss computation) and inference (decode_predictions).
        
        Args:
            bbox_logits: [..., 4] - (tx, ty, tw, th) in any shape
            stride: stride value (default: self.stride)
        
        Returns:
            bbox_decoded: [..., 4] - (cx_offset, cy_offset, w, h) in pixels
        """
        if stride is None:
            stride = self.stride
        
        # Unbind along last dimension
        tx = bbox_logits[..., 0]
        ty = bbox_logits[..., 1]
        tw = bbox_logits[..., 2]
        th = bbox_logits[..., 3]
        
        # Apply constraints
        cx_offset = torch.tanh(tx) * stride
        cy_offset = torch.tanh(ty) * stride
        w = F.softplus(tw)
        h = F.softplus(th)
        
        return torch.stack([cx_offset, cy_offset, w, h], dim=-1)
    
    def decode_predictions(self, obj_map, bbox_map, score_threshold=0.5, top_k=1):
        """Decode predictions to bounding boxes.
        
        Args:
            obj_map: [B, 1, H, W] - objectness logits
            bbox_map: [B, 4, H, W] - bbox predictions (tx, ty, tw, th)
            score_threshold: minimum score to consider
            top_k: return top k detections per image
        
        Returns:
            List of detections per image, each detection is dict with:
                - bbox: [x1, y1, x2, y2] in original image coordinates
                - score: confidence score
                - center: [cx, cy] grid cell indices
        """
        B, _, H, W = obj_map.shape
        stride = self.stride
        
        # Apply sigmoid to get probabilities
        obj_scores = torch.sigmoid(obj_map)  # [B, 1, H, W]
        
        detections_batch = []
        
        for b in range(B):
            scores = obj_scores[b, 0]  # [H, W]
            bboxes = bbox_map[b]  # [4, H, W]
            
            # Flatten
            scores_flat = scores.view(-1)  # [H*W]
            bboxes_flat = bboxes.view(4, -1).transpose(0, 1)  # [H*W, 4]
            
            # Filter by threshold
            valid_mask = scores_flat >= score_threshold
            valid_scores = scores_flat[valid_mask]
            valid_bboxes = bboxes_flat[valid_mask]
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            
            if len(valid_scores) == 0:
                detections_batch.append([])
                continue
            
            # Get top-k
            if len(valid_scores) > top_k:
                top_k_scores, top_k_idx = torch.topk(valid_scores, k=top_k)
                valid_scores = top_k_scores
                valid_bboxes = valid_bboxes[top_k_idx]
                valid_indices = valid_indices[top_k_idx]
            
            # Convert to image coordinates
            detections = []
            for idx, (score, bbox_pred) in enumerate(zip(valid_scores, valid_bboxes)):
                # Get grid cell position
                pos_idx = valid_indices[idx].item()
                grid_y = pos_idx // W
                grid_x = pos_idx % W
                
                # Grid cell center in image coordinates
                grid_center_x = (grid_x + 0.5) * stride
                grid_center_y = (grid_y + 0.5) * stride
                
                # Decode bbox using shared helper
                bbox_decoded = self.decode_bbox_logits(bbox_pred.unsqueeze(0), stride=stride).squeeze(0)
                cx_offset, cy_offset, w, h = bbox_decoded
                
                # Object center
                cx = grid_center_x + cx_offset.item()
                cy = grid_center_y + cy_offset.item()
                w = w.item()
                h = h.item()
                
                # Convert to [x1, y1, x2, y2]
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                
                # Clamp bbox to image boundaries to prevent visualization crashes
                # and invalid coordinates (standard practice in all detectors)
                W_img = W * stride
                H_img = H * stride
                x1 = max(0.0, min(W_img - 1, x1))
                y1 = max(0.0, min(H_img - 1, y1))
                x2 = max(0.0, min(W_img - 1, x2))
                y2 = max(0.0, min(H_img - 1, y2))
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': score.item(),
                    'center': [grid_x, grid_y],
                    'center_coords': [cx, cy]
                })
            
            detections_batch.append(detections)
        
        return detections_batch
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Helper function to create model
# ============================================================================

def build_convnext_refdet(fpn_channels=256, 
                          corr_channels=64,
                          det_channels=256,
                          template_fusion='attention',
                          pretrained_backbone=False):
    """Build ConvNeXt Reference Detection model.
    
    Args:
        fpn_channels: FPN output channels
        corr_channels: Correlation module internal channels
        det_channels: Detection head feature channels
        template_fusion: 'attention' or 'average'
        pretrained_backbone: whether to load ImageNet pretrained weights
    
    Returns:
        model: ConvNeXtRefDet instance
    """
    model = ConvNeXtRefDet(
        fpn_channels=fpn_channels,
        corr_channels=corr_channels,
        det_channels=det_channels,
        template_fusion=template_fusion
    )
    
    if pretrained_backbone:
        # TODO: Load pretrained ConvNeXt-Tiny weights
        # This would require downloading from timm or torchvision
        print("Warning: Pretrained backbone loading not implemented yet")
        pass
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Building ConvNeXt Reference Detection Model...")
    model = build_convnext_refdet(
        fpn_channels=256,
        corr_channels=64,
        det_channels=256,
        template_fusion='attention'
    )
    
    print(f"\nTotal parameters: {model.count_parameters() / 1e6:.2f}M")
    
    # Count backbone parameters
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"Backbone parameters: {backbone_params / 1e6:.2f}M")
    
    # Test forward pass
    batch_size = 2
    templates = [
        torch.randn(batch_size, 3, 256, 256),
        torch.randn(batch_size, 3, 256, 256),
        torch.randn(batch_size, 3, 256, 256)
    ]
    search = torch.randn(batch_size, 3, 1024, 576)
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        obj_map, bbox_map = model(templates, search)
    
    print(f"\nOutput shapes:")
    print(f"  Objectness map: {obj_map.shape}")
    print(f"  BBox map: {bbox_map.shape}")
    
    # Test decode
    print("\nTesting decode predictions...")
    detections = model.decode_predictions(obj_map, bbox_map, score_threshold=0.1, top_k=5)
    print(f"Number of detections per image: {[len(d) for d in detections]}")
    
    if len(detections[0]) > 0:
        print(f"\nFirst detection of first image:")
        print(f"  BBox: {detections[0][0]['bbox']}")
        print(f"  Score: {detections[0][0]['score']:.4f}")
        print(f"  Grid center: {detections[0][0]['center']}")
    
    print("\n✓ Model test completed successfully!")
