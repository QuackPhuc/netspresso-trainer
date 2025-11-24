# 3-Phase Progressive Training Strategy for License Plate Keypoint Detection
# Based on YOLOv5/v8 best practices and RTMPose architecture optimization

## Overview
This training strategy follows the principle: **Strong → Medium → Minimal Augmentation**
- Phase 1: Foundation (High LR + Strong Aug) - Learn robust features
- Phase 2: Refinement (Medium LR + Medium Aug) - Stabilize and improve
- Phase 3: Convergence (Low LR + Minimal Aug) - Pixel-perfect accuracy

---

## Phase 1: Foundation Building (Epochs 1-50)

### Objective
- Learn robust geometric invariant features
- Prevent overfitting to specific viewpoints
- Handle diverse lighting/quality conditions

### Configuration Files
- Augmentation: `config/augmentation/pose_estimation_phase1.yaml`
- Training: `config/training_phase1.yaml`
- Model: `config/model/rtmpose/rtmpose-pose_estimation.yaml` (unchanged)
- Environment: `config/environment.yaml` (adjust batch_size if needed)

### Key Parameters
- **Learning Rate**: 3e-4 (High for exploration)
- **Weight Decay**: 0.01 (Strong regularization)
- **Augmentation Strength**: HIGH
  - Rotation: ±45°
  - Scale: [0.6, 1.4]
  - Translation: 0.2
  - ColorJitter: p=0.8 (brightness/contrast=0.5)
  - RandomErasing: p=0.5 (simulate occlusion)
- **EMA**: Disabled (focus on learning)
- **Early Stopping**: 25 epochs

### Training Command
```bash
python train.py \
  --model_config config/model/rtmpose/rtmpose-pose_estimation.yaml \
  --augmentation_config config/augmentation/pose_estimation_phase1.yaml \
  --training_config config/training_phase1.yaml \
  --environment_config config/environment.yaml \
  --data_config config/data/YOUR_LICENSE_PLATE_DATA.yaml \
  --logging_config config/logging.yaml
```

### Expected Behavior
- Loss: Will be **higher** initially due to strong augmentation
- mAP: May plateau around 65-75% (this is normal)
- Goal: Learn diverse features, NOT achieve peak accuracy

---

## Phase 2: Stabilization & Refinement (Epochs 51-100)

### Objective
- Reduce augmentation noise
- Improve localization accuracy
- Stabilize batch normalization statistics

### Configuration Files
- Augmentation: `config/augmentation/pose_estimation_phase2.yaml`
- Training: `config/training_phase2.yaml`
- **Checkpoint**: Load best weights from Phase 1

### Key Parameters
- **Learning Rate**: 1e-4 (Reduced 3x)
- **Weight Decay**: 0.005 (Reduced)
- **Augmentation Strength**: MEDIUM
  - Rotation: ±30° (reduced)
  - Scale: [0.75, 1.25] (narrowed)
  - Translation: 0.15
  - ColorJitter: p=0.6 (reduced intensity)
  - RandomErasing: REMOVED (no more occlusion)
- **EMA**: Enabled (decay=0.9998) for smoother weights
- **Early Stopping**: 25 epochs

### Training Command
```bash
python train.py \
  --model_config config/model/rtmpose/rtmpose-pose_estimation.yaml \
  --augmentation_config config/augmentation/pose_estimation_phase2.yaml \
  --training_config config/training_phase2.yaml \
  --environment_config config/environment.yaml \
  --data_config config/data/YOUR_LICENSE_PLATE_DATA.yaml \
  --logging_config config/logging.yaml \
  --checkpoint_path outputs/phase1/best.pth  # Load Phase 1 best weights
```

### Expected Behavior
- Loss: Should decrease steadily
- mAP: Should improve to 78-85%
- Keypoint localization: Noticeably more precise

---

## Phase 3: Fine-tuning & Convergence (Epochs 101-150)

### Objective
- **CRITICAL**: Align training distribution with inference distribution
- Achieve pixel-perfect keypoint localization
- Final convergence for maximum mAP

### Configuration Files
- Augmentation: `config/augmentation/pose_estimation_phase3.yaml`
- Training: `config/training_phase3.yaml`
- **Checkpoint**: Load best weights from Phase 2

### Key Parameters
- **Learning Rate**: 1e-5 (Very low for micro-adjustments)
- **Weight Decay**: 0.001 (Minimal)
- **Augmentation Strength**: MINIMAL (almost like inference)
  - Rotation: ±10° (50% prob = 0, i.e., 50% no rotation)
  - Scale: [0.95, 1.05] (nearly fixed)
  - Translation: 0.05 (tiny)
  - ColorJitter: p=0.3 (light, low prob)
  - No geometric distortions
- **EMA**: Enabled (decay=0.9999) - use EMA weights for final evaluation
- **Early Stopping**: 25 epochs

### Training Command
```bash
python train.py \
  --model_config config/model/rtmpose/rtmpose-pose_estimation.yaml \
  --augmentation_config config/augmentation/pose_estimation_phase3.yaml \
  --training_config config/training_phase3.yaml \
  --environment_config config/environment.yaml \
  --data_config config/data/YOUR_LICENSE_PLATE_DATA.yaml \
  --logging_config config/logging.yaml \
  --checkpoint_path outputs/phase2/best.pth  # Load Phase 2 best weights
```

### Expected Behavior
- Loss: Minimal decrease (already converged)
- mAP: Should reach 85-92% (peak accuracy)
- **Key Improvement**: Coordinate precision (pixel-level accuracy)

---

## Critical Design Decisions Explained

### 1. NO Horizontal Flip for License Plates
**Reason**: License plates have directional features (e.g., EU blue strip on left, text orientation). Horizontal flipping breaks these semantic cues, confusing the model about left/right keypoint ordering.

### 2. Progressive Augmentation Reduction (Strong → Weak)
**Reason**: Based on YOLOv5/v8 strategy:
- Early phase: Heavy augmentation prevents overfitting, forces learning of invariant features
- Final phase: Minimal augmentation aligns batch norm statistics and predictions with real-world (clean) images, avoiding domain shift

### 3. EMA (Exponential Moving Average) in Phase 2 & 3
**Reason**: EMA smooths weight updates, reducing noise from mini-batch SGD. Critical for final convergence and stable predictions.

### 4. Gradient Clipping
- Phase 1-2: `max_norm=10.0` (prevent explosion during aggressive learning)
- Phase 3: `max_norm=5.0` (weights should be stable by now)

### 5. Weight Decay Schedule
- Phase 1: 0.01 (strong regularization during feature learning)
- Phase 2: 0.005 (moderate)
- Phase 3: 0.001 (minimal, avoid disrupting converged weights)

---

## Model Architecture Considerations

Your current setup:
- **Backbone**: MobileNetV3-Small (lightweight, good for edge deployment)
- **Head**: RTMcc (SimCC-based, sota for pose estimation)
- **Input Size**: 256×256 (good balance for license plates)

### Recommendations:
1. **Keep pretrained=True in Phase 1**: Transfer learning from ImageNet helps
2. **Batch Size**: 64 is good. If GPU memory allows, try 128 for more stable gradients
3. **Mixed Precision**: Keep enabled (faster training, no accuracy loss on modern GPUs)

---

## Expected Total Training Time
- Phase 1: ~50 epochs (may early stop at ~35-40)
- Phase 2: ~50 epochs (may early stop at ~30-35)
- Phase 3: ~50 epochs (may early stop at ~20-25)
- **Total**: ~85-100 actual epochs across 3 phases

---

## Monitoring & Validation

### What to Monitor:
1. **Training Loss**: Should decrease consistently within each phase
2. **Validation mAP**: Should plateau in Phase 1, improve in Phase 2, peak in Phase 3
3. **PCK@0.05** (Percentage of Correct Keypoints): Target > 95% for license plates
4. **OKS** (Object Keypoint Similarity): Target > 0.90

### Red Flags:
- **Phase 3 mAP drops**: Augmentation too strong, or LR too high → reduce further
- **Phase 1 doesn't converge**: LR too high or augmentation too aggressive → reduce rotation/scale range
- **All phases plateau early**: Model capacity too small → consider MobileNetV3-Large or add neck layers

---

## Final Checklist Before Starting

- [ ] Update `config/data/YOUR_LICENSE_PLATE_DATA.yaml` with correct dataset paths
- [ ] Verify keypoint order: top-left, top-right, bottom-right, bottom-left (clockwise)
- [ ] Set `flip_indices` in dataset config (if using horizontal flip - NOT recommended)
- [ ] Adjust `batch_size` in `environment.yaml` based on GPU memory
- [ ] Create output directories: `mkdir outputs/phase1 outputs/phase2 outputs/phase3`
- [ ] Enable TensorBoard/WandB logging in `logging.yaml` for visualization

---

## Advanced Tips (Optional)

1. **Test-Time Augmentation (TTA)**: In Phase 3, try averaging predictions from:
   - Original image
   - Slightly scaled versions ([0.95, 1.0, 1.05])
   
2. **Knowledge Distillation**: If you have a larger teacher model, distill into MobileNetV3-Small during Phase 3

3. **Post-processing**: Apply Gaussian smoothing to heatmaps for sub-pixel accuracy

4. **Ensemble**: Average predictions from Phase 2 (best) + Phase 3 (EMA weights) for final deployment

---

**Good luck with training! The key is patience - don't skip phases, each serves a specific purpose.**
