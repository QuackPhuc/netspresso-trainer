# License Plate Pose Estimation Scripts

This repository provides two standalone scripts for predicting and evaluating license plate keypoints using a trained RTMPose model.

## Files

- `predict.py`: Predict keypoints for images in a directory
- `evaluate.py`: Evaluate predictions against ground truth using PCK metric

## Prerequisites

- Python 3.8+
- PyTorch
- OpenCV
- PIL
- NumPy
- tqdm

Install dependencies:
```bash
pip install torch torchvision tqdm numpy pillow
```

## Usage

### 1. Prediction

Predict keypoints for all images in a directory:

```bash
python predict.py \
  --image_dir /path/to/license_plate_images \
  --output_dir /path/to/save_predictions \
  --checkpoint_path outputs/phase3/best.pth
```

**Parameters:**
- `--image_dir`: Directory containing input images (supports .jpg, .png, .jpeg)
- `--output_dir`: Directory to save prediction results (.txt files)
- `--checkpoint_path`: Path to trained model checkpoint (.pth file)
- `--batch_size`: Batch size for inference (default: 32)
- `--num_workers`: Number of data loading workers (default: 4)
- `--input_size`: Input image size (default: 256 256)

**Output Format:**
Each prediction is saved as a `.txt` file with the same name as the input image:
```
x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4 x_max y_max x_min y_min
```

Where:
- `x1,y1,v1` to `x4,y4,v4`: 4 keypoints (top-left, top-right, bottom-right, bottom-left) with confidence scores
- `x_max, y_max, x_min, y_min`: Bounding box coordinates derived from keypoints

### 2. Evaluation

Evaluate predictions against ground truth:

```bash
python evaluate.py \
  --pred_dir /path/to/predictions \
  --gt_dir /path/to/ground_truth_labels \
  --output_file evaluation_results.json
```

**Parameters:**
- `--pred_dir`: Directory containing prediction .txt files
- `--gt_dir`: Directory containing ground truth .txt files
- `--output_file`: Output JSON file for evaluation results (default: evaluation_results.json)
- `--pck_threshold`: PCK threshold (default: 0.05)
- `--input_size`: Input image size for normalization (default: 256 256)

**Ground Truth Format:**
Ground truth files should be in the same format as predictions:
```
x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4 x_max y_max x_min y_min
```

**Output:**
- Console output with PCK@0.05 scores
- JSON file with detailed results including:
  - Overall PCK@0.05 accuracy
  - Individual keypoint accuracies
  - Total number of evaluated samples

## Example Output

```json
{
  "PCK@0.05": 0.876,
  "total_samples": 1000,
  "keypoint_accuracies": {
    "top-left": 0.892,
    "top-right": 0.901,
    "bottom-right": 0.845,
    "bottom-left": 0.867
  }
}
```

## Notes

1. **Model Loading**: You need to implement the `load_model` function in `predict.py` based on your model's architecture and checkpoint format.

2. **Keypoint Order**: Assumes keypoints are ordered as: top-left, top-right, bottom-right, bottom-left (clockwise).

3. **Confidence Scores**: The evaluation uses confidence scores (v1-v4) for masking invalid keypoints.

4. **PCK Metric**: Uses PCK@0.05 (5% of image size) as the standard threshold.

5. **Batch Processing**: Prediction supports batch processing for efficiency.

## Troubleshooting

- **CUDA out of memory**: Reduce `--batch_size`
- **No predictions found**: Check that prediction .txt files match ground truth filenames
- **Low PCK scores**: Verify coordinate normalization and keypoint ordering
- **Model loading errors**: Ensure checkpoint path and format are correct

## Integration with Training Pipeline

These scripts are designed to work with the 3-phase training strategy:

1. Train model using the provided configs
2. Use `predict.py` for inference on test images
3. Use `evaluate.py` to compute final metrics

For production deployment, consider converting the PyTorch model to ONNX/TensorRT for better performance.

## Loading Trained Checkpoints for Multi-Phase Training

To continue training from Phase 1 to Phase 2:

1. **After Phase 1 completes**, locate the best checkpoint (usually `outputs/phase1/best.pth`)

2. **Create Phase 2 model config** by copying `rtmpose-pose_estimation.yaml` to `rtmpose-pose_estimation_phase2.yaml`

3. **Update the checkpoint path** in the Phase 2 config:
   ```yaml
   checkpoint:
     use_pretrained: True
     load_head: True
     path: outputs/phase1/best.pth  # Path to Phase 1 checkpoint
   ```

4. **Use the Phase 2 config** in training command:
   ```bash
   python train.py --model_config config/model/rtmpose/rtmpose-pose_estimation_phase2.yaml ...
   ```

Repeat the same process for Phase 3, loading from Phase 2 checkpoint.