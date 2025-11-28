"""
License Plate Keypoint Detection - Inference Script
===================================================
This script performs inference on trained pose estimation models and visualizes
predictions alongside ground truth keypoints on images.

Usage:
    python inference_kp.py --model_path weights/pose_estimation_mobilenet_v3_small_best_phase3.safetensors \
                           --data_config config/data/local/kp_detection.yaml \
                           --model_config config/model/rtmpose/rtmpose-pose_estimation.yaml \
                           --augmentation_config config/augmentation/pose_estimation_phase2.yaml \
                           --split val \
                           --output_dir outputs/inference_results \
                           --num_samples 10
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from netspresso_trainer.dataloaders import build_dataset
from netspresso_trainer.dataloaders.augmentation.registry import TRANSFORM_DICT
from netspresso_trainer.models import build_model, is_single_task_model
from netspresso_trainer.postprocessors.pose_estimation import (
    PoseEstimationPostprocessor,
)
from netspresso_trainer.utils.checkpoint import load_checkpoint


class KeypointInferencer:
    """Inferencer for keypoint detection models with visualization capabilities."""

    def __init__(
        self,
        model_path: str,
        data_config: str,
        model_config: str,
        augmentation_config: str,
        device: str = "cuda",
        input_size: Tuple[int, int] = (256, 256),
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        print(f"Using device: {self.device}")

        # Load configurations
        self.data_conf = OmegaConf.load(data_config).data
        self.model_conf = OmegaConf.load(model_config).model
        self.aug_conf = OmegaConf.load(augmentation_config).augmentation

        # Load model
        self.model = self._load_model(model_path)
        self.postprocessor = PoseEstimationPostprocessor(self.model_conf)

        # Load id_mapping for keypoint names
        self._load_id_mapping()

    def _load_id_mapping(self):
        """Load keypoint id mapping from dataset config."""
        root_path = Path(self.data_conf.path.root)
        id_mapping_path = root_path / self.data_conf.id_mapping
        with open(id_mapping_path, "r") as f:
            self.id_mapping = json.load(f)
        self.keypoint_names = [kp["name"] for kp in self.id_mapping]
        self.num_keypoints = len(self.keypoint_names)
        print(f"Keypoints: {self.keypoint_names}")

    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained model from checkpoint."""
        print(f"Loading model from: {model_path}")

        # Set checkpoint path in model config
        self.model_conf.checkpoint.path = model_path
        self.model_conf.checkpoint.use_pretrained = True
        self.model_conf.checkpoint.load_head = True
        self.model_conf.single_task_model = is_single_task_model(self.model_conf)

        # Build model
        model = build_model(
            self.model_conf,
            num_classes=4,  # 4 keypoints for license plate corners
            devices=self.device,
            distributed=False,
        )
        model.eval()
        return model

    def _build_transform(self):
        """Build inference transform pipeline."""
        transforms = []
        for t_conf in self.aug_conf.inference:
            t_name = t_conf.name.lower()
            if t_name in TRANSFORM_DICT:
                t_params = {k: v for k, v in t_conf.items() if k != "name"}
                transforms.append(TRANSFORM_DICT[t_name](**t_params))
        return transforms

    def preprocess_image(
        self, image_path: str, bbox: np.ndarray
    ) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        """Preprocess a single image for inference.

        Args:
            image_path: Path to the image
            bbox: Bounding box [x_min, y_min, x_max, y_max]

        Returns:
            tensor: Preprocessed image tensor
            warp_mat: Affine warp matrix for coordinate conversion
            org_size: Original image size (h, w)
        """
        img = Image.open(image_path).convert("RGB")
        org_size = (img.size[1], img.size[0])  # (h, w)
        img_np = np.array(img)

        # Compute affine transform from bbox to target size
        bbox_ = bbox.reshape(2, 2).copy()
        bbox_center = bbox_.sum(axis=0) / 2.0
        bbox_wh = bbox_[1] - bbox_[0]

        # Get warp matrix
        warp_mat = self._get_warp_matrix(bbox_center, bbox_wh, rot=0)

        # Apply affine transform
        img_warped = cv2.warpAffine(
            img_np, warp_mat, self.input_size, flags=cv2.INTER_LINEAR
        )
        img_pil = Image.fromarray(img_warped)

        # Apply normalization transforms
        transforms = self._build_transform()
        for t in transforms:
            if hasattr(t, "__call__"):
                if t.__class__.__name__ == "ToTensor":
                    result = t(img_pil)
                    if isinstance(result, tuple):
                        img_pil = result[0]
                    else:
                        img_pil = result
                elif t.__class__.__name__ == "Normalize":
                    result = t(img_pil)
                    if isinstance(result, tuple):
                        img_pil = result[0]
                    else:
                        img_pil = result
                elif t.__class__.__name__ == "PoseTopDownAffine":
                    # Skip - we already applied affine manually
                    continue

        if not isinstance(img_pil, torch.Tensor):
            # Manual conversion if transform didn't work
            img_pil = torch.from_numpy(
                np.array(img_pil).transpose(2, 0, 1).astype(np.float32) / 255.0
            )
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_pil = (img_pil - mean) / std

        return img_pil, warp_mat, org_size

    def _get_warp_matrix(
        self, box_center: np.ndarray, box_wh: np.ndarray, rot: float = 0
    ) -> np.ndarray:
        """Calculate affine transformation matrix."""

        def _rotate_point(pt, angle_rad):
            sn, cs = np.sin(angle_rad), np.cos(angle_rad)
            rot_mat = np.array([[cs, -sn], [sn, cs]])
            return rot_mat @ pt

        def _get_3rd_point(a, b):
            direction = a - b
            c = b + np.r_[-direction[1], direction[0]]
            return c

        src_w, src_h = box_wh[:2]
        dst_w, dst_h = self.input_size

        rot_rad = np.deg2rad(rot)
        src_dir = _rotate_point(np.array([src_w * -0.5, 0.0]), rot_rad)
        dst_dir = np.array([dst_w * -0.5, 0.0])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = box_center
        src[1, :] = box_center + src_dir

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2, :] = _get_3rd_point(src[0, :], src[1, :])
        dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return warp_mat

    def _get_inverse_warp_matrix(self, warp_mat: np.ndarray) -> np.ndarray:
        """Get inverse affine transformation matrix."""
        return cv2.invertAffineTransform(warp_mat)

    def predict_single(self, image_path: str, bbox: np.ndarray) -> np.ndarray:
        """Run prediction on a single image.

        Args:
            image_path: Path to the image
            bbox: Bounding box [x_min, y_min, x_max, y_max]

        Returns:
            keypoints: Predicted keypoints in normalized 256x256 space [K, 2]
        """
        tensor, warp_mat, org_size = self.preprocess_image(image_path, bbox)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)
            keypoints = self.postprocessor(out)

        return keypoints[0]  # [K, 2]

    def load_dataset(self, split: str) -> List[Dict]:
        """Load dataset samples.

        Args:
            split: 'val' or 'test'

        Returns:
            samples: List of sample dictionaries
        """
        data_root = Path(self.data_conf.path.root)

        if split == "val":
            image_dir = data_root / self.data_conf.path.valid.image
            label_dir = data_root / self.data_conf.path.valid.label
        elif split == "test":
            image_dir = data_root / self.data_conf.path.test.image
            label_dir = data_root / self.data_conf.path.test.label
        else:
            raise ValueError(f"Unknown split: {split}")

        samples = []
        skipped_count = 0
        img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        for img_path in sorted(image_dir.iterdir()):
            if img_path.suffix.lower() not in img_extensions:
                continue

            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            # Read label
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 12:
                    skipped_count += 1
                    continue

                keypoints = np.array(parts[:12]).reshape(-1, 3).astype(np.float32)

                if len(parts) >= 16:
                    bbox = np.array(parts[-4:]).astype(np.float32)
                else:
                    # No bbox provided - compute from keypoints
                    bbox = None

                # Validate bbox: check if it looks normalized (values too small)
                # or invalid (negative width/height)
                use_keypoints_bbox = False
                if bbox is not None:
                    bbox_wh = bbox[2:] - bbox[:2]
                    # If bbox looks normalized (max value < 2) or has invalid dimensions
                    if bbox.max() < 2 or bbox_wh.min() <= 0:
                        use_keypoints_bbox = True
                else:
                    use_keypoints_bbox = True

                if use_keypoints_bbox:
                    # Compute bbox from keypoints
                    xs = keypoints[:, 0]
                    ys = keypoints[:, 1]
                    bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])

                # Final validation: bbox should have positive width/height
                bbox_wh = bbox[2:] - bbox[:2]
                if bbox_wh.min() <= 0:
                    skipped_count += 1
                    continue

                samples.append(
                    {
                        "image": str(img_path),
                        "name": img_path.stem,
                        "keypoints": keypoints,  # [K, 3] with visibility
                        "bbox": bbox,  # [x_min, y_min, x_max, y_max]
                    }
                )

        print(f"Loaded {len(samples)} samples from {split} split")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} samples due to invalid bbox/keypoints")
        return samples

    def inference(
        self,
        samples: List[Dict],
        output_dir: str,
        num_samples: int = -1,
    ):
        """Run inference on samples and visualize results.

        Args:
            samples: List of sample dictionaries
            output_dir: Output directory for visualizations
            num_samples: Number of samples to process (-1 for all)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if num_samples > 0:
            samples = samples[:num_samples]

        print(f"\nRunning inference on {len(samples)} samples...")

        for i, sample in enumerate(tqdm(samples, desc="Inferring")):
            image_path = sample["image"]
            gt_keypoints = sample["keypoints"]  # [K, 3]
            bbox = sample["bbox"]

            # Run prediction
            pred_kp = self.predict_single(image_path, bbox)  # [K, 2] in 256x256 space

            # Transform GT keypoints to 256x256 space for comparison
            bbox_ = bbox.reshape(2, 2).copy()
            bbox_center = bbox_.sum(axis=0) / 2.0
            bbox_wh = bbox_[1] - bbox_[0]
            warp_mat = self._get_warp_matrix(bbox_center, bbox_wh, rot=0)

            gt_xy = gt_keypoints[:, :2].copy()
            gt_transformed = cv2.transform(gt_xy.reshape(1, -1, 2), warp_mat).reshape(
                -1, 2
            )

            # Visualize
            self._visualize_sample(
                sample,
                pred_kp,
                gt_transformed,
                output_path,
                sample_idx=i + 1,
            )

        print(f"\nInference completed. Visualizations saved to: {output_path}")

    def _visualize_sample(
        self,
        sample: Dict,
        pred_kp: np.ndarray,
        gt_kp: np.ndarray,
        output_dir: Path,
        sample_idx: int,
        alpha: float = 0.6,
    ):
        """Visualize a single sample with predictions and ground truth.

        Args:
            sample: Sample dictionary
            pred_kp: Predicted keypoints in 256x256 space [K, 2]
            gt_kp: Ground truth keypoints in 256x256 space [K, 2]
            output_dir: Output directory
            sample_idx: Sample index for naming
            alpha: Transparency for overlapping points (0.5-0.7)
        """
        # Load original image
        img = cv2.imread(sample["image"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get bbox and transform info
        bbox = np.array(sample["bbox"])
        bbox_ = bbox.reshape(2, 2).copy()
        bbox_center = bbox_.sum(axis=0) / 2.0
        bbox_wh = bbox_[1] - bbox_[0]
        warp_mat = self._get_warp_matrix(bbox_center, bbox_wh, rot=0)
        inv_warp_mat = self._get_inverse_warp_matrix(warp_mat)

        # Transform keypoints back to original image space
        pred_kp_orig = cv2.transform(pred_kp.reshape(1, -1, 2), inv_warp_mat).reshape(
            -1, 2
        )
        gt_kp_orig = cv2.transform(gt_kp.reshape(1, -1, 2), inv_warp_mat).reshape(-1, 2)

        # Create visualization
        fig_height, fig_width = img.shape[:2]
        vis_img = img.copy()

        # Draw bbox
        x_min, y_min, x_max, y_max = bbox.astype(int)
        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

        # Draw keypoints with alpha blending
        overlay = vis_img.copy()

        # Colors: Green for GT, Red for Pred
        gt_color = (0, 255, 0)  # Green
        pred_color = (255, 0, 0)  # Red

        point_radius = max(3, min(fig_height, fig_width) // 100)
        line_thickness = max(1, point_radius // 2)

        # Draw GT keypoints and polygon
        gt_points = gt_kp_orig.astype(np.int32)
        for j, (x, y) in enumerate(gt_points):
            cv2.circle(overlay, (int(x), int(y)), point_radius, gt_color, -1)

        # Draw GT polygon
        cv2.polylines(
            overlay, [gt_points.reshape(-1, 1, 2)], True, gt_color, line_thickness
        )

        # Draw Pred keypoints and polygon
        pred_points = pred_kp_orig.astype(np.int32)
        for j, (x, y) in enumerate(pred_points):
            cv2.circle(overlay, (int(x), int(y)), point_radius, pred_color, -1)

        # Draw Pred polygon
        cv2.polylines(
            overlay, [pred_points.reshape(-1, 1, 2)], True, pred_color, line_thickness
        )

        # Blend with alpha
        vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)

        # Add legend
        # legend_y = 30
        # cv2.putText(
        #     vis_img,
        #     "GT: Green",
        #     (10, legend_y),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     gt_color,
        #     2,
        # )
        # cv2.putText(
        #     vis_img,
        #     "Pred: Red",
        #     (10, legend_y + 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     pred_color,
        #     2,
        # )

        # Save visualization
        output_path = output_dir / f"inference_{sample_idx:04d}_{sample['name']}.jpg"
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), vis_img_bgr)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on keypoint detection model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="weights/pose_estimation_mobilenet_v3_small_epoch_21.safetensors",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="config/data/local/kp_detection.yaml",
        help="Path to data config",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/model/rtmpose/rtmpose-pose_estimation.yaml",
        help="Path to model config",
    )
    parser.add_argument(
        "--augmentation_config",
        type=str,
        default="config/augmentation/pose_estimation_phase2.yaml",
        help="Path to augmentation config",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to run inference on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/inference_results",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to process (-1 for all)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    # Create output directory with split name
    output_dir = Path(args.output_dir) / args.split

    # Initialize inferencer
    inferencer = KeypointInferencer(
        model_path=args.model_path,
        data_config=args.data_config,
        model_config=args.model_config,
        augmentation_config=args.augmentation_config,
        device=args.device,
    )

    # Load dataset
    samples = inferencer.load_dataset(args.split)

    # Run inference
    inferencer.inference(
        samples=samples,
        output_dir=str(output_dir),
        num_samples=args.num_samples,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
