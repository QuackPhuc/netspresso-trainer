"""
License Plate Keypoint Detection - Evaluation Script
=====================================================
This script evaluates trained pose estimation models on validation and test sets.
It calculates PCK (Percentage of Correct Keypoints) metrics and visualizes results.

Usage:
    python evaluate_kp.py --model_path weights/pose_estimation_mobilenet_v3_small_best_phase3.safetensors \
                          --data_config config/data/local/kp_detection.yaml \
                          --model_config config/model/rtmpose/rtmpose-pose_estimation.yaml \
                          --augmentation_config config/augmentation/pose_estimation_phase2.yaml \
                          --split all \
                          --output_dir outputs/evaluation_results \
                          --output_format csv
"""

import argparse
import csv
import json
import math
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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
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


def calc_iou_polygon(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """Calculate IoU between two quadrilaterals.

    Args:
        pts1: First polygon points [4, 2]
        pts2: Second polygon points [4, 2]

    Returns:
        iou: Intersection over Union
    """
    try:
        # Use cv2.contourArea for polygon area
        pts1 = pts1.astype(np.float32).reshape(-1, 1, 2)
        pts2 = pts2.astype(np.float32).reshape(-1, 1, 2)

        # Get bounding rects
        rect1 = cv2.boundingRect(pts1)
        rect2 = cv2.boundingRect(pts2)

        # Create masks
        max_x = max(rect1[0] + rect1[2], rect2[0] + rect2[2]) + 10
        max_y = max(rect1[1] + rect1[3], rect2[1] + rect2[3]) + 10

        mask1 = np.zeros((int(max_y), int(max_x)), dtype=np.uint8)
        mask2 = np.zeros((int(max_y), int(max_x)), dtype=np.uint8)

        cv2.fillPoly(mask1, [pts1.astype(np.int32)], 1)
        cv2.fillPoly(mask2, [pts2.astype(np.int32)], 1)

        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return float(intersection / union)
    except Exception as e:
        return 0.0


def keypoints_to_bbox(pts: np.ndarray) -> np.ndarray:
    """Convert keypoints to axis-aligned bounding box.

    Args:
        pts: Keypoints [K, 2]

    Returns:
        bbox: [x_min, y_min, x_max, y_max]
    """
    x_min = pts[:, 0].min()
    y_min = pts[:, 1].min()
    x_max = pts[:, 0].max()
    y_max = pts[:, 1].max()
    return np.array([x_min, y_min, x_max, y_max])


class KeypointEvaluator:
    """Evaluator for keypoint detection models with visualization capabilities."""

    def __init__(
        self,
        model_path: str,
        data_config: str,
        model_config: str,
        augmentation_config: str,
        device: str = "cuda",
        pck_threshold: float = 0.05,
        input_size: Tuple[int, int] = (256, 256),
        bbox_padding: float = 1.25,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pck_threshold = pck_threshold
        self.input_size = input_size
        self.bbox_padding = bbox_padding

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

    def calc_pck(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        mask: np.ndarray,
        norm_factors: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, int]:
        """Calculate PCK (Percentage of Correct Keypoints).

        Args:
            pred: Predicted keypoints [N, K, 2]
            gt: Ground truth keypoints [N, K, 2]
            mask: Visibility mask [N, K]
            norm_factors: Normalization factors [N, 2] (e.g., bbox width/height).
                         If None, uses input_size.

        Returns:
            acc: Per-keypoint accuracy [K]
            avg_acc: Average accuracy
            cnt: Number of valid samples
        """
        N, K, _ = pred.shape

        # Normalize by provided factors or input size
        if norm_factors is None:
            norm_factor = np.tile(
                np.array([[self.input_size[0], self.input_size[1]]]), (N, 1)
            )
        else:
            norm_factor = norm_factors

        # Calculate distances
        _mask = mask.copy()
        _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

        distances = np.full((N, K), -1, dtype=np.float32)
        norm_factor[np.where(norm_factor <= 0)] = 1e6
        distances[_mask] = np.linalg.norm(
            ((pred - gt) / norm_factor[:, None, :])[_mask], axis=-1
        )

        distances = distances.T  # [K, N]

        # Calculate accuracy per keypoint
        acc = np.zeros(K)
        for k in range(K):
            valid = distances[k] != -1
            if valid.sum() > 0:
                acc[k] = (distances[k][valid] < self.pck_threshold).sum() / valid.sum()
            else:
                acc[k] = -1

        valid_acc = acc[acc >= 0]
        cnt = len(valid_acc)
        avg_acc = valid_acc.mean() if cnt > 0 else 0.0

        return acc, avg_acc, cnt

    def calc_pck_single(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        mask: np.ndarray,
        norm_factor: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate PCK for a single sample.

        Args:
            pred: Predicted keypoints [K, 2]
            gt: Ground truth keypoints [K, 2]
            mask: Visibility mask [K]
            norm_factor: Normalization factor [2] (e.g., [bbox_w, bbox_h]).
                        If None, uses input_size.

        Returns:
            pck: PCK score for this sample
        """
        if norm_factor is None:
            norm_factor = np.array([self.input_size[0], self.input_size[1]])

        valid_count = 0
        correct_count = 0

        for k in range(len(pred)):
            if mask[k]:
                dist = np.linalg.norm((pred[k] - gt[k]) / norm_factor)
                if dist < self.pck_threshold:
                    correct_count += 1
                valid_count += 1

        if valid_count == 0:
            return 0.0
        return correct_count / valid_count

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
                    # Attempt to recover partial lines by padding with 0
                    # This ensures we don't skip samples that the training loop might include
                    parts = parts + ["0"] * (12 - len(parts))

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

                    # Apply padding to the bbox calculated from keypoints
                    if self.bbox_padding > 1.0:
                        c_x = (bbox[0] + bbox[2]) / 2.0
                        c_y = (bbox[1] + bbox[3]) / 2.0
                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]

                        w *= self.bbox_padding
                        h *= self.bbox_padding

                        bbox = np.array([c_x - w / 2, c_y - h / 2, c_x + w / 2, c_y + h / 2])

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

    def evaluate(
        self,
        samples: List[Dict],
        output_dir: str,
        visualize_worst: int = 10,
        output_format: str = "csv",
    ) -> Dict:
        """Run evaluation on samples.

        Args:
            samples: List of sample dictionaries
            output_dir: Output directory for results
            visualize_worst: Number of worst samples to visualize
            output_format: Output format for per-sample results ('csv' or 'json')

        Returns:
            results: Evaluation results dictionary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_preds = []
        all_gts = []
        all_masks = []

        sample_results = []

        print(f"\nEvaluating {len(samples)} samples...")

        for sample in tqdm(samples, desc="Evaluating"):
            image_path = sample["image"]
            gt_keypoints = sample["keypoints"]  # [K, 3]
            bbox = sample["bbox"]

            # Get GT keypoints and visibility mask
            gt_xy = gt_keypoints[:, :2].copy()  # [K, 2] in original image space
            mask = gt_keypoints[:, 2] > 0

            # Calculate bbox dimensions
            bbox_ = bbox.reshape(2, 2).copy()
            bbox_center = bbox_.sum(axis=0) / 2.0
            bbox_wh = bbox_[1] - bbox_[0]  # [width, height]

            # Get warp matrix for coordinate transformation
            warp_mat = self._get_warp_matrix(bbox_center, bbox_wh, rot=0)
            inv_warp_mat = self._get_inverse_warp_matrix(warp_mat)

            # Run prediction (output is in 256x256 space)
            pred_kp_256 = self.predict_single(image_path, bbox)  # [K, 2] in 256x256

            # Transform GT keypoints to 256x256 space for PCK calculation
            gt_transformed = cv2.transform(gt_xy.reshape(1, -1, 2), warp_mat).reshape(
                -1, 2
            )  # [K, 2] in 256x256 space

            # Convert prediction to original space for visualization
            pred_kp = cv2.transform(
                pred_kp_256.reshape(1, -1, 2), inv_warp_mat
            ).reshape(
                -1, 2
            )  # [K, 2] in original space

            # Calculate PCK in 256x256 space (consistent with training)
            pck_single = self.calc_pck_single(pred_kp_256, gt_transformed, mask)

            iou = calc_iou_polygon(pred_kp_256, gt_transformed)

            # Store results (only essential fields)
            sample_results.append(
                {
                    "name": sample["name"],
                    "image": sample["image"],
                    "bbox": bbox.tolist(),
                    "pred_keypoints": pred_kp.tolist(),
                    "gt_keypoints": gt_xy.tolist(),
                    "pck": pck_single,
                    "iou": iou,
                    "visibility": mask.tolist(),
                }
            )

            # Store 256x256 space data for overall PCK calculation
            all_preds.append(pred_kp_256)
            all_gts.append(gt_transformed)
            all_masks.append(mask)

        # Calculate overall metrics (in 256x256 space)
        all_preds = np.stack(all_preds, axis=0)  # [N, K, 2] in 256x256 space
        all_gts = np.stack(all_gts, axis=0)  # [N, K, 2] in 256x256 space
        all_masks = np.stack(all_masks, axis=0)  # [N, K]

        # Use default norm_factor (input_size) for consistent evaluation
        per_kp_acc, avg_acc, cnt = self.calc_pck(all_preds, all_gts, all_masks)

        # Calculate per-sample metrics distributions
        pck_scores = [r["pck"] for r in sample_results]
        iou_scores = [r["iou"] for r in sample_results]

        results = {
            "overall_pck": float(avg_acc),
            "overall_iou": float(np.mean(iou_scores)),
            "per_keypoint_pck": {
                self.keypoint_names[i]: float(per_kp_acc[i])
                for i in range(self.num_keypoints)
            },
            "num_samples": len(samples),
            "pck_threshold": self.pck_threshold,
            "input_size": self.input_size,
            "pck_distribution": {
                "mean": float(np.mean(pck_scores)),
                "std": float(np.std(pck_scores)),
                "min": float(np.min(pck_scores)),
                "max": float(np.max(pck_scores)),
                "median": float(np.median(pck_scores)),
            },
            "iou_distribution": {
                "mean": float(np.mean(iou_scores)),
                "std": float(np.std(iou_scores)),
                "min": float(np.min(iou_scores)),
                "max": float(np.max(iou_scores)),
                "median": float(np.median(iou_scores)),
            },
            "sample_results": sample_results,
        }

        # Print results using Rich
        console = Console()
        console.print()

        # Metrics Summary Table
        table = Table(
            title=f"[bold]Evaluation Results[/bold]  •  {len(samples)} samples  •  PCK@{self.pck_threshold}",
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
        )
        table.add_column("Metric", style="white", no_wrap=True)
        table.add_column("Value", justify="center")

        # PCK row
        table.add_row(
            "[bold]PCK[/bold]",
            f"[green]{avg_acc*100:.2f}%[/green]",
        )

        # IoU row
        table.add_row(
            "[bold]IoU[/bold]",
            f"[blue]{np.mean(iou_scores)*100:.2f}%[/blue]",
        )

        console.print(table)

        # Per-Keypoint PCK - Horizontal compact display
        kp_scores = " │ ".join(
            f"[cyan]{name}[/cyan]: {per_kp_acc[i]*100:.1f}%"
            for i, name in enumerate(self.keypoint_names)
        )
        console.print(
            Panel(
                kp_scores,
                title="Per-Keypoint PCK",
                border_style="dim",
                padding=(0, 1),
            )
        )

        # Save results
        results_path = output_path / "evaluation_results.json"
        with open(results_path, "w") as f:
            # Convert sample_results to be JSON serializable
            results_save = {k: v for k, v in results.items() if k != "sample_results"}
            json.dump(results_save, f, indent=2)
        console.print(f"\nResults saved to: {results_path}")

        # Save per-sample results
        if output_format == "csv":
            per_sample_path = output_path / "per_sample_results.csv"
            if sample_results:
                fieldnames = sample_results[0].keys()
                with open(per_sample_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(sample_results)
        else:  # json
            per_sample_path = output_path / "per_sample_results.json"
            with open(per_sample_path, "w") as f:
                json.dump(sample_results, f, indent=2)
        console.print(f"Per-sample results saved to: {per_sample_path}")

        # Visualize worst samples (by PCK)
        if visualize_worst > 0:
            self._visualize_worst_samples(
                sample_results,
                output_path / "worst_samples",
                num_samples=visualize_worst,
            )

        return results

    def _visualize_worst_samples(
        self,
        sample_results: List[Dict],
        output_dir: Path,
        num_samples: int = 10,
    ):
        """Visualize samples with lowest PCK scores.

        Args:
            sample_results: List of sample result dictionaries
            output_dir: Output directory for visualizations
            num_samples: Number of worst samples to visualize
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sort by original PCK (ascending, worst first)
        sorted_results = sorted(sample_results, key=lambda x: (x["pck"], x["iou"]))
        worst_samples = sorted_results[:num_samples]

        print(f"\nVisualizing {len(worst_samples)} worst samples (by PCK)...")

        for i, sample in enumerate(tqdm(worst_samples, desc="Visualizing")):
            try:
                self._visualize_sample(sample, output_dir, rank=i + 1)
            except Exception as e:
                print(f"Failed to visualize sample {sample['name']}: {e}")

        print(f"Visualizations saved to: {output_dir}")

    def _visualize_sample(
        self,
        sample: Dict,
        output_dir: Path,
        rank: int,
        alpha: float = 0.6,
    ):
        """Visualize a single sample with predictions and ground truth.

        Args:
            sample: Sample result dictionary
            output_dir: Output directory
            rank: Rank in worst samples list
            alpha: Transparency for overlapping points (0.5-0.7)
        """
        # Load original image
        img = cv2.imread(sample["image"])
        if img is None:
            print(
                f"Warning: Could not load image {sample['image']}, skipping visualization."
            )
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get bbox
        bbox = np.array(sample["bbox"])

        # Get keypoints directly in original space (already converted during evaluation)
        pred_kp_orig = np.array(sample["pred_keypoints"])  # [K, 2]
        gt_kp_orig = np.array(sample["gt_keypoints"])  # [K, 2] in original space

        # Create visualization
        fig_height, fig_width = img.shape[:2]
        vis_img = img.copy()

        # Draw bbox from GT keypoints
        x_min, y_min, x_max, y_max = bbox.astype(int)
        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

        # Draw keypoints with alpha blending
        overlay = vis_img.copy()

        # Colors: Green for GT, Red for Pred
        gt_color = (0, 255, 0)  # Green
        pred_color = (255, 0, 0)  # Red

        point_radius = max(3, min(fig_height, fig_width) // 100)
        line_thickness = max(1, point_radius // 2)

        # Draw GT keypoints
        gt_points = gt_kp_orig.astype(np.int32)
        for j, (x, y) in enumerate(gt_points):
            cv2.circle(overlay, (int(x), int(y)), point_radius, gt_color, -1)

        # Draw GT polygon
        cv2.polylines(
            overlay, [gt_points.reshape(-1, 1, 2)], True, gt_color, line_thickness
        )

        # Draw Pred keypoints
        pred_points = pred_kp_orig.astype(np.int32)
        for j, (x, y) in enumerate(pred_points):
            cv2.circle(overlay, (int(x), int(y)), point_radius, pred_color, -1)

        # Draw Pred polygon
        cv2.polylines(
            overlay, [pred_points.reshape(-1, 1, 2)], True, pred_color, line_thickness
        )

        # Blend with alpha
        vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)

        # Add info text
        pck_score = sample["pck"]
        iou_score = sample["iou"]

        # Save visualization
        output_path = (
            output_dir
            / f"worst_{rank:03d}_pck{pck_score:.4f}_iou{iou_score:.4f}_{sample['name']}.jpg"
        )
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), vis_img_bgr)


def main():
    parser = argparse.ArgumentParser(description="Evaluate keypoint detection model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="weights/pose_estimation_mobilenet_v3_small_best_phase3.safetensors",
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
        default="all",
        choices=["train", "val", "test", "all"],
        help="Dataset split to evaluate (train, val, test or all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--pck_threshold",
        type=float,
        default=0.05,
        help="PCK threshold (normalized distance)",
    )
    parser.add_argument(
        "--bbox_padding",
        type=float,
        default=1.25,
        help="Padding factor for bbox when inferred from keypoints (default: 1.25)",
    )
    parser.add_argument(
        "--visualize_worst",
        type=int,
        default=100,
        help="Number of worst samples to visualize",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="csv",
        choices=["csv", "json"],
        help="Output format for per-sample results (csv or json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device (e.g., 'cuda:0' or 'cpu')",
    )

    args = parser.parse_args()

    # Create output directory with split name
    output_dir = Path(args.output_dir) / args.split

    # Initialize evaluator
    evaluator = KeypointEvaluator(
        model_path=args.model_path,
        data_config=args.data_config,
        model_config=args.model_config,
        augmentation_config=args.augmentation_config,
        device=args.device,
        pck_threshold=args.pck_threshold,
        bbox_padding=args.bbox_padding,
    )

    # Load dataset and run evaluation
    if args.split == "all":
        # Evaluate val
        samples_val = evaluator.load_dataset("val")
        results_val = evaluator.evaluate(
            samples=samples_val,
            output_dir=str(output_dir / "val"),
            visualize_worst=args.visualize_worst,
            output_format=args.output_format,
        )

        # Evaluate test
        samples_test = evaluator.load_dataset("test")
        results_test = evaluator.evaluate(
            samples=samples_test,
            output_dir=str(output_dir / "test"),
            visualize_worst=args.visualize_worst,
            output_format=args.output_format,
        )

        # Summary comparison table
        console = Console()
        console.print()

        val_pck = results_val["overall_pck"]
        test_pck = results_test["overall_pck"]
        val_iou = results_val["overall_iou"]
        test_iou = results_test["overall_iou"]

        def _fmt_diff(diff: float) -> str:
            """Format difference with color."""
            if diff > 0:
                return f"[green]+{diff*100:.2f}%[/green]"
            elif diff < 0:
                return f"[red]{diff*100:.2f}%[/red]"
            return f"[dim]{diff*100:.2f}%[/dim]"

        comp_table = Table(
            title="[bold]VAL vs TEST Comparison[/bold]",
            show_header=True,
            header_style="bold cyan",
            border_style="green",
        )
        comp_table.add_column("Metric", style="white", no_wrap=True)
        comp_table.add_column("VAL", justify="center")
        comp_table.add_column("TEST", justify="center")
        comp_table.add_column("Δ", justify="center")

        comp_table.add_row(
            "PCK",
            f"{val_pck*100:.2f}%",
            f"{test_pck*100:.2f}%",
            _fmt_diff(test_pck - val_pck),
        )
        comp_table.add_row(
            "IoU",
            f"{val_iou*100:.2f}%",
            f"{test_iou*100:.2f}%",
            _fmt_diff(test_iou - val_iou),
        )

        console.print(comp_table)

        # Save comparison JSON
        comparison_path = output_dir / "comparison.json"
        comparison = {
            "val": {k: v for k, v in results_val.items() if k != "sample_results"},
            "test": {k: v for k, v in results_test.items() if k != "sample_results"},
            "comparison": {
                "pck": {
                    "val": val_pck,
                    "test": test_pck,
                    "diff": test_pck - val_pck,
                },
                "iou": {
                    "val": val_iou,
                    "test": test_iou,
                    "diff": test_iou - val_iou,
                },
            },
        }
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        console.print(f"\nComparison saved to: {comparison_path}")

    else:
        samples = evaluator.load_dataset(args.split)
        results = evaluator.evaluate(
            samples=samples,
            output_dir=str(output_dir),
            visualize_worst=args.visualize_worst,
            output_format=args.output_format,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
