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
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from netspresso_trainer.dataloaders import build_dataset, build_dataloader
from netspresso_trainer.models import build_model, is_single_task_model
from netspresso_trainer.postprocessors.pose_estimation import (
    PoseEstimationPostprocessor,
)


def calc_iou_polygon(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """Calculate IoU between two quadrilaterals."""
    try:
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
    except Exception:
        return 0.0


class KeypointEvaluator:
    """Evaluator for keypoint detection models using standard pipeline."""

    def __init__(
        self,
        model_path: str,
        data_config: str,
        model_config: str,
        augmentation_config: str,
        device: str = "cuda",
        pck_threshold: float = 0.05,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pck_threshold = pck_threshold

        print(f"Using device: {self.device}")

        # Load configurations
        self.data_conf = OmegaConf.load(data_config)
        self.model_conf = OmegaConf.load(model_config)
        self.aug_conf = OmegaConf.load(augmentation_config)

        # Merge configs into one 'conf' object mimicking the trainer
        self.conf = OmegaConf.create({
            "data": self.data_conf.data,
            "model": self.model_conf.model,
            "augmentation": self.aug_conf.augmentation,
            "environment": {
                "batch_size": 32, # Default eval batch size
                "num_workers": 4,
                "cache_data": False
            },
            "distributed": False,
            "world_size": 1,
            "rank": 0
        })

        # Correctly set input_size from model config if not provided
        if input_size is None:
            if hasattr(self.conf.model, 'input_size'):
                self.input_size = tuple(self.conf.model.input_size)
            else:
                self.input_size = (256, 256)
                print("Warning: input_size not found in config, defaulting to (256, 256)")
        else:
            self.input_size = input_size

        # Load model
        self.model = self._load_model(model_path)
        self.postprocessor = PoseEstimationPostprocessor(self.conf.model)

        # Load id_mapping
        self._load_id_mapping()

    def _load_id_mapping(self):
        root_path = Path(self.conf.data.path.root)
        id_mapping_path = root_path / self.conf.data.id_mapping
        with open(id_mapping_path, "r") as f:
            self.id_mapping = json.load(f)
        self.keypoint_names = [kp["name"] for kp in self.id_mapping]
        self.num_keypoints = len(self.keypoint_names)
        print(f"Keypoints: {self.keypoint_names}")

    def _load_model(self, model_path: str) -> nn.Module:
        print(f"Loading model from: {model_path}")
        self.conf.model.checkpoint.path = model_path
        self.conf.model.checkpoint.use_pretrained = True
        self.conf.model.checkpoint.load_head = True
        self.conf.model.single_task_model = is_single_task_model(self.conf.model)

        model = build_model(
            self.conf.model,
            num_classes=4,
            devices=self.device,
            distributed=False,
        )
        model.eval()
        return model

    def calc_pck(self, pred, gt, mask, norm_factors=None):
        N, K, _ = pred.shape
        if norm_factors is None:
            norm_factor = np.tile(
                np.array([[self.input_size[0], self.input_size[1]]]), (N, 1)
            )
        else:
            norm_factor = norm_factors

        _mask = mask.copy()
        _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

        distances = np.full((N, K), -1, dtype=np.float32)
        norm_factor[np.where(norm_factor <= 0)] = 1e6
        distances[_mask] = np.linalg.norm(
            ((pred - gt) / norm_factor[:, None, :])[_mask], axis=-1
        )
        distances = distances.T

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

    def calc_pck_single(self, pred, gt, mask, norm_factor=None):
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

    def evaluate(self, split: str, output_dir: str, visualize_worst: int = 10, output_format: str = "csv"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build dataset and dataloader using standard pipeline
        # We use mode='test' to get proper check behavior
        # Note: build_dataset returns (train, valid, test)
        # We need to construct conf such that the requested split is loaded

        # Clone conf to avoid side effects
        conf_local = self.conf.copy()

        # Manually adjust paths based on split to trick build_dataset if necessary
        # But build_dataset(..., mode='test') expects conf.data.path.test
        # build_dataset(..., mode='train') returns train/valid

        target_dataset = None
        if split == 'val':
             # Use mode='train' to get valid dataset
            _, valid_dataset, _ = build_dataset(
                conf_local.data, conf_local.augmentation,
                task=conf_local.data.task,
                model_name=conf_local.model.name,
                distributed=False, mode='train'
            )
            target_dataset = valid_dataset
        elif split == 'test':
            _, _, test_dataset = build_dataset(
                conf_local.data, conf_local.augmentation,
                task=conf_local.data.task,
                model_name=conf_local.model.name,
                distributed=False, mode='test'
            )
            target_dataset = test_dataset
        else:
            raise ValueError(f"Split {split} not supported in this simplified loader")

        if target_dataset is None:
            print(f"No dataset found for split {split}")
            return {}

        dataloader = build_dataloader(
            conf_local,
            task=conf_local.data.task,
            model_name=conf_local.model.name,
            dataset=target_dataset,
            phase='val' if split == 'val' else 'test'
        )

        all_preds = []
        all_gts = []
        all_masks = []
        sample_results = []

        print(f"\nEvaluating {len(target_dataset)} samples...")

        for batch in tqdm(dataloader, desc="Evaluating"):
            # Batch items
            images = batch['pixel_values'].to(self.device)
            keypoints = batch['keypoints'] # [B, K, 3] or [B, K, 2] depending on transform?
            # Standard loader returns keypoints

            # Run inference
            with torch.no_grad():
                out = self.model(images)
                preds = self.postprocessor(out) # [B, K, 2] in 256x256 space

            # Collect results
            batch_names = batch['name']
            batch_indices = batch['indices']
            batch_org_imgs = batch.get('org_img', None) # [B, H, W, 3]

            # Post-process batch
            for i in range(len(preds)):
                pred_kp_256 = preds[i] # [K, 2]
                gt_kp_transformed = keypoints[i].numpy() # [K, 3] (x, y, v) or [K, 2]?
                # Note: PoseEstimationCustomDataset.__getitem__ returns keypoint as (K, 3)
                # but 'keypoints' in batch are processed by ToTensor?
                # ToTensor keeps it as tensor.

                # Check dimensions
                if gt_kp_transformed.shape[-1] == 3:
                    gt_xy_transformed = gt_kp_transformed[:, :2]
                    mask = gt_kp_transformed[:, 2] > 0
                else:
                    gt_xy_transformed = gt_kp_transformed
                    mask = np.ones((gt_xy_transformed.shape[0],), dtype=bool)

                # Calculate PCK in 256x256 space
                pck_single = self.calc_pck_single(pred_kp_256, gt_xy_transformed, mask)
                iou = calc_iou_polygon(pred_kp_256, gt_xy_transformed)

                # For visualization and saving, we need original image and bbox
                # But since we are using standard pipeline, we don't readily have 'bbox' in batch unless we modify collate
                # PoseEstimationCustomDataset.__getitem__ does NOT return bbox in output dict if keypoint is present?
                # Wait, PoseEstimationCustomDataset returns:
                #   outputs.update({"pixel_values": out["image"], "keypoints": out["keypoint"][0]})
                #   outputs.update({"org_shape": (h, w), "org_img": ...})
                # It does NOT return 'bbox'.

                # However, we need to map back to original space for visualization?
                # Actually, the standard pipeline usually visualizes on the transformed image during training,
                # or uses inverse transform if it tracked it.
                # PoseTopDownAffine does not return inverse matrix.

                # But `evaluate_kp.py` was visualizing on ORIGINAL image.
                # To do that, we need the original keypoints and bbox.
                # The standard dataloader does not return them.

                # Reverting to visualization on the Warped (256x256) image?
                # Or skipping visualization on original image?

                # The user asked for "process exactly like validation".
                # Validation pipeline (EvaluationPipeline) saves predictions in JSON.

                # If I want to keep the visualization feature of evaluate_kp.py, I have a problem.
                # I cannot easily invert the transform without the parameters (bbox, rot, scale) used.
                # The dataloader consumes them.

                # Compromise: Visualize on the input image (256x256).
                # This is what the model sees.

                # Store results
                sample_res = {
                    "name": batch_names[i],
                    # "image": ... # Path is not available easily in batch
                    "pck": float(pck_single),
                    "iou": float(iou),
                    "pred_keypoints": pred_kp_256.tolist(),
                    "gt_keypoints": gt_xy_transformed.tolist(),
                    # "bbox": ... # Not available
                }
                sample_results.append(sample_res)

                all_preds.append(pred_kp_256)
                all_gts.append(gt_xy_transformed)
                all_masks.append(mask)

                # Visualization (Warped)
                if batch_org_imgs is not None:
                     # Wait, batch_org_imgs is the ORIGINAL image.
                     # But we don't have the warp matrix to map pred back to it.
                     pass

        # Overall Metrics
        all_preds = np.stack(all_preds, axis=0)
        all_gts = np.stack(all_gts, axis=0)
        all_masks = np.stack(all_masks, axis=0)

        per_kp_acc, avg_acc, cnt = self.calc_pck(all_preds, all_gts, all_masks)

        # Distributions
        pck_scores = [r["pck"] for r in sample_results]
        iou_scores = [r["iou"] for r in sample_results]

        results = {
            "overall_pck": float(avg_acc),
            "overall_iou": float(np.mean(iou_scores)),
            "per_keypoint_pck": {
                self.keypoint_names[i]: float(per_kp_acc[i])
                for i in range(self.num_keypoints)
            },
            "num_samples": len(target_dataset),
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
            "sample_results": sample_results
        }

        # Print tables (reused from old script)
        console = Console()
        console.print()
        table = Table(
            title=f"[bold]Evaluation Results ({split})[/bold]  •  {len(target_dataset)} samples  •  PCK@{self.pck_threshold}",
            show_header=True, header_style="bold cyan", border_style="blue",
        )
        table.add_column("Metric", style="white", no_wrap=True)
        table.add_column("Value", justify="center")
        table.add_row("[bold]PCK[/bold]", f"[green]{avg_acc*100:.2f}%[/green]")
        table.add_row("[bold]IoU[/bold]", f"[blue]{np.mean(iou_scores)*100:.2f}%[/blue]")
        console.print(table)

        kp_scores = " │ ".join(f"[cyan]{name}[/cyan]: {per_kp_acc[i]*100:.1f}%" for i, name in enumerate(self.keypoint_names))
        console.print(Panel(kp_scores, title="Per-Keypoint PCK", border_style="dim", padding=(0, 1)))

        # Save
        results_save = {k: v for k, v in results.items() if k != "sample_results"}
        results_path = output_path / f"evaluation_results_{split}.json"
        with open(results_path, "w") as f:
            json.dump(results_save, f, indent=2)

        if output_format == "csv":
             per_sample_path = output_path / f"per_sample_results_{split}.csv"
             if sample_results:
                 fieldnames = sample_results[0].keys()
                 with open(per_sample_path, "w", newline="") as f:
                     writer = csv.DictWriter(f, fieldnames=fieldnames)
                     writer.writeheader()
                     writer.writerows(sample_results)

        return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate keypoint detection model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_config", type=str, default="config/data/local/kp_detection.yaml")
    parser.add_argument("--model_config", type=str, default="config/model/rtmpose/rtmpose-pose_estimation.yaml")
    parser.add_argument("--augmentation_config", type=str, default="config/augmentation/pose_estimation_phase2.yaml")
    parser.add_argument("--split", type=str, default="all", choices=["val", "test", "all"])
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation_results")
    parser.add_argument("--pck_threshold", type=float, default=0.05)
    # bbox_padding removed as we use standard pipeline
    parser.add_argument("--visualize_worst", type=int, default=0, help="Visualization is currently disabled in this standard pipeline mode")
    parser.add_argument("--output_format", type=str, default="csv", choices=["csv", "json"])
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    evaluator = KeypointEvaluator(
        model_path=args.model_path,
        data_config=args.data_config,
        model_config=args.model_config,
        augmentation_config=args.augmentation_config,
        device=args.device,
        pck_threshold=args.pck_threshold,
    )

    if args.split == "all":
        evaluator.evaluate("val", args.output_dir, args.visualize_worst, args.output_format)
        evaluator.evaluate("test", args.output_dir, args.visualize_worst, args.output_format)
    else:
        evaluator.evaluate(args.split, args.output_dir, args.visualize_worst, args.output_format)

if __name__ == "__main__":
    main()
