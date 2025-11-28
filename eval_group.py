"""
Script to merge evaluation results with group data and analyze PCK/IoU by combo_group.
Displays results using rich for professional presentation and visualizes top worst cases.
"""

import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import sys
import cv2
import numpy as np
from PIL import Image
import ast
import argparse


def load_and_prepare_group_data(split: str):
    """Load val.csv or test.csv, create name column."""
    df = pd.read_csv(f"data/kp_detection/{split}.csv")
    df["name"] = df["dataset"] + "_" + df["image_id"].astype(str)
    return df[["name", "combo_group"]]


def load_results_data(split: str):
    """Load per_sample_results.csv for given split."""
    path = Path(f"outputs/evaluation_results/all/{split}/per_sample_results.csv")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def merge_data(group_df, results_df, split: str):
    """Merge group data with results for a split."""
    merged_df = pd.merge(results_df, group_df, on="name", how="left")
    merged_df["split"] = split
    return merged_df


def analyze_by_group(merged_df):
    """Group by combo_group, calculate mean PCK and IoU."""
    grouped = (
        merged_df.groupby("combo_group")
        .agg({"pck": "mean", "iou": "mean"})
        .reset_index()
    )
    return grouped


def visualize_sample(sample, output_dir: Path, rank: int, alpha: float = 0.6):
    """Visualize a single sample with predictions and ground truth.

    Args:
        sample: Sample row from DataFrame
        output_dir: Output directory
        rank: Rank in worst samples list
        alpha: Transparency for overlapping points (0.5-0.7)
    """
    # Load original image
    img_path = sample["image"]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load image {img_path}, skipping visualization.")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get bbox
    bbox = np.array(
        ast.literal_eval(sample["bbox"])
    )  # Assuming bbox is stored as string list

    # Get keypoints directly in original space (already converted during evaluation)
    pred_kp_orig = np.array(ast.literal_eval(sample["pred_keypoints"]))  # [K, 2]
    gt_kp_orig = np.array(
        ast.literal_eval(sample["gt_keypoints"])
    )  # [K, 2] in original space

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


def display_group_analysis(grouped_df, split: str):
    """Display group analysis using rich for a split."""
    console = Console()

    table = Table(
        title=f"[bold cyan]PCK và IoU Theo Nhóm (Combo Group) - {split.upper()}[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Combo Group", style="white", no_wrap=True)
    table.add_column("PCK (%)", style="green", justify="right")
    table.add_column("IoU (%)", style="yellow", justify="right")

    for _, row in grouped_df.iterrows():
        table.add_row(
            row["combo_group"],
            f"{row['pck'] * 100:.2f}",
            f"{row['iou'] * 100:.2f}",
        )

    console.print(table)


def visualize_top_worst_cases(merged_df, split: str, top_n=5):
    """Visualize top N worst cases per group for a split."""
    console = Console()
    output_base_dir = Path(f"outputs/visualizations/worst_cases/{split}")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    for combo_group, group_data in merged_df.groupby("combo_group"):
        group_output_dir = (
            combo_group.replace("|", "_")
            .replace("(0%)", "")
            .replace(" (>0%) ", "")
            .replace("0–10°", "s")
            .replace("10–30°", "m")
            .replace(">30°", "l")
            .replace(" ", "_")
        )

        group_output_dir = output_base_dir / group_output_dir
        group_output_dir.mkdir(parents=True, exist_ok=True)

        # Sort by PCK asc (worst first), then IoU asc
        sorted_group = group_data.sort_values(
            by=["pck", "iou"], ascending=[True, True]
        ).head(top_n)

        for rank, (_, sample) in enumerate(sorted_group.iterrows(), start=1):
            try:
                visualize_sample(sample, group_output_dir, rank)
            except Exception as e:
                console.print(f"[red]Failed to visualize {sample['name']}: {e}[/red]")

        # console.print()  # Add space - commented out to avoid extra lines


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate keypoint detection by group")
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of top worst cases to visualize per group (default: 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    console = Console()
    splits = ["val", "test"]

    for split in splits:
        try:
            console.print(f"\n[bold blue]Xử lý tập {split.upper()}[/bold blue]\n")

            # Load and prepare group data
            group_df = load_and_prepare_group_data(split)

            # Load results
            results_df = load_results_data(split)

            # Merge
            merged_df = merge_data(group_df, results_df, split)

            # Analyze by group
            grouped_df = analyze_by_group(merged_df)

            # Display
            display_group_analysis(grouped_df, split)
            visualize_top_worst_cases(merged_df, split, top_n=args.n)

        except Exception as e:
            console.print(f"[red]Error cho {split}: {e}[/red]")


if __name__ == "__main__":
    main()
