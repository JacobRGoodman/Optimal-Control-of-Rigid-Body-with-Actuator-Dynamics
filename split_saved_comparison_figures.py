#!/usr/bin/env python3
"""
Split the combined comparison figures emitted by the SO(3) simulation code
into one PNG file per subplot.

This is deliberately independent of the simulation code. Run it on an output
folder that already contains

    control_comparison_weighted.png
    control_and_error_comparison_extra.png

Example, from PowerShell:

    python split_saved_comparison_figures.py compare_bi_left_nonzero_spin_accel

It will create files such as

    control_comparison_weighted_panel_1_nominal_u_own_objective_norm.png
    control_comparison_weighted_panel_2_nominal_u_physical_J_norm.png
    control_comparison_weighted_panel_3_actual_u_physical_J_norm.png
    control_comparison_weighted_panel_4_torque_dual_norms.png

and

    control_and_error_comparison_extra_panel_1_actuator_command_M_norm.png
    control_and_error_comparison_extra_panel_2_desired_torque_rate_dual_norm.png
    control_and_error_comparison_extra_panel_3_tracking_errors_metric_weighted.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image


WEIGHTED_PANELS = [
    "control_comparison_weighted_panel_1_nominal_u_own_objective_norm",
    "control_comparison_weighted_panel_2_nominal_u_physical_J_norm",
    "control_comparison_weighted_panel_3_actual_u_physical_J_norm",
    "control_comparison_weighted_panel_4_torque_dual_norms",
]

EXTRA_PANELS = [
    "control_and_error_comparison_extra_panel_1_actuator_command_M_norm",
    "control_and_error_comparison_extra_panel_2_desired_torque_rate_dual_norm",
    "control_and_error_comparison_extra_panel_3_tracking_errors_metric_weighted",
]


def crop_grid(
    image_path: Path,
    rows: int,
    cols: int,
    output_stems: list[str],
    outdir: Path,
    trim_px: int = 0,
) -> list[Path]:
    """Crop image into equal grid cells and save one PNG per cell."""
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find {image_path}")

    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    cell_w = w / cols
    cell_h = h / rows

    if len(output_stems) != rows * cols:
        raise ValueError("Number of output names does not match grid shape.")

    saved: list[Path] = []
    k = 0
    for r in range(rows):
        for c in range(cols):
            left = int(round(c * cell_w))
            upper = int(round(r * cell_h))
            right = int(round((c + 1) * cell_w))
            lower = int(round((r + 1) * cell_h))

            # Optional small trim to remove duplicated whitespace at internal cuts.
            l2 = left + (trim_px if c > 0 else 0)
            u2 = upper + (trim_px if r > 0 else 0)
            r2 = right - (trim_px if c < cols - 1 else 0)
            b2 = lower - (trim_px if r < rows - 1 else 0)

            panel = img.crop((l2, u2, r2, b2))
            out_path = outdir / f"{output_stems[k]}.png"
            panel.save(out_path)
            saved.append(out_path)
            k += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split combined SO(3) comparison figures into one PNG per subplot."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Folder containing control_comparison_weighted.png and control_and_error_comparison_extra.png.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Optional output folder. Default: same as input folder.",
    )
    parser.add_argument(
        "--trim-px",
        type=int,
        default=0,
        help="Optional pixel trim at internal crop boundaries. Usually leave at 0.",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    outdir = Path(args.outdir) if args.outdir is not None else folder
    outdir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    saved += crop_grid(
        folder / "control_comparison_weighted.png",
        rows=2,
        cols=2,
        output_stems=WEIGHTED_PANELS,
        outdir=outdir,
        trim_px=args.trim_px,
    )
    saved += crop_grid(
        folder / "control_and_error_comparison_extra.png",
        rows=1,
        cols=3,
        output_stems=EXTRA_PANELS,
        outdir=outdir,
        trim_px=args.trim_px,
    )

    print("Saved individual subplot PNG files:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
