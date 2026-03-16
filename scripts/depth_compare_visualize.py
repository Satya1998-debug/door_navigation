#!/usr/bin/env python3
"""Visualize DepthAnything output vs. loaded depth image.

Usage:
  python depth_compare_visualize.py --rgb /path/to/rgb.jpg --depth /path/to/depth.png
"""

import argparse
import os
import sys

import cv2
import numpy as np

# Path setup for local imports
import rospkg
rospack = rospkg.RosPack()
PACKAGE_PATH = rospack.get_path('door_navigation')
script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from door_ros_interfaces import DoorDetector


def normalize_depth_for_vis(depth_m, min_m=None, max_m=None):
    """Normalize a depth map (meters) to 0-255 for visualization."""
    depth = depth_m.copy().astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0

    if min_m is None:
        min_m = np.percentile(depth[depth > 0], 5) if np.any(depth > 0) else 0.0
    if max_m is None:
        max_m = np.percentile(depth, 95) if np.any(depth > 0) else 1.0

    if max_m <= min_m:
        max_m = min_m + 1e-3

    depth = np.clip(depth, min_m, max_m)
    depth = (depth - min_m) / (max_m - min_m)
    depth_u8 = (depth * 255.0).astype(np.uint8)
    return depth_u8


def load_depth_image(depth_path):
    """Load depth image and convert to meters if needed."""
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to load depth image: {depth_path}")

    # If depth is 3-channel, use first channel.
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]

    if depth.dtype == np.uint16:
        depth_m = depth.astype(np.float32) / 1000.0
    else:
        depth_m = depth.astype(np.float32)
        # Heuristic: if values look like millimeters, convert to meters.
        if np.nanmax(depth_m) > 50.0:
            depth_m = depth_m / 1000.0

    return depth_m

def get_final_depth_1(depth_rs, depth_da):
    mask_reliable = (depth_rs > 0.6) & (depth_rs < 6.0)
    
    if np.sum(mask_reliable) > 500:
        # 1. Align using both Scale AND Shift
        depth_da_aligned, s, t = align_depth_affine(depth_rs, depth_da, mask_reliable)
        
        # 2. Advanced Glass Detection:
        # Instead of a fixed 0.5m, use a percentage-based error.
        # As you noticed, RS error increases with distance.
        error_threshold = 0.10 * depth_da_aligned + 0.2 # 10% error tolerance + 20cm base
        
        is_glass = (depth_rs > depth_da_aligned + error_threshold) | (depth_rs <= 0)
        
        final_depth = np.where(is_glass, depth_da_aligned, depth_rs)
    else:
        final_depth = depth_da # Total fallback
        
    return final_depth

def get_final_depth_0(depth_rs, depth_da):
    mask_reliable = (depth_rs > 0.6) & (depth_rs < 6.0)
    
    if np.sum(mask_reliable) > 100: # Ensure enough points for a fit
        # 1. Robustly estimate Scale and Shift using RANSAC or Least Squares
        # This handles the affine nature of Depth Anything
        x = depth_da[mask_reliable]
        y = depth_rs[mask_reliable]
        
        scale = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.var(x) + 1e-6)
        shift = np.mean(y) - scale * np.mean(x)
        
        depth_da_aligned = (depth_da * scale) + shift
        
        # 2. Use RS where valid, DA where RS is 0 (holes/glass)
        final_depth = np.where(depth_rs > 0, depth_rs, depth_da_aligned)
    else:
        final_depth = depth_da # Total fallback
        return final_depth

def get_final_depth_2(depth_rs, depth_da):

    mask_reliable = (depth_rs > 0.6) & (depth_rs < 6.0)

    if np.sum(mask_reliable) > 500:

        depth_da_aligned, s, t = align_depth_affine(depth_rs, depth_da, mask_reliable)

        # Difference between sensors
        diff = np.abs(depth_rs - depth_da_aligned)

        # Expected RS error grows with distance
        sigma = 0.1 * depth_da_aligned + 0.15

        # Confidence weight for RS
        w_rs = np.exp(-(diff**2) / (sigma**2))

        # If RS invalid -> zero confidence
        w_rs[depth_rs <= 0] = 0

        # Weighted fusion
        final_depth = w_rs * depth_rs + (1 - w_rs) * depth_da_aligned

    else:
        final_depth = depth_da

    return final_depth

def align_depth_affine(depth_rs, depth_da, mask):
    # Flatten the arrays based on the reliable mask
    rs_pts = depth_rs[mask].flatten()
    da_pts = depth_da[mask].flatten()
    
    # Solve for RS = s * DA + t
    # Using a simple linear fit: y = mx + c
    A = np.vstack([da_pts, np.ones(len(da_pts))]).T
    s, t = np.linalg.lstsq(A, rs_pts, rcond=None)[0]
    
    # Apply to the whole DA frame
    da_aligned = (depth_da * s) + t
    return da_aligned, s, t


def main(img_id):
    rgb_path = f"/home/satya/MT/catkin_ws/src/door_navigation/scripts/data_new/latest_image_color_lab_{img_id}.jpg"
    depth_path = f"/home/satya/MT/catkin_ws/src/door_navigation/scripts/data_new/latest_image_depth_lab_{img_id}.png"

    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise RuntimeError(f"Failed to load RGB image: {rgb_path}")

    depth_rs = load_depth_image(depth_path)

    # DepthAnything inference
    detector = DoorDetector()
    depth_da = detector.run_depth_anything_v2_on_image(rgb_image=rgb)
    #depth_da_corr = depth_da.copy()
    
    # callibrate the raw DA depth
    depth_da_corr = detector.get_corrected_depth_image(depth_da=depth_da, model="quad")
    
    #depth_da_corr = get_final_depth_0(depth_rs, depth_da_corr)
    
    # get final corrected depth
    # depth_da_corr = get_final_depth_1(depth_rs, depth_da_corr)

    # Compute RS-based normalization range (use 5-95 percentiles)
    valid_rs = depth_rs[np.isfinite(depth_rs) & (depth_rs > 0)]
    if valid_rs.size:
        vmin = float(np.percentile(valid_rs, 5))
        vmax = float(np.percentile(valid_rs, 95))
        if vmax <= vmin:
            vmax = vmin + 1e-3
    else:
        vmin, vmax = None, None

    # Visualize both using RS range so colors are comparable
    depth_rs_vis = normalize_depth_for_vis(depth_rs, vmin, vmax)
    depth_da_vis = normalize_depth_for_vis(depth_da, vmin, vmax)
    depth_da_corr_vis = normalize_depth_for_vis(depth_da_corr, vmin, vmax)

    depth_rs_color = cv2.applyColorMap(depth_rs_vis, cv2.COLORMAP_TURBO)
    depth_da_color = cv2.applyColorMap(depth_da_vis, cv2.COLORMAP_TURBO)
    depth_da_corr_color = cv2.applyColorMap(depth_da_corr_vis, cv2.COLORMAP_TURBO)

    # Resize to same size if needed
    if depth_da_color.shape[:2] != depth_rs_color.shape[:2]:
        depth_da_color = cv2.resize(depth_da_color, (depth_rs_color.shape[1], depth_rs_color.shape[0]))

    if depth_da_corr_color.shape[:2] != depth_rs_color.shape[:2]:
        depth_da_corr_color = cv2.resize(depth_da_corr_color, (depth_rs_color.shape[1], depth_rs_color.shape[0]))

    # Build 2x2 grid:
    # [ RGB | RS depth ]
    # [ DA  | DA corrected ]
    rgb_vis = rgb.copy()
    if rgb_vis.shape[:2] != depth_rs_color.shape[:2]:
        rgb_vis = cv2.resize(rgb_vis, (depth_rs_color.shape[1], depth_rs_color.shape[0]))

    top_row = np.hstack([rgb_vis, depth_rs_color])
    bottom_row = np.hstack([depth_da_color, depth_da_corr_color])
    grid = np.vstack([top_row, bottom_row])

    # Stats for RS depth (reusing valid_rs)
    if valid_rs.size:
        stats_text_rs = (
            f"RS depth (m): min={np.min(valid_rs):.2f}, max={np.max(valid_rs):.2f}, "
            f"mean={np.mean(valid_rs):.2f}, median={np.median(valid_rs):.2f}"
        )
    else:
        stats_text_rs = "RS depth (m): no valid pixels"

    # Stats for DepthAnything
    valid_da = depth_da[np.isfinite(depth_da) & (depth_da > 0)]
    if valid_da.size:
        stats_text_da = (
            f"DA depth (m): min={np.min(valid_da):.2f}, max={np.max(valid_da):.2f}, "
            f"mean={np.mean(valid_da):.2f}, median={np.median(valid_da):.2f}"
        )
    else:
        stats_text_da = "DA depth (m): no valid pixels"

    # Stats for corrected DepthAnything
    valid_da_corr = depth_da_corr[np.isfinite(depth_da_corr) & (depth_da_corr > 0)]
    if valid_da_corr.size:
        stats_text_da_corr = (
            f"DA Corrected depth (m): min={np.min(valid_da_corr):.2f}, max={np.max(valid_da_corr):.2f}, "
            f"mean={np.mean(valid_da_corr):.2f}, median={np.median(valid_da_corr):.2f}"
        )
    else:
        stats_text_da_corr = "DA Corrected depth (m): no valid pixels"

    window_name = "2x2: RGB | RS | DA | DA Corrected"

    # Mouse hover: show depth values at cursor for all depth panes
    cursor_info = {"x": -1, "y": -1, "px": -1, "py": -1, "pane": "NA", "rs": None, "da": None, "da_corr": None, "diff": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            h, w = depth_rs.shape[:2]
            pane = "NA"
            px, py = None, None

            if 0 <= x < w and 0 <= y < h:
                pane = "RGB"
                px, py = x, y
            elif w <= x < 2 * w and 0 <= y < h:
                pane = "RS"
                px, py = x - w, y
            elif 0 <= x < w and h <= y < 2 * h:
                pane = "DA"
                px, py = x, y - h
            elif w <= x < 2 * w and h <= y < 2 * h:
                pane = "DA_CORR"
                px, py = x - w, y - h

            if px is not None and py is not None:
                rs_val = depth_rs[py, px]
                da_val = depth_da[py, px]
                da_corr_val = depth_da_corr[py, px]
            else:
                rs_val = None
                da_val = None
                da_corr_val = None

            # compute diff RS - DA_corr for hover
            if px is not None and py is not None:
                try:
                    diff_val = depth_rs[py, px] - depth_da_corr[py, px]
                except Exception:
                    diff_val = None
            else:
                diff_val = None

            cursor_info["x"] = x
            cursor_info["y"] = y
            cursor_info["px"] = px if px is not None else -1
            cursor_info["py"] = py if py is not None else -1
            cursor_info["pane"] = pane
            cursor_info["rs"] = float(rs_val) if (rs_val is not None and np.isfinite(rs_val)) else None
            cursor_info["da"] = float(da_val) if (da_val is not None and np.isfinite(da_val)) else None
            cursor_info["da_corr"] = float(da_corr_val) if (da_corr_val is not None and np.isfinite(da_corr_val)) else None
            cursor_info["diff"] = float(diff_val) if (diff_val is not None and np.isfinite(diff_val)) else None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 1000)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = grid.copy()

        # Pane labels
        cv2.putText(display, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "RS Depth", (depth_rs.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "DA Depth", (10, depth_rs.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "DA Corrected", (depth_rs.shape[1] + 10, depth_rs.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Overlay stats
        cv2.putText(display, stats_text_rs, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, stats_text_da, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, stats_text_da_corr, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Compute signed RS - DA_corr difference stats (no color overlay)
        try:
            depth_diff = depth_rs - depth_da_corr
            valid_mask = np.isfinite(depth_diff)
            valid_vals = depth_diff[valid_mask]
            if valid_vals.size:
                stats_text_diff = (
                    f"RS-DAcorr diff (m): min={np.min(valid_vals):.3f}, max={np.max(valid_vals):.3f}, "
                    f"mean={np.mean(valid_vals):.3f}, median={np.median(valid_vals):.3f}"
                )
            else:
                stats_text_diff = "RS-DAcorr diff (m): no valid pixels"

            cv2.putText(display, stats_text_diff, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        except Exception:
            pass

        # Overlay cursor depth for both RS and DA at the same pixel
        if cursor_info["rs"] is not None or cursor_info["da"] is not None or cursor_info["da_corr"] is not None:
            rs_text = "RS z=NA"
            da_text = "DA z=NA"
            da_corr_text = "DA Corrected z=NA"
            if cursor_info["rs"] is not None:
                rs_text = f"RS z={cursor_info['rs']:.3f} m"
            if cursor_info["da"] is not None:
                da_text = f"DA z={cursor_info['da']:.3f} m"
            if cursor_info["da_corr"] is not None:
                da_corr_text = f"DA Corrected z={cursor_info['da_corr']:.3f} m"

            diff_text = "diff=NA"
            if cursor_info.get("diff") is not None:
                diff_text = f"RS-DAcorr={cursor_info['diff']:.3f} m"

            text = (
                f"hover pane={cursor_info['pane']}  panel_xy=({cursor_info['x']},{cursor_info['y']}) "
                f"img_xy=({cursor_info['px']},{cursor_info['py']})  {rs_text} | {da_text} | {da_corr_text} | {diff_text}"
            )
            cv2.putText(display, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    for img_id in [19, 20, 26, 27, 35, 38]:
        main(img_id)
