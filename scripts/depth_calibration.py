# using inverse depth model
# calibration of DepthAnythingV2 model with RealSense depth camera

import sys
import numpy as np
import cv2
import os
import glob

import torch


try:
    import rospkg
    rospack = rospkg.RosPack()
    PACKAGE_PATH = rospack.get_path('door_navigation')
except (ImportError, Exception):
    # Fallback: use script location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PACKAGE_PATH = os.path.dirname(SCRIPT_DIR)

# Add py_packages to path
py_packages_path = os.path.join(PACKAGE_PATH, 'src/door_navigation/py_packages')
if py_packages_path not in sys.path:
    sys.path.insert(0, py_packages_path)

depth_anything_v2_path = os.path.join(py_packages_path, 'depth_anything_v2')
if depth_anything_v2_path not in sys.path:
    sys.path.insert(0, depth_anything_v2_path)  

COEF_QUAD = (0.8863810300827026, 0.9585662484169006, 0.08955039829015732)
COEF_LINEAR = (1.9691135883331299, -0.16483189165592194)

def get_valid_depth_mask(depth_da, depth_rs):
    # calculate valid depth mask using Edge-Rejection (reject edge pixel during calibration)
    gx = cv2.Sobel(depth_rs, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_rs, cv2.CV_32F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(gx**2 + gy**2)
    edge_threshold = 0.1  # in meters
    valid_mask = (grad_magnitude < edge_threshold) & (depth_rs > 0.1) & (depth_da > 0.1)
    return valid_mask

def collect_samples(depth_da, depth_rs):
    # collect depth samples from both DepthAnything and RealSense depth maps
    valid_mask = get_valid_depth_mask(depth_da, depth_rs)

    depth_da_samples = depth_da[valid_mask]
    depth_rs_samples = depth_rs[valid_mask]

    # inverse depth samples
    inv_depth_da_samples = 1.0 / depth_da_samples
    inv_depth_rs_samples = 1.0 / depth_rs_samples
    return inv_depth_da_samples, inv_depth_rs_samples, depth_rs_samples

def fit_inverse_depth_linear(inv_depth_da_samples, inv_depth_rs_samples, depth_rs_samples, epsilon=1e-6):
    # fit inverse depth linear model: depth_rs = (b * (1 / depth_da)) + a

    # weights for least squares (inverse weighting by depth squared to emphasize closer depths)
    w = 1.0 / (depth_rs_samples ** 2) # (N,)
    W = w # use of broadcasting to avoid constructing large diagonal matrix (N,)

    X= np.vstack([inv_depth_da_samples, np.ones_like(inv_depth_da_samples)]).T # (N,2)

    # x = 1/depth_da, y = 1/depth_rs

    XtW = X.T * W  # (2,N)
    XtWX = XtW @ X  # (2,2)
    XtWy = XtW @ inv_depth_rs_samples  # (2,)

    a, b = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
    return float(a), float(b)

def fit_inverse_depth_quad(inv_depth_da_samples, inv_depth_rs_samples, depth_rs_samples, epsilon=1e-6):
    # fit inverse depth quadratic model: depth_rs = c2 * (1 / depth_da)^2 + c1 * (1 / depth_da) + c0

    # weights for least squares (inverse weighting by depth squared to emphasize closer depths)
    w = 1.0 / (depth_rs_samples ** 2)  # (N,)
    W = w # use of broadcasting to avoid constructing large diagonal matrix (N,)

    X= np.vstack([inv_depth_da_samples**2, inv_depth_da_samples, np.ones_like(inv_depth_da_samples)]).T # (N,3)

    XtW = X.T * W  # (3,N)
    XtWX = XtW @ X  # (3,3)
    XtWy = XtW @ inv_depth_rs_samples  # (3,)

    c0, c1, c2 = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
    coef_quad = (float(c0), float(c1), float(c2))
    return coef_quad

def apply_inverse_depth_correction(depth_da, coef, epsilon=1e-6, model='linear'):
    # apply inverse depth correction to DepthAnything depth map
    inv_depth_da = 1.0 / depth_da
    if model == 'linear':
        a, b = coef
        inv_depth_corr = a * inv_depth_da + b
    elif model == 'quad':
        c2, c1, c0 = coef
        inv_depth_corr = c0 + c1 * inv_depth_da + c2 * (inv_depth_da **2)

    depth_corrected = 1.0 / (inv_depth_corr + epsilon) # in meters
    return depth_corrected

def fused_depth(depth_da, depth_rs): # can be used after calibration
    fused_depth = depth_rs.copy() # values from RealSense
    invalid = (depth_rs <= 0.0) | (depth_rs > 20.0) # here depth is not trustworthy
    fused_depth[invalid] = depth_da[invalid] # use DA values where RealSense is invalid
    return fused_depth

def run_depth_anything_v2_on_image(img_dir=None, rgb_image=None):
    # img_dir = caliberation_dataset

    from depth_anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
    
    # Disable xformers for CPU inference (xformers requires CUDA)
    import depth_anything_v2.metric_depth.depth_anything_v2.dinov2_layers.attention as attn_module
    if not torch.cuda.is_available():
        attn_module.XFORMERS_AVAILABLE = False

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vits' # or 'vits', 'vitb'
    dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20 # 20 for indoor model, 80 for outdoor model

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    checkpoint_path = os.path.join(PACKAGE_PATH, f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth')
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False))
    model.eval()

    if rgb_image is not None: # RGB image CV2 BGR format
        print(f"Processing provided RGB image")
        depth = model.infer_image(rgb_image) # HxW depth map in meters in numpy
        # Save depth map as .npy file
        depth = depth.astype(np.float32) # depth in meters
        print(f"Depth map processing completed.")
        return depth

    if img_dir is not None:  # only used during testing/calibration
        for image_path in sorted(glob.glob(os.path.join(img_dir, "color", '*.jpg'))):
            print(f"Processing image: {image_path}")

            depth_img_save_path = image_path.replace("color", "depth_da").replace(".jpg", ".npy").replace("/color/", "/depth_da/")

            raw_img = cv2.imread(image_path)

            depth = model.infer_image(raw_img) # HxW depth map in meters in numpy
            # Save depth map as .npy file
            depth = depth.astype(np.float32) # depth in meters
            np.save(depth_img_save_path, depth)
            print(f"Depth map saved to: {depth_img_save_path}")
            return None

    

def calibrate_depth_anything_v2(model):

    # load all images from the dataset directory
    calibration_ds_dir = "/home/satya/MT/catkin_ws/src/door_navigation/scripts/caliberation_dataset_ias"
    # color_dir = os.path.join(calibration_ds_dir, "color")
    depth_rs_dir = os.path.join(calibration_ds_dir, "depth_rs")
    depth_da_dir = os.path.join(calibration_ds_dir, "depth_da")

    # load all depth images
    depth_rs_images = sorted(glob.glob(os.path.join(depth_rs_dir, '*.png')))
    depth_da_images = sorted(glob.glob(os.path.join(depth_da_dir, '*.npy')))
    all_inv_depth_da_samples = []
    all_inv_depth_rs_samples = []
    all_depth_rs_samples = []
    for drs, dda in zip(depth_rs_images, depth_da_images):
        print(f"Comparing RealSense depth: {drs} with DepthAnything depth: {dda}")
        depth_rs = cv2.imread(drs, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # convert mm to meters
        depth_da = np.load(dda).astype(np.float32)  # depth in meters
        assert depth_rs.shape == depth_da.shape, "Depth map shapes do not match!"

        # collect samples for each image pair (get inv values)

        inv_depth_da_samples, inv_depth_rs_samples, depth_rs_samples = collect_samples(depth_da=depth_da, depth_rs=depth_rs)

        if inv_depth_da_samples.size == 0 or inv_depth_rs_samples.size == 0:
            print("No valid depth samples found for this image pair, skipping...")
            continue

        all_inv_depth_da_samples.append(inv_depth_da_samples)
        all_inv_depth_rs_samples.append(inv_depth_rs_samples)
        all_depth_rs_samples.append(depth_rs_samples)

    # concatenate all samples
    all_inv_depth_da_samples = np.concatenate(all_inv_depth_da_samples)
    all_inv_depth_rs_samples = np.concatenate(all_inv_depth_rs_samples)
    all_depth_rs_samples = np.concatenate(all_depth_rs_samples)

    # fit linear model
    if model == "linear":
        coef_linear = fit_inverse_depth_linear(all_inv_depth_da_samples, all_inv_depth_rs_samples, all_depth_rs_samples)
        print(f"Fitted inverse depth linear model: 1/depth_rs = {coef_linear[0]:.4f} * (1/depth_da) + {coef_linear[1]:.4f}")
        return coef_linear
    elif model == "quad":
        coef_quad = fit_inverse_depth_quad(all_inv_depth_da_samples, all_inv_depth_rs_samples, all_depth_rs_samples)
        print(f"Fitted inverse depth quadratic model: 1/depth_rs = {coef_quad[0]:.4f} * (1/depth_da)^2 + {coef_quad[1]:.4f} * (1/depth_da) + {coef_quad[2]:.4f}")
        return coef_quad

def get_corrected_depth_image(coef=COEF_QUAD, cal_dir=None, model="quad", is_test=False, depth_da=None):

    if is_test: # only if testing
        actual_depth_da_dir = os.path.join(cal_dir, "depth_da")  # example path, adjust as needed

        os.makedirs(os.path.join(cal_dir, f"depth_da_correct_{model}"), exist_ok=True)

        for depth_img_path in sorted(glob.glob(os.path.join(actual_depth_da_dir, '*.npy'))):
            print(f"Applying correction to DepthAnything depth image: {depth_img_path}")
            depth_da = np.load(depth_img_path).astype(np.float32)  # depth in meters
            depth_corrected = apply_inverse_depth_correction(depth_da, coef, model=model)
            depth_corrected_save_path = depth_img_path.replace("/depth_da", f"/depth_da_correct_{model}")
            np.save(depth_corrected_save_path, depth_corrected)
            print(f"Corrected depth map saved to: {depth_corrected_save_path}")
            return
    else: # in production
        depth_corrected = apply_inverse_depth_correction(depth_da, coef, model=model)
        return depth_corrected

def compare_depth_calibration(depth_da_path, depth_rs_path, color_img_path=None, model='linear'):
    """
    Compare depth maps from DepthAnything (before/after calibration) with RealSense.
    Mouse hover shows depth values in meters for easy comparison.
    Loads pre-saved corrected depth maps instead of computing correction on-the-fly.
    
    Args:
        depth_da_path: Path to DepthAnything depth map (.npy file)
        depth_rs_path: Path to RealSense depth map (.png file)
        color_img_path: Path to color image (.jpg file, optional)
        model: 'linear' or 'quad' (calibration model type for loading corrected depth)
    """
    # Load depth maps
    depth_da_raw = np.load(depth_da_path).astype(np.float32)  # meters
    depth_rs = cv2.imread(depth_rs_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # convert mm to meters
    
    # Load color image if path not provided, try to infer from depth paths
    if color_img_path is None:
        # Try to infer color image path from depth_da_path or depth_rs_path
        color_img_path = depth_da_path.replace("/depth_da/", "/color/").replace("_depth_da", "_color").replace(".npy", ".jpg")
        if not os.path.exists(color_img_path):
            color_img_path = depth_rs_path.replace("/depth_rs/", "/color/").replace("_depth", "_color").replace(".png", ".jpg")
    
    # Load color image
    color_img = None
    if color_img_path and os.path.exists(color_img_path):
        color_img = cv2.imread(color_img_path)
        if color_img is not None:
            print(f"Loaded color image from: {color_img_path}")
    else:
        print(f"Warning: Color image not found at {color_img_path}")
    
    # Load pre-saved corrected depth map instead of computing
    depth_da_corrected_path = depth_da_path.replace("/depth_da/", f"/depth_da_correct_{model}/")
    depth_da_corrected = None
    if os.path.exists(depth_da_corrected_path):
        depth_da_corrected = np.load(depth_da_corrected_path).astype(np.float32)
        print(f"Loaded corrected depth from: {depth_da_corrected_path}")
    else:
        print(f"Warning: Corrected depth not found at {depth_da_corrected_path}")
        print(f"Showing only raw DA and RealSense depths.")
    
    # Validate shapes
    if depth_da_raw.shape != depth_rs.shape:
        print(f"Error: Depth map shapes do not match! DA: {depth_da_raw.shape}, RS: {depth_rs.shape}")
        return
    
    # Prepare visualization
    h, w = depth_da_raw.shape
    
    # Normalize depths for visualization (0-5 meters for better indoor visualization)
    def normalize_depth_for_vis(depth, vmin=0.0, vmax=5.0):
        depth_clipped = np.clip(depth, vmin, vmax)
        depth_norm = (depth_clipped - vmin) / (vmax - vmin)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    
    # Create colormaps
    depth_da_vis = normalize_depth_for_vis(depth_da_raw)
    depth_rs_vis = normalize_depth_for_vis(depth_rs)
    
    # Resize color image to match depth dimensions if needed
    if color_img is not None:
        if color_img.shape[:2] != (h, w):
            color_img = cv2.resize(color_img, (w, h))
    else:
        # Create a blank color image if not available
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(color_img, "Color Image Not Found", (w//2 - 100, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Create corrected depth visualization
    if depth_da_corrected is not None:
        depth_da_corrected_vis = normalize_depth_for_vis(depth_da_corrected)
    else:
        # Create a placeholder if corrected depth not available
        depth_da_corrected_vis = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(depth_da_corrected_vis, "Corrected Depth Not Found", (w//2 - 120, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Create 2x2 grid layout
    # Row 1: Color Image | RS Raw
    # Row 2: DA Raw | DA Corrected
    row1 = np.hstack([color_img, depth_rs_vis])
    row2 = np.hstack([depth_da_vis, depth_da_corrected_vis])
    comparison = np.vstack([row1, row2])
    
    # Labels for 2x2 grid: [top-left, top-right, bottom-left, bottom-right]
    labels_grid = [
        ["Color Image", "RS Raw"],
        ["DA Raw", "DA Corrected"]
    ]
    
    # Add labels to each quadrant
    for row_idx, row_labels in enumerate(labels_grid):
        for col_idx, label in enumerate(row_labels):
            x_offset = col_idx * w + 10
            y_offset = row_idx * h + 30
            cv2.putText(comparison, label, (x_offset, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add range info
    cv2.putText(comparison, "Range: 0.0m - 5.0m", (10, 2*h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Mouse callback for hover depth display
    comparison_display = comparison.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal comparison_display
        if event == cv2.EVENT_MOUSEMOVE:
            comparison_display = comparison.copy()
            
            # Determine which panel the mouse is over in 2x2 grid
            row_idx = y // h  # 0 for top row, 1 for bottom row
            col_idx = x // w  # 0 for left column, 1 for right column
            
            # Local coordinates within the panel
            y_local = y % h
            x_local = x % w
            
            # Determine panel name based on grid position
            if row_idx < 2 and col_idx < 2:
                panel_name = labels_grid[row_idx][col_idx]
                
                # Get depth value only for depth panels (not color)
                depth_value = None
                if panel_name == "RS Raw":
                    depth_value = depth_rs[y_local, x_local]
                elif panel_name == "DA Raw":
                    depth_value = depth_da_raw[y_local, x_local]
                elif panel_name == "DA Corrected" and depth_da_corrected is not None:
                    depth_value = depth_da_corrected[y_local, x_local]
                # Color Image panel: no depth value
                
                # Draw crosshair
                cv2.line(comparison_display, (x, 0), (x, comparison_display.shape[0]), (0, 255, 0), 1)
                cv2.line(comparison_display, (0, y), (comparison_display.shape[1], y), (0, 255, 0), 1)
                
                # Display depth value at cursor (if applicable)
                if depth_value is not None:
                    label = f"{panel_name}: {depth_value:.3f}m"
                else:
                    label = f"{panel_name}"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Position label near cursor, adjust if near edges
                label_x = x + 15
                label_y = y - 15
                if label_x + label_size[0] > comparison_display.shape[1]:
                    label_x = x - label_size[0] - 15
                if label_y < label_size[1]:
                    label_y = y + label_size[1] + 15
                
                # Draw label background
                cv2.rectangle(comparison_display,
                            (label_x - 5, label_y - label_size[1] - 5),
                            (label_x + label_size[0] + 5, label_y + 5),
                            (0, 0, 0), -1)
                
                # Draw label text
                cv2.putText(comparison_display, label, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Depth Comparison", comparison_display)
    
    cv2.imshow("Depth Comparison", comparison_display)
    cv2.setMouseCallback("Depth Comparison", mouse_callback)
    
    print("\nDepth Comparison Viewer")
    print("=" * 60)
    if color_img is not None:
        print(f"Color Image:   {color_img_path}")
    print(f"DA Raw:        {depth_da_path}")
    print(f"RealSense:     {depth_rs_path}")
    if depth_da_corrected is not None:
        print(f"DA Corrected:  {depth_da_corrected_path}")
        print(f"Calibration:   {model} model")
    print("=" * 60)
    print("Hover mouse over the depth maps to see values in meters.")
    print("Press any key to exit.\n")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    calibration_dataset_dir = "/home/satya/MT/catkin_ws/src/door_navigation/scripts/calibration_dataset_ias"
    # run_depth_anything_v2_on_image(calibration_dataset_dir)
    # coef = calibrate_depth_anything_v2(model="quad")
    # print("Calibration completed. Coefficients:", coef)

    # Linear Coef
    # 1/depth_rs = 1.9691 * (1/depth_da) - 0.1648
    coef_linear = (1.9691135883331299, -0.16483189165592194)
    A = coef_linear[0]
    B = coef_linear[1]

    # Quadratic Coef
    # 1/depth_rs = 0.8864 * (1/depth_da)^2 + 0.9586 * (1/depth_da) + 0.0896
    coef_quad = (0.8863810300827026, 0.9585662484169006, 0.08955039829015732)
    C2 = coef_quad[0]
    C1 = coef_quad[1]
    C0 = coef_quad[2]

    # Apply correction to images
    # get_corrected_depth_image(calibration_dataset_dir, coef_quad, model="quad")
    
    # Compare depth maps with visualization
    # Example: compare a specific image
    image_id = 1  # Change this to an existing image ID
    sample_da_path = os.path.join(calibration_dataset_dir, f"depth_da/latest_image_depth_da_lab_{image_id}.npy")
    sample_rs_path = os.path.join(calibration_dataset_dir, f"depth_rs/latest_image_depth_lab_{image_id}.png")
    
    # Check if files exist before running comparison
    if not os.path.exists(sample_da_path):
        print(f"Error: DA depth file not found: {sample_da_path}")
        print(f"Please check the calibration_dataset_dir and image_id")
        exit(1)
    if not os.path.exists(sample_rs_path):
        print(f"Error: RS depth file not found: {sample_rs_path}")
        print(f"Please check the calibration_dataset_dir and image_id")
        exit(1)
    
    # Run comparison (loads pre-saved corrected depth)
    compare_depth_calibration(sample_da_path, sample_rs_path, model='quad')


