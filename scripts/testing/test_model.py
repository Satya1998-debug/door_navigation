#!/home/satya/MT/uv_ros_py38/bin python3

from ultralytics import YOLO
import sys
import os 
import cv2
import numpy as np
import rospy

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import rospkg
# from config import LABEL_MAP

LABEL_MAP = {0: 'door_double', 1: 'door_single', 2: 'handle'}

# Get package path using rospkg (works with rosrun)
rospack = rospkg.RosPack()
PACKAGE_PATH = rospack.get_path('door_navigation')

# Add py_packages to path so we can import depth_anything_v2, etc.
py_packages_path = os.path.join(PACKAGE_PATH, 'src/door_navigation/py_packages')
if py_packages_path not in sys.path:
    sys.path.insert(0, py_packages_path)

# Add grounding_dino to path for its internal imports (import groundingdino.xxx)
grounding_dino_path = os.path.join(py_packages_path, 'grounding_dino')
if grounding_dino_path not in sys.path:
    sys.path.insert(0, grounding_dino_path)

depth_anything_v2_path = os.path.join(py_packages_path, 'depth_anything_v2')
if depth_anything_v2_path not in sys.path:
    sys.path.insert(0, depth_anything_v2_path)    

depth_anything_v3_path = os.path.join(py_packages_path, 'depth_anything_v3/src')
if depth_anything_v3_path not in sys.path:
    sys.path.insert(0, depth_anything_v3_path) 
#----------------------------------------------------------------#


# Update these after running calibration
DEPTH_CORRECTION_POLY = [0.054848948720128736, -0.7970283765287229, 2.252882064094957]

# IMAGE_PATH = os.path.join(PACKAGE_PATH, 'scripts/data/latest_image_color_11.jpg')
IMAGE_PATH = os.path.join(PACKAGE_PATH, 'scripts/data_new/latest_image_color_lab_35.jpg')

# detector parameters
MODEL_PATH = os.path.join(PACKAGE_PATH, 'weights/last_yolo11m_ias12.pt')
CONFIDENCE_THRESHOLD = 0.5
IMG_SIZE = 640  # input image size for the model
DOOR_DETECTION_TOPIC = "/door_detections"  # assuming door class id is 0 in the model

def depth_anything_v3(img_path_given=True):
    import torch
    from depth_anything_3.api import DepthAnything3

    IMAGE_PATH = "/home/satya/MT/catkin_ws/src/door_navigation/scripts/data_new/latest_image_color_11.jpg"

    # Load model from Hugging Face Hub
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/da3-base")
    model = model.to(device=device)

    # Run inference on images - must pass as list
    prediction = model.inference(
        [IMAGE_PATH],  # Pass as list, not single string
        align_to_input_ext_scale=True,
        export_dir="output",
        export_format="npz"  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
    )

    # Access results
    print(f"Depth shape: {prediction.depth.shape}")        # Depth maps: [N, H, W] float32
    print(f"Conf shape: {prediction.conf.shape}")         # Confidence maps: [N, H, W] float32
    print(f"Extrinsics shape: {prediction.extrinsics.shape}")   # Camera poses (w2c): [N, 3, 4] float32
    print(f"Intrinsics shape: {prediction.intrinsics.shape}")   # Camera intrinsics: [N, 3, 3] float32
    
    # Extract the depth map for visualization
    import cv2
    import numpy as np
    
    depth = prediction.depth[0]  # Get first (and only) depth map
    
    # Normalize for visualization
    depth_min = np.min(depth)
    depth_max = np.max(depth)
    depth_norm = (depth - depth_min) / (depth_max - depth_min)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    
    # Load and display input image
    img = cv2.imread(IMAGE_PATH)
    cv2.imshow("Input Image", img)
    cv2.imshow("Depth Map", depth_colormap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def apply_depth_correction(depth_raw, method='polynomial'):
    """
    Apply calibrated depth correction.
    
    Args:
        depth_raw: Raw depth map from Depth Anything V2 (numpy array)
        method: 'polynomial', 'linear', or 'offset'
    
    Returns:
        Corrected depth map
    """
    global DEPTH_CORRECTION_POLY
    
    if method == 'polynomial' and DEPTH_CORRECTION_POLY is not None:
        # Apply polynomial correction
        error = np.polyval(DEPTH_CORRECTION_POLY, depth_raw)
        return depth_raw - error
    elif method == 'linear':
        # Apply linear calibration: depth_corrected = (depth - intercept) / slope
        # These values should be updated based on your calibration results
        slope = 1.0  # Update from calibration
        intercept = 0.845  # Update from calibration
        return (depth_raw - intercept) / slope
    elif method == 'offset':
        # Simple offset correction
        offset = 0.845  # Update from calibration
        return depth_raw - offset
    else:
        print("Warning: No calibration applied. Run calibration first or use 'offset'/'linear' method.")
        return depth_raw

def run_yolo_model():
    try:

        # load the model
        model = YOLO(MODEL_PATH)

        valid_boxes = []
        jsonable_valid_boxes = []

        detections = dict()

        # run inference
        results = model(source=IMAGE_PATH, imgsz=IMG_SIZE, conf=CONFIDENCE_THRESHOLD)

        # print results
        for result in results:
            print(f"got {len(result.boxes)} boxes in test image")
            for i, box in enumerate(result.boxes):
                print("for box-{}:".format(i))
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # get bbox coordinates
                print(f"Class ID: {cls_id}, Confidence: {conf:.2f}, BBox: {bbox}")

                # filter boxes above confidence threshold
                if conf >= CONFIDENCE_THRESHOLD:
                    print(f"Detected door with confidence {conf:.2f} at bbox {bbox}")
                    valid_boxes.append({
                        'cls_id': cls_id,  # 0 for door, 1 for handle
                        'conf': conf,
                        'bbox': bbox
                    })
                    jsonable_valid_boxes.append({
                        'cls_id': cls_id,  # 0 for door, 1 for handle
                        'conf': conf,
                        'bbox': bbox.tolist()  # convert numpy array to list for json serialization
                    })
            detections.update({
                f"{IMAGE_PATH.split('/')[-1].replace('.jpg', '')}": 
                {
                    'image_path': IMAGE_PATH,
                    'boxes': jsonable_valid_boxes}
            })

        # write detections to json file
        with open(os.path.join(PACKAGE_PATH, 'scripts/detections.json'), 'w') as f:
            import json
            json.dump(detections, f, indent=4)

        color_image = cv2.imread(IMAGE_PATH)
        for vb in valid_boxes:
            x1, y1, x2, y2 = map(int, vb['bbox'])
            conf = vb['conf']
            cls_id = vb['cls_id']
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{LABEL_MAP.get(cls_id, 'Unknown')} {conf:.2f}"
            cv2.putText(color_image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        cv2.imshow("Test Image Detections", color_image)
        # handle cv2 events and check for ESC key to exit
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error testing model: %s", e)

# zero shot object detection with grounding dino base
def grounding_dino_base_hf(image):
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    
    # Check for cats and remote controls
    # VERY important: text queries need to be lowercased + end with a dot
    text = "a door."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    return results

def grounding_dino_base():
    # Simple import since grounding_dino is in sys.path
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    import cv2

    weigths_path = os.path.join(PACKAGE_PATH, 'weights/groundingdino_swint_ogc.pth')
    model_path = os.path.join(PACKAGE_PATH, 'src/door_navigation/py_packages/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py')

    model = load_model(model_path, weigths_path)

    TEXT_PROMPT = "door ."
    BOX_TRESHOLD = 0.60
    TEXT_TRESHOLD = 0.60

    image_source, image = load_image(IMAGE_PATH)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imshow("Grounding DINO Detections", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# depth estimation with DepthAnything
def depth_anything():
    # Simple import since py_packages is in sys.path
    from depth_anything.depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vitb'  # choose encoder type
    model = DepthAnythingV2(**model_configs[encoder])
    
    # Use relative path from package
    checkpoint_path = os.path.join(PACKAGE_PATH, f'checkpoints/depth_anything_v2_{encoder}.pth')
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False))
    model.eval()

    raw_img = cv2.imread(IMAGE_PATH)
    depth = model.infer_image(raw_img) # HxW raw depth map

    # raw image
    cv2.imshow("Input Image", raw_img)
    # depth map
    # Prepare depth for display (robust to NaNs/outliers)
    # Robustly handle NaNs/Infs
    depth_display = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    # First, check the raw depth values
    print(f"Raw depth statistics:")
    print(f"  Shape: {depth.shape}")
    print(f"  Dtype: {depth.dtype}")
    print(f"  Min: {np.min(depth):.6f}")
    print(f"  Max: {np.max(depth):.6f}")
    print(f"  Mean: {np.mean(depth):.6f}")
    print(f"  Median: {np.median(depth):.6f}")

    # If depth is in uint16 mm, convert to meters first:
    if depth_display.dtype == np.uint16:
        depth_display = depth_display.astype(np.float32) / 1000.0

    # Option: autoscale using percentiles (robust to outliers)
    vmin = np.nanpercentile(depth_display, 2)
    vmax = np.nanpercentile(depth_display, 98)
    if vmax <= vmin:
        # fallback to fixed range if percentiles collapse
        vmin, vmax = 0.0, max(1.0, np.nanmax(depth_display))

    # Normalize to 0..1 then to 0..255 uint8
    depth_norm = (depth_display - vmin) / (vmax - vmin)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    depth_uint8 = (depth_norm * 255.0).astype(np.uint8)


    depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    cv2.imshow("Depth Image", depth_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def depth_anything_v2_metric(compare_with_realsense=False, img_path_given=False, save_depth=False):

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

    if img_path_given:
        raw_img = cv2.imread(IMAGE_PATH)
    else: # get live image from camera
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge

        bridge = CvBridge()
        rospy.init_node('depth_anything_v2_metric_tester')
        print("Depth Anything V2 Metric Tester Node Started.")

        color_image_topic = "/camera/color/image_raw"
        print(f"Waiting for image on topic: {color_image_topic}")
        msg = rospy.wait_for_message(color_image_topic, Image)
        raw_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        print("Received image from camera.")

    depth = model.infer_image(raw_img) # HxW depth map in meters in numpy

    if save_depth:
        # Save depth map as .npy file
        depth = depth.astype(np.float32)
        depth_save_path = IMAGE_PATH.replace('.jpg', '_da2_metric_depth.npy')
        np.save(depth_save_path, depth)
        print(f"Depth map saved to: {depth_save_path}")
        return
    
    # Load RealSense depth if comparison requested
    realsense_depth = None
    if compare_with_realsense:
        depth_image_path = IMAGE_PATH.replace('color', 'depth').replace('.jpg', '.png')
        realsense_depth_raw = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        if realsense_depth_raw is not None:
            # Convert RealSense depth from mm to meters, default of realsense is uint16 in mm
            realsense_depth = realsense_depth_raw.astype(np.float32) / 1000.0
            print(f"\nRealSense depth loaded from: {depth_image_path}")
            print(f"RealSense depth statistics:")
            print(f"  Shape: {realsense_depth.shape}")
            print(f"  Dtype: {realsense_depth.dtype}")
            print(f"  Min: {np.min(realsense_depth[realsense_depth > 0]):.6f}m")
            print(f"  Max: {np.max(realsense_depth):.6f}m")
            print(f"  Mean: {np.mean(realsense_depth[realsense_depth > 0]):.6f}m")
            print(f"  Median: {np.median(realsense_depth[realsense_depth > 0]):.6f}m")
        else:
            print(f"Warning: Could not load RealSense depth from {depth_image_path}")
            compare_with_realsense = False
    
    # Display raw image
    cv2.imshow("Input RGB Image", raw_img)
    
    # First, check the raw depth values
    print(f"\nDepth Anything V2 statistics:")
    print(f"  Shape: {depth.shape}")
    print(f"  Dtype: {depth.dtype}")
    print(f"  Min: {np.min(depth):.6f}m")
    print(f"  Max: {np.max(depth):.6f}m")
    print(f"  Mean: {np.mean(depth):.6f}m")
    print(f"  Median: {np.median(depth):.6f}m")
    
    # Compare with RealSense if requested
    if compare_with_realsense and realsense_depth is not None:
        compare_with_depth_rgbd(depth, realsense_depth)
    
    # Keep the original depth values for display
    depth_display = depth.copy()
    
    # For visualization only: normalize to 0-255 for colormap
    depth_min = np.min(depth)
    depth_max = 5.0  # cap max depth for better visualization
    depth_normalized = (depth - depth_min) / (depth_max - depth_min) if depth_max > depth_min else np.zeros_like(depth)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Apply colormap (blue=close, red=far)
    depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    
    # Add depth range info on the image
    cv2.putText(depth_colormap, f"Range: {depth_min:.2f}m - {depth_max:.2f}m", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Mouse callback function to display depth value on hover
    depth_colormap_display = depth_colormap.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal depth_colormap_display
        if event == cv2.EVENT_MOUSEMOVE:
            # Make a fresh copy
            depth_colormap_display = depth_colormap.copy()
            
            # Check if mouse is within image bounds
            if 0 <= y < depth_display.shape[0] and 0 <= x < depth_display.shape[1]:
                depth_value = depth_display[y, x]  # Actual depth value in meters
                
                # Draw crosshair
                cv2.line(depth_colormap_display, (x, 0), (x, depth_colormap_display.shape[0]), (0, 255, 0), 1)
                cv2.line(depth_colormap_display, (0, y), (depth_colormap_display.shape[1], y), (0, 255, 0), 1)
                
                # Display depth value at cursor
                label = f"Depth: {depth_value:.3f}m ({x}, {y})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Position label near cursor, adjust if near edges
                label_x = x + 15
                label_y = y - 15
                if label_x + label_size[0] > depth_colormap_display.shape[1]:
                    label_x = x - label_size[0] - 15
                if label_y < label_size[1]:
                    label_y = y + label_size[1] + 15
                
                # Draw label background
                cv2.rectangle(depth_colormap_display, 
                            (label_x - 5, label_y - label_size[1] - 5),
                            (label_x + label_size[0] + 5, label_y + 5),
                            (0, 0, 0), -1)
                
                # Draw label text
                cv2.putText(depth_colormap_display, label, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Depth Map (Metric)", depth_colormap_display)
    
    cv2.imshow("Depth Map (Metric)", depth_colormap_display)
    cv2.setMouseCallback("Depth Map (Metric)", mouse_callback)
    
    print("\nHover mouse over the depth map to see depth values. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def depth_anything_with_hf():
    from transformers import pipeline
    from PIL import Image

    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    image = Image.open(IMAGE_PATH)
    depth = pipe(image)["depth"]

    raw_img = cv2.imread(IMAGE_PATH)

    # raw image
    cv2.imshow("Input Image", raw_img)
    # depth map
    # Prepare depth for display (robust to NaNs/outliers)
    # Robustly handle NaNs/Infs
    depth_display = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    # If depth is in uint16 mm, convert to meters first:
    if depth_display.dtype == np.uint16:
        depth_display = depth_display.astype(np.float32) / 1000.0

    # Option: autoscale using percentiles (robust to outliers)
    vmin = np.nanpercentile(depth_display, 2)
    vmax = np.nanpercentile(depth_display, 98)
    if vmax <= vmin:
        # fallback to fixed range if percentiles collapse
        vmin, vmax = 0.0, max(1.0, np.nanmax(depth_display))

    # Normalize to 0..1 then to 0..255 uint8
    depth_norm = (depth_display - vmin) / (vmax - vmin)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    depth_uint8 = (depth_norm * 255.0).astype(np.uint8)

    depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    cv2.imshow("Depth Image", depth_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_mode_obj_detect():
    image = Image.open(IMAGE_PATH)
    
    results = grounding_dino_base_hf(image) # object detection results

    print(f"Detected {len(results[0]['boxes'])} objects.")

    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for box, score, label in zip(results[0]['boxes'], results[0]['scores'], results[0]['labels']):
        box = box.cpu().numpy().astype(int)
        score = score.cpu().numpy()
        # label is already a string, no need to convert
        print(f"Box: {box}, Score: {score:.3f}, Label: {label}")

        x1, y1, x2, y2 = box
        cv2.rectangle(cv2_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label_text = f"Door {score:.2f}"
        cv2.putText(cv2_image, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Test Image Detections", cv2_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def analyze_depth_distribution(depth_anything_image, realsense_depth, valid_mask, difference):
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Extract valid data, here valid_mask is a boolean mask such that depth values > 0.1
    da_valid = depth_anything_image[valid_mask]
    rs_valid = realsense_depth[valid_mask]
    diff_valid = difference
    
    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Scatter plot: Predicted vs Ground Truth
    ax1 = plt.subplot(2, 2, 1)
    scatter = ax1.scatter(rs_valid, da_valid, c=np.abs(diff_valid), 
                         cmap='hot', alpha=0.3, s=1)
    ax1.plot([rs_valid.min(), rs_valid.max()], 
             [rs_valid.min(), rs_valid.max()], 
             'g--', linewidth=2, label='Perfect prediction')
    
    # Fit linear regression to find systematic bias
    from numpy.polynomial import Polynomial
    p = Polynomial.fit(rs_valid, da_valid, 1)
    slope, intercept = p.convert().coef
    x_line = np.linspace(rs_valid.min(), rs_valid.max(), 100)
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, 'r-', linewidth=2, 
             label=f'Fit: y={slope:.3f}x+{intercept:.3f}')
    
    ax1.set_xlabel('RealSense Depth (m)', fontsize=11)
    ax1.set_ylabel('Depth Anything V2 (m)', fontsize=11)
    ax1.set_title('Predicted vs Ground Truth Depth', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Absolute Error (m)')
    
    # 2. Error distribution histogram
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(diff_valid, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(diff_valid), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(diff_valid):.3f}m')
    ax2.axvline(np.median(diff_valid), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(diff_valid):.3f}m')
    ax2.set_xlabel('Error (DA - RS) [m]', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Absolute error vs depth with polynomial fit
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(rs_valid, np.abs(diff_valid), c='coral', alpha=0.3, s=1)
    
    # Fit polynomial to error vs depth (2nd degree)
    poly_degree = 2
    poly_error = Polynomial.fit(rs_valid, diff_valid, poly_degree)
    poly_coefs = poly_error.convert().coef
    
    # Plot polynomial fit
    x_poly = np.linspace(rs_valid.min(), rs_valid.max(), 100)
    y_poly = np.polyval(poly_coefs[::-1], x_poly)
    ax3.plot(x_poly, np.abs(y_poly), 'purple', linewidth=3, 
             label=f'Polynomial fit (degree {poly_degree})')
    
    # Compute binned statistics
    depth_bins = np.linspace(rs_valid.min(), rs_valid.max(), 20)
    bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
    bin_means = []
    for i in range(len(depth_bins)-1):
        mask = (rs_valid >= depth_bins[i]) & (rs_valid < depth_bins[i+1])
        if np.sum(mask) > 0:
            bin_means.append(np.mean(np.abs(diff_valid[mask])))
        else:
            bin_means.append(0)
    
    ax3.plot(bin_centers, bin_means, 'b-', linewidth=3, marker='o', markersize=8,
             label='Mean abs error per bin')
    ax3.set_xlabel('RealSense Depth (m)', fontsize=11)
    ax3.set_ylabel('Absolute Error (m)', fontsize=11)
    ax3.set_title('Error vs Depth Range', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics summary table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Compute additional statistics
    percentile_50 = np.percentile(np.abs(diff_valid), 50)
    percentile_75 = np.percentile(np.abs(diff_valid), 75)
    percentile_95 = np.percentile(np.abs(diff_valid), 95)
    rmse = np.sqrt(np.mean(diff_valid**2))
    
    # Check for systematic bias
    bias_direction = "overestimating" if np.mean(diff_valid) > 0 else "underestimating"
    
    # Format polynomial coefficients
    poly_str = f"{poly_coefs[0]:.4f}"
    for i in range(1, len(poly_coefs)):
        sign = "+" if poly_coefs[i] >= 0 else ""
        poly_str += f" {sign} {poly_coefs[i]:.4f}*x^{i}"
    
    stats_text = f"""
    DEPTH ANALYSIS SUMMARY
    ═══════════════════════════════════
    
    Error Statistics:
    • Mean Error: {np.mean(diff_valid):.3f} m
    • MAE: {np.mean(np.abs(diff_valid)):.3f} m
    • RMSE: {rmse:.3f} m
    • Std Dev: {np.std(diff_valid):.3f} m
    
    Percentiles (Absolute Error):
    • 50th: {percentile_50:.3f} m
    • 75th: {percentile_75:.3f} m
    • 95th: {percentile_95:.3f} m
    
    Linear Fit (DA = a*RS + b):
    • Slope: {slope:.3f}
    • Intercept: {intercept:.3f} m
    
    Polynomial Error Model (degree {poly_degree}):
    • Error(x) = {poly_str[:50]}...
    
    Systematic Bias:
    • Direction: {bias_direction}
    • Magnitude: {np.abs(np.mean(diff_valid)):.3f} m
    
    RECOMMENDATIONS:
    """
    
    # Add recommendations based on analysis
    recommendations = []
    
    if np.abs(slope - 1.0) > 0.1:
        recommendations.append("• Scale correction needed (slope ≠ 1)")
    
    if np.abs(intercept) > 0.2:
        recommendations.append(f"• Constant offset correction: -{intercept:.3f}m")
    
    if np.abs(np.mean(diff_valid)) > 0.3:
        recommendations.append("• Apply bias correction (systematic error)")
    
    if np.std(diff_valid) > 0.2:
        recommendations.append("• High variance - consider depth-dependent correction")
    
    # Check if error increases with depth
    if len(bin_means) > 0 and bin_means[-1] > bin_means[0] * 1.5:
        recommendations.append("• Error increases with depth - use range-based calibration")
    
    if not recommendations:
        recommendations.append("• Model performing well, minor calibration may help")
    
    stats_text += "\n".join(recommendations)
    
    # Print actionable calibration formula
    print(f"\n{'='*60}")
    print("CALIBRATION FORMULAS")
    print(f"{'='*60}")
    print(f"1. Simple offset correction:")
    print(f"   depth_corrected = depth_anything - {np.mean(diff_valid):.3f}")
    print(f"\n2. Linear calibration:")
    print(f"   depth_corrected = (depth_anything - {intercept:.3f}) / {slope:.3f}")
    print(f"\n3. Polynomial correction (degree {poly_degree}) - RECOMMENDED:")
    print(f"   error = {poly_coefs[0]:.4f}", end="")
    for i in range(1, len(poly_coefs)):
        sign = "+" if poly_coefs[i] >= 0 else ""
        print(f" {sign} {poly_coefs[i]:.4f}*depth^{i}", end="")
    print(f"\n   depth_corrected = depth_anything - error")
    print(f"\n   Python code:")
    print(f"   poly_coefs = {list(poly_coefs[::-1])}")
    print(f"   error = np.polyval(poly_coefs, depth_anything)")
    print(f"   depth_corrected = depth_anything - error")
    print(f"{'='*60}\n")
    
    # Store polynomial coefficients globally for use in correction
    global DEPTH_CORRECTION_POLY
    DEPTH_CORRECTION_POLY = poly_coefs[::-1]  # Store in numpy polyval order
    print(f"✓ Polynomial coefficients stored in DEPTH_CORRECTION_POLY")
    # Print actionable calibration formula
    print(f"\n{'='*60}")
    print("CALIBRATION FORMULA")
    print(f"{'='*60}")
    print(f"Simple offset correction:")
    print(f"  depth_corrected = depth_anything - {np.mean(diff_valid):.3f}")
    print(f"\nLinear calibration (more accurate):")
    print(f"  depth_corrected = (depth_anything - {intercept:.3f}) / {slope:.3f}")
    print(f"{'='*60}\n")

def compare_with_depth_rgbd(depth_anything_image, realsense_depth):    
    # HxW numpy arrays in meters
    if depth_anything_image.shape != realsense_depth.shape:
        print("Depth map shapes do not match for comparison.")
        return

    valid_mask = realsense_depth > 0.1  # ignorung very small/zero depths
    
    if np.sum(valid_mask) == 0:
        print("No valid RealSense depth values to compare.")
        return
    
    # Compute difference only on valid pixels
    difference = depth_anything_image[valid_mask] - realsense_depth[valid_mask]
    abs_difference = np.abs(difference)

    # Compute statistics
    mean_diff = np.mean(difference)
    mean_abs_diff = np.mean(abs_difference)
    std_diff = np.std(difference)
    median_diff = np.median(difference)
    max_diff = np.max(abs_difference)

    print(f"\n{'='*60}")
    print(f"Depth Comparison: Depth Anything V2 vs RealSense")
    print(f"{'='*60}")
    print(f"Valid pixels compared: {np.sum(valid_mask):,} / {valid_mask.size:,}")
    print(f"Mean Difference: {mean_diff:.3f}m")
    print(f"Mean Absolute Error: {mean_abs_diff:.3f}m")
    print(f"Std Deviation: {std_diff:.3f}m")
    print(f"Median Difference: {median_diff:.3f}m")
    print(f"Max Absolute Error: {max_diff:.3f}m")
    print(f"{'='*60}\n")
    
    # Call detailed analysis
    analyze_depth_distribution(depth_anything_image, realsense_depth, valid_mask, difference)

    # Create full-size difference map
    diff_map = np.zeros_like(depth_anything_image)
    diff_map[valid_mask] = np.abs(depth_anything_image[valid_mask] - realsense_depth[valid_mask])
    
    # Visualize differences
    diff_norm = (diff_map - np.min(diff_map)) / (np.max(diff_map) - np.min(diff_map) + 1e-6)
    diff_uint8 = (diff_norm * 255).astype(np.uint8)
    diff_colormap = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_HOT)
    
    # Add text overlay
    cv2.putText(diff_colormap, f"Max Error: {max_diff:.2f}m, MAE: {mean_abs_diff:.2f}m",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Visualize RealSense depth
    rs_display = realsense_depth.copy()
    rs_min = np.min(rs_display[rs_display > 0])
    rs_max = np.max(rs_display)
    rs_normalized = (rs_display - rs_min) / (rs_max - rs_min) if rs_max > rs_min else np.zeros_like(rs_display)
    rs_uint8 = (rs_normalized * 255).astype(np.uint8)
    rs_colormap = cv2.applyColorMap(rs_uint8, cv2.COLORMAP_JET)
    
    cv2.putText(rs_colormap, f"RealSense Depth: {rs_min:.2f}m - {rs_max:.2f}m",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Mouse callback for RealSense depth
    rs_colormap_display = rs_colormap.copy()
    
    def rs_mouse_callback(event, x, y, flags, param):
        nonlocal rs_colormap_display
        if event == cv2.EVENT_MOUSEMOVE:
            rs_colormap_display = rs_colormap.copy()
            
            if 0 <= y < realsense_depth.shape[0] and 0 <= x < realsense_depth.shape[1]:
                depth_value = realsense_depth[y, x]
                
                # Draw crosshair
                cv2.line(rs_colormap_display, (x, 0), (x, rs_colormap_display.shape[0]), (0, 255, 0), 1)
                cv2.line(rs_colormap_display, (0, y), (rs_colormap_display.shape[1], y), (0, 255, 0), 1)
                
                # Display depth value at cursor
                label = f"Depth: {depth_value:.3f}m ({x}, {y})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Position label near cursor
                label_x = x + 15
                label_y = y - 15
                if label_x + label_size[0] > rs_colormap_display.shape[1]:
                    label_x = x - label_size[0] - 15
                if label_y < label_size[1]:
                    label_y = y + label_size[1] + 15
                
                # Draw label background
                cv2.rectangle(rs_colormap_display, 
                            (label_x - 5, label_y - label_size[1] - 5),
                            (label_x + label_size[0] + 5, label_y + 5),
                            (0, 0, 0), -1)
                
                # Draw label text
                cv2.putText(rs_colormap_display, label, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("RealSense Depth", rs_colormap_display)
    
    # Mouse callback for difference map
    diff_colormap_display = diff_colormap.copy()
    
    def diff_mouse_callback(event, x, y, flags, param):
        nonlocal diff_colormap_display
        if event == cv2.EVENT_MOUSEMOVE:
            diff_colormap_display = diff_colormap.copy()
            
            if 0 <= y < diff_map.shape[0] and 0 <= x < diff_map.shape[1]:
                error_value = diff_map[y, x]
                da_value = depth_anything_image[y, x]
                rs_value = realsense_depth[y, x]
                
                # Draw crosshair
                cv2.line(diff_colormap_display, (x, 0), (x, diff_colormap_display.shape[0]), (0, 255, 0), 1)
                cv2.line(diff_colormap_display, (0, y), (diff_colormap_display.shape[1], y), (0, 255, 0), 1)
                
                # Display error value at cursor
                label = f"Error: {error_value:.3f}m (DA:{da_value:.2f} RS:{rs_value:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Position label near cursor
                label_x = x + 15
                label_y = y - 15
                if label_x + label_size[0] > diff_colormap_display.shape[1]:
                    label_x = x - label_size[0] - 15
                if label_y < label_size[1]:
                    label_y = y + label_size[1] + 15
                
                # Draw label background
                cv2.rectangle(diff_colormap_display, 
                            (label_x - 5, label_y - label_size[1] - 5),
                            (label_x + label_size[0] + 5, label_y + 5),
                            (0, 0, 0), -1)
                
                # Draw label text
                cv2.putText(diff_colormap_display, label, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Absolute Difference (Hot=High Error)", diff_colormap_display)
    
    cv2.imshow("RealSense Depth", rs_colormap_display)
    cv2.setMouseCallback("RealSense Depth", rs_mouse_callback)
    
    cv2.imshow("Absolute Difference (Hot=High Error)", diff_colormap_display)
    cv2.setMouseCallback("Absolute Difference (Hot=High Error)", diff_mouse_callback)

def depth_anything_v3(img_path_given=True):
    import torch
    from depth_anything_v3.src.depth_anything_3.api import DepthAnything3

    # Load model from Hugging Face Hub
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/da3metric-small")
    model = model.to(device=device)

    # Run inference on images
    prediction = model.inference(
        IMAGE_PATH,
        export_dir="output",
        export_format="npz"  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
    )

    # Access results
    print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
    print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
    print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
    print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32

def test_polynomial_correction():
    """Test polynomial depth correction on calibrated data"""
    from depth_anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
    
    # Disable xformers for CPU
    import depth_anything_v2.metric_depth.depth_anything_v2.dinov2_layers.attention as attn_module
    if not torch.cuda.is_available():
        attn_module.XFORMERS_AVAILABLE = False
    
    # Load model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    }
    encoder = 'vits'
    dataset = 'hypersim'
    max_depth = 20
    
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    checkpoint_path = os.path.join(PACKAGE_PATH, f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth')
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False))
    model.eval()
    
    # Get image and depth
    raw_img = cv2.imread(IMAGE_PATH)
    depth_raw = model.infer_image(raw_img)
    
    # Apply polynomial correction
    depth_corrected = apply_depth_correction(depth_raw, method='polynomial')
    
    # Load RealSense ground truth
    depth_image_path = IMAGE_PATH.replace('color', 'depth').replace('.jpg', '.png')
    realsense_depth_raw = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    realsense_depth = realsense_depth_raw.astype(np.float32) / 1000.0
    
    # Compute errors before and after correction
    valid_mask = realsense_depth > 0.1
    
    error_before = depth_raw[valid_mask] - realsense_depth[valid_mask]
    error_after = depth_corrected[valid_mask] - realsense_depth[valid_mask]
    
    mae_before = np.mean(np.abs(error_before))
    mae_after = np.mean(np.abs(error_after))
    rmse_before = np.sqrt(np.mean(error_before**2))
    rmse_after = np.sqrt(np.mean(error_after**2))
    
    print(f"\n{'='*60}")
    print("POLYNOMIAL CORRECTION RESULTS")
    print(f"{'='*60}")
    print(f"Before correction:")
    print(f"  MAE: {mae_before:.3f}m")
    print(f"  RMSE: {rmse_before:.3f}m")
    print(f"\nAfter polynomial correction:")
    print(f"  MAE: {mae_after:.3f}m")
    print(f"  RMSE: {rmse_after:.3f}m")
    print(f"\nImprovement:")
    print(f"  MAE reduced by: {(1 - mae_after/mae_before)*100:.1f}%")
    print(f"  RMSE reduced by: {(1 - rmse_after/rmse_before)*100:.1f}%")
    print(f"{'='*60}\n")
    
    # Visualize side by side
    def normalize_depth(depth):
        vmin, vmax = np.percentile(depth[depth > 0], [2, 98])
        normalized = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
        return (normalized * 255).astype(np.uint8)
    
    raw_vis = cv2.applyColorMap(normalize_depth(depth_raw), cv2.COLORMAP_JET)
    corrected_vis = cv2.applyColorMap(normalize_depth(depth_corrected), cv2.COLORMAP_JET)
    rs_vis = cv2.applyColorMap(normalize_depth(realsense_depth), cv2.COLORMAP_JET)
    
    # Add labels
    cv2.putText(raw_vis, f"Raw (MAE: {mae_before:.3f}m)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(corrected_vis, f"Corrected (MAE: {mae_after:.3f}m)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(rs_vis, "RealSense Ground Truth", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Stack horizontally
    comparison = np.hstack([raw_vis, corrected_vis, rs_vis])
    
    cv2.imshow("Depth Correction Comparison", comparison)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_yolo_model()
    # test_mode_obj_detect()
    # grounding_dino_base()
    # depth_anything()  # need to do this now internet needed
    # depth_anything_with_hf() # hugging face needs internet access
    
    # Use locally cached HuggingFace model (no internet needed)
    # depth_anything_v3(img_path_given=True)
    
    # Set to True to compare with RealSense depth and generate polynomial calibration
    depth_anything_v2_metric(compare_with_realsense=True, img_path_given=True, save_depth=False)
    
    # After running calibration, test the polynomial correction
    # test_polynomial_correction()