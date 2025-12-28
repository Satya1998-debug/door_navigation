from ultralytics import YOLO
import sys
import os
import cv2
import numpy as np
import torch
import rospkg
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from config import *
from PIL import Image

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

#----------------------------------------------------------------#

def load_yolo():
    try:
        # load the model
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        print("Error in load_yolo: %s", e)
        return None

# zero shot object detection with grounding dino base
def grounding_dino_base_hf(image):
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    return model, processor
    
def load_grounding_dino_base():
    try:
        # Simple import since grounding_dino is in sys.path
        from groundingdino.util.inference import load_model
        weigths_path = os.path.join(PACKAGE_PATH, 'weights/groundingdino_swint_ogc.pth')
        model_path = os.path.join(PACKAGE_PATH, 'src/door_navigation/py_packages/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py')
        model = load_model(model_path, weigths_path)
        return model
    except Exception as e:
        print("Error in grounding_dino_base: %s", e)
        return None

# depth estimation with DepthAnything
def load_depth_anything_v2(): # depth estimation with DepthAnything v2 with relative depth values
    # Simple import since py_packages is in sys.path
    from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vits'  # choose encoder type
    model = DepthAnythingV2(**model_configs[encoder])
    
    # Use relative path from package
    checkpoint_path = os.path.join(PACKAGE_PATH, f'checkpoints/depth_anything_v2_{encoder}.pth')
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False))
    model.eval()
    return model

def depth_anything_v2_metric():  # depth estimation with DepthAnything v2 metric with depth values in meters

    from depth_anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

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
    return model

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

def visualize_depth(depth_img):
    # Keep the original depth values for display
    depth_display = depth_img.copy()
    
    # For visualization only: normalize to 0-255 for colormap
    depth_min = np.min(depth_img)
    depth_max = np.max(depth_img)
    depth_normalized = (depth_img - depth_min) / (depth_max - depth_min) if depth_max > depth_min else np.zeros_like(depth_img)
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

