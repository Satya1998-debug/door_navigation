import numpy as np
import os
import torch
import torch.onnx
import sys
import rospkg
from ultralytics import YOLO

# ------ path setup -----
try:
    rospack = rospkg.RosPack()
    PACKAGE_PATH = rospack.get_path('door_navigation')
except (rospkg.ResourceNotFound, rospkg.common.ResourceNotFound):
    PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"[ROS-Interface] rospkg not available, using relative path: {PACKAGE_PATH}")

script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    
# Add py_packages to path so we can import depth_anything_v2, etc.
py_packages_path = os.path.join(PACKAGE_PATH, 'src/door_navigation/py_packages')
if py_packages_path not in sys.path:
    sys.path.insert(0, py_packages_path)

depth_anything_v2_path = os.path.join(py_packages_path, 'depth_anything_v2')
if depth_anything_v2_path not in sys.path:
    sys.path.insert(0, depth_anything_v2_path)


def export_DAv2_onnx():    
    encoder = "vits"
    input_size = 518
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Exporting Depth-Anything-V2 with encoder {encoder} to ONNX format on device {DEVICE}...")
    
    # we are undergoing company review procedures to release Depth-Anything-Giant checkpoint
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    from depth_anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_hypersim_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Define dummy input data
    dummy_input = torch.randn(1, 3, input_size, input_size).to(DEVICE)
    # Provide an example input to the model, this is necessary for exporting to ONNX
    example_output = depth_anything.forward(dummy_input)

    onnx_path = f'/home/ias/satya/catkin_ws/src/door_navigation/checkpoints/depth_anything_v2_{encoder}.onnx'

    # Export the PyTorch model to ONNX format
    torch.onnx.export(depth_anything, 
                      dummy_input, 
                      onnx_path, 
                      opset_version=11, 
                      input_names=["input"], 
                      output_names=["output"], 
                      verbose=False)

    print(f"Model exported to {onnx_path}")


def export_yolo_trt():
    from ultralytics import YOLO

    # Load your trained custom model
    model = YOLO('/home/ias/satya/catkin_ws/src/door_navigation/weights/last_yolo11m_ias_door_type1.pt')

    # Export to TensorRT with Jetson-stable settings.
    # Notes:
    # - dynamic=True can cause profile/runtime mismatch issues on some TRT builds.
    # - nms=True bakes end-to-end NMS into engine and can be fragile across stacks.
    # - fixed shape + nms=False is the most robust baseline.
    path = model.export(
        format='engine',
        device='cuda:0',
        imgsz=640,
        half=True,
        dynamic=False,
        nms=False,
        batch=1,
        simplify=False,
        workspace=4,
    )

    print(f"TensorRT model saved at: {path}")
    
def export_yolo_onnx():
    from ultralytics import YOLO

    # Load your trained custom model
    model = YOLO('/home/ias/satya/catkin_ws/src/door_navigation/weights/last_yolo11m_ias_door_type1.pt')

    # Export to ONNX format
    path = model.export(
        format='onnx',
        device=0,
        half=True,
        dynamic=False,
        batch=1,
        simplify=True,
        opset=11,
    )

    print(f"ONNX model saved at: {path}")

if __name__ == "__main__":
    # export_DAv2_onnx()
    # export_yolo_trt()
    export_yolo_onnx()