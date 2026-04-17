import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import os
import sys
import importlib

# TensorRT Python bindings in some versions still reference np.bool.
if "bool" not in np.__dict__:
    np.bool = np.bool_

import tensorrt as trt
import pycuda.driver as cuda


def _ensure_depth_anything_import_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.abspath(os.path.join(script_dir, ".."))
    py_packages_path = os.path.join(package_root, "src", "door_navigation", "py_packages")
    da_path = os.path.join(py_packages_path, "depth_anything_v2")

    if py_packages_path not in sys.path:
        sys.path.insert(0, py_packages_path)
    if da_path not in sys.path:
        sys.path.insert(0, da_path)


def infer_depth_anything_torch(
    bgr_image,
    checkpoint_path,
    encoder="vits",
    max_depth=20,
    device=None,
    return_timing=False,
):
    _ensure_depth_anything_import_path()
    dpt_module = importlib.import_module("depth_anything_v2.metric_depth.depth_anything_v2.dpt")
    DepthAnythingV2 = dpt_module.DepthAnythingV2

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    if encoder not in model_configs:
        raise ValueError(f"Unsupported encoder '{encoder}'. Use one of: {list(model_configs.keys())}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    model = DepthAnythingV2(**{**model_configs[encoder], "max_depth": max_depth})
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    t1 = time.perf_counter()

    # Warmup helps avoid first-call overhead contaminating timing too much.
    _ = model.infer_image(bgr_image)

    t2 = time.perf_counter()
    depth = model.infer_image(bgr_image).astype(np.float32)
    t3 = time.perf_counter()

    timing = {
        "torch_model_load_ms": (t1 - t0) * 1000.0,
        "torch_infer_ms": (t3 - t2) * 1000.0,
        "torch_e2e_ms": (t3 - t0) * 1000.0,
    }

    if return_timing:
        return depth, timing
    return depth


def compare_depth_outputs(depth_trt, depth_torch):
    if depth_trt.shape != depth_torch.shape:
        raise ValueError(f"Shape mismatch: TRT={depth_trt.shape}, Torch={depth_torch.shape}")

    trt = depth_trt.astype(np.float32)
    ref = depth_torch.astype(np.float32)

    diff = trt - ref
    abs_diff = np.abs(diff)

    valid = np.isfinite(trt) & np.isfinite(ref) & (ref > 1e-6)
    if not np.any(valid):
        return {
            "valid_pixels": 0,
            "mae_m": float("nan"),
            "rmse_m": float("nan"),
            "abs_rel": float("nan"),
            "max_abs_m": float("nan"),
        }, diff

    mae = float(np.mean(abs_diff[valid]))
    rmse = float(np.sqrt(np.mean((diff[valid]) ** 2)))
    abs_rel = float(np.mean(abs_diff[valid] / np.maximum(ref[valid], 1e-6)))
    max_abs = float(np.max(abs_diff[valid]))

    metrics = {
        "valid_pixels": int(np.sum(valid)),
        "mae_m": mae,
        "rmse_m": rmse,
        "abs_rel": abs_rel,
        "max_abs_m": max_abs,
    }
    return metrics, diff

class DepthAnythingTRT:
    def __init__(self, engine_path):
        cuda.init()
        self.cuda_ctx = cuda.Device(0).make_context()

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 1. Load the engine
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 2. Allocate GPU memory
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        for i in range(self.engine.num_bindings):
            is_input = self.engine.binding_is_input(i)
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            size = np.prod(shape)
            
            # Allocate memory on GPU
            nbytes = int(size) * np.dtype(dtype).itemsize
            allocation = cuda.mem_alloc(int(nbytes))
            self.allocations.append(int(allocation))
            
            binding = {
                'name': name,
                'dtype': dtype,
                'shape': shape,
                'allocation': allocation,
                'size': size
            }
            
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        self.input_h = int(self.inputs[0]['shape'][-2])
        self.input_w = int(self.inputs[0]['shape'][-1])
        
        # CRITICAL: Pop the context so it's not "blocking" the thread
        self.cuda_ctx.pop()

    def __del__(self):
        if hasattr(self, "cuda_ctx") and self.cuda_ctx is not None:
            try:
                self.cuda_ctx.pop()
            except Exception:
                pass

    def preprocess(self, bgr_image):
        """
        Prepares raw OpenCV image for the engine.

        Note: DepthAnything PyTorch infer_image uses keep_aspect_ratio=True with
        resize_method='lower_bound'. For fixed-shape TRT engines (e.g. 518x518),
        keep_aspect_ratio must be False.
        """
        target_w, target_h = self.input_w, self.input_h

        # Convert BGR to RGB
        img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Resize to engine's expected input using INTER_CUBIC to match DepthAnything preprocess.
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Normalize (0-1 range and ImageNet stats)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # HWC to CHW format
        img = img.transpose(2, 0, 1)
        # Add batch dimension to match DepthAnything input
        img = np.ascontiguousarray(img[None, ...])
        return img

    def postprocess(self, output_data, original_shape):
        """
        Converts raw engine output back to a usable depth map (metric depth-like model output).
        """
        # Squeeze singleton dimensions first (e.g., [1,1,H,W] -> [H,W]).
        depth_map = np.squeeze(output_data).astype(np.float32)

        if depth_map.ndim != 2:
            if depth_map.size == self.input_h * self.input_w:
                depth_map = depth_map.reshape(self.input_h, self.input_w)
            else:
                raise ValueError(
                    f"Unexpected TRT output shape {output_data.shape}; "
                    f"cannot map to depth of size ({self.input_h}, {self.input_w})."
                )

        out_h, out_w = original_shape[:2]
        depth_tensor = torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0)
        depth_up = F.interpolate(
            depth_tensor,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=True,
        )
        return depth_up[0, 0].cpu().numpy().astype(np.float32)

    @staticmethod
    def normalize_for_visualization(depth_map):
        """Return 0..1 normalized depth only for visualization."""
        depth_min, depth_max = float(depth_map.min()), float(depth_map.max())
        if depth_max - depth_min < 1e-6:
            return np.zeros_like(depth_map, dtype=np.float32)
        return ((depth_map - depth_min) / (depth_max - depth_min)).astype(np.float32)

    def infer_image_no_cudacontext_management(self, bgr_image):
        """
        Main pipeline: Pre -> Inference -> Post
        """
        orig_shape = bgr_image.shape
        
        # 1. Preprocess
        input_data = self.preprocess(bgr_image)
        
        # 2. Inference
        cuda.memcpy_htod(self.inputs[0]['allocation'], input_data)
        
        # Execute
        self.context.execute_v2(self.allocations)
        
        # Copy output back to CPU
        output_data = np.zeros(self.outputs[0]['shape'], dtype=self.outputs[0]['dtype'])
        cuda.memcpy_dtoh(output_data, self.outputs[0]['allocation'])
        
        # 3. Postprocess
        depth = self.postprocess(output_data, orig_shape)

        return depth
    
    def infer_image(self, bgr_image):
        orig_shape = bgr_image.shape
        input_data = self.preprocess(bgr_image)
        
        # --- ACTIVATE ---
        self.cuda_ctx.push()
        
        try:
            # Move to GPU
            cuda.memcpy_htod(self.inputs[0]['allocation'], input_data)
            
            # Run math
            self.context.execute_v2(self.allocations)
            
            # Bring results back to CPU
            output_data = np.zeros(self.outputs[0]['shape'], dtype=self.outputs[0]['dtype'])
            cuda.memcpy_dtoh(output_data, self.outputs[0]['allocation'])
        finally:
            # --- DEACTIVATE ---
            # Using 'finally' ensures the context is popped even if the code crashes
            self.cuda_ctx.pop()
        
        # Postprocess (on CPU)
        depth = self.postprocess(output_data, orig_shape)
        return depth

# Example Usage:
if __name__ == "__main__":
    engine_file = "/home/ias/satya/catkin_ws/src/door_navigation/checkpoints/depth_anything_v2_vits.engine"
    checkpoint_file = "/home/ias/satya/catkin_ws/src/door_navigation/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
    model = DepthAnythingTRT(engine_file)
    
    test_image_path = "/home/ias/satya/catkin_ws/src/door_navigation/scripts/data_new/latest_image_color_lab_19.jpg"
    img = cv2.imread(test_image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load test image: {test_image_path}")
    
    depth, timing = model.infer_image_trt(img, return_timing=True)
    print(f"Depth map shape: {depth.shape}")
    print(
        f"Depth stats: min={float(depth.min()):.4f}, "
        f"max={float(depth.max()):.4f}, mean={float(depth.mean()):.4f}"
    )
    print(
        "Timing (ms): "
        f"pre={timing['preprocess_ms']:.2f}, "
        f"trt={timing['trt_infer_ms']:.2f}, "
        f"post={timing['postprocess_ms']:.2f}, "
        f"e2e={timing['e2e_ms']:.2f}"
    )

    # Torch reference inference and TRT-vs-Torch depth comparison.
    depth_torch, torch_timing = infer_depth_anything_torch(
        bgr_image=img,
        checkpoint_path=checkpoint_file,
        encoder="vits",
        max_depth=20,
        device=None,
        return_timing=True,
    )
    print(
        "Torch Timing (ms): "
        f"load={torch_timing['torch_model_load_ms']:.2f}, "
        f"infer={torch_timing['torch_infer_ms']:.2f}, "
        f"e2e={torch_timing['torch_e2e_ms']:.2f}"
    )

    metrics, diff = compare_depth_outputs(depth, depth_torch)
    print(
        "TRT vs Torch: "
        f"valid={metrics['valid_pixels']}, "
        f"MAE={metrics['mae_m']:.4f} m, "
        f"RMSE={metrics['rmse_m']:.4f} m, "
        f"AbsRel={metrics['abs_rel']:.4f}, "
        f"MaxAbs={metrics['max_abs_m']:.4f} m"
    )

    depth_viz = (DepthAnythingTRT.normalize_for_visualization(depth) * 255.0).astype(np.uint8)
    depth_viz_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_TURBO)
    out_path = "/home/ias/satya/catkin_ws/src/door_navigation/scripts/data_new/depth_trt_viz_lab_19_color.png"
    cv2.imwrite(out_path, depth_viz_color)
    print(f"Saved color visualization: {out_path}")

    torch_viz = (DepthAnythingTRT.normalize_for_visualization(depth_torch) * 255.0).astype(np.uint8)
    torch_viz_color = cv2.applyColorMap(torch_viz, cv2.COLORMAP_TURBO)
    torch_out_path = "/home/ias/satya/catkin_ws/src/door_navigation/scripts/data_new/depth_torch_viz_lab_19_color.png"
    cv2.imwrite(torch_out_path, torch_viz_color)
    print(f"Saved torch color visualization: {torch_out_path}")

    diff_abs_viz = (DepthAnythingTRT.normalize_for_visualization(np.abs(diff)) * 255.0).astype(np.uint8)
    diff_out_path = "/home/ias/satya/catkin_ws/src/door_navigation/scripts/data_new/depth_trt_vs_torch_absdiff_lab_19.png"
    cv2.imwrite(diff_out_path, diff_abs_viz)
    print(f"Saved abs-diff visualization: {diff_out_path}")