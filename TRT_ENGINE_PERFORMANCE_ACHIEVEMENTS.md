# TensorRT Engine Conversion Achievements

Date: 2026-04-16

## Summary
The Depth Anything V2 TensorRT engine pipeline is now fully operational and benchmarked against the PyTorch reference path on the same test image.

## Key Achievements

1. Successful TensorRT deployment
- ONNX model converted and TensorRT engine generated successfully.
- Runtime pipeline executes end-to-end without crashes.

2. Real image end-to-end inference working
- Inference now runs on a real RGB test image (not dummy data).
- Depth outputs are produced at original image resolution.

3. Full timing instrumentation added
- Per-stage timing is now captured:
  - Preprocess
  - TensorRT inference (H2D + execute + D2H)
  - Postprocess
  - End-to-end

4. Numerical validation against PyTorch reference
- Added side-by-side TRT vs PyTorch comparison on the same input.
- Added quantitative error metrics and saved visual artifacts.

5. Visualization and diagnostics improved
- Color depth visualization (TURBO colormap) is saved.
- Torch visualization and absolute-difference maps are saved for review.

## Measured Performance (Latest Run)

### TensorRT (single-image)
- Preprocess: 12.89 ms
- TRT compute path (H2D + execute + D2H): 66.25 ms
- Postprocess: 7.87 ms
- End-to-end: 87.02 ms

### PyTorch reference (single-image)
- Model load: 2850.79 ms
- Inference: 120.70 ms
- End-to-end (including load): 4673.42 ms

## Accuracy / Agreement (TRT vs PyTorch)
- Valid pixels: 307200
- MAE: 0.0517 m
- RMSE: 0.0560 m
- AbsRel: 0.0141
- MaxAbs: 0.1690 m

Interpretation:
- TensorRT provides strong runtime gains while maintaining close numerical agreement with the reference model.
- Error levels are low enough to support practical depth-based downstream tasks.

## Runtime Outputs Generated
- TRT depth color visualization:
  - scripts/data_new/depth_trt_viz_lab_19_color.png
- PyTorch depth color visualization:
  - scripts/data_new/depth_torch_viz_lab_19_color.png
- TRT vs PyTorch absolute-difference map:
  - scripts/data_new/depth_trt_vs_torch_absdiff_lab_19.png

## Engineering Improvements Completed During Integration
- Fixed NumPy compatibility issue with TensorRT bindings (`np.bool` deprecation behavior).
- Fixed GPU buffer allocation type casting for PyCUDA (`int` byte sizes).
- Implemented explicit CUDA context management for stable standalone execution.
- Hardened output shape handling in postprocess with validation and fallback reshape.
- Matched Depth Anything postprocess behavior more closely using bilinear upsampling with align_corners=True.

## Recommended Next Benchmark Step
For publication-grade benchmarking, run warmup + multi-run averages (for example 50-200 iterations) and report:
- Mean latency
- P50 / P90 / P95 / P99 latency
- Throughput (FPS)
- GPU memory usage
