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

if __name__ == "__main__":
    depth_anything_v3() # uses huggingface python 3.10 venv, only estimates relative depth