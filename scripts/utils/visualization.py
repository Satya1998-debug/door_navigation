import numpy as np
import cv2
from door_navigation.scripts.utils.utils import crop_to_bbox_rgb
import open3d as o3d
from config import CX, FX

def visualize_roi(rgb_rs, door_bbox, roi_depth):
    try:
        roi_rgb = crop_to_bbox_rgb(rgb_rs, door_bbox) # crop RGB for visualization
        roi_depth_clean = np.nan_to_num(roi_depth, nan=0.0) # replace nan with 0 for visualization
        depth_max = np.max(roi_depth_clean)
        if depth_max > 0:
            depth_normalized = np.clip(roi_depth_clean / depth_max, 0, 1)
            depth_viz = (depth_normalized * 255).astype(np.uint8)
            depth_viz_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
        else:
            depth_viz_color = np.zeros_like(roi_rgb)
            
        cv2.imshow("ROI RGB", roi_rgb)
        cv2.imshow("ROI Depth", depth_viz_color)
        # visualize ring mask on original image
        mask_viz = rgb_rs.copy()
        cv2.imshow("Wall Region on Image", mask_viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error in visualize_roi: {e}")
        return 
    
def visualize_plane_with_normal(inlier_points, normal_vector):
    try:
        inlier_centre = np.median(inlier_points, axis=0).astype(np.float32)  # inlier/door/wall center point
        print(f"Door/Inlier center: X={inlier_centre[0]:.3f}m, Y={inlier_centre[1]:.3f}m, Z={inlier_centre[2]:.3f}m")
        print(f"Normal vector: [{normal_vector[0]:.3f}, {normal_vector[1]:.3f}, {normal_vector[2]:.3f}]")
        
        # point cloud from inliers
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inlier_points)
        pcd.paint_uniform_color([0.0, 0.7, 0.0])  # green for visualization of inlier points
        
        # plane mesh (large rectangle aligned with plane)
        extent = np.max(np.abs(inlier_points - inlier_centre), axis=0)
        size = np.max(extent) * 1.5
        
        # tangent vectors perpendicular to normal for plane rectangle
        if abs(normal_vector[0]) < 0.9:
            tangent1 = np.cross(normal_vector, [1, 0, 0])
        else:
            tangent1 = np.cross(normal_vector, [0, 1, 0])
        tangent1 = tangent1 / np.linalg.norm(tangent1)
        tangent2 = np.cross(normal_vector, tangent1)
        
        # 4 corners of plane rectangle
        vertices = [
            inlier_centre + size * tangent1 + size * tangent2,
            inlier_centre - size * tangent1 + size * tangent2,
            inlier_centre - size * tangent1 - size * tangent2,
            inlier_centre + size * tangent1 - size * tangent2,
        ]
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        plane_mesh.triangles = o3d.utility.Vector3iVector([[0,1,2], [0,2,3]])
        plane_mesh.paint_uniform_color([0.8, 0.8, 0.0])  # yellow plane
        plane_mesh.compute_vertex_normals()
        
        # normal vector arrow (RED)
        arrow_length = 0.7  # 70cm arrow
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.02,
            cone_radius=0.05,
            cylinder_height=arrow_length * 0.7,
            cone_height=arrow_length * 0.3
        )
        arrow.paint_uniform_color([1.0, 0.0, 0.0])  # red arrow for normal vector
        
        # Rotate arrow to align with normal vector
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, normal_vector)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(z_axis, normal_vector), -1, 1))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
            arrow.rotate(R, center=[0, 0, 0])
        arrow.translate(inlier_centre)
        
        # Camera viewing direction arrow (BLUE) - shows where camera is looking
        camera_origin = np.array([0.0, 0.0, 0.0])
        camera_dir = inlier_centre / np.linalg.norm(inlier_centre)  # unit vector to door
        camera_arrow_length = 0.8
        
        camera_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.015,
            cone_radius=0.03,
            cylinder_height=camera_arrow_length * 0.7,
            cone_height=camera_arrow_length * 0.3
        )
        camera_arrow.paint_uniform_color([0.0, 0.5, 1.0])  # blue arrow for camera direction
        
        # Rotate camera arrow to point at door
        rotation_axis_cam = np.cross(z_axis, camera_dir)
        if np.linalg.norm(rotation_axis_cam) > 1e-6:
            rotation_axis_cam = rotation_axis_cam / np.linalg.norm(rotation_axis_cam)
            angle_cam = np.arccos(np.clip(np.dot(z_axis, camera_dir), -1, 1))
            R_cam = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis_cam * angle_cam)
            camera_arrow.rotate(R_cam, center=[0, 0, 0])
        camera_arrow.translate(camera_origin)
        
        # Line from camera to door center (CYAN)
        camera_to_door_line = o3d.geometry.LineSet()
        camera_to_door_line.points = o3d.utility.Vector3dVector([camera_origin, inlier_centre])
        camera_to_door_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        camera_to_door_line.colors = o3d.utility.Vector3dVector([[0, 1, 1]])  # cyan
        
        # coordinate frame at camera origin
        coord_frame_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=camera_origin
        )
        
        # coordinate frame at door center
        coord_frame_door = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=inlier_centre
        )
        
        print("\nVisualization Legend:")
        print("  - Green points: Plane inlier points")
        print("  - Yellow mesh: Fitted plane")
        print("  - RED arrow at door: Surface normal (points away from surface)")
        print("  - BLUE arrow from camera: Camera viewing direction")
        print("  - Cyan line: Camera to door center")
        print("  - RGB axes: X=Right, Y=Down, Z=Forward (ROS optical frame)")
        
        # visualization
        o3d.visualization.draw_geometries(
            [pcd, plane_mesh, arrow, camera_arrow, camera_to_door_line, 
             coord_frame_camera, coord_frame_door],
            window_name="Door Plane & Normal Visualization",
            width=1280, height=720,
            point_show_normal=False
        )
    except Exception as e:
        print(f"Error in visualize_plane_with_normal: {e}")
        return

def visualize_door_passability(
    depth, bbox,
    xv_valid, yv_valid,
    slab_xv, slab_yv,
    X, Y, X_all, Y_all,
    z, slab_z,
    z_center,
    depth_slab_thickness,
    robot_effective_width,
    passable,
    num_bins=30,
    visualize_2d=True,
    visualize_3d=True
):
    """
    Visualize door passability using GAP-BASED logic.
    Shows the largest FREE opening instead of occupied span.
    """

    x1, y1, x2, y2 = map(int, bbox)

    # -----------------------------
    # GAP COMPUTATION (shared)
    # -----------------------------
    x_min, x_max = np.min(X), np.max(X)
    bins = np.linspace(x_min, x_max, num_bins + 1)

    occupied = np.zeros(num_bins, dtype=bool)
    for x in X:
        idx = np.searchsorted(bins, x) - 1
        if 0 <= idx < num_bins:
            occupied[idx] = True

    # Find largest free gap
    max_free_bins = 0
    current_free = 0
    end_bin = 0

    for i, occ in enumerate(occupied):
        if not occ:
            current_free += 1
            if current_free > max_free_bins:
                max_free_bins = current_free
                end_bin = i
        else:
            current_free = 0

    start_bin = end_bin - max_free_bins + 1
    bin_width = (x_max - x_min) / num_bins
    max_free_width = max_free_bins * bin_width

    gap_x_start = bins[start_bin]
    gap_x_end = bins[end_bin + 1]

    # =============================
    # 3D VISUALIZATION
    # =============================
    if visualize_3d:
        import open3d as o3d

        # All points
        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(
            np.stack([X_all, Y_all, z], axis=1)
        )
        pcd_all.paint_uniform_color([0.7, 0.7, 0.7])

        # Slab points
        pcd_slab = o3d.geometry.PointCloud()
        pcd_slab.points = o3d.utility.Vector3dVector(
            np.stack([X, Y, slab_z], axis=1)
        )
        pcd_slab.paint_uniform_color([0.0, 1.0, 0.0])

        # Slab bounding box
        y_min, y_max = np.min(Y), np.max(Y)
        z_min = z_center - depth_slab_thickness / 2
        z_max = z_center + depth_slab_thickness / 2

        slab_bbox_points = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max],
        ]

        slab_bbox_lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ]

        slab_bbox = o3d.geometry.LineSet()
        slab_bbox.points = o3d.utility.Vector3dVector(slab_bbox_points)
        slab_bbox.lines = o3d.utility.Vector2iVector(slab_bbox_lines)
        slab_bbox.colors = o3d.utility.Vector3dVector([[1, 1, 0]] * len(slab_bbox_lines))

        # FREE GAP line (GREEN)
        y_mid = np.median(Y)
        z_mid = z_center

        gap_line = o3d.geometry.LineSet()
        gap_line.points = o3d.utility.Vector3dVector([
            [gap_x_start, y_mid, z_mid],
            [gap_x_end, y_mid, z_mid]
        ])
        gap_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        gap_line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )

        print("\n3D Visualization (GAP-BASED):")
        print(f"  - Max free opening: {max_free_width:.3f} m")
        print(f"  - Robot needs: {robot_effective_width:.3f} m")
        print(f"  - Passable: {passable}")

        o3d.visualization.draw_geometries(
            [pcd_all, pcd_slab, slab_bbox, gap_line, coord_frame],
            window_name=f"Door Passability | Free gap: {max_free_width:.2f}m (Need: {robot_effective_width:.2f}m)",
            width=1280, height=720
        )

    # ---- BIN VISUALIZATION (3D) ----
    bin_lines = []
    bin_colors = []

    y_mid = np.median(Y)
    z_mid = z_center

    for i in range(num_bins):
        x_start = bins[i]
        x_end = bins[i + 1]
        x_center = 0.5 * (x_start + x_end)

        # Vertical line per bin
        p1 = [x_center, y_mid - 0.2, z_mid]
        p2 = [x_center, y_mid + 0.2, z_mid]

        bin_lines.append([p1, p2])

        if occupied[i]:
            bin_colors.append([1, 0, 0])   # red = occupied
        else:
            bin_colors.append([0, 1, 0])   # green = free

    # Convert to Open3D LineSet
    bin_points = []
    bin_indices = []

    for i, (p1, p2) in enumerate(bin_lines):
        bin_points.append(p1)
        bin_points.append(p2)
        bin_indices.append([2*i, 2*i + 1])

    bin_lineset = o3d.geometry.LineSet()
    bin_lineset.points = o3d.utility.Vector3dVector(bin_points)
    bin_lineset.lines = o3d.utility.Vector2iVector(bin_indices)
    bin_lineset.colors = o3d.utility.Vector3dVector(bin_colors)


    # =============================
    # 2D VISUALIZATION
    # =============================
    if visualize_2d:
        import cv2

        viz_depth = depth.copy()
        h, w = viz_depth.shape

        slab_viz = np.zeros((h, w, 3), dtype=np.uint8)

        # All valid points (blue)
        slab_viz[yv_valid, xv_valid] = [255, 0, 0]

        # Slab points (green)
        slab_viz[slab_yv, slab_xv] = [0, 255, 0]

        depth_clean = np.nan_to_num(viz_depth, nan=0.0)
        depth_max = np.max(depth_clean)

        if depth_max > 0:
            depth_norm = np.clip(depth_clean / depth_max, 0, 1)
            depth_img = (depth_norm * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
        else:
            depth_color = np.zeros((h, w, 3), dtype=np.uint8)

        blended = cv2.addWeighted(depth_color, 0.5, slab_viz, 0.5, 0)

        cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 255, 255), 2)

        info = [
            f"Slab center: {z_center:.2f} m",
            f"Thickness: ±{depth_slab_thickness/2:.2f} m",
            f"Max free gap: {max_free_width:.2f} m",
            f"Robot needs: {robot_effective_width:.2f} m",
            f"Passable: {passable}"
        ]

        # ---- BIN VISUALIZATION (2D overlay) ----
        for i in range(num_bins):
            # Map bin x range back to pixel x
            x_bin_start = int((bins[i] * FX / z_center) + CX)
            x_bin_end   = int((bins[i+1] * FX / z_center) + CX)

            color = (0, 255, 0) if not occupied[i] else (0, 0, 255)

            cv2.rectangle(
                blended,
                (x_bin_start, y1),
                (x_bin_end, y2),
                color,
                1
            )

        # Highlight largest free gap
        x_gap_start_px = int((gap_x_start * FX / z_center) + CX)
        x_gap_end_px   = int((gap_x_end   * FX / z_center) + CX)

        cv2.rectangle(
            blended,
            (x_gap_start_px, y1),
            (x_gap_end_px, y2),
            (0, 255, 255),
            3
        )

        for i, txt in enumerate(info):
            cv2.putText(blended, txt, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Door Passability (Gap-Based)", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
