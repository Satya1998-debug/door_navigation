
import ctypes
import sys
import os
# fix for PyTorch libgomp TLS allocation issue - preload libgomp before torch imports
try:
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass  # libgomp already loaded or not found



import base64
import time
import json
from ollama import chat
import cv2
import numpy as np

# Path setup
import rospkg
try:
    rospack = rospkg.RosPack()
    PACKAGE_PATH = rospack.get_path('door_navigation')
except (rospkg.ResourceNotFound, rospkg.common.ResourceNotFound):
    PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"[door-pose-estimator] rospkg not available, using relative path: {PACKAGE_PATH}")


from door_ros_interfaces import DoorDetector
from door_pose_estimator_utils import fit_plane, project_to_3d, visualize_plane_with_normal
from utils.visualization import visualize_door_passability, visualize_roi
from utils.utils import crop_to_bbox_depth, expand_bbox, divide_bbox, ring_mask

LEAF_MAX_NORMAL_DEV_DEG = 25.0   # leaf normal within this angle of the reference plane >> leaf considered closed
LEAF_MAX_DEPTH_GAP_M = 0.40      # leaf median depth farther than the reference by more than this >> leaf considered open
LEAF_MIN_POINTS = 100            # minimum valid depth points on a leaf ROI to attempt a plane fit

# (used when no reliable wall/frame is available i.e. recessed doors, or a doorframe whose surrounding wall is perpendicular to the door plane)
LEAF_PAIR_MAX_ANGLE_DEG = 15.0   # threshold for the leaf-vs-leaf coplanarity test used to build a reference plane 
LEAF_PAIR_MAX_DEPTH_GAP_M = 0.30 # ...and at similar median depth

# the wall/frame plane is only trusted as a "closed reference" when its normal
# is within this angle of at least one fitted leaf normal. Rejects the classic
# false-negative case where the ring around the door picks up an alcove side-wall or protruding frame that is perpendicular to the actual door plane
WALL_ALIGNMENT_MAX_DEG = 30.0

ROBOT_WIDTH = 0.5  # in meters, robot width for door pass check
EXPANSION_RATIO = 0.2  # ratio to expand door bbox for wall plane fitting
LABEL_MAP = {0: 'door_double', 1: 'door_single', 2: 'handle'}
S_DOOR_OPEN_THRESHOLD = 60
S_DOOR_CLOSED_THRESHOLD = 15
S_DOOR_MAX_ANGLE = 90
D_DOOR_OPEN_THRESHOLD = 60
D_DOOR_CLOSED_THRESHOLD = 30
SAFETY_MARGIN_WIDTH = 0.1  # in meters, additional margin for door pass check
SLAB_THICKNESS = 0.3  # in meters, thickness of depth slab for door pass check
MIN_POINTS_PASS_CHECK = 30  # minimum valid depth points in slab region for door pass check


def warmup_ollama_vlm(model='qwen3-vl:4b-instruct'):
    """Warm up Ollama VLM once to reduce first-call latency."""
    try:
        s_time = time.time()
        _ = chat(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': 'Reply only with valid JSON: {"status":"ok"}'
                }
            ],
            format='json'
        )
        print(f"VLM warmup completed in {time.time() - s_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"VLM warmup failed: {e}")
        return False

def estimate_door_state_ollama_vlm(rgb_img, is_passable="", door_open_percent="", door_wall_angle="", left_right_door_angle="", door_type=""):
    # directly use ollama api to estimate door state
    try:
        # encode OpenCV image (BGR) as JPEG
        ok, buf = cv2.imencode('.jpg', rgb_img)
        if not ok:
            raise RuntimeError(f"Failed to encode image.")
        rgb_img_bytes = buf.tobytes() 
        img_b64 = base64.b64encode(rgb_img_bytes).decode('utf-8')

        prompt = f"""
            You are a robot perception assistant. Visually verify the door state in the image.

            Classify door state as: "open", "semi_open", "closed", or "unknown"
            Detect if a human is clearly visible near the door.

            Preliminary data (may be noisy):
            - is_passable: {is_passable}, open_percent: {door_open_percent}, wall_angle: {door_wall_angle}
            - lr_angle: {left_right_door_angle}, type: {door_type}

            Rules:
            - Fully open or unobstructed → "open"
            - Partially open → "semi_open"
            - Clearly closed → "closed"
            - Ambiguous/occluded → "unknown"
            - Human visible → "yes", otherwise "no"

            Generate a SHORT, polite spoken sentence ALWAYS:
            - If human present → request appropriate action (open door, open more, or keep open)
            - If no human → briefly describe the door scene (e.g., "The door appears closed" or "I see an open doorway"), then request if anyone can please open.
            
            Output ONLY valid JSON:
            {{
                "door_state": "<open|semi_open|closed|unknown>",
                "human_present": "<yes|no>",
                "conversation": "<single short sentence, always required>"
            }}
            """

        response = chat(
            model='qwen3-vl:4b-instruct',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64],
                }
            ],
            format="json"
        )

        """
        res = {
            'door_state': 'open',
            'human_present': 'no',
            'conversation': 'please open the door'
        }
        """
        # print(f"Ollama API response: {response}")

        res = response.message.content.strip()
        if res:
            parsed = json.loads(res)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)

            door_state = str(parsed.get("door_state", "unknown")).strip().lower()
            human_present = str(parsed.get("human_present", "no")).strip().lower()
            conversation = str(parsed.get("conversation", "Please open the door.")).strip()

            return {
                "door_state": door_state,
                "human_present": human_present,
                "conversation": conversation,
            }
        else:
            print("No valid response received from Ollama API.")
            return None
    
    except Exception as e:
        print(f"Error during estimate_door_state_ollama_api: {e}")
        return None

def make_fallback_conversation(door_state, door_type, is_passable):
    """Human-friendly spoken sentence used when VLM is disabled or fails.

    Called on the geometric path so the coordinator's `_speak(response.conversation)`
    plays something meaningful instead of the literal string "NA".
    """
    kind = "double door" if door_type == "double" else "door"
    state = (door_state or "").lower()
    if state == "open":
        if is_passable:
            return f"The {kind} is open. I can pass through."
        # open but not passable → something is blocking the opening
        return f"The {kind} appears open but the opening is too narrow to pass."
    if state == "semi_open":
        return f"The {kind} is only partially open. Could someone please open it fully?"
    if state == "closed":
        return f"The {kind} is closed. Could someone please open it?"
    return f"I cannot clearly tell the {kind} state. Please check."

def calculate_door_opening_angle(n1, n2):
    ang = np.arccos(np.clip(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)), -1.0, 1.0))
    angle_deg = np.degrees(ang)
    return angle_deg

def calculate_door_state_single(angle_deg, 
                                open_threshold=S_DOOR_OPEN_THRESHOLD, 
                                closed_threshold=S_DOOR_CLOSED_THRESHOLD):
    # calculate door opening percentage
    door_open_percent = angle_deg / S_DOOR_MAX_ANGLE * 100.0
    if door_open_percent > 100:
        door_open_percent = -1.0 # this means invalid value

    if angle_deg >= open_threshold:
        return 'open', door_open_percent
    elif angle_deg <= closed_threshold:
        return 'closed', door_open_percent
    else:
        return 'semi_open', door_open_percent
    
def calculate_door_state_double(angle_deg, 
                                open_threshold=D_DOOR_OPEN_THRESHOLD, 
                                closed_threshold=D_DOOR_CLOSED_THRESHOLD,
                                is_passable=None):
    # if door is passable, return open or semi_open based on opening angle
    if is_passable:
        if angle_deg >= open_threshold:
            return 'open'
        else:
            return 'semi_open'
    else:  #  not passable, return closed or semi_open based on opening angle
        if angle_deg <= closed_threshold:
            return 'closed'
        else:
            return 'semi_open'

def is_door_passable(depth, bbox, FX, CX, 
                     robot_width=ROBOT_WIDTH, safety_margin=SAFETY_MARGIN_WIDTH, 
                     depth_slab_thickness=SLAB_THICKNESS, stride=2, min_points=MIN_POINTS_PASS_CHECK,
                     visualize=False, visualize_3d=False, intrinsics=None):
    
    x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

    # depth ponts inside bbox in the slab region
    xs = np.arange(x1, x2, stride) # interms of full image coordinates
    ys = np.arange(y1, y2, stride) # interms of full image coordinates
    xv, yv = np.meshgrid(xs, ys)

    z = depth[yv, xv]  # depth values (obtained for the 2D meshgrid points) from Full depth
    
    valid_mask = np.isfinite(z) & (z > 0)
    if np.sum(valid_mask) < min_points:
        # not enough valid depth points in slab region for door pass check, so default to not-passable
        print(f"Not enough valid depth points in slab region for door pass check "
              f"({int(np.sum(valid_mask))} < {min_points}). Defaulting to not-passable.")
        return False
    
    xv_valid = xv[valid_mask] # keeping only those meshgrid (horizontal) x points (which has valid depths)
    yv_valid = yv[valid_mask] # keeping only those meshgrid (vertical) y points (which has valid depths)
    z = z[valid_mask] # depth values for the 2d meshgrid points

    # door depth points in meters
    z_center = np.median(z) # centre depth for slab placement
    print(f"Slab center depth: {z_center:.2f} meters")
    print(f"Slab thickness: {depth_slab_thickness:.2f} meters (±{depth_slab_thickness/2:.2f}m)")

    # slab mask (about depth centre)
    slab_mask = (z >= (z_center - depth_slab_thickness/2)) & (z <= (z_center + depth_slab_thickness/2))
    slab_xv = xv_valid[slab_mask]
    slab_yv = yv_valid[slab_mask]
    slab_z = z[slab_mask]

    print(f"Valid points in slab: {len(slab_xv)} / {len(z)} total valid points")

    # convert to 3D points, slab 3D points
    X = (slab_xv - intrinsics['CX']) * slab_z / intrinsics['FX']  # Using CX and FX
    Y = (slab_yv - intrinsics['CY']) * slab_z / intrinsics['FY']  # Using CY and FY

    # all valid points in 3D
    X_all = (xv_valid - intrinsics['CX']) * z / intrinsics['FX']
    Y_all = (yv_valid - intrinsics['CY']) * z / intrinsics['FY']

    # gap based passibility
    num_bins = 30  # tuneable
    x_min, x_max = np.min(X), np.max(X) # to check horizontal width
    bins = np.linspace(x_min, x_max, num_bins + 1)

    occupied = np.zeros(num_bins, dtype=bool)

    for x in X:
        idx = np.searchsorted(bins, x) - 1
        if 0 <= idx < num_bins:
            occupied[idx] = True

    # largest continuous free gap
    # https://websites.umich.edu/~ykoren/uploads/The_Vector_Field_HistogramuFast_Obstacle_Avoidance.pdf
    max_free_bins = 0
    current_free = 0

    for occ in occupied:
        if not occ:
            current_free += 1
            max_free_bins = max(max_free_bins, current_free)
        else:
            current_free = 0

    bin_width = (x_max - x_min) / num_bins
    max_free_width = max_free_bins * bin_width

    print(f"Max free opening width: {max_free_width:.2f} meters")
    robot_effective_width = robot_width + safety_margin  # add margin
    passable = max_free_width >= robot_effective_width
    print(f"Robot effective width (with margin): {robot_effective_width:.2f} meters")
    
    # door is passable if the measured opening width is >= robot's required width
    print(f"Passability result: {passable}")

    # for visualization
    if visualize or visualize_3d:
        visualize_door_passability(depth, bbox, xv_valid, yv_valid, slab_xv, slab_yv,
                                   X, Y, X_all, Y_all, z, slab_z, z_center, depth_slab_thickness, robot_effective_width,
                                   passable, num_bins=30, visualize_2d=True, visualize_3d=True)
    
    return passable

def estimate_single_door_state(door_bbox, rgb_rs, roi_depth, full_depth, visualize=True, use_vlm=False, intrinsics=None):
    try:
        # get bbox coordinates (also the inner bbox for wall fitting)
        if len(door_bbox) == 0:
            print("Empty door box provided for single door state estimation.")
            return None
        
        x1, y1, x2, y2 = (int(door_bbox[0]), int(door_bbox[1]), int(door_bbox[2]), int(door_bbox[3]))
        inner_bbox = (x1, y1, x2, y2)
        print(f"Inner bbox for wall fitting: {inner_bbox}")

        # expand ROI for wall plane fitting, formulate ring mask (only 2 sided wall region)
        img_height, img_width = rgb_rs.shape[:2]
        print(f"Image dimensions: width={img_width}, height={img_height}")

        # get outer bbox (sideways expansion only)
        outer_bbox = expand_bbox(x1, x2, y1, y2, exp_ratio=EXPANSION_RATIO, img_width=img_width, img_height=img_height)
        print(f"Outer bbox for wall fitting: {outer_bbox}")
        
        exp_mask = ring_mask(img_width, img_height, inner_bbox, outer_bbox)
        print(f"Ring mask shape: {exp_mask.shape}")

        if visualize: # visualize ROI
            visualize_roi(rgb_rs, door_bbox, roi_depth, disp_text="single-door")
        
        # fit plane for door
        s_time = time.time()
        points_3d_door = project_to_3d(x1, y1, valid_mask=None, depth=roi_depth, intrinsics=intrinsics)
        door_inliers, door_n, _ = fit_plane(points_3d_door, "singledoor_roi_plane-door")
        if door_n is None or door_inliers is None:
            print("Door plane fit failed")
            return None

        # visualize door plane with normal
        if visualize:
            visualize_plane_with_normal(door_inliers, normal_vector=door_n, disp_text="single-door")

        # fit wall plane
        x1_o, y1_o, _, _ = outer_bbox
        points_3d_wall = project_to_3d(x1_o, y1_o, valid_mask=exp_mask, depth=full_depth, intrinsics=intrinsics)
        wall_inliers, wall_n, _ = fit_plane(points_3d_wall, "singledoor_roi_plane-wall")
        if wall_n is None or wall_inliers is None:
            print("Wall plane fit failed")
            return None
        
        # visualize wall plane with normal
        if visualize:
            visualize_plane_with_normal(wall_inliers, normal_vector=wall_n, disp_text="single-door-wall-plane")

        # calculate door opening angle
        door_opening_angle = calculate_door_opening_angle(door_n, wall_n)
        print(f"Estimated door opening angle: {door_opening_angle} degrees")
        print(f"Plane fitting & angle calculation time: {time.time() - s_time:.2f} seconds")

        # door pass check
        s_time = time.time()
        is_passable = is_door_passable(full_depth, door_bbox, intrinsics['FX'], intrinsics['CX'], visualize=visualize, visualize_3d=visualize, intrinsics=intrinsics)
        print(f"Door passability check time: {time.time() - s_time:.2f} seconds")
        # door state, open percent (NOTE: geometrically to take decision)
        door_state, door_open_percent = calculate_door_state_single(door_opening_angle)
        print(f"Door state based on angle thresholds: {door_state}, open percent: {door_open_percent:.2f}%")

        # VLM based door state estimation (falls back to geometric)

        if use_vlm:
            s_time = time.time()
            door_state_res = estimate_door_state_ollama_vlm(rgb_rs, is_passable=is_passable, 
                                                            door_open_percent=door_open_percent,
                                                            door_wall_angle=door_opening_angle,
                                                            door_type="single")
            print(f"VLM door state estimation time: {time.time() - s_time:.2f} seconds")
            if isinstance(door_state_res, dict):
                door_state_res["is_passable"] = is_passable
                return door_state_res
            print("Invalid VLM response format. Falling back to geometric state.")

        # calculate post door pose
        conversation = make_fallback_conversation(door_state, "single", is_passable)
        return {"door_state": door_state, "human_present": "no",
                "conversation": conversation, "is_passable": is_passable}

    except Exception as e:
        print(f"Error in estimate_single_door_state: {e}")
        return None


def _fit_leaf_plane(leaf_bbox, full_depth, intrinsics, tag):
    """Fit a plane to a single door leaf using its own bbox on the full depth image.

    Returns (inliers, normal, median_z, n_points). Any of these may be None if the
    leaf ROI is too sparse or a plane cannot be fitted.
    """
    leaf_depth = crop_to_bbox_depth(full_depth, leaf_bbox)
    if leaf_depth is None or leaf_depth.size == 0:
        print(f"[{tag}] empty leaf depth crop")
        return None, None, None, 0

    x1, y1, _, _ = int(leaf_bbox[0]), int(leaf_bbox[1]), int(leaf_bbox[2]), int(leaf_bbox[3])
    pts = project_to_3d(x1, y1, valid_mask=None, depth=leaf_depth, intrinsics=intrinsics) # project 2D points to 3D points
    n_pts = 0 if pts is None else len(pts)
    print(f"[{tag}] valid 3D points: {n_pts}")
    if n_pts < LEAF_MIN_POINTS:
        return None, None, None, n_pts

    inliers, normal, _ = fit_plane(pts, "", min_points=LEAF_MIN_POINTS)
    if inliers is None or normal is None:
        # retry once with looser thresholds (helps on textured/noisy leaves)
        print(f"[{tag}] strict plane fit failed; retrying with laxer thresholds")
        inliers, normal, _ = fit_plane(pts, "", distance_threshold=0.04, min_inlier_ratio=0.18, min_points=LEAF_MIN_POINTS,)
    if inliers is None or normal is None:
        return None, None, None, n_pts

    median_z = float(np.median(np.asarray(inliers)[:, 2]))
    return inliers, normal, median_z, n_pts # inliers: 3D points, normal: normal vector, median_z: median depth, n_pts: number of valid points


def _classify_leaf(leaf_normal, leaf_z, wall_normal, wall_z, tag):
    """Classify a single leaf as 'open' or 'closed' relative to a wall/frame reference plane."""
    if leaf_normal is None:
        # plane fit failed >>> the region is almost certainly not a coherent door surface,
        # which is what we expect when that leaf is open.
        print(f"[{tag}] no leaf plane → treating as OPEN")
        return "open"
    if wall_normal is None:
        return None  # caller falls back to pairwise-angle logic (i.e. if wall plane fit failed)
    # both normals are flipped to face camera by fit_plane, so a direct dot is fine
    cos_ang = float(np.clip(np.dot(leaf_normal, wall_normal), -1.0, 1.0))
    ang_deg = float(np.degrees(np.arccos(cos_ang)))
    depth_gap = None if (leaf_z is None or wall_z is None) else float(leaf_z - wall_z)
    if ang_deg > LEAF_MAX_NORMAL_DEV_DEG:
        print(f"[{tag}] normal deviates {ang_deg:.1f}° from ref (> {LEAF_MAX_NORMAL_DEV_DEG:.1f}°) → OPEN")
        return "open"
    if depth_gap is not None and depth_gap > LEAF_MAX_DEPTH_GAP_M:
        print(f"[{tag}] {depth_gap:.2f} m behind ref (> {LEAF_MAX_DEPTH_GAP_M:.2f} m) → OPEN")
        return "open"
    print(f"[{tag}] angle={ang_deg:.1f}°, depth_gap={depth_gap if depth_gap is None else f'{depth_gap:.2f}m'} → CLOSED")
    return "closed"


def _angle_between_deg(a, b):
    """Angle in degrees between two unit vectors, clipped for numerical safety."""
    cos_a = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _pick_closed_reference(left_n, left_z, right_n, right_z, wall_n, wall_z):
    """Pick the most trustworthy "what a CLOSED leaf looks like" reference.

    Priority order (each step falls through to the next only if the current
    signal is unreliable or unavailable):

      1. Wall/frame plane, IF its normal is within ``WALL_ALIGNMENT_MAX_DEG``
         of at least one fitted leaf normal. This is the original flush-wall
         path — preserved unchanged for that case.
      2. The average of both fitted leaves, IF they pass the pairwise
         coplanarity test (angle ≤ ``LEAF_PAIR_MAX_ANGLE_DEG`` and depth
         difference ≤ ``LEAF_PAIR_MAX_DEPTH_GAP_M``). This catches recessed
         doors and doors whose wall/frame is perpendicular to the leaves.
      3. Whichever single leaf plane fit succeeded. A tentative reference
         used only to decide the other (failed) leaf as OPEN.
      4. Wall alone, IF no leaves fit at all (degenerate but better than
         nothing).
      5. No reference — caller falls back to ``is_passable``.

    Returns ``(ref_normal, ref_z, source_tag)``. ``ref_normal`` and ``ref_z``
    are ``None`` when no reference is available.
    """
    # priority 1: wall plane, validated against any fitted leaf
    if wall_n is not None:
        fitted_leaves = [n for n in (left_n, right_n) if n is not None]
        if fitted_leaves:
            min_wall_leaf_ang = min(_angle_between_deg(wall_n, ln) for ln in fitted_leaves)
            if min_wall_leaf_ang <= WALL_ALIGNMENT_MAX_DEG:
                return wall_n, wall_z, f"wall (angle to closest leaf {min_wall_leaf_ang:.1f}°)"

    # priority 2: two fitted leaves in agreement
    if left_n is not None and right_n is not None:
        angle_lr = _angle_between_deg(left_n, right_n)
        depth_diff = abs(float(left_z - right_z))
        if angle_lr <= LEAF_PAIR_MAX_ANGLE_DEG and depth_diff <= LEAF_PAIR_MAX_DEPTH_GAP_M:
            ref_normal = (left_n + right_n) / 2.0
            norm = float(np.linalg.norm(ref_normal))
            if norm > 1e-6:
                ref_normal = ref_normal / norm
                ref_z = (left_z + right_z) / 2.0
                return ref_normal, ref_z, (
                    f"coplanar_leaves (Δang={angle_lr:.1f}°, "
                    f"Δz={depth_diff:.2f}m)")

    # priority 3: exactly one leaf fitted >>> use it as a tentative reference
    # {this is intentionally optimistic: a fitted plane is more likely a real
    # surface than random background, so we lean toward calling the fitted
    # leaf CLOSED and the failed leaf OPEN. In the rare pathological case where the "fitted" side is actually open (background wall) and the "closed" side failed, 
    # we'll only mislabel to semi_open >>> is_passable will typically catch it and prevent unsafe traversal.}
    if left_n is not None and right_n is None:
        return left_n, left_z, "left_leaf_only (right plane fit failed)" # right plane fit failed
    if right_n is not None and left_n is None:
        return right_n, right_z, "right_leaf_only (left plane fit failed)" # left plane fit failed

    # priority 4: wall alone (no leaves fitted) >>> degenerate but preserves the previous behavior of "wall exists >>> use it"
    if wall_n is not None and left_n is None and right_n is None:
        return wall_n, wall_z, "wall (no leaves to cross-check)"

    # priority 5: nothing usable >>> caller falls back to is_passable
    return None, None, "no_reference"


def _combine_leaf_states(left_state, right_state, is_passable):
    """Combine per-leaf open/closed labels into an overall double-door state."""
    if left_state == "closed" and right_state == "closed":
        return "closed"
    if left_state == "open" and right_state == "open":
        return "open" if is_passable else "semi_open"
    # exactly one leaf open
    return "open" if is_passable else "semi_open"


def estimate_double_door_state(door_bbox, rgb_rs, roi_depth, full_depth, visualize=True, use_vlm=False, intrinsics=None):
    try:

        if len(door_bbox) == 0:
            print("Empty door box provided for double door state estimation.")
            return None

        # get bbox coordinates
        x1, y1, x2, y2 = (int(door_bbox[0]), int(door_bbox[1]), int(door_bbox[2]), int(door_bbox[3]))

        img_height, img_width = rgb_rs.shape[:2]
        print(f"Image dimensions: width={img_width}, height={img_height}")

        # divide the double door bbox into two single door bboxes
        left_bbox, right_bbox = divide_bbox(rgb_rs, x1, x2, y1, y2,
                                            exp_ratio=EXPANSION_RATIO,
                                            visualize_bbox=visualize,
                                            img_width=img_width, img_height=img_height)
        print(f"Left door bbox: {left_bbox}, Right door bbox: {right_bbox}")

        if visualize:  # visualize ROI
            visualize_roi(rgb_rs, door_bbox, roi_depth, disp_text="double-door")

        # reference wall/frame plane from a ring around the whole double-door bbox
        s_time = time.time()
        outer_bbox = expand_bbox(x1, x2, y1, y2, exp_ratio=EXPANSION_RATIO, img_width=img_width, img_height=img_height)
        exp_mask = ring_mask(img_width, img_height, (x1, y1, x2, y2), outer_bbox)
        x1_o, y1_o, _, _ = outer_bbox
        points_3d_wall = project_to_3d(x1_o, y1_o, valid_mask=exp_mask, depth=full_depth, intrinsics=intrinsics)
        wall_inliers, wall_n, _ = fit_plane(points_3d_wall, "")
        wall_z = float(np.median(np.asarray(wall_inliers)[:, 2])) if wall_inliers is not None else None
        if wall_n is None:
            print("Wall/frame plane fit failed; will fall back to pairwise leaf-angle method")
        else:
            print(f"Wall plane fitted (median z={wall_z:.2f} m)")
            if visualize:
                visualize_plane_with_normal(wall_inliers, normal_vector=wall_n,
                                            disp_text="double-door-wall-plane")

        # per-leaf plane fits from full_depth using each leaf's own bbox 
        left_inliers,  left_n,  left_z,  _ = _fit_leaf_plane(left_bbox,  full_depth, intrinsics, "left-leaf")
        right_inliers, right_n, right_z, _ = _fit_leaf_plane(right_bbox, full_depth, intrinsics, "right-leaf")

        if visualize and left_inliers is not None:
            visualize_plane_with_normal(left_inliers, normal_vector=left_n, disp_text="double-left-door")
        if visualize and right_inliers is not None:
            visualize_plane_with_normal(right_inliers, normal_vector=right_n, disp_text="double-right-door")

        print(f"Plane fitting time: {time.time() - s_time:.2f} seconds")

        # passability check on the full opening
        s_time = time.time()
        is_passable = is_door_passable(full_depth, door_bbox,
                                       intrinsics['FX'], intrinsics['CX'],
                                       visualize=visualize, visualize_3d=visualize,
                                       intrinsics=intrinsics)
        print(f"Door passability check time: {time.time() - s_time:.2f} seconds")

        # decide overall state
        side_doors_angle = 0.0  # kept for logging / VLM prompt

        # pick the most trustworthy CLOSED-reference we can build.
        # Order: wall (if aligned with a leaf) > coplanar-leaves > single leaf > wall-alone.
        ref_n, ref_z, ref_src = _pick_closed_reference(
            left_n, left_z, right_n, right_z, wall_n, wall_z)
        print(f"CLOSED-reference source: {ref_src}")

        if ref_n is not None:
            # existing classifier semantics preserved: leaf normal + depth vs
            # reference plane. The "reference" is just no-longer-forced to be
            # the wall.
            left_state  = _classify_leaf(left_n,  left_z,  ref_n, ref_z, "left-leaf")
            right_state = _classify_leaf(right_n, right_z, ref_n, ref_z, "right-leaf")
            door_state = _combine_leaf_states(left_state, right_state, is_passable)
            print(f"Leaf states → left={left_state}, right={right_state} ⇒ door_state={door_state}")
        else:
            # No trustworthy reference at all: preserve the previous fallback
            # exactly so the behavior on truly-unreadable frames is unchanged.
            if left_n is not None and right_n is not None:
                side_doors_angle = calculate_door_opening_angle(left_n, right_n)
                print(f"Estimated door opening angle (leaf-vs-leaf): {side_doors_angle:.2f} degrees")
                door_state = calculate_door_state_double(side_doors_angle, is_passable=is_passable)
            elif left_n is None and right_n is None:
                print("Both leaf plane fits failed and no wall reference; treating as unknown/open based on passability")
                door_state = "open" if is_passable else "unknown"
            else:  # when one leaf plane fit failed, we treat the door as semi_open/open based on passability
                print("One leaf plane fit failed and no wall reference; treating as semi_open/open based on passability")
                door_state = "open" if is_passable else "semi_open"

        # optional VLM refinement
        if use_vlm:
            s_time = time.time()
            door_state_res = estimate_door_state_ollama_vlm(rgb_rs, is_passable=is_passable,
                                                            left_right_door_angle=side_doors_angle,
                                                            door_type="double")
            print(f"VLM door state estimation time: {time.time() - s_time:.2f} seconds")
            if isinstance(door_state_res, dict):
                door_state_res["is_passable"] = is_passable
                return door_state_res
            print("Invalid VLM response format. Falling back to geometric state.")

        conversation = make_fallback_conversation(door_state, "double", is_passable)
        return {"door_state": door_state, "human_present": "no",
                "conversation": conversation, "is_passable": is_passable}

    except Exception as e:
        print(f"Error in estimate_double_door_state: {e}")
        import traceback
        traceback.print_exc()
        return None

def estimate_door_state_test(img_path, depth_path, visualize=True, use_vlm=False, intrinsics=None):
    # NOTE: this is executed at the Pre-Pose stage, before robot moves through the door
    try:
        # loads RGB 
        rgb_rs = cv2.imread(img_path) # numpy array HWC
        
        # loads depth map
        # depth_rs = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # convert mm to meters

        door_detector = DoorDetector()  # initialize door detector
        # get RAW depth from DA model (in meters)
        s_time = time.time()
        depth_da = door_detector.run_depth_anything_v2_on_image(rgb_image=rgb_rs, use_trt=False)
        print(f"Depth Anything v2 inference time: {time.time() - s_time:.2f} seconds")
        # apply correction to depth_da_raw using pre-computed calibration coefficients
        depth_da_corr = door_detector.get_corrected_depth_image(depth_da=depth_da, model="quad")

        # get bounding box, make detection object
        detections = door_detector.run_yolo_model(rgb_image=rgb_rs, visualize=visualize) # runs YOLO model and returns detections
        # decide the door type based on detection (single/double)
        # since door state estimation will run infront of the door, we assume only one door is present in the scene
        door_detections = [(item, LABEL_MAP[item["cls_id"]]) for item in detections if item["cls_id"] in [0, 1]]  # class_id 0 is door_double, class_id 1 is door_single
        if len(door_detections) == 0:
            print("No door detected in the image.")
            return
        
        door_box = door_detections[0][0] # we only have one door in the scene (safe assumption for now)
        door_type = door_detections[0][1]
        print(f"Detected door type: {door_type} with confidence {door_box['conf']:.2f}")
        
        # crop ROI for depth, based on actual bbox
        roi_depth = crop_to_bbox_depth(depth_da_corr, door_box)
        full_depth = depth_da_corr

        # NOTE: door estimation for Single Door
        if door_type == 'door_single':
            door_state = estimate_single_door_state(door_box.get("bbox", []), rgb_rs, roi_depth, 
                                                    full_depth, visualize=visualize, 
                                                    use_vlm=use_vlm, intrinsics=intrinsics)
            print(f"Estimated single door state: {door_state}")
            return door_state

        # NOTE: door estimation for Double Door
        elif door_type == 'door_double':
            door_state = estimate_double_door_state(door_box.get("bbox", []), rgb_rs, roi_depth, 
                                                    full_depth, visualize=visualize, 
                                                    use_vlm=use_vlm, intrinsics=intrinsics)
            print(f"Estimated double door state: {door_state}")
            return door_state
        else:
            print(f"Unknown door type: {door_type}, cannot estimate door state.")
            return None
        # subsequent plane fitting and door state estimation logic

    except Exception as e:
        print(f"Error during door_state_estimate: {e}")
        return None

        
if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # single door: 19(normal-closed), 63(normal-semi-open), 66(glass-closed)
    # double door: 27(glass-closed), 30(glass-closed), 35(glass-semi-open)
    img_id = 35
    img_path = os.path.join(script_dir, f"data_new/latest_image_color_lab_{img_id}.jpg")
    depth_path = os.path.join(script_dir, f"data_new/latest_image_depth_lab_{img_id}.png")
    estimate_door_state_test(img_path, depth_path, visualize=True, use_vlm=False)