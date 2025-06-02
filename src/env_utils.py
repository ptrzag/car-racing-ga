import numpy as np
import cv2

def to_hsv(obs_frame: np.ndarray) -> np.ndarray:
    # Convert rgb to bgr
    bgr = cv2.cvtColor(obs_frame, cv2.COLOR_RGB2BGR)

    # Convert bgr to hsv
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    return hsv

def mask_asphalt(hsv_frame: np.ndarray) -> np.ndarray:
    # HSV range for asphalt
    lower_asphalt = np.array([0,   0,  50])
    upper_asphalt = np.array([180, 30, 150])

    return cv2.inRange(hsv_frame, lower_asphalt, upper_asphalt)

def compute_horizontal_distances(mask: np.ndarray, window_size: int = 10) -> (float, float):
    # Determine the size and center
    height, width = mask.shape
    center_x = width // 2

    # Finding the first row from the top where the asphalt is visible
    front_y = None
    for y in range(height):
        # We only check a narrow strip of ±5 pixels from the center (to avoid bits of asphalt on the side of the screen)
        x_left = max(center_x - 5, 0)
        x_right = min(center_x - 5, width - 1)
        if np.any(mask[y, x_left : x_right + 1] == 255):
            front_y = y
            break

    # If the asphalt is not visible at all
    if front_y is None:
        return 0.0, 0.0
    
    # Set the "center point" of the window
    center_y = (front_y + (height - 1)) // 2

    # Calculate window boundaries
    half_window = window_size // 2
    window_start = max(center_y - half_window, 0)
    window_end = min(center_y + window_size, height)

    # We make sure that window actually has a window_size of rows
    if window_end - window_start < window_size:
        window_start = max(window_end - window_size, 0)

    left_positions = []
    right_positions = []

    # Processing each row from the window and segmenting the asphalt
    for y in range(window_start, window_end):
        asphalt_xs = np.where(mask[y] == 255)[0]
        if asphalt_xs.size == 0:
            # No asphalt in this line - we skip it
            continue

        # Division of asphalt_xs into continuous segments
        # e,g. array([10,11,12,  20,21,22,23,  50,51]) -> [ [10,11,12], [20,21,22,23], [50,51] ]
        splits = np.where(np.diff(asphalt_xs) > 1)[0] + 1
        segments = np.split(asphalt_xs, splits)

        # For each segment we calculate its "center" and select the segment whose center is closest to center_x.
        best_seg = None
        min_dist_to_center = float('inf')
        for seg in segments:
            if seg.size ==  0:
                continue
            seg_center = float(np.mean(seg))
            dist_center = abs(seg_center - center_x)
            if dist_center < min_dist_to_center:
                min_dist_to_center = dist_center
                best_seg = seg

            # If no segment could be selected, skip the row
            if best_seg is None or best_seg.size == 0:
                continue

            # Determine x_min and x_max from the selected segment
            x_min = int(best_seg.min())
            x_max = int(best_seg.max())

            # Calculate the distance in pixels from the center of the image
            dist_left = center_x - x_min
            dist_right = x_max - center_x

            left_positions.append(dist_left)
            right_positions.append(dist_right)

        # If asphalt was not found in any of the selected rows
        if len(left_positions) == 0 or len(right_positions) == 0:
            return 0.0, 0.0
        
        # Normalization of results [0; +1]
        left_med = float(np.median(left_positions))
        right_med = float(np.median(right_positions))
        max_dist  = float(width // 2)
        d_left_norm = np.clip(left_med / max_dist, 0.0, 1.0)
        d_right_norm = np.clip(right_med / max_dist, 0.0, 1.0)
        
        return d_left_norm, d_right_norm

def compute_front_distance(mask: np.ndarray) -> float:
    # Determine the size and center
    height, width = mask.shape
    center_x = width // 2

    # Find the first row y where there are asphalt pixels in our narrow central strip
    front_y = None
    for y in range(height):
        x_left = max(center_x - 5, 0)
        x_right = min(center_x + 5, width - 1)
        if np.any(mask[y, x_left : x_right] == 255):
            front_y = y
            break 
    
    # Checks if there is asphalt above the car
    if front_y is None:
        return 0.0
    
    # Normalization of results [0; +1]
    raw_dist = (height - 1) - front_y
    d_front_norm = raw_dist / float(height - 1)

    return np.clip(d_front_norm, 0.0, 1.0)

def extract_distances(obs_frame: np.ndarray) -> np.ndarray:
    hsv = to_hsv(obs_frame)
    mask = mask_asphalt(hsv)
    d_left_norm, d_right_norm = compute_horizontal_distances(mask)
    d_front_norm = compute_front_distance(mask)

    return np.array([d_left_norm, d_right_norm, d_front_norm], dtype = np.float32)

def mask_grass(hsv_frame: np.ndarray) -> np.ndarray:
    # HSV range for grass
    lower_grass = np.array([55, 50, 50], dtype = np.uint8)
    upper_grass = np.array([65, 255, 255], dtype = np.uint8)

    return cv2.inRange(hsv_frame, lower_grass, upper_grass)

def is_on_grass(obs_frame: np.ndarray) -> bool:
    hsv = to_hsv(obs_frame)
    grass_mask = mask_grass(hsv)

    # Contact coordinate
    contact_y = 90
    contact_x = 48

    return grass_mask[contact_y, contact_x] == 255
