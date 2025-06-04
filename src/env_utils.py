import numpy as np
import cv2
import gymnasium as gym

# HSV thresholds for asphalt and grass
ASPHALT_LOWER = np.array([0, 0, 50], dtype=np.uint8)
ASPHALT_UPPER = np.array([180, 30, 150], dtype=np.uint8)
GRASS_LOWER = np.array([55, 50, 50], dtype=np.uint8)
GRASS_UPPER = np.array([65, 255, 255], dtype=np.uint8)


def make_env(render: bool = False):
    mode = "human" if render else None
    return gym.make("CarRacing-v3", render_mode=mode)


def to_hsv(obs_frame: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(obs_frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv


def mask_asphalt(hsv_frame: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv_frame, ASPHALT_LOWER, ASPHALT_UPPER)


def mask_grass(hsv_frame: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv_frame, GRASS_LOWER, GRASS_UPPER)


def compute_horizontal_distances(mask: np.ndarray, window_size: int = 10) -> (float, float):
    height, width = mask.shape
    center_x = width // 2

    front_y = None
    for y in range(height):
        strip = mask[y, max(center_x - 5, 0): min(center_x + 5, width - 1) + 1]
        if np.any(strip == 255):
            front_y = y
            break

    if front_y is None:
        return 0.0, 0.0

    window_start = front_y
    window_end = min(front_y + window_size, height)

    left_dists = []
    right_dists = []
    max_dist = float(width // 2)

    for y in range(window_start, window_end):
        asphalt_xs = np.where(mask[y] == 255)[0]
        if asphalt_xs.size == 0:
            continue
        x_min = int(asphalt_xs.min())
        x_max = int(asphalt_xs.max())
        dist_left = center_x - x_min
        dist_right = x_max - center_x
        left_dists.append(dist_left)
        right_dists.append(dist_right)

    if not left_dists or not right_dists:
        return 0.0, 0.0

    left_med = float(np.median(left_dists))
    right_med = float(np.median(right_dists))

    d_left_norm = np.clip(left_med / max_dist, 0.0, 1.0)
    d_right_norm = np.clip(right_med / max_dist, 0.0, 1.0)
    return d_left_norm, d_right_norm


def compute_front_distance(mask: np.ndarray) -> float:
    height, width = mask.shape
    center_x = width // 2

    front_y = None
    for y in range(height):
        strip = mask[y, max(center_x - 5, 0): min(center_x + 5, width - 1) + 1]
        if np.any(strip == 255):
            front_y = y
            break

    if front_y is None:
        return 0.0

    raw_dist = (height - 1) - front_y
    d_front_norm = raw_dist / float(height - 1)
    return np.clip(d_front_norm, 0.0, 1.0)


def extract_distances(obs_frame: np.ndarray) -> np.ndarray:
    hsv = to_hsv(obs_frame)
    mask = mask_asphalt(hsv)

    height, width = mask.shape
    center_x = width // 2
    positions = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    distances = []

    for offset in positions:
        x = np.clip(center_x + offset, 0, width - 1)
        col = mask[:, x]
        white_indices = np.where(col == 255)[0]
        if white_indices.size == 0:
            distances.append(0.0)
        else:
            norm_dist = (height - 1 - white_indices[0]) / float(height - 1)
            distances.append(np.clip(norm_dist, 0.0, 1.0))

    return np.array(distances, dtype=np.float32)


def is_on_grass(obs_frame: np.ndarray) -> bool:
    hsv = to_hsv(obs_frame)
    grass_mask = mask_grass(hsv)
    contact_y = 90
    contact_x = 48
    return bool(grass_mask[contact_y, contact_x] == 255)
