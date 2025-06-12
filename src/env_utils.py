import numpy as np
import cv2

# Cached constants
ASPHALT_LOWER = np.array([0, 0, 50], dtype=np.uint8)
ASPHALT_UPPER = np.array([180, 30, 150], dtype=np.uint8)
GRASS_LOWER   = np.array([55, 50, 50], dtype=np.uint8)
GRASS_UPPER   = np.array([65, 255, 255], dtype=np.uint8)
ANGLES = np.array([-60, -30, 0, 30, 60], dtype=np.float32)

def to_hsv(frame: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

def mask_asphalt(hsv: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv, ASPHALT_LOWER, ASPHALT_UPPER)

def mask_grass(hsv: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER)

def compute_front_distance(mask: np.ndarray) -> float:
    h, w = mask.shape
    cx = w // 2
    for y in range(h):
        if np.any(mask[y, cx-5:cx+5] == 255):
            return np.clip((h-1 - y) / float(h-1), 0.0, 1.0)
    return 0.0

def compute_horizontal_distances(mask: np.ndarray, window=10) -> tuple[float,float]:
    h, w = mask.shape
    cx = w // 2
    # find first row with asphalt in center strip
    front = None
    for y in range(h):
        if np.any(mask[y, cx-5:cx+5] == 255):
            front = y
            break
    if front is None:
        return 0.0, 0.0
    lefts, rights = [], []
    maxd = float(cx)
    for y in range(front, min(front+window, h)):
        xs = np.where(mask[y] == 255)[0]
        if xs.size:
            lefts .append(cx - xs.min())
            rights.append(xs.max() - cx)
    if not lefts or not rights:
        return 0.0, 0.0
    return (float(np.median(lefts ))/maxd, float(np.median(rights))/maxd)

def compute_angular_rays(mask: np.ndarray, angles=None) -> list[float]:
    if angles is None:
        angles = [-60, -30, 0, 30, 60]
    h, w = mask.shape
    cx, cy = w//2, h-1
    maxd = float(h-1)
    rays = []
    for deg in angles:
        rad = np.deg2rad(deg)
        dx, dy = np.sin(rad), -np.cos(rad)
        dist = 0.0
        while True:
            x = int(round(cx + dx*dist))
            y = int(round(cy + dy*dist))
            if x<0 or x>=w or y<0 or y>=h:
                rays.append(1.0)
                break
            if mask[y,x] == 255:
                rays.append(np.clip(dist/maxd,0.0,1.0))
                break
            dist += 1.0
            if dist > maxd:
                rays.append(1.0)
                break
    return rays

def extract_distances(frame: np.ndarray) -> np.ndarray:
    hsv  = to_hsv(frame)
    mask = mask_asphalt(hsv)
    h, w = mask.shape
    cx = w//2

    # 9 vertical columns
    offs = [-40,-30,-20,-10,0,10,20,30,40]
    cols = []
    for o in offs:
        x = np.clip(cx+o,0,w-1)
        ys = np.where(mask[:,x]==255)[0]
        cols.append((h-1-ys[0])/float(h-1) if ys.size else 0.0)

    # summary
    front = compute_front_distance(mask)
    left, right = compute_horizontal_distances(mask)

    # angular rays
    rays = compute_angular_rays(mask)

    return np.array(cols + [front, left, right] + rays, dtype=np.float32)

def is_on_grass(frame: np.ndarray) -> bool:
    hsv = to_hsv(frame)
    return bool(mask_grass(hsv)[90,48] == 255)
