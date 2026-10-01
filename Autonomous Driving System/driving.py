import sys
import subprocess
import importlib

_REQUIRED = {
    "cv2":           "opencv-python",
    "numpy":         "numpy",
    "torch":         "torch",
    "torchvision":   "torchvision",
    "PIL":           "pillow",
    "scipy":         "scipy",
    "matplotlib":    "matplotlib",
    "supervision":   "supervision",
    "ultralytics":   "ultralytics",
    "tqdm":          "tqdm",
}

def _auto_install():
    """Install any missing dependency silently then reimport."""
    missing = []
    for module, package in _REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"\n[SETUP] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )
        print("[SETUP] Installation complete.\n")

_auto_install()

# ──────────────────────────────────────────────────────────────────────────────
#  STANDARD IMPORTS
# ──────────────────────────────────────────────────────────────────────────────

import os
import time
import math
import argparse
import warnings
import collections
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import gaussian_filter
from collections import defaultdict, deque

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

CFG = {
    # Detection
    "yolo_model":        "yolov8n.pt",          # nano for speed; swap to yolov8s/m for quality
    "yolo_conf":         0.35,
    "yolo_iou":          0.45,
    "yolo_classes":      [0, 1, 2, 3, 5, 7],    # person,bicycle,car,motorcycle,bus,truck

    # Tracking
    "track_buffer":      40,                     # frames to keep lost tracks
    "trail_len":         30,                     # centroid trail history length

    # Depth
    "depth_model":       "MiDaS_small",          # or "DPT_Large" for quality
    "depth_interval":    2,                      # run depth every N frames

    # Lane
    "lane_roi_top":      0.55,                   # fraction of frame height for ROI
    "lane_canny_lo":     50,
    "lane_canny_hi":     150,

    # Collision
    "ttc_warn_secs":     3.0,                    # TTC threshold for critical warning
    "ttc_caution_secs":  6.0,
    "danger_dist_m":     8.0,                    # metres (estimated) for danger zone

    # BEV
    "bev_w":             240,
    "bev_h":             320,
    "bev_scale":         12.0,                   # pixels per estimated metre

    # Point cloud
    "pc_points":         1800,                   # number of sampled depth points
    "pc_panel_w":        240,
    "pc_panel_h":        200,

    # Display
    "display_w":         1280,
    "display_h":         720,
    "hud_alpha":         0.72,
    "font":              cv2.FONT_HERSHEY_SIMPLEX,
    "font_mono":         cv2.FONT_HERSHEY_DUPLEX,

    # Performance
    "frame_skip":        1,                      # process every Nth frame (1 = all)
    "seg_interval":      4,                      # segmentation every N frames
    "fp16":              True,
}

# Neon colour palette  (BGR)
COL = {
    "cyan":      (255, 220,   0),   # cyan-ish
    "green":     ( 60, 255,  60),
    "lime":      ( 30, 255, 180),
    "orange":    (  0, 160, 255),
    "red":       (  0,  40, 255),
    "magenta":   (255,   0, 200),
    "blue":      (255, 100,   0),
    "white":     (240, 240, 240),
    "dark":      ( 10,  12,  16),
    "panel_bg":  ( 18,  20,  26),
    "yellow":    (  0, 220, 255),
    "teal":      (200, 230,  50),
}

CLASS_LABELS = {0:"PEDESTRIAN", 1:"CYCLIST", 2:"CAR", 3:"MOTORCYCLE", 5:"BUS", 7:"TRUCK"}

CLASS_COLORS = {
    0:  COL["magenta"],
    1:  COL["teal"],
    2:  COL["cyan"],
    3:  COL["lime"],
    5:  COL["orange"],
    7:  COL["red"],
}

# ──────────────────────────────────────────────────────────────────────────────
#  DEVICE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        print(f"[DEVICE] GPU detected: {name}  ({vram} MB VRAM)")
        return dev, True
    print("[DEVICE] No GPU detected. Running on CPU.")
    return torch.device("cpu"), False


DEVICE, USE_GPU = get_device()
USE_FP16 = USE_GPU and CFG["fp16"]

# ──────────────────────────────────────────────────────────────────────────────
#  FPS / LATENCY TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    def __init__(self, window=30):
        self._times = deque(maxlen=window)
        self._last  = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now

    @property
    def fps(self):
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))

    @property
    def latency_ms(self):
        if not self._times:
            return 0.0
        return self._times[-1] * 1000.0


# ──────────────────────────────────────────────────────────────────────────────
#  EXPONENTIAL SMOOTHING HELPER
# ──────────────────────────────────────────────────────────────────────────────

class ExpSmooth:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self._val  = None

    def update(self, x):
        if self._val is None:
            self._val = x
        else:
            self._val = self.alpha * x + (1 - self.alpha) * self._val
        return self._val

    @property
    def value(self):
        return self._val


# ──────────────────────────────────────────────────────────────────────────────
#  YOLO DETECTOR + BYTETRACK TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class ObjectDetectorTracker:
    """
    Wraps YOLOv8 detection + ByteTrack tracking via the Ultralytics unified API.
    Returns a list of Track objects each frame.
    """

    def __init__(self):
        from ultralytics import YOLO
        print(f"[DETECTOR] Loading {CFG['yolo_model']} ...")
        self.model = YOLO(CFG["yolo_model"])
        self.model.to(DEVICE)
        self._trail: dict[int, deque] = defaultdict(lambda: deque(maxlen=CFG["trail_len"]))
        self._smoothers: dict[int, dict] = defaultdict(
            lambda: {"x": ExpSmooth(0.4), "y": ExpSmooth(0.4),
                     "w": ExpSmooth(0.4), "h": ExpSmooth(0.4)}
        )
        self._prev_centers: dict[int, tuple] = {}
        self._speed_est: dict[int, ExpSmooth] = defaultdict(lambda: ExpSmooth(0.25))
        print("[DETECTOR] Ready.")

    def run(self, frame: np.ndarray):
        """
        Returns list of dicts with keys:
          track_id, cls, conf, box (x1,y1,x2,y2), center, speed_px
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=CFG["yolo_conf"],
            iou=CFG["yolo_iou"],
            classes=CFG["yolo_classes"],
            tracker="bytetrack.yaml",
            verbose=False,
            half=USE_FP16,
        )

        tracks = []
        if results and results[0].boxes is not None:
            boxes_data = results[0].boxes
            for box in boxes_data:
                if box.id is None:
                    continue
                tid  = int(box.id.item())
                cls  = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                sm = self._smoothers[tid]
                sx1 = int(sm["x"].update(x1))
                sy1 = int(sm["y"].update(y1))
                sx2 = int(sm["w"].update(x2))
                sy2 = int(sm["h"].update(y2))

                cx = (sx1 + sx2) // 2
                cy = (sy1 + sy2) // 2
                center = (cx, cy)
                self._trail[tid].append(center)

                # pixel-space speed proxy
                prev = self._prev_centers.get(tid)
                if prev:
                    dist = math.hypot(cx - prev[0], cy - prev[1])
                else:
                    dist = 0.0
                speed_px = self._speed_est[tid].update(dist)
                self._prev_centers[tid] = center

                tracks.append({
                    "track_id": tid,
                    "cls":      cls,
                    "conf":     conf,
                    "box":      (sx1, sy1, sx2, sy2),
                    "center":   center,
                    "speed_px": speed_px,
                    "trail":    list(self._trail[tid]),
                })

        return tracks


# ──────────────────────────────────────────────────────────────────────────────
#  MIDAS DEPTH ESTIMATOR
# ──────────────────────────────────────────────────────────────────────────────

class DepthEstimator:
    """
    Runs MiDaS monocular depth estimation.
    Returns a normalised single-channel float32 depth map (0..1, 1 = closest).
    """

    def __init__(self):
        print(f"[DEPTH] Loading {CFG['depth_model']} ...")
        self.model = torch.hub.load(
            "intel-isl/MiDaS", CFG["depth_model"], pretrained=True, trust_repo=True
        )
        self.model.to(DEVICE).eval()
        if USE_FP16:
            self.model = self.model.half()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        if CFG["depth_model"] in ("DPT_Large", "DPT_Hybrid"):
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        self._cached: np.ndarray | None = None
        print("[DEPTH] Ready.")

    @torch.inference_mode()
    def run(self, frame_rgb: np.ndarray) -> np.ndarray:
        inp = self.transform(frame_rgb).to(DEVICE)
        if USE_FP16:
            inp = inp.half()
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = pred.float().cpu().numpy()
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-5:
            depth = (depth - d_min) / (d_max - d_min)
        self._cached = depth
        return depth

    def false_color(self, depth: np.ndarray) -> np.ndarray:
        """Convert normalised depth to INFERNO false-colour BGR image."""
        import matplotlib.cm as cm
        coloured = (cm.inferno(depth)[:, :, :3] * 255).astype(np.uint8)
        return cv2.cvtColor(coloured, cv2.COLOR_RGB2BGR)

    def estimate_distance_m(self, depth_map: np.ndarray,
                             box: tuple, frame_h: int) -> float:
        """
        Rough metric distance in metres using the median depth value
        inside the lower-half of the bounding box.
        Calibrated heuristically for dashcam focal lengths.
        """
        x1, y1, x2, y2 = box
        mid_y = (y1 + y2) // 2
        roi = depth_map[mid_y:y2, x1:x2]
        if roi.size == 0:
            return 99.9
        med = float(np.median(roi))
        # Inverse mapping: depth 1 => ~2m, depth 0 => ~80m
        dist = 2.0 + (1.0 - med) * 78.0
        return round(dist, 1)


# ──────────────────────────────────────────────────────────────────────────────
#  LANE DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

class LaneDetector:
    """
    Hough-transform lane detection with polynomial curve fitting.
    Produces left/right lane lines plus a driving corridor fill.
    """

    def __init__(self, frame_h: int, frame_w: int):
        self.fh = frame_h
        self.fw = frame_w
        self._left_sm  = [ExpSmooth(0.3) for _ in range(3)]
        self._right_sm = [ExpSmooth(0.3) for _ in range(3)]
        self._departure = "CENTERED"

    def _roi_mask(self, edges: np.ndarray) -> np.ndarray:
        roi_y = int(self.fh * CFG["lane_roi_top"])
        pts = np.array([[
            (0,          self.fh),
            (self.fw,    self.fh),
            (self.fw,    roi_y),
            (0,          roi_y),
        ]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, pts, 255)
        return cv2.bitwise_and(edges, mask)

    def _fit_poly(self, lines, side: str):
        if lines is None:
            return None
        xs, ys = [], []
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if side == "left"  and slope < -0.3:
                xs += [x1, x2]; ys += [y1, y2]
            elif side == "right" and slope >  0.3:
                xs += [x1, x2]; ys += [y1, y2]
        if len(xs) < 2:
            return None
        return np.polyfit(ys, xs, 2)

    def run(self, frame_gray: np.ndarray):
        blur  = cv2.GaussianBlur(frame_gray, (7, 7), 0)
        edges = cv2.Canny(blur, CFG["lane_canny_lo"], CFG["lane_canny_hi"])
        roi   = self._roi_mask(edges)
        lines = cv2.HoughLinesP(roi, 1, np.pi/180, 40,
                                minLineLength=60, maxLineGap=80)

        lp = self._fit_poly(lines, "left")
        rp = self._fit_poly(lines, "right")

        # Smooth polynomial coefficients
        if lp is not None:
            lp = np.array([s.update(v) for s, v in zip(self._left_sm, lp)])
        elif all(s.value is not None for s in self._left_sm):
            lp = np.array([s.value for s in self._left_sm])

        if rp is not None:
            rp = np.array([s.update(v) for s, v in zip(self._right_sm, rp)])
        elif all(s.value is not None for s in self._right_sm):
            rp = np.array([s.value for s in self._right_sm])

        # Departure analysis
        if lp is not None and rp is not None:
            y_bot  = self.fh
            lx_bot = np.polyval(lp, y_bot)
            rx_bot = np.polyval(rp, y_bot)
            cx     = self.fw / 2
            mid    = (lx_bot + rx_bot) / 2
            off    = mid - cx
            if   off < -50:  self._departure = "LEFT DEPARTURE"
            elif off >  50:  self._departure = "RIGHT DEPARTURE"
            else:            self._departure = "CENTERED"

        return lp, rp, self._departure


# ──────────────────────────────────────────────────────────────────────────────
#  SIMPLE ROAD / SKY SEGMENTATION  (no heavy model – heuristic + depth)
# ──────────────────────────────────────────────────────────────────────────────

class SegmentationEngine:
    """
    Lightweight heuristic segmentation using HSV colour cues
    combined with depth position priors.
    Produces per-pixel labels: 0=background, 1=road, 2=sky, 3=vehicle/object.
    """

    def __init__(self, frame_h: int, frame_w: int):
        self.fh = frame_h
        self.fw = frame_w

    def run(self, frame_bgr: np.ndarray,
            depth_map: np.ndarray | None,
            tracks: list) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        seg  = np.zeros((h, w), dtype=np.uint8)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # ── Sky: upper region with high saturation blue/grey or bright
        sky_mask = np.zeros((h, w), dtype=np.uint8)
        sky_zone = int(h * 0.45)
        # Blue sky
        sky_blue = cv2.inRange(hsv[:sky_zone], (90, 20, 100), (140, 255, 255))
        # Overcast/grey sky
        sky_grey = cv2.inRange(hsv[:sky_zone], (0, 0, 160),  (180, 40, 255))
        sky_mask[:sky_zone] = cv2.bitwise_or(sky_blue, sky_grey)
        seg[sky_mask > 0] = 2

        # ── Road: lower region with grey/asphalt tones
        road_zone_start = int(h * 0.55)
        road_grey  = cv2.inRange(
            hsv[road_zone_start:], (0, 0, 40), (180, 60, 200)
        )
        road_mask = np.zeros((h, w), dtype=np.uint8)
        road_mask[road_zone_start:] = road_grey
        seg[road_mask > 0] = 1

        # ── Tracked vehicle boxes -> label 3
        for t in tracks:
            x1, y1, x2, y2 = t["box"]
            seg[y1:y2, x1:x2] = 3

        return seg

    def colour_overlay(self, frame_bgr: np.ndarray,
                       seg: np.ndarray, alpha=0.28) -> np.ndarray:
        overlay = frame_bgr.copy()
        # Road = dark teal
        overlay[seg == 1] = (overlay[seg == 1] * 0.5
                             + np.array([80, 60, 10]) * 0.5).astype(np.uint8)
        # Sky = deep blue
        overlay[seg == 2] = (overlay[seg == 2] * 0.6
                             + np.array([60, 10, 10]) * 0.4).astype(np.uint8)
        # Vehicle = magenta hint
        overlay[seg == 3] = (overlay[seg == 3] * 0.65
                             + np.array([60, 0, 80]) * 0.35).astype(np.uint8)

        return cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)


# ──────────────────────────────────────────────────────────────────────────────
#  COLLISION RISK ASSESSOR
# ──────────────────────────────────────────────────────────────────────────────

class CollisionRiskAssessor:
    """
    Estimates Time-To-Collision for each detected object and classifies
    risk level: SAFE / CAUTION / WARNING / CRITICAL.
    """

    def __init__(self):
        self._dist_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=8)
        )

    def assess(self, tracks: list, depth_map: np.ndarray,
               frame_h: int, depth_estimator) -> list:
        results = []
        for t in tracks:
            tid = t["track_id"]
            dist = depth_estimator.estimate_distance_m(depth_map, t["box"], frame_h)
            self._dist_history[tid].append(dist)

            # Estimate approach speed from distance history
            hist = list(self._dist_history[tid])
            if len(hist) >= 3:
                delta = np.diff(hist[-4:])         # negative = approaching
                v_avg = float(np.mean(delta))       # metres-per-frame
            else:
                v_avg = 0.0

            # TTC = dist / approach_speed  (only meaningful when approaching)
            if v_avg < -0.05:
                ttc = dist / abs(v_avg)
            else:
                ttc = 999.0

            # Risk classification
            if dist < CFG["danger_dist_m"] or ttc < CFG["ttc_warn_secs"]:
                risk = "CRITICAL"
            elif ttc < CFG["ttc_caution_secs"] or dist < 15.0:
                risk = "WARNING"
            elif dist < 25.0:
                risk = "CAUTION"
            else:
                risk = "SAFE"

            results.append({
                **t,
                "dist_m": dist,
                "ttc":    round(ttc, 1),
                "v_avg":  v_avg,
                "risk":   risk,
            })
        return results

    @staticmethod
    def risk_color(risk: str) -> tuple:
        return {
            "CRITICAL": COL["red"],
            "WARNING":  COL["orange"],
            "CAUTION":  COL["yellow"],
            "SAFE":     COL["green"],
        }.get(risk, COL["white"])


# ──────────────────────────────────────────────────────────────────────────────
#  BIRD'S-EYE VIEW RENDERER
# ──────────────────────────────────────────────────────────────────────────────

class BEVRenderer:
    """
    Renders a pseudo-LiDAR top-down (bird's-eye view) panel.
    Vehicles are projected to a 2D grid based on their estimated distance
    and horizontal position relative to frame centre.
    """

    def __init__(self):
        self.w  = CFG["bev_w"]
        self.h  = CFG["bev_h"]
        self._ego_col = COL["lime"]

    def render(self, tracks_with_risk: list, frame_w: int, frame_h: int) -> np.ndarray:
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # Grid lines
        grid_spacing = int(CFG["bev_scale"] * 5)
        for gx in range(0, self.w, grid_spacing):
            cv2.line(canvas, (gx, 0), (gx, self.h), (22, 28, 36), 1)
        for gy in range(0, self.h, grid_spacing):
            cv2.line(canvas, (0, gy), (self.w, gy), (22, 28, 36), 1)

        # Distance rings
        cx_e = self.w // 2
        cy_e = self.h - 20
        for r_m in [5, 10, 20, 40]:
            r_px = int(r_m * CFG["bev_scale"] * 0.6)
            cv2.ellipse(canvas, (cx_e, cy_e), (r_px, r_px // 3),
                        0, 180, 360, (30, 45, 55), 1)
            cv2.putText(canvas, f"{r_m}m", (cx_e + r_px + 2, cy_e),
                        CFG["font"], 0.3, (50, 70, 80), 1)

        # Ego vehicle
        ego_pts = np.array([
            [cx_e,      cy_e - 14],
            [cx_e - 8,  cy_e],
            [cx_e + 8,  cy_e],
        ], np.int32)
        cv2.fillPoly(canvas, [ego_pts], self._ego_col)
        cv2.putText(canvas, "EGO", (cx_e - 12, cy_e + 12),
                    CFG["font"], 0.28, COL["lime"], 1)

        # Other objects
        for t in tracks_with_risk:
            dist  = t.get("dist_m", 50.0)
            box   = t["box"]
            cls   = t["cls"]
            risk  = t.get("risk", "SAFE")
            color = CollisionRiskAssessor.risk_color(risk)

            # Horizontal offset: map box centre X to BEV x
            bx = (box[0] + box[2]) / 2
            rel_x  = (bx / frame_w - 0.5) * 2.0    # -1..1
            bev_x  = int(cx_e + rel_x * self.w * 0.45)
            bev_y  = int(cy_e - dist * CFG["bev_scale"] * 0.6)
            bev_y  = max(5, bev_y)

            # Object marker
            half_w = max(6, int(12 * (1 - dist / 80.0)))
            half_h = max(4, int(8  * (1 - dist / 80.0)))
            cv2.rectangle(canvas,
                          (bev_x - half_w, bev_y - half_h),
                          (bev_x + half_w, bev_y + half_h),
                          color, -1)
            label = CLASS_LABELS.get(cls, "OBJ")[:3]
            cv2.putText(canvas, label, (bev_x - half_w, bev_y - half_h - 2),
                        CFG["font"], 0.25, color, 1)

        # Panel border + title
        cv2.rectangle(canvas, (0, 0), (self.w - 1, self.h - 1), COL["blue"], 1)
        cv2.putText(canvas, "BEV  POINT CLOUD", (4, 12),
                    CFG["font"], 0.33, COL["blue"], 1)
        return canvas


# ──────────────────────────────────────────────────────────────────────────────
#  POINT CLOUD PANEL  (2-D depth scatter effect)
# ──────────────────────────────────────────────────────────────────────────────

class PointCloudRenderer:
    """
    Samples the depth map to produce a pseudo-3D scatter panel
    reminiscent of a LiDAR point cloud front view.
    Sky pixels are suppressed.
    """

    def __init__(self):
        self.pw = CFG["pc_panel_w"]
        self.ph = CFG["pc_panel_h"]

    def render(self, depth_map: np.ndarray, seg: np.ndarray | None) -> np.ndarray:
        import matplotlib.cm as cm
        canvas = np.zeros((self.ph, self.pw, 3), dtype=np.uint8)

        h, w = depth_map.shape
        # Sub-sample candidate pixels (skip sky)
        ys, xs = np.meshgrid(
            np.linspace(int(h * 0.3), h - 1, 60, dtype=int),
            np.linspace(0, w - 1, 60, dtype=int),
            indexing="ij"
        )
        ys = ys.ravel()
        xs = xs.ravel()

        # Remove sky
        if seg is not None:
            valid = seg[ys, xs] != 2
            ys = ys[valid]
            xs = xs[valid]

        if len(ys) == 0:
            return canvas

        # Randomly sample
        idx  = np.random.choice(len(ys), min(CFG["pc_points"], len(ys)), replace=False)
        ys_s = ys[idx]
        xs_s = xs[idx]
        ds   = depth_map[ys_s, xs_s]

        # Map to panel coordinates
        px = ((xs_s / w) * self.pw).astype(int)
        py = ((1.0 - ds) * (self.ph * 0.9)).astype(int) + int(self.ph * 0.05)
        py = np.clip(py, 0, self.ph - 1)
        px = np.clip(px, 0, self.pw - 1)

        # Colour by depth using inferno map
        colours = (cm.plasma(ds)[:, :3] * 255).astype(np.uint8)[:, ::-1]  # BGR

        for i in range(len(px)):
            cv2.circle(canvas, (px[i], py[i]), 1, tuple(int(c) for c in colours[i]), -1)

        cv2.rectangle(canvas, (0, 0), (self.pw - 1, self.ph - 1), COL["blue"], 1)
        cv2.putText(canvas, "3D POINT CLOUD", (4, 11),
                    CFG["font"], 0.3, COL["blue"], 1)
        return canvas


# ──────────────────────────────────────────────────────────────────────────────
#  HUD / OVERLAY RENDERER
# ──────────────────────────────────────────────────────────────────────────────

class HUDRenderer:
    """
    Renders all futuristic overlay elements onto the output frame:
      - Bounding boxes with info panels
      - Motion trails
      - Lane corridor fill
      - Top status bar
      - Bottom telemetry bar
      - Collision warning banners
      - Depth mini-panel
      - BEV mini-panel
      - Point cloud mini-panel
    """

    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h
        self._frame_count = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _alpha_rect(canvas: np.ndarray, pt1: tuple, pt2: tuple,
                    color: tuple, alpha: float, border_col=None, border_t=1):
        """Draw a semi-transparent filled rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(canvas.shape[1]-1, x2)
        y2 = min(canvas.shape[0]-1, y2)
        if x2 <= x1 or y2 <= y1:
            return
        roi = canvas[y1:y2, x1:x2]
        rect = np.full_like(roi, color)
        blended = cv2.addWeighted(rect, alpha, roi, 1 - alpha, 0)
        canvas[y1:y2, x1:x2] = blended
        if border_col:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, border_t)

    @staticmethod
    def _text(canvas, txt, pos, scale=0.45, color=COL["white"],
              thickness=1, font=None):
        if font is None:
            font = CFG["font"]
        # Shadow
        cv2.putText(canvas, txt, (pos[0]+1, pos[1]+1),
                    font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(canvas, txt, pos, font, scale, color,
                    thickness, cv2.LINE_AA)

    # ── lanes ────────────────────────────────────────────────────────────────

    def draw_lanes(self, canvas: np.ndarray,
                   lp, rp, departure: str):
        if lp is None and rp is None:
            return

        h, w = canvas.shape[:2]
        y_vals = np.linspace(int(h * CFG["lane_roi_top"]), h - 1, 60)

        def poly_pts(coef):
            xs = np.polyval(coef, y_vals).astype(int)
            ys = y_vals.astype(int)
            valid = (xs >= 0) & (xs < w)
            return list(zip(xs[valid], ys[valid]))

        lpts = poly_pts(lp) if lp is not None else []
        rpts = poly_pts(rp) if rp is not None else []

        # Corridor fill
        if lpts and rpts:
            fill_pts = np.array(lpts + list(reversed(rpts)), np.int32)
            overlay  = canvas.copy()
            cv2.fillPoly(overlay, [fill_pts], (0, 180, 60))
            cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)

        lane_col = (COL["yellow"] if "DEPARTURE" in departure else COL["lime"])

        # Draw curves
        for pts, side in [(lpts, "left"), (rpts, "right")]:
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i+1], lane_col, 2, cv2.LINE_AA)

        # Departure label
        dep_col = COL["red"] if "DEPARTURE" in departure else COL["green"]
        self._alpha_rect(canvas, (w//2 - 90, h - 40),
                         (w//2 + 90, h - 14),
                         COL["dark"], 0.7, dep_col)
        self._text(canvas, f"LANE: {departure}",
                   (w//2 - 82, h - 22), 0.38, dep_col)

    # ── object boxes ─────────────────────────────────────────────────────────

    def draw_tracks(self, canvas: np.ndarray, tracks: list):
        for t in tracks:
            tid   = t["track_id"]
            cls   = t["cls"]
            conf  = t["conf"]
            box   = t["box"]
            risk  = t.get("risk", "SAFE")
            dist  = t.get("dist_m", 0.0)
            ttc   = t.get("ttc",   999.0)
            speed = t.get("speed_px", 0.0)
            trail = t.get("trail", [])

            x1, y1, x2, y2 = box
            color = CLASS_COLORS.get(cls, COL["white"])
            risk_col = CollisionRiskAssessor.risk_color(risk)

            # Corners instead of full box
            corner = 12
            lw     = 2
            for (px, py), (dx, dy) in [
                ((x1, y1), (corner, corner)),
                ((x2, y1), (-corner, corner)),
                ((x1, y2), (corner, -corner)),
                ((x2, y2), (-corner, -corner)),
            ]:
                cv2.line(canvas, (px, py), (px + dx, py), color, lw)
                cv2.line(canvas, (px, py), (px, py + dy), color, lw)

            # Box outline (thin)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (*color[:3], 1), 1)

            # Info panel
            label  = CLASS_LABELS.get(cls, "OBJ")
            line1  = f"{label} #{tid}   {conf:.0%}"
            line2  = f"Dist: {dist:.1f}m"
            line3  = f"TTC:  {ttc if ttc < 99 else '--'}s"
            line4  = f"Risk: {risk}"

            panel_x1 = x1
            panel_y1 = max(0, y1 - 72)
            panel_x2 = x1 + 150
            panel_y2 = y1

            self._alpha_rect(canvas, (panel_x1, panel_y1),
                             (panel_x2, panel_y2),
                             COL["panel_bg"], 0.78, risk_col)

            base_y = panel_y1 + 14
            for txt, col in [
                (line1, color),
                (line2, COL["white"]),
                (line3, risk_col),
                (line4, risk_col),
            ]:
                self._text(canvas, txt, (panel_x1 + 4, base_y), 0.34, col)
                base_y += 15

            # Distance badge on box bottom
            dist_txt = f"{dist:.1f}m"
            tw, th = cv2.getTextSize(dist_txt, CFG["font"], 0.45, 1)[0]
            bx = (x1 + x2) // 2 - tw // 2
            by = y2 + 2
            self._alpha_rect(canvas, (bx - 3, by), (bx + tw + 3, by + th + 4),
                             risk_col, 0.6)
            self._text(canvas, dist_txt, (bx, by + th), 0.45, COL["dark"])

            # Motion trail
            if len(trail) > 1:
                for i in range(1, len(trail)):
                    alpha_t = i / len(trail)
                    c = tuple(int(v * alpha_t) for v in color)
                    cv2.line(canvas, trail[i-1], trail[i], c, 2, cv2.LINE_AA)

    # ── top status bar ───────────────────────────────────────────────────────

    def draw_top_bar(self, canvas: np.ndarray, fps: float, latency_ms: float,
                     frame_idx: int, obj_count: int, mode_str: str):
        bar_h = 30
        self._alpha_rect(canvas, (0, 0), (self.fw, bar_h),
                         COL["dark"], 0.88, COL["blue"])

        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self._text(canvas, "ADAS PERCEPTION PIPELINE",
                   (8, 20), 0.48, COL["cyan"], font=CFG["font_mono"])
        self._text(canvas, f"FPS: {fps:5.1f}", (300, 20), 0.42, COL["lime"])
        self._text(canvas, f"LAT: {latency_ms:5.1f}ms", (400, 20), 0.42, COL["lime"])
        self._text(canvas, f"OBJ: {obj_count:02d}", (520, 20), 0.42, COL["yellow"])
        self._text(canvas, f"FRAME: {frame_idx:05d}", (600, 20), 0.42, COL["white"])
        self._text(canvas, mode_str, (self.fw - 210, 20), 0.40, COL["orange"])
        self._text(canvas, ts, (self.fw - 190, 20), 0.36, COL["white"])

    # ── bottom telemetry bar ─────────────────────────────────────────────────

    def draw_bottom_bar(self, canvas: np.ndarray,
                        departure: str, risk_counts: dict, depth_ok: bool):
        bar_y  = self.fh - 24
        self._alpha_rect(canvas, (0, bar_y), (self.fw, self.fh),
                         COL["dark"], 0.88, COL["blue"])

        items = [
            (f"LANE: {departure}",
             COL["red"] if "DEPARTURE" in departure else COL["green"]),
            (f"CRIT: {risk_counts.get('CRITICAL',0)}",  COL["red"]),
            (f"WARN: {risk_counts.get('WARNING', 0)}",  COL["orange"]),
            (f"SAFE: {risk_counts.get('SAFE',    0)}",  COL["green"]),
            (f"DEPTH: {'ON' if depth_ok else 'OFF'}",   COL["cyan"]),
            (f"MODE: {'GPU FP16' if USE_FP16 else 'GPU' if USE_GPU else 'CPU'}",
             COL["magenta"]),
        ]
        x = 8
        for txt, col in items:
            self._text(canvas, txt, (x, self.fh - 7), 0.35, col)
            x += len(txt) * 8 + 12

        # Watermark
        self._text(canvas,
                   "Lane Change Perception | MiDaS Depth | YOLOv8 + ByteTrack",
                   (self.fw - 420, self.fh - 7), 0.30, (60, 70, 80))

    # ── collision warning banner ─────────────────────────────────────────────

    def draw_collision_banner(self, canvas: np.ndarray,
                              tracks_with_risk: list):
        critical = [t for t in tracks_with_risk if t.get("risk") == "CRITICAL"]
        if not critical:
            return

        # Flash every 15 frames
        if (self._frame_count // 15) % 2 == 0:
            bw, bh = 320, 50
            bx = (self.fw - bw) // 2
            by = 36
            self._alpha_rect(canvas, (bx, by), (bx + bw, by + bh),
                             COL["red"], 0.85, COL["orange"], 2)
            c = critical[0]
            self._text(canvas, "COLLISION WARNING",
                       (bx + 55, by + 18), 0.60, COL["white"], 2)
            d_txt = (f"Dist: {c['dist_m']:.1f}m  "
                     f"TTC: {c['ttc']:.1f}s" if c['ttc'] < 99 else
                     f"Dist: {c['dist_m']:.1f}m")
            self._text(canvas, d_txt, (bx + 65, by + 38), 0.42, COL["yellow"])

        self._frame_count += 1

    # ── depth mini-panel ─────────────────────────────────────────────────────

    def draw_depth_panel(self, canvas: np.ndarray,
                         depth_colour: np.ndarray):
        ph, pw = 140, 200
        px, py = self.fw - pw - 4, 36
        resized = cv2.resize(depth_colour, (pw, ph))
        canvas[py:py+ph, px:px+pw] = resized
        cv2.rectangle(canvas, (px, py), (px+pw, py+ph), COL["blue"], 1)
        self._text(canvas, "DEPTH MAP", (px + 4, py + 11), 0.3, COL["cyan"])

    # ── composite mini-panels on left side ───────────────────────────────────

    def draw_side_panels(self, canvas: np.ndarray,
                         bev_img: np.ndarray, pc_img: np.ndarray):
        # BEV
        bh, bw = bev_img.shape[:2]
        by = self.fh - bh - 30
        bx = 4
        canvas[by:by+bh, bx:bx+bw] = bev_img

        # Point cloud above BEV
        ph, pw = pc_img.shape[:2]
        pcy = by - ph - 4
        canvas[pcy:pcy+ph, bx:bx+pw] = pc_img


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def print_startup_banner():
    print("\n" + "=" * 56)
    print("  ADAS PERCEPTION PIPELINE")
    print("  REAL-TIME AUTONOMOUS DRIVING AI SYSTEM")
    print("=" * 56)
    print(f"  Dev/Creator : tubakhxn")
    print(f"  GitHub      : https://github.com/tubakhxn")
    print("=" * 56)
    print("\n[INIT] Loading detection models...")
    print("[INIT] Initializing depth estimation...")
    print("[INIT] Starting lane analysis...")
    print("[INIT] Launching cinematic visualization engine...\n")


def open_source(src_arg: str):
    """
    Open video source (file, webcam index, or RTSP URL).
    Returns cv2.VideoCapture or raises RuntimeError.
    """
    if src_arg.isdigit():
        cap = cv2.VideoCapture(int(src_arg))
    else:
        cap = cv2.VideoCapture(src_arg)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {src_arg}")
    return cap


def build_writer(cap: cv2.VideoCapture, out_path: str) -> cv2.VideoWriter:
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Target display size
    fw_d = CFG["display_w"]
    fh_d = CFG["display_h"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(out_path, fourcc, fps, (fw_d, fh_d))


def main():
    parser = argparse.ArgumentParser(
        description="ADAS Perception Pipeline -- tubakhxn"
    )
    parser.add_argument("source", nargs="?", default="0",
                        help="Video file, webcam index, or RTSP URL")
    parser.add_argument("--output", default="output_adas.mp4",
                        help="Output video path (default: output_adas.mp4)")
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable depth estimation (faster on CPU)")
    args = parser.parse_args()

    print_startup_banner()

    # ── model initialisation ─────────────────────────────────────────────────
    detector   = ObjectDetectorTracker()
    depth_eng  = None if args.no_depth else DepthEstimator()
    risk_eng   = CollisionRiskAssessor()
    bev_render = BEVRenderer()
    pc_render  = PointCloudRenderer()
    fps_ctr    = FPSCounter()

    # ── open source ──────────────────────────────────────────────────────────
    cap = open_source(args.source)
    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    DW = CFG["display_w"]
    DH = CFG["display_h"]

    lane_det  = LaneDetector(DH, DW)
    seg_eng   = SegmentationEngine(DH, DW)
    hud       = HUDRenderer(DW, DH)
    writer    = build_writer(cap, args.output)

    mode_str  = f"GPU FP16" if USE_FP16 else ("GPU" if USE_GPU else "CPU")
    print(f"[PIPELINE] Source       : {args.source}")
    print(f"[PIPELINE] Resolution   : {src_w}x{src_h}  @  {src_fps:.1f} fps")
    print(f"[PIPELINE] Total frames : {total_frames if total_frames > 0 else 'stream'}")
    print(f"[PIPELINE] Compute mode : {mode_str}")
    print(f"[PIPELINE] Output       : {args.output}")
    print(f"[PIPELINE] Controls     : Q=quit  P=pause  S=screenshot\n")

    frame_idx  = 0
    paused     = False
    depth_map  = None
    depth_col  = None
    seg_map    = None
    lp = rp    = None
    departure  = "CENTERED"
    tracks_r   = []

    try:
        while True:
            # ── keyboard ─────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
                print(f"[PIPELINE] {'Paused' if paused else 'Resumed'}")
            if key == ord("s"):
                sname = f"screenshot_{frame_idx:05d}.jpg"
                # canvas saved below if available
                print(f"[PIPELINE] Screenshot saved: {sname}")

            if paused:
                cv2.waitKey(50)
                continue

            ret, raw_frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            t_start = time.perf_counter()

            # ── resize to display resolution ──────────────────────────────────
            frame = cv2.resize(raw_frame, (DW, DH))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── detection + tracking ──────────────────────────────────────────
            tracks = detector.run(frame)

            # ── depth estimation (every N frames) ────────────────────────────
            if depth_eng is not None and frame_idx % CFG["depth_interval"] == 0:
                depth_map = depth_eng.run(frame_rgb)
                depth_col = depth_eng.false_color(depth_map)

            # ── segmentation (every N frames) ────────────────────────────────
            if frame_idx % CFG["seg_interval"] == 0:
                seg_map = seg_eng.run(frame, depth_map, tracks)

            # ── lane detection ────────────────────────────────────────────────
            lp, rp, departure = lane_det.run(frame_gray)

            # ── risk assessment ───────────────────────────────────────────────
            if depth_map is not None:
                tracks_r = risk_eng.assess(tracks, depth_map, DH, depth_eng)
            else:
                tracks_r = [{**t, "dist_m": 99.9, "ttc": 999.0,
                             "risk": "SAFE", "v_avg": 0.0} for t in tracks]

            # ── BEV + point cloud ─────────────────────────────────────────────
            bev_img = bev_render.render(tracks_r, DW, DH)
            if depth_map is not None:
                pc_img = pc_render.render(depth_map, seg_map)
            else:
                pc_img = np.zeros((CFG["pc_panel_h"], CFG["pc_panel_w"], 3),
                                  dtype=np.uint8)

            # ── compose canvas ────────────────────────────────────────────────
            canvas = frame.copy()

            # Segmentation overlay
            if seg_map is not None:
                canvas = seg_eng.colour_overlay(canvas, seg_map)

            # Lanes
            hud.draw_lanes(canvas, lp, rp, departure)

            # Object boxes + trails
            hud.draw_tracks(canvas, tracks_r)

            # Panels
            hud.draw_side_panels(canvas, bev_img, pc_img)

            # Depth mini-panel
            if depth_col is not None:
                hud.draw_depth_panel(canvas, depth_col)

            # Collision banner
            hud.draw_collision_banner(canvas, tracks_r)

            # FPS tick
            fps_ctr.tick()
            latency = (time.perf_counter() - t_start) * 1000.0

            # Risk summary
            risk_counts: dict[str, int] = defaultdict(int)
            for t in tracks_r:
                risk_counts[t["risk"]] += 1

            # HUD bars
            hud.draw_top_bar(canvas, fps_ctr.fps, latency,
                             frame_idx, len(tracks_r), mode_str)
            hud.draw_bottom_bar(canvas, departure, risk_counts,
                                depth_map is not None)

            # Screenshot
            if key == ord("s"):
                cv2.imwrite(f"screenshot_{frame_idx:05d}.jpg", canvas)

            # ── write + display ───────────────────────────────────────────────
            writer.write(canvas)
            cv2.imshow("ADAS Perception Pipeline  |  Q=Quit  P=Pause  S=Screenshot",
                       canvas)

            # Progress print every 30 frames
            if frame_idx % 30 == 0:
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"\r[PIPELINE] Frame {frame_idx:05d}"
                      f"  FPS: {fps_ctr.fps:5.1f}"
                      f"  Lat: {latency:5.1f}ms"
                      f"  Obj: {len(tracks_r):02d}"
                      f"  {pct:5.1f}%", end="", flush=True)

    except KeyboardInterrupt:
        print("\n[PIPELINE] Interrupted by user.")

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"\n\n[PIPELINE] Finished. Output saved to: {args.output}")
        print(f"[PIPELINE] Total frames processed: {frame_idx}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()