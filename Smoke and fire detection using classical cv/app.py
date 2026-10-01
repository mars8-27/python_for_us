import sys
import subprocess
import importlib

_REQUIRED = {
    "cv2":         "opencv-python",
    "numpy":       "numpy",
    "torch":       "torch",
    "torchvision": "torchvision",
    "PIL":         "pillow",
    "ultralytics": "ultralytics",
}

def _auto_install():
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
from datetime import datetime
from collections import deque, defaultdict

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

CFG = {
    # Display
    "display_w":      1280,
    "display_h":      720,

    # YOLO (scene context)
    "yolo_model":     "yolov8n.pt",
    "yolo_conf":      0.40,
    "yolo_classes":   [0, 2, 5, 7],          # person, car, bus, truck

    # Fire colour detection (HSV)
    # Range 1: red-orange flames
    "fire_h1_lo":     np.array([0,   120, 120]),
    "fire_h1_hi":     np.array([22,  255, 255]),
    # Range 2: yellow-white core
    "fire_h2_lo":     np.array([22,   80, 180]),
    "fire_h2_hi":     np.array([40,  255, 255]),
    "fire_min_area":  600,                   # px² minimum blob area

    # Smoke detection
    "smoke_h_lo":     np.array([0,    0, 100]),
    "smoke_h_hi":     np.array([180, 50, 220]),
    "smoke_min_area": 2500,
    "smoke_diff_thr": 22,                    # frame-diff threshold
    "smoke_blur":     21,                    # blur kernel for diff

    # Temporal smoothing
    "smooth_alpha":   0.35,

    # Performance
    "fp16":           True,
    "scale_w":        640,                   # inference width
    "depth_interval": 1,

    # Risk thresholds (confidence 0..1)
    "crit_thresh":    0.55,
    "warn_thresh":    0.30,
    "caut_thresh":    0.10,

    # Fonts
    "font":           cv2.FONT_HERSHEY_SIMPLEX,
    "font_mono":      cv2.FONT_HERSHEY_DUPLEX,
}

# Neon colour palette (BGR)
COL = {
    "red":        (  0,  30, 255),
    "orange":     (  0, 130, 255),
    "yellow":     (  0, 215, 255),
    "green":      ( 40, 220,  40),
    "cyan":       (230, 220,   0),
    "blue":       (255, 100,  20),
    "magenta":    (220,   0, 200),
    "white":      (230, 230, 230),
    "dark":       (  8,  10,  14),
    "panel_bg":   ( 14,  16,  22),
    "fire_glow":  (  0,  80, 255),
    "smoke_glow": (160, 160, 180),
    "teal":       (200, 220,  40),
}

# ──────────────────────────────────────────────────────────────────────────────
#  DEVICE DETECTION cuda
# ──────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        dev  = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        print(f"[DEVICE] GPU: {name}  ({vram} MB VRAM)")
        return dev, True
    print("[DEVICE] No GPU found. Running on CPU.")
    return torch.device("cpu"), False

DEVICE, USE_GPU = get_device()
USE_FP16 = USE_GPU and CFG["fp16"]

# ──────────────────────────────────────────────────────────────────────────────
#  UTILITIES
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
        return (self._times[-1] * 1000.0) if self._times else 0.0


class ExpSmooth:
    """Single-value exponential moving average."""
    def __init__(self, alpha=0.35, init=0.0):
        self.alpha = alpha
        self._v    = init

    def update(self, x):
        self._v = self.alpha * x + (1.0 - self.alpha) * self._v
        return self._v

    @property
    def value(self):
        return self._v


# ──────────────────────────────────────────────────────────────────────────────
#  YOLO SCENE CONTEXT (people / vehicles)
# ──────────────────────────────────────────────────────────────────────────────

class SceneDetector:
    """YOLOv8 wrapper for general scene objects (people, vehicles)."""

    _LABELS = {0: "PERSON", 2: "CAR", 5: "BUS", 7: "TRUCK"}
    _COLORS = {0: COL["magenta"], 2: COL["cyan"],
               5: COL["orange"],  7: COL["teal"]}

    def __init__(self):
        from ultralytics import YOLO
        print(f"[YOLO] Loading {CFG['yolo_model']} ...")
        self.model = YOLO(CFG["yolo_model"])
        self.model.to(DEVICE)
        print("[YOLO] Ready.")

    def run(self, frame: np.ndarray) -> list:
        results = self.model(
            frame,
            conf=CFG["yolo_conf"],
            classes=CFG["yolo_classes"],
            verbose=False,
            half=USE_FP16,
        )
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls  = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    "cls":  cls,
                    "conf": conf,
                    "box":  (x1, y1, x2, y2),
                })
        return detections


# ──────────────────────────────────────────────────────────────────────────────
#  FIRE DETECTOR  (HSV colour + morphology)
# ──────────────────────────────────────────────────────────────────────────────

class FireDetector:
    """
    Detects fire regions using HSV colour segmentation.
    Returns a list of contour bounding boxes and an overall confidence score.
    """

    def __init__(self):
        self._conf_smooth = ExpSmooth(CFG["smooth_alpha"])

    def run(self, frame_bgr: np.ndarray):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Two HSV fire ranges
        m1 = cv2.inRange(hsv, CFG["fire_h1_lo"], CFG["fire_h1_hi"])
        m2 = cv2.inRange(hsv, CFG["fire_h2_lo"], CFG["fire_h2_hi"])
        mask = cv2.bitwise_or(m1, m2)

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blobs = []
        total_area = 0
        h, w = frame_bgr.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < CFG["fire_min_area"]:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Brightness check: mean V channel inside box must be high
            roi_v = hsv[y:y+bh, x:x+bw, 2]
            if roi_v.size and roi_v.mean() < 140:
                continue
            blobs.append({
                "box":      (x, y, x + bw, y + bh),
                "area":     area,
                "contour":  cnt,
            })
            total_area += area

        # Confidence: fire pixel fraction of frame (capped at 1)
        raw_conf = min(1.0, total_area / max(1, h * w * 0.05))
        conf = self._conf_smooth.update(raw_conf)

        return blobs, conf, mask


# ──────────────────────────────────────────────────────────────────────────────
#  SMOKE DETECTOR  (frame differencing + grey HSV masking)
# ──────────────────────────────────────────────────────────────────────────────

class SmokeDetector:
    """
    Detects smoke using temporal frame differencing combined with
    grey-tone HSV masking to isolate diffuse light-grey/white regions.
    """

    def __init__(self):
        self._prev_gray    = None
        self._conf_smooth  = ExpSmooth(CFG["smooth_alpha"])
        self._bg_sub       = cv2.createBackgroundSubtractorMOG2(
                                 history=120, varThreshold=40,
                                 detectShadows=False)

    def run(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w = frame_bgr.shape[:2]

        # Motion mask via MOG2
        fg_mask = self._bg_sub.apply(frame_bgr)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE,
                                   np.ones((9, 9), np.uint8), iterations=2)

        # Colour mask: low-saturation, mid-high brightness (grey/white smoke)
        colour_mask = cv2.inRange(hsv, CFG["smoke_h_lo"], CFG["smoke_h_hi"])

        # Frame differencing (extra motion cue)
        if self._prev_gray is not None:
            diff  = cv2.absdiff(gray, self._prev_gray)
            diff  = cv2.GaussianBlur(diff, (CFG["smoke_blur"], CFG["smoke_blur"]), 0)
            _, diff_mask = cv2.threshold(diff, CFG["smoke_diff_thr"], 255, cv2.THRESH_BINARY)
        else:
            diff_mask = np.zeros_like(gray)
        self._prev_gray = gray.copy()

        # Combined: must be motion AND grey-toned
        combined = cv2.bitwise_and(colour_mask,
                                   cv2.bitwise_or(fg_mask, diff_mask))

        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blobs = []
        total_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < CFG["smoke_min_area"]:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            blobs.append({
                "box":     (x, y, x + bw, y + bh),
                "area":    area,
                "contour": cnt,
            })
            total_area += area

        raw_conf = min(1.0, total_area / max(1, h * w * 0.08))
        conf = self._conf_smooth.update(raw_conf)

        return blobs, conf, combined


# ──────────────────────────────────────────────────────────────────────────────
#  RISK CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

def classify_risk(fire_conf: float, smoke_conf: float) -> str:
    combined = max(fire_conf, smoke_conf * 0.7 + fire_conf * 0.3)
    if combined >= CFG["crit_thresh"]:
        return "CRITICAL"
    elif combined >= CFG["warn_thresh"]:
        return "WARNING"
    elif combined >= CFG["caut_thresh"]:
        return "CAUTION"
    return "CLEAR"

def risk_color(risk: str) -> tuple:
    return {
        "CRITICAL": COL["red"],
        "WARNING":  COL["orange"],
        "CAUTION":  COL["yellow"],
        "CLEAR":    COL["green"],
    }.get(risk, COL["white"])


# ──────────────────────────────────────────────────────────────────────────────
#  HUD RENDERER
# ──────────────────────────────────────────────────────────────────────────────

class HUDRenderer:
    """
    Renders all visual elements:
      - Fire / smoke bounding boxes with glow outlines
      - Hazard zone fills
      - Top status bar
      - Bottom telemetry bar
      - Side status panel
      - Animated DANGER / WARNING banners
      - Scene object boxes (people, vehicles)
    """

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self._flash_frame = 0

    # ── low-level helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _alpha_rect(canvas, pt1, pt2, color, alpha,
                    border_col=None, border_t=1):
        x1, y1 = max(0, pt1[0]), max(0, pt1[1])
        x2 = min(canvas.shape[1] - 1, pt2[0])
        y2 = min(canvas.shape[0] - 1, pt2[1])
        if x2 <= x1 or y2 <= y1:
            return
        roi     = canvas[y1:y2, x1:x2]
        fill    = np.full_like(roi, color)
        canvas[y1:y2, x1:x2] = cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0)
        if border_col:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, border_t)

    @staticmethod
    def _text(canvas, txt, pos, scale=0.44, color=COL["white"],
              thickness=1, font=None):
        if font is None:
            font = CFG["font"]
        cv2.putText(canvas, txt, (pos[0]+1, pos[1]+1),
                    font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(canvas, txt, pos, font, scale, color,
                    thickness, cv2.LINE_AA)

    def _glow_rect(self, canvas, x1, y1, x2, y2, color, layers=3):
        """Draw a multi-layer glow outline around a rectangle."""
        for i in range(layers, 0, -1):
            alpha = 0.15 * i
            pad   = i * 2
            c_dim = tuple(max(0, int(v * (0.3 + 0.7 * alpha))) for v in color)
            cv2.rectangle(canvas,
                          (x1 - pad, y1 - pad),
                          (x2 + pad, y2 + pad),
                          c_dim, 1 + i, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # ── fire boxes ───────────────────────────────────────────────────────────

    def draw_fire_blobs(self, canvas, blobs, conf):
        for b in blobs:
            x1, y1, x2, y2 = b["box"]
            # Hazard zone fill
            self._alpha_rect(canvas, (x1, y1), (x2, y2),
                             COL["fire_glow"], 0.22)
            # Glow outline
            self._glow_rect(canvas, x1, y1, x2, y2, COL["red"])
            # Corner markers
            cl = 14
            for (px, py), (dx, dy) in [
                ((x1, y1), ( cl,  cl)),
                ((x2, y1), (-cl,  cl)),
                ((x1, y2), ( cl, -cl)),
                ((x2, y2), (-cl, -cl)),
            ]:
                cv2.line(canvas, (px, py), (px + dx, py), COL["red"],  2)
                cv2.line(canvas, (px, py), (px, py + dy), COL["red"],  2)
            # Label
            lbl = f"FIRE  {conf:.0%}"
            tw  = cv2.getTextSize(lbl, CFG["font"], 0.46, 1)[0][0]
            bx  = x1
            self._alpha_rect(canvas, (bx, y1 - 22), (bx + tw + 10, y1),
                             COL["dark"], 0.82, COL["red"])
            self._text(canvas, lbl, (bx + 4, y1 - 6), 0.46, COL["red"])

    # ── smoke boxes ──────────────────────────────────────────────────────────

    def draw_smoke_blobs(self, canvas, blobs, conf):
        for b in blobs:
            x1, y1, x2, y2 = b["box"]
            self._alpha_rect(canvas, (x1, y1), (x2, y2),
                             COL["smoke_glow"], 0.18)
            self._glow_rect(canvas, x1, y1, x2, y2, COL["smoke_glow"])
            lbl = f"SMOKE {conf:.0%}"
            tw  = cv2.getTextSize(lbl, CFG["font"], 0.46, 1)[0][0]
            self._alpha_rect(canvas, (x1, y1 - 22), (x1 + tw + 10, y1),
                             COL["dark"], 0.82, COL["smoke_glow"])
            self._text(canvas, lbl, (x1 + 4, y1 - 6), 0.46, COL["smoke_glow"])

    # ── scene objects (YOLO) ─────────────────────────────────────────────────

    def draw_scene_objects(self, canvas, detections):
        _LABELS = SceneDetector._LABELS
        _COLORS = SceneDetector._COLORS
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            cls   = d["cls"]
            conf  = d["conf"]
            color = _COLORS.get(cls, COL["white"])
            label = _LABELS.get(cls, "OBJ")
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            lbl = f"{label} {conf:.0%}"
            self._alpha_rect(canvas, (x1, y1 - 18), (x1 + 110, y1),
                             COL["panel_bg"], 0.75, color)
            self._text(canvas, lbl, (x1 + 3, y1 - 4), 0.36, color)

    # ── top status bar ────────────────────────────────────────────────────────

    def draw_top_bar(self, canvas, fps, latency_ms, frame_idx,
                     fire_conf, smoke_conf, risk, mode_str):
        self._alpha_rect(canvas, (0, 0), (self.w, 32),
                         COL["dark"], 0.90, COL["red"])
        rc = risk_color(risk)
        self._text(canvas, "INDUSTRIAL FIRE & SMOKE DETECTION AI",
                   (8, 21), 0.50, COL["red"], font=CFG["font_mono"])
        self._text(canvas, f"FPS:{fps:5.1f}", (390, 21), 0.42, COL["green"])
        self._text(canvas, f"LAT:{latency_ms:5.1f}ms", (470, 21), 0.42, COL["green"])
        self._text(canvas, f"FRAME:{frame_idx:05d}", (580, 21), 0.42, COL["white"])
        self._text(canvas, f"RISK: {risk}", (700, 21), 0.46, rc)
        mode_x = self.w - 160
        self._text(canvas, mode_str, (mode_x, 21), 0.40, COL["orange"])
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self._text(canvas, ts, (self.w - 168, 21), 0.34, (70, 80, 90))

    # ── side status panel ─────────────────────────────────────────────────────

    def draw_side_panel(self, canvas, fire_conf, smoke_conf,
                        risk, fire_count, smoke_count, scene_count):
        pw, ph = 195, 210
        px = self.w - pw - 4
        py = 36
        self._alpha_rect(canvas, (px, py), (px + pw, py + ph),
                         COL["panel_bg"], 0.85, COL["red"])

        self._text(canvas, "HAZARD STATUS", (px + 8, py + 16),
                   0.42, COL["red"], font=CFG["font_mono"])

        rc = risk_color(risk)
        items = [
            ("FIRE  CONF", f"{fire_conf:5.1%}",
             COL["red"] if fire_conf > CFG["caut_thresh"] else COL["green"]),
            ("SMOKE CONF", f"{smoke_conf:5.1%}",
             COL["orange"] if smoke_conf > CFG["caut_thresh"] else COL["green"]),
            ("RISK LEVEL", risk,  rc),
            ("FIRE  ZONES", str(fire_count),  COL["yellow"]),
            ("SMOKE ZONES", str(smoke_count), COL["smoke_glow"]),
            ("SCENE OBJ",  str(scene_count),  COL["cyan"]),
            ("GPU MODE",   "ON" if USE_GPU else "OFF",
             COL["green"] if USE_GPU else COL["orange"]),
        ]
        y = py + 34
        for label, val, col in items:
            self._text(canvas, label, (px + 8, y), 0.34, (120, 130, 140))
            self._text(canvas, val,   (px + 128, y), 0.38, col)
            y += 24

        # Confidence bars
        for label, conf, col in [
            ("FIRE",  fire_conf,  COL["red"]),
            ("SMOKE", smoke_conf, COL["smoke_glow"]),
        ]:
            self._text(canvas, label, (px + 8, y), 0.32, (100, 110, 120))
            bar_x = px + 60
            bar_y = y - 8
            bar_w = pw - 68
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8),
                          (30, 35, 42), -1)
            fill_w = int(bar_w * conf)
            if fill_w > 0:
                cv2.rectangle(canvas, (bar_x, bar_y),
                              (bar_x + fill_w, bar_y + 8), col, -1)
            y += 18

    # ── bottom telemetry bar ──────────────────────────────────────────────────

    def draw_bottom_bar(self, canvas, fire_conf, smoke_conf, risk):
        bar_y = self.h - 24
        self._alpha_rect(canvas, (0, bar_y), (self.w, self.h),
                         COL["dark"], 0.90, COL["red"])
        rc = risk_color(risk)
        items = [
            (f"FIRE: {fire_conf:.0%}",
             COL["red"] if fire_conf > CFG["caut_thresh"] else COL["green"]),
            (f"SMOKE: {smoke_conf:.0%}",
             COL["orange"] if smoke_conf > CFG["caut_thresh"] else COL["green"]),
            (f"ALERT: {risk}", rc),
            (f"MODE: {'FP16 GPU' if USE_FP16 else 'GPU' if USE_GPU else 'CPU'}",
             COL["cyan"]),
        ]
        x = 8
        for txt, col in items:
            self._text(canvas, txt, (x, self.h - 7), 0.36, col)
            x += len(txt) * 8 + 20
        self._text(canvas,
                   "Dev/Creator: tubakhxn  |  github.com/tubakhxn",
                   (self.w - 330, self.h - 7), 0.30, (50, 60, 70))

    # ── animated DANGER banner ────────────────────────────────────────────────

    def draw_danger_banner(self, canvas, risk, fire_conf, smoke_conf):
        self._flash_frame += 1
        if risk not in ("CRITICAL", "WARNING"):
            return
        # Flash on/off
        if (self._flash_frame // 12) % 2 == 0 and risk == "CRITICAL":
            return

        bw, bh = 360, 56
        bx = (self.w - bw) // 2
        by = 36
        rc = risk_color(risk)
        self._alpha_rect(canvas, (bx, by), (bx + bw, by + bh),
                         rc, 0.25, rc, 2)
        # Inner border pulse
        pulse = abs(math.sin(self._flash_frame * 0.15))
        border_col = tuple(int(v * (0.5 + 0.5 * pulse)) for v in rc)
        cv2.rectangle(canvas, (bx + 3, by + 3),
                      (bx + bw - 3, by + bh - 3), border_col, 1)

        banner_txt = f"  {risk} -- HAZARD DETECTED  "
        self._text(canvas, banner_txt, (bx + 14, by + 22),
                   0.62, COL["white"], 2)
        sub = f"Fire: {fire_conf:.0%}   Smoke: {smoke_conf:.0%}"
        self._text(canvas, sub, (bx + 60, by + 44), 0.42, rc)

    # ── scan-line overlay (cinematic effect) ──────────────────────────────────

    @staticmethod
    def draw_scanlines(canvas, alpha=0.06):
        h, w = canvas.shape[:2]
        lines = np.zeros((h, w, 3), dtype=np.uint8)
        lines[::3, :] = 30
        cv2.addWeighted(lines, alpha, canvas, 1.0, 0, canvas)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def print_startup_banner():
    print("\n" + "=" * 44)
    print("  INDUSTRIAL FIRE & SMOKE DETECTION AI")
    print("=" * 44)
    print(f"  Dev/Creator : tubakhxn")
    print("=" * 44)
    print("\n[INIT] Loading AI models...")
    print("[INIT] Initializing safety monitoring...")
    print("[INIT] Starting real-time hazard detection...\n")


def open_source(src_arg: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(src_arg) if src_arg.isdigit() else src_arg)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {src_arg}")
    return cap


def build_writer(cap: cv2.VideoCapture, path: str,
                 dw: int, dh: int) -> cv2.VideoWriter:
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                           fps, (dw, dh))


def main():
    parser = argparse.ArgumentParser(
        description="Industrial Fire & Smoke Detection System -- tubakhxn"
    )
    parser.add_argument("source", nargs="?", default="0",
                        help="Video file, webcam index, or RTSP URL")
    parser.add_argument("--output", default="output_detected.mp4")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Skip YOLO scene detection (faster)")
    args = parser.parse_args()

    print_startup_banner()

    # Model init
    fire_det  = FireDetector()
    smoke_det = SmokeDetector()
    scene_det = None if args.no_yolo else SceneDetector()

    DW = CFG["display_w"]
    DH = CFG["display_h"]

    hud = HUDRenderer(DW, DH)
    fps_ctr = FPSCounter()

    cap    = open_source(args.source)
    writer = build_writer(cap, args.output, DW, DH)

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_f   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mode_str  = "GPU FP16" if USE_FP16 else ("GPU" if USE_GPU else "CPU")

    print(f"[PIPELINE] Source   : {args.source}")
    print(f"[PIPELINE] FPS      : {src_fps:.1f}")
    print(f"[PIPELINE] Frames   : {total_f if total_f > 0 else 'stream'}")
    print(f"[PIPELINE] Mode     : {mode_str}")
    print(f"[PIPELINE] Output   : {args.output}")
    print(f"[PIPELINE] Controls : Q=quit  P=pause  S=screenshot\n")

    frame_idx  = 0
    paused     = False
    scene_dets = []
    canvas_ref = None          # for screenshot

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
                print(f"\n[PIPELINE] {'Paused' if paused else 'Resumed'}")
            if key == ord("s") and canvas_ref is not None:
                sname = f"screenshot_{frame_idx:05d}.jpg"
                cv2.imwrite(sname, canvas_ref)
                print(f"\n[PIPELINE] Screenshot saved: {sname}")

            if paused:
                cv2.waitKey(30)
                continue

            ret, raw = cap.read()
            if not ret:
                break

            frame_idx += 1
            t0 = time.perf_counter()

            frame = cv2.resize(raw, (DW, DH))

            # ── detections ────────────────────────────────────────────────────
            fire_blobs,  fire_conf,  _fm = fire_det.run(frame)
            smoke_blobs, smoke_conf, _sm = smoke_det.run(frame)

            # YOLO scene objects (every 2nd frame for speed)
            if scene_det is not None and frame_idx % 2 == 0:
                scene_dets = scene_det.run(frame)

            risk = classify_risk(fire_conf, smoke_conf)

            # ── compose canvas ────────────────────────────────────────────────
            canvas = frame.copy()

            # Subtle vignette darkening at edges
            vign = np.ones((DH, DW), np.float32)
            cx, cy = DW // 2, DH // 2
            Y, X = np.ogrid[:DH, :DW]
            dist_map = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
            vign = np.clip(1.0 - dist_map * 0.45, 0.55, 1.0)
            for c in range(3):
                canvas[:, :, c] = (canvas[:, :, c] * vign).astype(np.uint8)

            # Draw detections
            hud.draw_smoke_blobs(canvas, smoke_blobs, smoke_conf)
            hud.draw_fire_blobs(canvas, fire_blobs,   fire_conf)
            hud.draw_scene_objects(canvas, scene_dets)

            # Cinematic scanlines
            hud.draw_scanlines(canvas)

            # HUD layers
            fps_ctr.tick()
            lat_ms = (time.perf_counter() - t0) * 1000.0

            hud.draw_danger_banner(canvas, risk, fire_conf, smoke_conf)
            hud.draw_top_bar(canvas, fps_ctr.fps, lat_ms, frame_idx,
                             fire_conf, smoke_conf, risk, mode_str)
            hud.draw_side_panel(canvas, fire_conf, smoke_conf, risk,
                                len(fire_blobs), len(smoke_blobs), len(scene_dets))
            hud.draw_bottom_bar(canvas, fire_conf, smoke_conf, risk)

            canvas_ref = canvas

            # ── write + display ───────────────────────────────────────────────
            writer.write(canvas)
            cv2.imshow(
                "Industrial Fire & Smoke Detection AI  |  Q=Quit  P=Pause  S=Screenshot",
                canvas
            )

            if frame_idx % 30 == 0:
                pct = (frame_idx / total_f * 100) if total_f > 0 else 0
                print(f"\r[PIPELINE] Frame {frame_idx:05d}"
                      f"  FPS:{fps_ctr.fps:5.1f}"
                      f"  Fire:{fire_conf:.0%}"
                      f"  Smoke:{smoke_conf:.0%}"
                      f"  Risk:{risk:<8}"
                      f"  {pct:5.1f}%", end="", flush=True)

    except KeyboardInterrupt:
        print("\n[PIPELINE] Interrupted.")

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"\n\n[PIPELINE] Done. Output: {args.output}")
        print(f"[PIPELINE] Frames processed: {frame_idx}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()