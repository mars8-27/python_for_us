import sys, subprocess, importlib

def _ensure(pkg, import_as=None):
    try:
        importlib.import_module(import_as or pkg)
    except ImportError:
        print(f"  [AUTO-INSTALL] {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

print("\n" + "="*52)
print("  SMART CROWD PANIC DETECTION AI")
print("  REAL-TIME CROWD SAFETY & BEHAVIOR SYSTEM")
print("="*52)
print("\n  [INIT] Checking dependencies...\n")

for pkg, imp in [
    ("ultralytics", "ultralytics"),
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("Pillow", "PIL"),
    ("supervision", "supervision"),
]:
    _ensure(pkg, imp)

print("\n  Loading YOLO detection engine...")
print("  Initializing motion analysis system...")
print("  Starting crowd intelligence pipeline...")
print("  Activating panic detection module...\n")
print("="*52 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, time, math, argparse, datetime, collections, warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE / CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
NEON_CYAN   = (0,   255, 255)
NEON_GREEN  = (57,  255,  20)
NEON_RED    = (255,  50,  50)
NEON_ORANGE = (255, 140,   0)
NEON_PINK   = (255,  20, 147)
NEON_BLUE   = ( 30, 144, 255)
NEON_YELLOW = (255, 255,   0)
NEON_PURPLE = (180,  50, 255)
PANEL_BG    = (6,   10,  18)
GRID_COLOR  = (0,   60, 100)

OUTPUT_FILE   = "output_crowd_panic_ai.mp4"
BEV_W, BEV_H  = 300, 400
TRAIL_LEN     = 24
PERSON_CLASS  = 0

# Thresholds
SPEED_SPIKE_THRESH    = 18.0   # px/frame → panic velocity
CHAOS_THRESH_WARN     = 0.45
CHAOS_THRESH_PANIC    = 0.70
DENSITY_MODERATE      = 8
DENSITY_HIGH          = 18
DENSITY_CRITICAL      = 35
CONVERGENCE_THRESH    = 0.60   # flow convergence → stampede risk


# ─────────────────────────────────────────────────────────────────────────────
# LIGHTWEIGHT IoU TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class PersonTrack:
    def __init__(self, tid, bbox):
        self.id      = tid
        self.bbox    = bbox
        self.hits    = 1
        self.misses  = 0
        self.history = collections.deque(maxlen=TRAIL_LEN)
        self.vel_history = collections.deque(maxlen=12)
        self.speed   = 0.0
        self.direction = 0.0   # radians
        cx, cy = self._center(bbox)
        self.history.append((cx, cy))

    @staticmethod
    def _center(b):
        return int((b[0]+b[2])/2), int((b[1]+b[3])/2)

    def update(self, bbox):
        px, py = self.history[-1]
        cx, cy = self._center(bbox)
        dx, dy = cx - px, cy - py
        spd = math.hypot(dx, dy)
        self.vel_history.append((dx, dy, spd))
        self.speed = spd
        self.direction = math.atan2(dy, dx)
        self.bbox = bbox
        self.history.append((cx, cy))
        self.hits  += 1
        self.misses = 0

    @property
    def avg_speed(self):
        if not self.vel_history: return 0.0
        return np.mean([v[2] for v in self.vel_history])

    @property
    def direction_variance(self):
        """How erratic are direction changes — key panic indicator."""
        if len(self.vel_history) < 3: return 0.0
        dirs = [math.atan2(v[1], v[0]) for v in self.vel_history if v[2] > 1]
        if len(dirs) < 2: return 0.0
        diffs = [abs(math.sin(dirs[i]-dirs[i-1])) for i in range(1, len(dirs))]
        return float(np.mean(diffs))


class IoUTracker:
    def __init__(self, iou_thresh=0.30, max_misses=6):
        self.tracks     = {}
        self.next_id    = 1
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses

    @staticmethod
    def _iou(a, b):
        ix1 = max(a[0],b[0]); iy1 = max(a[1],b[1])
        ix2 = min(a[2],b[2]); iy2 = min(a[3],b[3])
        iw  = max(0, ix2-ix1); ih = max(0, iy2-iy1)
        inter = iw*ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter/ua if ua > 0 else 0

    def update(self, bboxes):
        matched_t, matched_d = set(), set()
        bboxes = list(bboxes)
        for tid, t in self.tracks.items():
            best_iou, best_di = 0, -1
            for di, bb in enumerate(bboxes):
                if di in matched_d: continue
                iou = self._iou(t.bbox, bb)
                if iou > best_iou:
                    best_iou, best_di = iou, di
            if best_iou >= self.iou_thresh:
                t.update(bboxes[best_di])
                matched_t.add(tid)
                matched_d.add(best_di)
        for tid in list(self.tracks):
            if tid not in matched_t:
                self.tracks[tid].misses += 1
        for di, bb in enumerate(bboxes):
            if di not in matched_d:
                self.tracks[self.next_id] = PersonTrack(self.next_id, bb)
                self.next_id += 1
        self.tracks = {t: v for t, v in self.tracks.items()
                       if v.misses <= self.max_misses}
        return self.tracks


# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL FLOW ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class OpticalFlowEngine:
    def __init__(self):
        self.prev_gray = None
        self.flow      = None

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        self.flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=12,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )
        self.prev_gray = gray
        return self.flow

    def render_flow(self, frame, step=18):
        if self.flow is None: return
        h, w = frame.shape[:2]
        fh, fw = self.flow.shape[:2]
        sx, sy = w/fw, h/fh
        for y in range(0, fh, step):
            for x in range(0, fw, step):
                dx, dy = self.flow[y, x]
                mag = math.hypot(dx, dy)
                if mag < 0.5: continue
                ox, oy = int(x*sx), int(y*sy)
                ex = int(ox + dx*sx*2)
                ey = int(oy + dy*sy*2)
                alpha = min(1.0, mag/8.0)
                color = (
                    int(255*alpha),
                    int((1-alpha)*200),
                    int((1-alpha)*255)
                )
                cv2.arrowedLine(frame, (ox,oy), (ex,ey), color, 1,
                                tipLength=0.35, line_type=cv2.LINE_AA)

    def compute_convergence(self):
        """Flow convergence score → stampede indicator (0→1)."""
        if self.flow is None: return 0.0
        dx = self.flow[:,:,0]
        dy = self.flow[:,:,1]
        # divergence  ∂u/∂x + ∂v/∂y
        div_x = np.gradient(dx, axis=1)
        div_y = np.gradient(dy, axis=0)
        divergence = div_x + div_y
        # Strong negative divergence → convergent flow → stampede risk
        conv_score = float(np.clip(-np.mean(divergence) * 5, 0, 1))
        return conv_score

    def compute_motion_entropy(self):
        """How chaotic are motion directions? (0=uniform, 1=total chaos)"""
        if self.flow is None: return 0.0
        dx = self.flow[:,:,0].ravel()
        dy = self.flow[:,:,1].ravel()
        mag = np.hypot(dx, dy)
        mask = mag > 0.5
        if mask.sum() < 10: return 0.0
        angles = np.arctan2(dy[mask], dx[mask])
        hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi), density=True)
        hist += 1e-9
        entropy = -np.sum(hist * np.log(hist + 1e-9))
        return float(np.clip(entropy / math.log(16), 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# PANIC DETECTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class PanicDetector:
    def __init__(self):
        self.panic_history = collections.deque(maxlen=20)

    def analyze(self, tracks, flow_engine, crowd_count):
        """Returns (panic_state, panic_score, metrics_dict)"""

        # ── Velocity spike ratio ──
        speeds = [t.avg_speed for t in tracks.values()]
        if speeds:
            spike_ratio = sum(s > SPEED_SPIKE_THRESH for s in speeds) / len(speeds)
            avg_spd = np.mean(speeds)
        else:
            spike_ratio, avg_spd = 0.0, 0.0

        # ── Direction chaos ──
        dir_vars = [t.direction_variance for t in tracks.values()]
        dir_chaos = float(np.mean(dir_vars)) if dir_vars else 0.0

        # ── Optical flow metrics ──
        motion_entropy  = flow_engine.compute_motion_entropy()
        convergence     = flow_engine.compute_convergence()

        # ── Density pressure ──
        density_pressure = min(1.0, crowd_count / max(DENSITY_CRITICAL, 1))

        # ── Composite panic score ──
        panic_score = (
            spike_ratio     * 0.25 +
            dir_chaos       * 0.25 +
            motion_entropy  * 0.20 +
            convergence     * 0.15 +
            density_pressure* 0.15
        )
        panic_score = float(np.clip(panic_score, 0.0, 1.0))
        self.panic_history.append(panic_score)
        smoothed = float(np.mean(self.panic_history))

        # ── State classification ──
        if smoothed < 0.25:
            state = "NORMAL"
        elif smoothed < 0.50:
            state = "ALERT"
        elif smoothed < 0.72:
            state = "PANIC RISK"
        else:
            state = "MASS PANIC"

        metrics = dict(
            spike_ratio=spike_ratio,
            dir_chaos=dir_chaos,
            motion_entropy=motion_entropy,
            convergence=convergence,
            density_pressure=density_pressure,
            avg_speed=avg_spd,
        )
        return state, smoothed, metrics


# ─────────────────────────────────────────────────────────────────────────────
# HEATMAP ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class HeatmapEngine:
    def __init__(self, h, w):
        self.acc = np.zeros((h, w), dtype=np.float32)
        self.panic_acc = np.zeros((h, w), dtype=np.float32)

    def update(self, tracks, panic_score):
        for t in tracks.values():
            if not t.history: continue
            cx, cy = t.history[-1]
            if 0<=cy<self.acc.shape[0] and 0<=cx<self.acc.shape[1]:
                self.acc[cy, cx]       += 1.0
                self.panic_acc[cy, cx] += panic_score

        self.acc       *= 0.97
        self.panic_acc *= 0.95

    def render(self, frame, alpha=0.40):
        # Base density heatmap (blue-green)
        blurred = gaussian_filter(self.acc, sigma=22)
        mx = blurred.max()
        if mx > 1e-6:
            norm = np.clip(blurred/mx, 0, 1)
            heat8 = (norm*255).astype(np.uint8)
            colored = cv2.applyColorMap(heat8, cv2.COLORMAP_OCEAN)
            mask = (norm > 0.08).astype(np.float32)[:,:,None]
            frame[:] = (frame*(1-mask*alpha) + colored*mask*alpha).clip(0,255).astype(np.uint8)

        # Panic hotspot overlay (red-orange)
        pb = gaussian_filter(self.panic_acc, sigma=16)
        pmx = pb.max()
        if pmx > 1e-6:
            pnorm = np.clip(pb/pmx, 0, 1)
            ph8 = (pnorm*255).astype(np.uint8)
            pc = cv2.applyColorMap(ph8, cv2.COLORMAP_HOT)
            pmask = (pnorm > 0.15).astype(np.float32)[:,:,None]
            frame[:] = (frame*(1-pmask*0.3) + pc*pmask*0.3).clip(0,255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# BIRD'S-EYE VIEW PANEL
# ─────────────────────────────────────────────────────────────────────────────

class BirdsEyeView:
    def __init__(self, src_h, src_w):
        self.src_h = src_h
        self.src_w = src_w

    def _person_color(self, t, panic_score):
        if t.avg_speed > SPEED_SPIKE_THRESH:
            return NEON_RED
        if panic_score > 0.60:
            return NEON_ORANGE
        if panic_score > 0.35:
            return NEON_YELLOW
        return NEON_GREEN

    def render(self, tracks, panic_score, crowd_count):
        panel = np.full((BEV_H, BEV_W, 3), PANEL_BG, dtype=np.uint8)

        # Grid
        for y in range(0, BEV_H, 35):
            cv2.line(panel, (0,y), (BEV_W,y), GRID_COLOR, 1)
        for x in range(0, BEV_W, 35):
            cv2.line(panel, (x,0), (x,BEV_H), GRID_COLOR, 1)

        # Crowd dots
        for t in tracks.values():
            if not t.history: continue
            cx, cy = t.history[-1]
            bx = int(cx / self.src_w * BEV_W)
            by = int(cy / self.src_h * BEV_H)
            bx = max(3, min(BEV_W-3, bx))
            by = max(3, min(BEV_H-3, by))
            col = self._person_color(t, panic_score)

            # Glow
            cv2.circle(panel, (bx,by), 9, tuple(c//5 for c in col), -1)
            cv2.circle(panel, (bx,by), 5, col, -1)
            cv2.circle(panel, (bx,by), 2, (255,255,255), -1)

            # Velocity arrow
            if t.vel_history:
                dx, dy, _ = t.vel_history[-1]
                ex = int(bx + dx*2)
                ey = int(by + dy*2)
                cv2.arrowedLine(panel, (bx,by), (ex,ey), col, 1,
                                tipLength=0.4, line_type=cv2.LINE_AA)

        # Panic level ring
        ring_x, ring_y, ring_r = BEV_W//2, BEV_H-36, 18
        ring_col = _panic_color(panic_score)
        cv2.circle(panel, (ring_x,ring_y), ring_r, ring_col, 2)
        filled_angle = int(360 * panic_score)
        if filled_angle > 0:
            cv2.ellipse(panel, (ring_x,ring_y), (ring_r,ring_r),
                        -90, 0, filled_angle, ring_col, -1)
        glow_text(panel, f"{panic_score*100:.0f}%",
                  ring_x-12, ring_y+5, 0.35, (255,255,255))

        cv2.rectangle(panel, (0,0),(BEV_W-1,BEV_H-1), NEON_CYAN, 1)
        glow_text(panel, "CROWD MAP", 6, 14, 0.38, NEON_CYAN)
        glow_text(panel, f"N={crowd_count}", BEV_W-50, 14, 0.38, NEON_YELLOW)
        return panel


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.4):
    h, w = frame.shape[:2]
    x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
    if x2<=x1 or y2<=y1: return
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    roi = roi*(1-alpha) + np.array(color, np.float32)*alpha
    frame[y1:y2, x1:x2] = roi.clip(0,255).astype(np.uint8)

def glow_text(frame, text, x, y, fs=0.45, color=NEON_CYAN, thick=1):
    gc = tuple(int(c*0.35) for c in color)
    cv2.putText(frame, text, (x-1,y+1), cv2.FONT_HERSHEY_SIMPLEX, fs, gc, thick+2, cv2.LINE_AA)
    cv2.putText(frame, text, (x,y),   cv2.FONT_HERSHEY_SIMPLEX, fs, color, thick, cv2.LINE_AA)

def corner_box(frame, x1, y1, x2, y2, color, L=14, T=2):
    for px,py in [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]:
        dx = L if px==x1 else -L
        dy = L if py==y1 else -L
        cv2.line(frame,(px,py),(px+dx,py), color, T, cv2.LINE_AA)
        cv2.line(frame,(px,py),(px,py+dy), color, T, cv2.LINE_AA)

def draw_trail(frame, history, color):
    pts = list(history)
    for i in range(1, len(pts)):
        a = i/len(pts)
        c = tuple(int(ch*a) for ch in color)
        cv2.line(frame, pts[i-1], pts[i], c, max(1,int(2*a)), cv2.LINE_AA)

def _panic_color(score):
    if score < 0.25: return NEON_GREEN
    if score < 0.50: return NEON_YELLOW
    if score < 0.72: return NEON_ORANGE
    return NEON_RED

def _panic_state_color(state):
    return {
        "NORMAL":    NEON_GREEN,
        "ALERT":     NEON_YELLOW,
        "PANIC RISK":NEON_ORANGE,
        "MASS PANIC":NEON_RED,
    }.get(state, NEON_CYAN)

def _density_color(state):
    return {
        "LOW DENSITY":        NEON_GREEN,
        "MODERATE":           NEON_YELLOW,
        "HIGH DENSITY":       NEON_ORANGE,
        "CRITICAL OVERCROWD": NEON_RED,
    }.get(state, NEON_CYAN)

def _density_state(n):
    if n < DENSITY_MODERATE: return "LOW DENSITY"
    if n < DENSITY_HIGH:     return "MODERATE"
    if n < DENSITY_CRITICAL: return "HIGH DENSITY"
    return "CRITICAL OVERCROWD"


# ─────────────────────────────────────────────────────────────────────────────
# HUD RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(frame, stats, panic_score, panic_state):
    h, w = frame.shape[:2]

    # ── Left panel ──
    pw, ph = 268, 320
    alpha_rect(frame, 0, 0, pw, ph, PANEL_BG, alpha=0.85)
    cv2.rectangle(frame, (0,0),(pw,ph), NEON_CYAN, 1)
    cv2.line(frame, (0,22),(pw,22), NEON_CYAN, 1)
    glow_text(frame, "CROWD PANIC AI  v1.0", 7, 16, 0.46, NEON_PINK)

    rows = [
        ("FPS",         f"{stats['fps']:.1f}",               NEON_GREEN),
        ("LATENCY",     f"{stats['latency']*1000:.1f} ms",   NEON_CYAN),
        ("FRAME",       f"#{stats['frame']}",                NEON_CYAN),
        ("PEOPLE",      str(stats['count']),                  NEON_YELLOW),
        ("TRACKS",      str(stats['tracks']),                 NEON_CYAN),
        ("DENSITY",     stats['density_state'],               _density_color(stats['density_state'])),
        ("AVG SPEED",   f"{stats['avg_speed']:.1f} px/f",   NEON_ORANGE),
        ("ENTROPY",     f"{stats['entropy']:.2f}",           NEON_PURPLE),
        ("CONVERGENCE", f"{stats['convergence']:.2f}",       NEON_BLUE),
        ("PANIC SCORE", f"{panic_score*100:.1f}%",           _panic_color(panic_score)),
        ("PANIC STATE", panic_state,                          _panic_state_color(panic_state)),
        ("MODE",        stats['device'].upper(),              NEON_GREEN),
    ]
    for i, (lbl, val, col) in enumerate(rows):
        y = 36 + i*23
        glow_text(frame, f"{lbl:<12}", 7, y, 0.36, (100,120,140))
        glow_text(frame, val, 120, y, 0.40, col)

    # timestamp
    ts = datetime.datetime.now().strftime("%H:%M:%S  %d/%m/%Y")
    glow_text(frame, ts, 7, ph-7, 0.34, NEON_CYAN)

    # ── Top accent bar ──
    alpha_rect(frame, 0, 0, w, 4, NEON_PINK, alpha=0.95)

    # ── Panic score bar (bottom) ──
    bar_y = h - 30
    alpha_rect(frame, 0, bar_y, w, h, PANEL_BG, alpha=0.75)
    bar_fill = int(w * panic_score)
    pcol = _panic_color(panic_score)
    cv2.rectangle(frame, (0, bar_y+4), (bar_fill, bar_y+14), pcol, -1)
    cv2.rectangle(frame, (0, bar_y+4), (w,        bar_y+14), NEON_CYAN, 1)
    glow_text(frame, f"PANIC INDEX  {panic_score*100:.1f}%",
              8, h-8, 0.42, pcol)
    glow_text(frame, panic_state,
              w//2-60, h-8, 0.50, _panic_state_color(panic_state))

    # ── Mass panic emergency flash ──
    fi = stats['frame']
    if panic_state == "MASS PANIC" and (fi//8) % 2 == 0:
        alpha_rect(frame, 0, 0, w, h, (120,0,0), alpha=0.10)
        alpha_rect(frame, w//2-200, 46, w//2+200, 84, (160,0,0), alpha=0.70)
        cv2.rectangle(frame, (w//2-200,46),(w//2+200,84), NEON_RED, 2)
        glow_text(frame, "⚠  MASS PANIC DETECTED — EMERGENCY  ⚠",
                  w//2-192, 74, 0.58, NEON_RED)
    elif panic_state == "PANIC RISK":
        alpha_rect(frame, w//2-170, 46, w//2+170, 78, (80,30,0), alpha=0.60)
        glow_text(frame, "⚠  CROWD PANIC RISK — MONITOR NOW",
                  w//2-162, 70, 0.50, NEON_ORANGE)
    elif panic_state == "ALERT":
        alpha_rect(frame, w//2-130, 46, w//2+130, 74, (50,40,0), alpha=0.55)
        glow_text(frame, "CROWD ALERT — ELEVATED MOTION",
                  w//2-122, 68, 0.46, NEON_YELLOW)

    # ── Convergence warning (stampede) ──
    if stats['convergence'] > CONVERGENCE_THRESH:
        glow_text(frame, "▲ STAMPEDE FLOW DETECTED",
                  w-240, h-50, 0.44, NEON_RED)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class CrowdPanicSystem:
    def __init__(self, source, model_path="yolov8n.pt"):
        self.source = source

        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = self.device == "cuda"
        print(f"  [DEVICE] {self.device.upper()}" + (" | FP16" if self.use_fp16 else ""))

        print("  [MODEL] Loading YOLOv8n...")
        self.model = YOLO(model_path)
        if self.use_fp16:
            self.model.model.half()

        src = int(source) if source.isdigit() else source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open: {source}")

        self.W  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H  = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

        # Sub-systems
        self.tracker     = IoUTracker()
        self.flow_engine = OpticalFlowEngine()
        self.heatmap     = HeatmapEngine(self.H, self.W)
        self.panic_det   = PanicDetector()
        self.bev         = BirdsEyeView(self.H, self.W)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, self.src_fps,
                                       (self.W, self.H))

        self.frame_idx    = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.fps_buf      = collections.deque([self.src_fps]*8, maxlen=8)
        self.paused       = False
        self._last_frame  = None
        self._start_time  = None

        # ── Speed optimisation ──
        self.INFER_EVERY      = 2       # YOLO every N frames
        self.INFER_SIZE       = 416     # smaller inference res
        self.HEATMAP_EVERY    = 4       # heatmap gaussian every N frames
        self.FLOW_EVERY       = 2       # optical flow every N frames
        self._last_bboxes     = []
        self._cached_bev      = None
        self._cached_heat     = None
        self._scanline_mask   = None
        _empty_metrics = dict(avg_speed=0.0, motion_entropy=0.0, convergence=0.0,
                              spike_ratio=0.0, dir_chaos=0.0, density_pressure=0.0)
        self._last_panic      = (0.0, "NORMAL", _empty_metrics)
        self._last_metrics    = _empty_metrics

    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        print(f"\n  [GO] Source: {self.source}")
        print(f"  [OUT] → {OUTPUT_FILE}")
        print("  [CTRL] Q=Quit  P=Pause  S=Screenshot\n")

        cv2.namedWindow("Crowd Panic AI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Crowd Panic AI", min(1280,self.W), min(720,self.H))

        self._start_time = time.time()
        total = self.total_frames

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('p'): self.paused = not self.paused
            if key == ord('s') and self._last_frame is not None:
                fn = f"screenshot_{self.frame_idx}.png"
                cv2.imwrite(fn, self._last_frame)
                print(f"\n  [SAVE] {fn}")

            if self.paused:
                time.sleep(0.03)
                continue

            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret: break

            self.frame_idx += 1
            out = self._process(frame, t0)
            self._last_frame = out.copy()
            self.writer.write(out)
            cv2.imshow("Crowd Panic AI", out)

            # ── Terminal progress bar ──
            fps_now = float(np.mean(self.fps_buf))
            elapsed = time.time() - self._start_time
            if total > 0:
                pct   = self.frame_idx / total * 100
                eta_s = (total - self.frame_idx) / max(fps_now, 0.1)
                eta_m, eta_s2 = divmod(int(eta_s), 60)
                bar_filled = int(pct / 5)
                bar = "█" * bar_filled + "░" * (20 - bar_filled)
                line = (f"\r  [PROCESSING]  [{bar}]  "
                        f"{pct:5.1f}%  |  "
                        f"Frame {self.frame_idx}/{total}  |  "
                        f"{fps_now:.1f} FPS  |  "
                        f"ETA {eta_m}m {eta_s2:02d}s   ")
            else:
                line = (f"\r  [PROCESSING]  Frame {self.frame_idx}  |  "
                        f"{fps_now:.1f} FPS  |  "
                        f"Elapsed {int(elapsed)}s   ")
            sys.stdout.write(line)
            sys.stdout.flush()

        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()
        print(f"\n\n  [DONE] Saved → {OUTPUT_FILE}")

    # ─────────────────────────────────────────────────────────────────────────
    def _process(self, frame, t0):
        do_infer   = (self.frame_idx % self.INFER_EVERY   == 0)
        do_heatmap = (self.frame_idx % self.HEATMAP_EVERY == 0)
        do_bev     = (self.frame_idx % 3 == 0)
        do_flow    = (self.frame_idx % self.FLOW_EVERY    == 0)

        # ── Optical flow (every 2 frames) ──
        if do_flow:
            self.flow_engine.update(frame)

        # ── YOLO (every 2 frames, on downscaled image) ──
        if do_infer:
            small = cv2.resize(frame, (self.INFER_SIZE, self.INFER_SIZE))
            results = self.model(small, verbose=False, device=self.device,
                                  half=self.use_fp16, conf=0.30, iou=0.45,
                                  classes=[PERSON_CLASS])
            sx = self.W / self.INFER_SIZE
            sy = self.H / self.INFER_SIZE
            bboxes = []
            for r in results:
                if r.boxes is None: continue
                for box in r.boxes:
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    bboxes.append((int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)))
            self._last_bboxes = bboxes
        else:
            bboxes = self._last_bboxes

        # ── Track ──
        tracks      = self.tracker.update(bboxes)
        crowd_count = len(tracks)

        # ── Panic analysis (every 2 frames) ──
        if do_infer:
            panic_state, panic_score, metrics = self.panic_det.analyze(
                tracks, self.flow_engine, crowd_count)
            self._last_panic   = (panic_score, panic_state, metrics)
            self._last_metrics = metrics
        else:
            panic_score, panic_state, metrics = self._last_panic

        # ── Heatmap ──
        self.heatmap.update(tracks, panic_score)
        if do_heatmap:
            heat_copy = frame.copy()
            self.heatmap.render(heat_copy)
            self._cached_heat = heat_copy
        if self._cached_heat is not None:
            np.copyto(frame, self._cached_heat)

        # ── Optical flow overlay (every 2 frames) ──
        if do_flow:
            self.flow_engine.render_flow(frame, step=24)

        # ── Draw people ──
        self._draw_tracks(frame, tracks, panic_score)

        # ── BEV (every 3 frames) ──
        if do_bev or self._cached_bev is None:
            self._cached_bev = self.bev.render(tracks, panic_score, crowd_count)

        # ── Timing ──
        latency = time.time() - t0
        fps_now = 1.0 / max(latency, 1e-6)
        self.fps_buf.append(fps_now)
        fps = float(np.mean(self.fps_buf))

        density_state = _density_state(crowd_count)
        stats = dict(
            fps=fps, latency=latency, frame=self.frame_idx,
            count=crowd_count, tracks=len(tracks),
            density_state=density_state,
            avg_speed=metrics['avg_speed'],
            entropy=metrics['motion_entropy'],
            convergence=metrics['convergence'],
            device=self.device,
        )

        draw_hud(frame, stats, panic_score, panic_state)

        # ── Embed BEV (bottom-right) ──
        bx = self.W - BEV_W - 8
        by = self.H - BEV_H - 36
        if by > 0 and bx > 0:
            frame[by:by+BEV_H, bx:bx+BEV_W] = self._cached_bev

        # ── Scanlines (pre-built, zero alloc) ──
        if self._scanline_mask is None:
            self._scanline_mask = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            self._scanline_mask[::3, :] = 8
        cv2.subtract(frame, self._scanline_mask, frame)

        return frame

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_tracks(self, frame, tracks, panic_score):
        for t in tracks.values():
            spd = t.avg_speed
            if spd > SPEED_SPIKE_THRESH:
                color = NEON_RED
            elif panic_score > 0.60:
                color = NEON_ORANGE
            elif panic_score > 0.35:
                color = NEON_YELLOW
            else:
                color = NEON_GREEN

            draw_trail(frame, list(t.history), color)

            x1,y1,x2,y2 = t.bbox
            corner_box(frame, x1,y1,x2,y2, color, L=12, T=2)
            alpha_rect(frame, x1,y1,x2,y2, color, alpha=0.05)

            lbl = f"#{t.id}  {spd:.1f}px/f"
            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            alpha_rect(frame, x1, y1-th-8, x1+tw+8, y1, PANEL_BG, alpha=0.80)
            glow_text(frame, lbl, x1+4, y1-5, 0.36, color)

            # Direction arrow from center
            if t.vel_history:
                dx, dy, _ = t.vel_history[-1]
                cx = (x1+x2)//2; cy_c = (y1+y2)//2
                ex = int(cx + dx*3); ey = int(cy_c + dy*3)
                cv2.arrowedLine(frame, (cx,cy_c), (ex,ey), color, 1,
                                tipLength=0.35, line_type=cv2.LINE_AA)

            # Panic ring on high-speed individuals
            if spd > SPEED_SPIKE_THRESH:
                cx = (x1+x2)//2; cy_c = (y1+y2)//2
                r  = max(30, (y2-y1)//2 + 12)
                cv2.circle(frame, (cx,cy_c), r, NEON_RED, 1, cv2.LINE_AA)
                glow_text(frame, "PANIC", cx-20, y2+14, 0.34, NEON_RED)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Smart Crowd Panic Detection AI")
    parser.add_argument("source", nargs="?", default="0",
                        help="Video file, camera index, or RTSP URL")
    args = parser.parse_args()

    print(f"\n  [SYSTEM] Crowd Panic AI starting...")
    print(f"  [SOURCE] {args.source}\n")

    system = CrowdPanicSystem(args.source)
    system.run()


if __name__ == "__main__":
    main()