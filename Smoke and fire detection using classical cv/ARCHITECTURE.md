# System Architecture

```mermaid
flowchart TD
    A[Video Source<br/>Webcam / File / RTSP] --> B[Frame Capture]
    B --> C[Frame Resize & Preprocessing]
    C --> D[Fire Detector]
    C --> E[Smoke Detector]
    C --> F[YOLOv8n Scene Detector]
    D --> D1[HSV Segmentation]
    D1 --> D2[Morphology + Contours]
    E --> E1[MOG2 Background Subtraction]
    E1 --> E2[Frame Difference + HSV Mask]
    F --> F1[Person / Car / Bus / Truck]
    D2 --> G[Confidence Smoothing]
    E2 --> G
    G --> H[Risk Classifier]
    F1 --> I[Scene Context]
    H --> J[HUD Renderer]
    I --> J
    J --> K[Live Monitoring]
    J --> L[Video Output]
```

## Components

| Component | Responsibility |
|---|---|
| Frame Capture | Reads webcam, file, or stream input |
| Fire Detector | Finds fire-like colour/brightness regions |
| Smoke Detector | Combines motion and grey/white-region cues |
| YOLOv8n | Adds scene-object context |
| Confidence Smoothing | Reduces frame-to-frame instability |
| Risk Classifier | Maps signals to CLEAR/CAUTION/WARNING/CRITICAL |
| HUD Renderer | Produces the monitoring interface |
| Video Writer | Saves processed output |
