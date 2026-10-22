# Lips Webcam

Standalone OpenCV and MediaPipe demo for mouth-shape tracking.

## Setup

No downloaded model is required. MediaPipe Face Mesh is bundled with the Python
package.

```powershell
python prototypes\lips_webcam\app.py
```

## Use

- A webcam window opens.
- The overlay reports closed, rounded, or open mouth goals.
- Press `Esc` or `Q` to close the window.

This is the smallest prototype for checking camera and facial-landmark support
before running the multimodal assessment.
