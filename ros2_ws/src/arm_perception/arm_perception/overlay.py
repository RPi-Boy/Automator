"""Drawing helpers for the demo views (numpy + OpenCV only, no ROS), so they can be unit-tested anywhere."""
import cv2
import numpy as np

CLASS_NAMES = ['sphere', 'cube', 'cylinder', 'brick']
GREEN, RED, WHITE, GRAY = (60, 200, 60), (230, 50, 50), (255, 255, 255), (40, 40, 40)   # RGB


def draw_detections(rgb, results, threshold=0.75, timing_ms=None):
    """results: [{'bbox': (x, y, w, h), 'cls': int, 'unc': float}]. Known objects green, flagged ones red."""
    out = rgb.copy()
    for r in results:
        x, y, w, h = r['bbox']
        flagged = r['unc'] > threshold
        color = RED if flagged else GREEN
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        name = 'UNKNOWN' if flagged else CLASS_NAMES[r['cls'] % len(CLASS_NAMES)]
        label = f"{name}  u={r['unc']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y - th - 6, 0)
        cv2.rectangle(out, (x, top), (x + tw + 6, top + th + 6), color, -1)
        cv2.putText(out, label, (x + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    if timing_ms is not None:
        cv2.putText(out, f'perception {timing_ms:.1f} ms', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
    return out


def decode_esp32_status(value):
    """/esp32/status Int32: bit0 e-stop latched, bit1 motors enabled, bit2 fresh command (< 200 ms),
    bit3 hardware e-stop line asserted; bits 8-23 count of /joint_commands received."""
    return {'estop_latched': bool(value & 1), 'motors_enabled': bool(value & 2), 'cmd_fresh': bool(value & 4),
            'hw_estop_line': bool(value & 8), 'cmd_count': (value >> 8) & 0xFFFF}


def compose_view(left, right, lines, width=640, height=480, bar_height=96):
    """Side-by-side panels (resized to width x height) under a text bar. All images RGB uint8."""
    def fit(img):
        if img is None:
            img = np.full((height, width, 3), 30, np.uint8)
            cv2.putText(img, 'waiting for image...', (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    bar = np.full((bar_height, width * 2, 3), GRAY, np.uint8)
    for i, (text, color) in enumerate(lines[:4]):
        cv2.putText(bar, text, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return np.vstack([np.hstack([fit(left), fit(right)]), bar])
