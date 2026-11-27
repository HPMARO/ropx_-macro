from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float
    label: str

    @property
    def center(self):
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass
class Template:
    label: str
    image: np.ndarray

    def __post_init__(self):
        if self.image is None or min(self.image.shape[:2]) < 12:
            raise ValueError("기준 이미지는 가로·세로 12픽셀 이상이어야 합니다.")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if float(self.gray.std()) < 5:
            raise ValueError("단색에 가까운 이미지입니다. 슬라임 전체 윤곽을 포함해 주세요.")
        if max(self.image.shape[:2]) > 500:
            raise ValueError("기준 이미지가 너무 큽니다. 슬라임 부분만 작게 잘라 주세요.")

    @classmethod
    def load(cls, path: Path):
        data = np.fromfile(str(path), dtype=np.uint8)
        return cls(path.stem, cv2.imdecode(data, cv2.IMREAD_COLOR))


def normalized_roi(frame, roi):
    """Normalized ROI keeps the same field of view on differently sized windows."""
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, round(roi[0] * w)))
    y1 = max(0, min(h - 1, round(roi[1] * h)))
    x2 = max(x1 + 1, min(w, round(roi[2] * w)))
    y2 = max(y1 + 1, min(h, round(roi[3] * h)))
    return frame[y1:y2, x1:x2], (x1, y1)


def iou(a: Detection, b: Detection):
    x = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    y = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    intersection = x * y
    return intersection / max(1, a.w * a.h + b.w * b.h - intersection)


class Detector:
    """Multi-scale template matching; no process memory or game API access."""

    def __init__(self, templates, max_width=800):
        self.templates = list(templates)
        self.max_width = max_width

    def detect(self, frame, roi, threshold=.84):
        if not self.templates:
            return []
        crop, (ox, oy) = normalized_roi(frame, roi)
        ratio = min(1.0, self.max_width / crop.shape[1])
        small = cv2.resize(crop, None, fx=ratio, fy=ratio) if ratio < 1 else crop
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        found = []
        for template in self.templates:
            for scale in (.65, .8, 1.0, 1.2, 1.45):
                tw = round(template.gray.shape[1] * scale * ratio)
                th = round(template.gray.shape[0] * scale * ratio)
                if min(tw, th) < 10 or tw > gray.shape[1] or th > gray.shape[0]:
                    continue
                needle = cv2.resize(template.gray, (tw, th))
                scores = cv2.matchTemplate(gray, needle, cv2.TM_CCOEFF_NORMED)
                for _ in range(3):
                    _, score, _, pos = cv2.minMaxLoc(scores)
                    if not math.isfinite(score) or score < threshold:
                        break
                    x, y = pos
                    # Grayscale alone confuses differently colored background objects.
                    patch = small[y:y + th, x:x + tw]
                    original_hsv = cv2.cvtColor(template.image, cv2.COLOR_BGR2HSV)
                    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                    ah = cv2.calcHist([original_hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
                    bh = cv2.calcHist([patch_hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
                    cv2.normalize(ah, ah)
                    cv2.normalize(bh, bh)
                    color_distance = cv2.compareHist(ah, bh, cv2.HISTCMP_BHATTACHARYYA)
                    if color_distance < .65:
                        found.append(Detection(round(x / ratio) + ox, round(y / ratio) + oy,
                                               round(tw / ratio), round(th / ratio), score,
                                               template.label))
                    scores[max(0, y - th // 2): min(scores.shape[0], y + th // 2 + 1),
                           max(0, x - tw // 2): min(scores.shape[1], x + tw // 2 + 1)] = -1
        picked = []
        for candidate in sorted(found, key=lambda d: d.score, reverse=True):
            if all(iou(candidate, prev) < .3 for prev in picked):
                picked.append(candidate)
        return picked[:20]


class Tracker:
    def __init__(self):
        self.last = None
        self.hits = 0

    def reset(self):
        self.last = None
        self.hits = 0

    def update(self, detections, shape):
        if not detections:
            self.reset()
            return None, 0
        h, w = shape[:2]
        # Choose from the current image; previous identity does not own the target.
        # History only confirms stability before inputs are allowed.
        anchor = (w / 2, h / 2)
        target = min(detections, key=lambda d: math.dist(d.center, anchor) / max(w, h) - .10 * d.score)
        same = self.last and self.last.label == target.label and math.dist(self.last.center, target.center) < max(w, h) * .15
        self.hits = self.hits + 1 if same else 1
        self.last = target
        return target, self.hits


def preview(frame, detections, target, roi):
    canvas = frame.copy()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (round(roi[i] * (w if i % 2 == 0 else h)) for i in range(4))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 180, 30), 2)
    for d in detections:
        color = (0, 240, 255) if d is target else (40, 220, 60)
        cv2.rectangle(canvas, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
        cv2.putText(canvas, f"{d.score:.2f}", (d.x, max(15, d.y - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    cv2.drawMarker(canvas, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 18, 1)
    return canvas


def dashboard(windows, pictures, labels):
    """Each tile is a separate preview, never fed back into target detection."""
    columns = min(3, max(1, math.ceil(math.sqrt(len(windows)))))
    rows = math.ceil(len(windows) / columns)
    cell_w, cell_h = 320, 224
    canvas = np.full((rows * cell_h, columns * cell_w, 3), (43, 33, 23), np.uint8)
    for n, window in enumerate(windows):
        left, top = (n % columns) * cell_w, (n // columns) * cell_h
        frame = pictures.get(window.hwnd)
        if frame is not None:
            ratio = min(cell_w / frame.shape[1], 192 / frame.shape[0])
            resized = cv2.resize(frame, (max(1, round(frame.shape[1] * ratio)), max(1, round(frame.shape[0] * ratio))))
            h, w = resized.shape[:2]
            canvas[top + 30:top + 30 + h, left:left + w] = resized
        label = f'#{n+1} PID {window.pid} | {labels.get(window.hwnd, "WAIT")}'
        cv2.putText(canvas, label, (left + 5, top + 19), cv2.FONT_HERSHEY_SIMPLEX, .40, (230, 230, 230), 1)
    return canvas
