"""
Spot the Difference - core logic (OpenCV + NumPy only).

Idea:  two photos of the SAME scene (taken from the same place)  ->  subtract them  ->
       where the result is big, something changed  ->  draw a box there.
"""
import cv2
import numpy as np


def match_size(before, after):
    """Make 'after' the same width/height as 'before' (Lab 2: resize). INTER_AREA is the shrinking method."""
    h, w = before.shape[:2]
    if after.shape[:2] != (h, w):
        after = cv2.resize(after, (w, h), interpolation=cv2.INTER_AREA)
    return after


def match_brightness(before, after):
    """If the 2nd photo is a bit darker/brighter overall, shift every channel so the averages match
    (Lab 3: brightness = add a constant, done in int16 + clip so values never wrap around)."""
    out = after.astype(np.int16)
    for c in range(3):
        out[:, :, c] += int(round(float(before[:, :, c].mean()) - float(after[:, :, c].mean())))
    return np.clip(out, 0, 255).astype(np.uint8)


def find_differences(before, after, threshold=40, min_area_pct=0.05, blur=5,
                     ignore_brightness=True, roi=(0, 0, 100, 100)):
    """
    before, after : BGR images
    threshold     : how big a pixel change (0-255) counts as 'different'
    min_area_pct  : ignore changed regions smaller than this % of the image (noise)
    blur          : Gaussian blur size, removes camera noise before subtracting
    roi           : (x1%, y1%, x2%, y2%) - only compare this part of the image (Lab 2: ROI)
    Returns: after_resized, diff_map (grayscale), mask (0/255), list of boxes (x, y, w, h)
    """
    after = match_size(before, after)
    h, w = before.shape[:2]

    a, b = before, after
    if ignore_brightness:
        b = match_brightness(a, b)
    k = int(blur) | 1                                   # blur size must be odd
    if k > 1:
        a, b = cv2.GaussianBlur(a, (k, k), 0), cv2.GaussianBlur(b, (k, k), 0)

    # absdiff = |a - b| computed safely.  (Plain NumPy a-b on uint8 WRAPS AROUND: 10-20 = 246!  Lab 3, Part 12)
    diff3 = cv2.absdiff(a, b)
    diff = diff3.max(axis=2)                            # biggest change among B, G, R at every pixel

    # Mask: 255 where the change is bigger than the threshold, 0 elsewhere (Lab 3: masking)
    _, mask = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)

    # Keep only the chosen region (ROI): everything outside is set to 0
    x1, y1, x2, y2 = roi
    region = np.zeros_like(mask)
    region[int(h * y1 / 100):int(h * y2 / 100), int(w * x1 / 100):int(w * x2 / 100)] = 255
    mask = cv2.bitwise_and(mask, region)

    # Grow the white spots so pieces of the same object join into ONE blob
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, min(h, w) // 60) | 1,) * 2)
    mask = cv2.dilate(mask, ker, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (min_area_pct / 100.0) * h * w
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    boxes.sort(key=lambda r: (r[1], r[0]))              # top-to-bottom, left-to-right
    return after, diff, mask, boxes


def draw_boxes(img, boxes):
    """Red rectangle + number around every difference (Lab 1: drawing)."""
    out = img.copy()
    t = max(2, min(out.shape[:2]) // 200)
    for i, (x, y, w, h) in enumerate(boxes, 1):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), t)
        cv2.putText(out, str(i), (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t / 2, (255, 255, 255), t + 2, cv2.LINE_AA)
        cv2.putText(out, str(i), (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t / 2, (0, 0, 255), t, cv2.LINE_AA)
    return out


def amplify(diff):
    """Make the difference map easier to see (multiply by 4, clip to 255)."""
    return np.clip(diff.astype(np.int16) * 4, 0, 255).astype(np.uint8)
