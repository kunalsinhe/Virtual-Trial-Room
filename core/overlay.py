import cv2
import numpy as np
from config.settings import SHIRT_IMAGE_PATH

shirt_img = cv2.imread(SHIRT_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

def overlay_shirt(frame, x1, y1, x2, y2):
    shoulder_width = abs(x2 - x1)
    shirt_width = shoulder_width + 60
    shirt_height = int(shirt_width * 1.3)

    shirt_resized = cv2.resize(shirt_img, (shirt_width, shirt_height))
    h, w, _ = shirt_resized.shape
    x_offset = min(x1, x2) - 30
    y_offset = y1 + 10

    if (x_offset >= 0 and y_offset >= 0 and
        x_offset + w <= frame.shape[1] and
        y_offset + h <= frame.shape[0]):

        roi = frame[y_offset:y_offset + h, x_offset:x_offset + w]
        shirt_rgb = shirt_resized[:, :, :3]
        shirt_alpha = shirt_resized[:, :, 3] / 255.0

        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1 - shirt_alpha) + shirt_rgb[:, :, c] * shirt_alpha

        frame[y_offset:y_offset + h, x_offset:x_offset + w] = roi
        return (x_offset, y_offset, w, h), shoulder_width

    return None, None
