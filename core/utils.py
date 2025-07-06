import cv2

def draw_keypoints(frame, x1, y1, x2, y2):
    cv2.circle(frame, (x1, y1), 8, (255, 0, 0), -1)
    cv2.circle(frame, (x2, y2), 8, (0, 0, 255), -1)

def get_size(shoulder_width):
    if shoulder_width > 160:
        return "Large"
    elif shoulder_width > 130:
        return "Medium"
    else:
        return "Small"
