from core.detector import detect_person_and_overlay
from config.settings import VIDEO_SOURCE

import cv2

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    detect_person_and_overlay(cap)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
