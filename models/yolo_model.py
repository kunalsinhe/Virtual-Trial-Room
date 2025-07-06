from ultralytics import YOLO
from ultralytics.utils import LOGGER
from config.settings import MODEL_PATH

# Suppress YOLO logs
LOGGER.setLevel(50)

def load_model():
    return YOLO(MODEL_PATH)
