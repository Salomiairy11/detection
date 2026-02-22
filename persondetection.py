import numpy as np
from ultralytics import YOLO


class DetectorAPI:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def processFrame(self, image):
        results = self.model(image)

        boxes_list = []
        scores = []
        classes = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Convert to same format your try.py expects
                boxes_list.append((y1, x1, y2, x2))
                scores.append(conf)
                classes.append(cls)

        return boxes_list, scores, classes, len(boxes_list)

    def close(self):
        pass
