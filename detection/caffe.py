import cv2
import imutils
import numpy as np

class CaffeDetector:

    @staticmethod
    def get_model_paths():
        return {
            "protoPath" : "./models/deploy.prototxt",
            "modelPath" : "./models/res10_300x300_ssd_iter_140000.caffemodel",
        }

    def __init__(self):
        model_paths = CaffeDetector.get_model_paths()
        self.detector = cv2.dnn.readNetFromCaffe(model_paths["protoPath"], model_paths["modelPath"])

    def detect_face_from_image(self, image, threshold=0.5):
        
        (h, w) = image.shape[:2]
        image = imutils.resize(image, width=600)
        
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.detector.setInput(blob)
        detector_result = self.detector.forward()

        detections = []

        for i in range(0, detector_result.shape[2]):
            confidence = detector_result[0, 0, i, 2]
            if confidence < threshold:
                continue

            temp_box = detector_result[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = [max(point, 0) for point in temp_box.astype("int")]

            box_w = endX - startX
            box_h = endY - startY
            
            if endX > w or endY > h:
                print("w",w,"h", h, confidence, temp_box)
                continue
            detection = {
                "box": [startX, startY, box_w, box_h],
                "confidence": confidence,
            }
        
            detections.append(detection)

        return detections