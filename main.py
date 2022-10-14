import os
import cv2

from detection.caffe import CaffeDetector
from deepface import DeepFace

from utils import drawer


# DeepFace.stream(db_path = "database/")

cap = cv2.VideoCapture(0)
detector = CaffeDetector()

while True:
    ret, frame = cap.read()

    if ret:
        detections = detector.detect_face_from_image(frame)

        df = DeepFace.find(
            img_path=frame,
            model_name="Facenet512",
            db_path="database/",
            enforce_detection=False
        )
        df = df.values.tolist()
        names = [(os.path.split(item[0])[0]).replace("database/", "") for item in df]
        if not names:
            names.append("unknown")
        detections = [ dict(item, **{'person':names[0]}) for item in detections]
        frame = drawer(frame, detections)
        cv2.imshow("Face Recognition", frame)


        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()