from mtcnn.mtcnn import MTCNN
import cv2

class MTCNNDetector:

    # [
    #     {
    #         'box': [520, 289, 141, 197],
    #         'confidence': 0.999991774559021,
    #         'keypoints': {
    #             'left_eye': (593, 363),
    #             'right_eye': (650, 370),
    #             'nose': (634, 406),
    #             'mouth_left': (585, 437),
    #             'mouth_right': (635, 441)
    #         }
    #     },
    # ]

    def __init__(self):
        self.mtcnn_face_detector = MTCNN()

    def detect_face_from_image(self, image):
        result = self.mtcnn_face_detector.detect_faces(image)
        return result

    def detect_face_from_path(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result = self.detect_face_from_image(image)
        return result