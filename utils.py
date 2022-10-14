import cv2


def drawer(frame, detections):

    for detect in detections:
        startX, startY, box_w, box_y = detect["box"]
        endX = startX + box_w
        endY = startY + box_y

        text = "{}: {:.2f}%".format(detect["person"], detect["confidence"] * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

    return frame