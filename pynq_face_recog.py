import logging
logging.basicConfig(level=logging.INFO)


logging.info("Program başlatılıyor.")

# Paketler projeye dahil edilir.
from pynq import Overlay
import numpy as np
import cv2
from pynq.drivers.video import HDMI


Overlay("base.bit").download()

# Hdmi çıkış konfigürasyonları yapılır.
hdmi_out = HDMI('out', video_mode=HDMI.VMODE_640x480)
hdmi_out.start()


# Monitör çıktı arabellek boyutu belirlenir.
frame_out_w = 1920
frame_out_h = 1080
# Kamera girdi boyutu belirlenir.
frame_in_w = 640
frame_in_h = 480

# OpenCV kullanarak kamerayı başlatır.
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);

logging.info("Kamera açık/kapalı durumu: " + str(webcam.isOpened()))


logging.info("Yüz tanıma paketleri ve yüzler ekleniyor.")

import face_recognition
ilayda_image = face_recognition.load_image_file("database/ilayda/1.jpg")
ilayda_face_encoding = face_recognition.face_encodings(ilayda_image)[0]

kadir_image = face_recognition.load_image_file("database/kadir/1.jpg")
kadir_face_encoding = face_recognition.face_encodings(kadir_image)[0]


known_face_encodings = [
    ilayda_face_encoding,
    kadir_face_encoding
]

known_face_names = [
    "Ilayda",
    "Kadir"
]

face_locations = []
face_encodings = []
face_names = []

logging.info("Tüm yüzler sisteme başarıyla tanımlandı.")


logging.info("Yüz tanıma başlatılıyor.")

from pynq.board import Button

# 0 numaralı buton sistemi durdurmak için tanımlandı.
stopButton = Button(0)

frame_counter = 0
ret, frame = webcam.read()

while True:
    ret, frame = webcam.read()
    frame_1080p = np.zeros((1080,1920,3)).astype(np.uint8)       
    
    inputBGR = cv2.resize(frame, (0,0),fx=0.25, fy=0.25)
    rgb_small_frame = inputBGR[:,:,::-1]
    

    if stopButton.read():
        logging.info("Program durduruldu.")
        break
    
    # Her 5 frameden biri için yüz tanıma algoritması çalışacaktır. Böylece sistem yavaşlamamış olacaktır.
    if frame_counter%5==0:
        # Frame içindeki yüzleri ve lokasyonlarını bulur.
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Sisteme tanımlamış olduğumuz yüzler ile framede bulduklarını karşılaştırır.
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Bilinmiyor"

            # Tanınmış yüzleri isimleriyle, diğerlerini bilinmiyor olarak diziye ekler.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)


        # Sonuçları ekranda gösterir.
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Yüz tanımayı 1/4 oranınıda yaptığımız için yüz konumlarını tekrardan eski orana çevirir.
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Yüzü belirten bir dikdörtgen çizer.
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Çizilen dikdörtgenin yanına yüzün adını yazar.
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left, bottom - 10), font, 1.0, (255, 255, 255), 1)
        
        frame_counter+=1
    
    frame_1080p = np.zeros((1080,1920,3)).astype(np.uint8)       
    frame_1080p[0:480,0:640,:] = frame[0:480,0:640,:]
    hdmi_out.frame_raw(bytearray(frame_1080p.astype(np.int8)))
    

logging.info("Sistem kapatılıyor.")

# Tanımlanmış olan camera değişkeni ve hdmi çıkışı silinir.
webcam.release()
hdmi_out.stop()
del hdmi_out