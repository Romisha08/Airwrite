import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import webbrowser
import time

# Webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Classifier
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["W", "L", "M", "Y" , "Ct"]

# Canvas for drawing
canvas = np.ones((480, 640, 3), np.uint8) * 255
PINCH_DIST = 40   # pixels for pinch detection
prev_point = None
writing = False

# Cooldown timer
last_action_time = 0
cooldown = 6  # seconds

# Mapping characters to websites
sites = {
    "W": "https://web.whatsapp.com/",
    "L": "https://www.linkedin.com/",
    "Y": "https://www.youtube.com/",
    "C": "https://chatgpt.com/" ,
    "M": "https://mail.google.com/"  
}

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # mirror effect
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]["lmList"]
        x1, y1 = lmList[8][:2]   # index tip
        x2, y2 = lmList[4][:2]   # thumb tip

        dist, _, _ = detector.findDistance((x1, y1), (x2, y2), img)

        if dist < PINCH_DIST:  # Writing mode
            if prev_point is None:
                prev_point = (x1, y1)
            cv2.line(canvas, prev_point, (x1, y1), (0, 0, 0), 6, cv2.LINE_AA)
            prev_point = (x1, y1)
            writing = True
        else:
            if writing:  # Finished writing, classify the canvas
                gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(gray, (300, 300))  # match classifier size
                img_input = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

                prediction, index = classifier.getPrediction(img_input, draw=False)
                predicted = labels[index]

                print(f"Predicted: {predicted}")

                # Cooldown check
                current_time = time.time()
                if current_time - last_action_time > cooldown:
                    if predicted in sites:
                        webbrowser.open(sites[predicted])
                        last_action_time = current_time

                # Reset canvas after classification
                canvas = np.ones((480, 640, 3), np.uint8) * 255

            writing = False
            prev_point = None

        # Show fingertip
        cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)

    # Show windows
    cv2.imshow("Webcam", img)
    cv2.imshow("Air Writing", canvas)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
