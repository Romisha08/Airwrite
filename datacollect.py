import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import os

# Initialize
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Drawing canvas
canvas = np.ones((480, 640, 3), np.uint8) * 255  # white background

# Folder to save
folder = "data/youtube"   # <-- change for each word
os.makedirs(folder, exist_ok=True)
counter = 0

# Threshold for pinch detection
PINCH_DIST = 40  # pixels

writing = False
prev_point = None

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # mirror for natural movement
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]["lmList"]
        x1, y1 = lmList[8][:2]   # index tip
        x2, y2 = lmList[4][:2]   # thumb tip

        dist, _, _ = detector.findDistance((x1, y1), (x2, y2), img)

        if dist < PINCH_DIST:  # start writing
            writing = True
            if prev_point is None:
                prev_point = (x1, y1)
            cv2.line(canvas, prev_point, (x1, y1), (0, 0, 0), 4)
            prev_point = (x1, y1)
        else:
            writing = False
            prev_point = None

        # Draw circle on fingertip
        cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)

    # Show windows
    cv2.imshow("Webcam", img)
    cv2.imshow("Air Writing", canvas)

    key = cv2.waitKey(1)

    if key == ord("s"):  # save word
        counter += 1
        filename = f"{folder}/word_{time.time()}.png"
        cv2.imwrite(filename, canvas)
        print(f"Saved {filename} ({counter})")
        # Reset canvas after saving
        canvas = np.ones((480, 640, 3), np.uint8) * 255
        
    elif key == ord("r"):  # reset canvas without saving
        print("Canvas reset")
        canvas = np.ones((480, 640, 3), np.uint8) * 255

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
