# Airwrite
AirWrite Launcher uses a webcam + hand tracking to let you write letters in the air (pinch + move). When you finish a character, the app classifies the drawn letter and opens a website mapped to that letter (with a cooldown to avoid accidental repeats).

# ✨ AirWrite Launcher (Teachable Machine Edition)

> **Write alphabets in the air using your fingers — recognized with Google Teachable Machine — to instantly open mapped websites.**

---

## 📌 Features
- ✋ Track your hand in real-time using webcam.  
- 🖊️ Air-write alphabets with finger pinch gesture.  
- 🧠 Predict letters using **Google Teachable Machine** model.  
- 🌐 Instantly open mapped websites (e.g., YouTube, Netflix, WhatsApp).  

---

## 📂 Project Structure

AirWrite/
├─ Model/
│  ├─ keras_model.h5
│  └─ labels.txt
├─ test.py
├─datacollect.py
├─ requirements.txt
└─ README.md

⚙️ Requirements

Python 3.8+
mediapipe
opencv-python
cvzone
tensorflow
numpy
