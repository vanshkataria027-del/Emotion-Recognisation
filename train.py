import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.title("Emotion Recognition")

@st.cache_resource
def get_model():
    return load_model("emotion_model.h5")

model = get_model()
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

img_file = st.camera_input("Photo kheencho")
if img_file is None:
    img_file = st.file_uploader("Ya photo upload karo", type=["jpg", "jpeg", "png"])

if img_file is not None:
    img = np.array(Image.open(img_file).convert("RGB"))
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("Koi face nahi mila, dobara try karo.")

    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y+h, x:x+w], (48, 48)) / 255.0
        face = np.reshape(face, (1, 48, 48, 1))
        pred = model.predict(face, verbose=0)
        emotion = emotions[int(np.argmax(pred))]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, max(y-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
