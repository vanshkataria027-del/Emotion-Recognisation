import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

st.title("Emotion Recognition")

@st.cache_resource
def get_model():
    return load_model("emotion_model.h5")

model = get_model()
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

img_file = st.camera_input("Photo kheencho (chehra beech mein rakho)")
if img_file is None:
    img_file = st.file_uploader("Ya photo upload karo", type=["jpg", "jpeg", "png"])

if img_file is not None:
    img = Image.open(img_file).convert("L")
    img = ImageOps.fit(img, (48, 48))
    arr = np.array(img) / 255.0
    arr = np.reshape(arr, (1, 48, 48, 1))
    pred = model.predict(arr, verbose=0)[0]
    st.image(img_file)
    st.subheader("Emotion: " + emotions[int(np.argmax(pred))])
    st.bar_chart(dict(zip(emotions, [float(p) for p in pred])))
