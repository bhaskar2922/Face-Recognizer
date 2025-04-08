import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load model and Haar Cascade
fr = cv2.face.LBPHFaceRecognizer_create()
fr.read('face_trained.yml')
haar_cascade = cv2.CascadeClassifier('haar_face.xml')
people = ['bhaskar', 'sindhu', 'subhash']
st.title("🤖 Face Recognition Web App")
option = st.sidebar.selectbox("Choose Mode", ["Static Image", "Webcam"])
def recognize_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frec= haar_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in frec:
        fedg = gray[y:y+h, x:x+w]
        lab, conf = fr.predict(fedg)
        if conf < 80:  # Only display if prediction is confident
            n = people[lab]
        else:
            n = "Unknown"
        # name = people[label]
        cv2.putText(img, f'{n} ({int(conf)})', (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 1), 2)
    
    return img

# ---- Static Image Mode ----
if option == "Static Image":
    st.subheader("upload an image")
    f = st.file_uploader("choose an image...", type=["jpg", "jpeg", "png"])

    if f is not None:
        img = Image.open(f)
        imgnp = np.array(img)
        imgbgr = cv2.cvtColor(imgnp, cv2.COLOR_RGB2BGR)

        resimg = recognize_faces(imgbgr)
        st.image(cv2.cvtColor(resimg, cv2.COLOR_BGR2RGB), channels="RGB", caption="Predicted Image")

# ---- Webcam Mode ----
else:
    st.subheader("Live Face Recognition")
    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        isTrue, frame = cap.read()
        if not isTrue:
            st.write("Webcam not setup.")
            break
        resf = recognize_faces(frame)
        rgbf = cv2.cvtColor(resf, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(rgbf)

    cap.release()
