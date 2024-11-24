import cv2
from keras.models import model_from_json
import numpy as np
import threading
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import base64

capturing = False
window_name = "Output"
emotion_stack = []

client_id = 'a42ce6c619654d6e90e85bc2ae9cc2d3'
client_secret = 'fd154a7dac254d449a5846230bc91827'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def search_playlists(keyword):
    results = sp.search(q=keyword, type='playlist')
    playlists = results['playlists']['items']
    return playlists

def capture_emotion():
    global capturing, emotion_stack
    json_file = open("emotiondetector.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    final_emotion = ""

    model.load_weights("emotiondetector.h5")
    hear_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(hear_file)

    def extract_features(image):
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0

    webcam = cv2.VideoCapture(0)
    labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

    while True:
        i, im = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(im, 1.3, 5)
        try:
            for (p, q, r, s) in faces:
                image = gray[q: q + s, p: p + r]
                cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                # Adjust text size based on frame size
                text_size = min(2, max(0.5, im.shape[1] / 640))
                cv2.putText(im, prediction_label, (p - 10, q - 10), 
                           cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size,
                           (0, 0, 255))
                final_emotion = prediction_label
                if final_emotion:
                    emotion_stack.append(final_emotion)
            cv2.imshow("Output", im)
            if cv2.waitKey(1) & 0xFF == ord('m'):
                break
        except cv2.error:
            pass

    webcam.release()
    cv2.destroyAllWindows()

def capture_and_recommend():
    capture_emotion()

    while True:
        if not emotion_stack:
            continue
        detected_emotion = emotion_stack[-1]
        emotion_stack.clear()

        recommendation_links = list(search_playlists(detected_emotion))

        if recommendation_links:
            st.markdown(f"Detected Emotion: {detected_emotion}")
            st.markdown("""<h3 style='padding-top: 20px'>🎶Recommendation Links:</h3>""", unsafe_allow_html=True)
            
            # Create a grid layout for playlists
            cols = st.columns(2)
            for idx, playlist in enumerate(recommendation_links):
                with cols[idx % 2]:
                    st.write(playlist['name'], playlist['external_urls']['spotify'])
            break

# Frontend Setup
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide",
)

# Custom CSS for responsiveness while maintaining original colors
st.markdown("""
<style>
    /* General responsive styles */
    .stApp {
        max-width: 100vw;
        padding: 1rem;
    }
    
    /* Header styles */
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-title {
        color: white;
        font-family: impact;
        font-size: clamp(2rem, 5vw, 4rem);
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: clamp(1rem, 2vw, 1.5rem);
        font-family: arial;
        font-style: italic;
        margin-top: 2px;
    }
    
    /* Button styles */
    .stButton > button {
        padding: clamp(1rem, 3vw, 2rem);
        width: clamp(100px, 20vw, 150px);
        height: clamp(100px, 20vw, 150px);
        font-size: clamp(1rem, 2vw, 1.25rem);
        border-radius: 700px;
        margin: 0 auto;
        display: block;
    }
    
    /* Stop message styling */
    .stop-message {
        text-align: center;
        color: red;
        font-size: clamp(1rem, 1.5vw, 1.25rem);
        font-weight: bold;
        padding-top: 107px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
        <div class="main-title">Your Mood, Your Music</div>
        <div class="subtitle">~ Tune in to your emotions ~</div>
    </div>
""", unsafe_allow_html=True)

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("Let's Go"):
        capture_and_recommend()

st.markdown(
    '<div class="stop-message">To Stop press m</div>',
    unsafe_allow_html=True
)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        min-height: 100vh;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('./background.png')