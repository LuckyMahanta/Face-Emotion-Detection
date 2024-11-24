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
emotion_stack = []  # Use a list to simulate a stack

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
                cv2.putText(im, ' %s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                            (0, 0, 255))
                final_emotion = prediction_label
                if final_emotion:
                    emotion_stack.append(final_emotion)  # Add emotion to the stack
            cv2.imshow("Output", im)
            if cv2.waitKey(1) & 0xFF == ord('m'):
                break
        except cv2.error:
            pass

    webcam.release()
    cv2.destroyAllWindows()

def capture_and_recommend():
    # capture_thread = threading.Thread(target=capture_emotion)
    # capture_thread.start()

    capture_emotion()


   # playlists=[]

    while True:
        if not emotion_stack:
            continue  # Wait until an emotion is captured
        detected_emotion = emotion_stack[-1]  # Get the last captured emotion
        emotion_stack.clear()  # Clear the stack for the next capture

       
        recommendation_links = list(search_playlists(detected_emotion))


        if recommendation_links:
            st.markdown(f"Detected Emotion: {detected_emotion}")
            st.markdown("""<h3 style='padding-top: 20px'>🎶Recommendation Links:</h3>""", unsafe_allow_html=True)
            for playlist in recommendation_links:
                st.write(playlist['name'], playlist['external_urls']['spotify'])

            break

# Frontend

st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide",  
)

# st.markdown("<p style='text-align: center;color : white; font-family:impact; font-size: 60px'>Your Mood, Your Music</p>", unsafe_allow_html=True)
# st.markdown(" <p style='text-align: center;font-style: italic;font-size: 20px; font-family:verdana;'>~ Tune in to your emotions ~</p>", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center;'>
        <p style='margin-bottom: 0px; color: white; font-family: impact; font-size: 60px;'>Your Mood, Your Music</p>
        <p style='margin-top: 2px; font-size: 20px; font-family: arial;'>~ Tune in to your emotions ~</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
  .container {
    display: flex; 
    justify-content: flex-end;
    align-items: center; 
  }
  
  .stButton>button {
    padding: 20px 30px; 
    font-size: 20px; 
    height: 150px; 
    width: 150px;
    display : flex;
    margin-left : 490px;
    border-radius: 700px;
  },
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

button_text = "Let's Go"
if col1.button(button_text):
    capture_and_recommend()

center_text = "<center style= 'color: red;font-size: 20px; font-weight:bold; padding-top: 107px'>To Stop press m </center>"
st.write(center_text, unsafe_allow_html=True)

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
    background-size: 1280px 700px;
    background-repeat : no-repeat;    
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background('./background.png')
