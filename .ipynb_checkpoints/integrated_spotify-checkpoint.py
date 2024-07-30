import cv2
from keras.models import model_from_json
import numpy as np
#import webbrowser
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


json_file= open("emotiondetector.json","r")
model_json=json_file.read()
json_file.close()
model = model_from_json(model_json)

final_emotion=""

model.load_weights("emotiondetector.h5")
hear_file= cv2.data.haarcascades+ "haarcascade_frontalface_default.xml"
#to detect our face from camera
face_cascade= cv2.CascadeClassifier(hear_file)

def extract_features(image):
    feature= np.array(image)
    feature= feature.reshape(1,48,48,1)
    return feature/255.0

webcam = cv2.VideoCapture(0)
labels= {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"neutral", 5:"sad", 6:"surprise"}

happy_playlist=["https://open.spotify.com/playlist/0jrlHA5UmxRxJjoykf7qRY?si=a3db07b1f11348a8", "https://open.spotify.com/playlist/1ZKdvf5yXvwVyzgznVz2Nl?si=9070d20d742e4708" , "https://open.spotify.com/playlist/0x3wglohDy4DHednSb91wA?si=89ba36b7c66b4285"]

angry_playlist= ["https://open.spotify.com/playlist/609gQW5ztNwAkKnoZplkao?si=ef737bf8546946d3","https://open.spotify.com/playlist/4wd8PRPwNXDURHTrhTHlA4?si=18be2ee9b3134a55","https://open.spotify.com/playlist/3kZwG7OlYDSmoHiem6Fbd9?si=6e5f36435f274f41"]

disgust_playlist=["https://open.spotify.com/playlist/2e4Ca2HBIQNs2G2n9IMtnR?si=27e25aec06ce4b78","https://open.spotify.com/playlist/37i9dQZF1E8KEaf5o7wGZB?si=4c5b5d3d585c4a81","https://open.spotify.com/playlist/30WK5IrD3u1G0sRWr8IFgQ?si=c102f1e67763415e"]

fear_playlist=[]

emotion_playlist_mapping = {
    "angry": "https://open.spotify.com/playlist/0YMghmr5hSy2rrkL4BVuHP?si=cb993302b1554fbb",
    "disgust":"https://open.spotify.com/playlist/1KQiNjmLkHeDxkvRqlUYD2?si=cf7341f8b34c4bd1",
    "fear":"https://open.spotify.com/playlist/6PfWQuxhmEnmTJ7oQJlA1T?si=053433121851470f",
    "happy": happy_playlist,
    "neutral" :"https://open.spotify.com/playlist/4PFwZ4h1LMAOwdwXqvSYHd?si=539b740312254206",
    "sad" :"https://open.spotify.com/playlist/25ZzkJkOuYir9kHr2CqwPQ?si=0c47b4144ccf49fe" ,
    "surpise" :"https://open.spotify.com/playlist/7oszvIc5rxQGxmwpMK1PID?si=1f5339c907e9467a"
}

while True:
    i,im= webcam.read()
    gray= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(im, 1.3, 5)
    try:
        for (p,q,r,s) in faces:
            image= gray[q: q+s, p:p+r]
            cv2.rectangle(im, (p,q),(p+r, q+s),(255,0,0),2)
            image= cv2.resize(image,(48,48))
            #resized the image to 48x48
            img = extract_features(image)
            pred= model.predict(img)
            prediction_label= labels[pred.argmax()]
            cv2.putText(im,' %s' %(prediction_label),(p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255))
            print(prediction_label," is the emotion as this point of time.")
            final_emotion=prediction_label
        cv2.imshow("Output", im)
        #cv2.waitKey(27)
        if cv2.waitKey(1) & 0xFF==ord('q') :
            break
    
        
        

        
    except cv2.error:
        pass

'''
def open_spotify_playlist(playlist_id):
    playlist_uri = f"spotify:playlist:{playlist_id}"
    webbrowser.open(playlist_uri)


def open_spotify_playlist(playlist_id):
    playlist_url = f"{playlist_id}"
    webbrowser.open(playlist_url)


def open_spotify_playlists(playlist_ids):
    for playlist_id in playlist_ids:
        print(playlist_id)
        open_spotify_playlist(playlist_id)


'''           
if final_emotion in emotion_playlist_mapping.keys():
            print(emotion_playlist_mapping[final_emotion])
            #open_spotify_playlists(emotion_playlist_mapping[final_emotion])

client_id = 'a42ce6c619654d6e90e85bc2ae9cc2d3'
client_secret = 'fd154a7dac254d449a5846230bc91827'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def search_playlists(keyword):
    results = sp.search(q=keyword, type='playlist')
    playlists = results['playlists']['items']
    return playlists

keyword = final_emotion
playlists = search_playlists(keyword)
for playlist in playlists:
    print(playlist['name'], playlist['external_urls']['spotify'])
