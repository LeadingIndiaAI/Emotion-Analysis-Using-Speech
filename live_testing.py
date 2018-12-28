import pyaudio
import wave
import speech_recognition as sr

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

RECORD_SECONDS = 5

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,channels=2,rate=44100, input=True, frames_per_buffer=1024)

print("**** recording")

frames = []

for i in range(0, int(44100 / 1024 * RECORD_SECONDS)):
    data = stream.read(1024)
    frames.append(data)

print("**** done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open('output.wav', 'wb')
wf.setnchannels(2)
wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
wf.setframerate(44100)
wf.writeframes(b''.join(frames))
wf.close()

harvard = sr.AudioFile('output.wav')
r = sr.Recognizer()
with harvard as source:
      audio = r.record(source)
      text = r.recognize_google(audio)
try:
    print("You said: " +text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
    
    

data, sampling_rate = librosa.load('output.wav')

##% pylab inline
import os
import pandas as pd
import librosa
import glob 

plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

#livedf= pd.DataFrame(columns=['feature'])
X, sample_rate = librosa.load('output.wav', res_type='kaiser_fast',duration=5,sr=22050*2,offset=0)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive
livedf2= pd.DataFrame(data=livedf2)

livedf2 = livedf2.stack().to_frame().T
twodim= np.expand_dims(livedf2, axis=2)
livepreds = loaded_model.predict(twodim, 
                         batch_size=32, 
                         verbose=1)

livepreds1=livepreds.argmax(axis=1)

liveabc = livepreds1.astype(int).flatten()
print(livepreds[0])
livepredictions = (lb.inverse_transform((liveabc)))
print(livepredictions[0][0:3])


import pandas as pd

df = pd.read_csv('/home/amitkmaurya16/Downloads/Emotion/train_data.csv')
df.head()
df.isnull().sum() 
df['sentiment'].value_counts()

from sklearn.model_selection import train_test_split

X = df['content']
y = df['sentiment']

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=42)


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
#from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])

text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)

#from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
#print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))

print('Emotion detected through words used :'+ text_clf.predict([text]))
    