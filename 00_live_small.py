import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os,time,librosa,warnings,glob

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from keras.models import Model,Sequential,load_model, model_from_json
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import pyaudio
import wave
from array import array
import struct
import time

import streamlit as st

st.title("Prediction of Emotions with your Voice")

def add_noise(data):
    noise_value = 0.015 * np.random.uniform() * np.amax(data)
    data = data + noise_value * np.random.normal(size=data.shape[0])
    return data

def stretch_process(data,rate=0.8):
    return librosa.effects.time_stretch(data,rate)

def shift_process(data):
    shift_range = int(np.random.uniform(low=-5,high=5) * 1000)
    return np.roll(data,shift_range)

def pitch_process(data,sampling_rate,pitch_factor=0.7):
    return librosa.effects.pitch_shift(data,sampling_rate,pitch_factor)

def extract_process(data, sample_rate):
    
    output_result = np.array([])
    mean_zero = np.mean(librosa.feature.zero_crossing_rate(y=data).T,axis=0)
    output_result = np.hstack((output_result,mean_zero))
    
    stft_out = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft_out,sr=sample_rate).T,axis=0)
    output_result = np.hstack((output_result,chroma_stft))
    
    mfcc_out = np.mean(librosa.feature.mfcc(y=data,sr=sample_rate).T,axis=0)
    output_result = np.hstack((output_result,mfcc_out))
    
    root_mean_out = np.mean(librosa.feature.rms(y=data).T,axis=0)
    output_result = np.hstack((output_result,root_mean_out))
    
    mel_spectogram = np.mean(librosa.feature.melspectrogram(y=data,sr=sample_rate).T,axis=0)
    output_result = np.hstack((output_result,mel_spectogram))
    
    return output_result

def export_process(path):
    
    data,sample_rate = librosa.load(path,duration=2.5,offset=0.6)
    
    output_1 = extract_process(data, sample_rate)
    result = np.array(output_1)
    
    noise_out = add_noise(data)
    output_2 = extract_process(noise_out, sample_rate)
    result = np.hstack((result))

    new_out = stretch_process(data)
    strectch_pitch = pitch_process(new_out,sample_rate)
    output_3 = extract_process(strectch_pitch, sample_rate)
    result = np.vstack((result,output_3))
    
    return result

def create_X_test(path):
    scaler = StandardScaler()

    features = export_process(path)
    test_1_f = pd.DataFrame(features)
    #test_1_scaled = scaler.fit_transform(test_1_f)
    test_1_exp = np.expand_dims(test_1_f, axis=2)
    return test_1_exp



json_file= open("./models/model_seq_71.json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./models/model_seq_71.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



#
emotions  = {"angry": 0, "happy": 1, "neural": 2, "sad":3}
#emotions = {0 : 'neutral', 1 : 'calm', 2 : 'happy', 3 : 'sad', 4 : 'angry', 5 : 'fearful', 6 : 'disgust', 7 : 'suprised' }
emo_list = list(emotions.values())
emo_list_names = list(emotions)


print("** session started")
st.write("the session has started")
time.sleep(1)
st.write("please start talking for at least 5 seconds")
time.sleep(1)


rate = 24414
chunk = 512
record_seconds = 5
format = pyaudio.paInt32 #change it maybe
channels = 1
wave_output = "outputs/test.wav"

print("* recording...")

p = pyaudio.PyAudio()
stream = p.open(format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk)

total_predictions = []

frames = []
for i in range(0, int(rate/chunk * record_seconds)):
    data = array("l", stream.read(chunk)) 
    frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()


wf = wave.open(wave_output, "wb")
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(format))
wf.setframerate(rate)
wf.writeframes(b"".join(frames))

st.write("done recording")
time.sleep(1)
print("* done recording")

x = create_X_test(wave_output) 

predictions = loaded_model.predict(x, use_multiprocessing=True)
#prediction_nonseen = loaded_model.predict(x)
#arg_prediction_nonseen = prediction_nonseen.argmax(axis=-1)
pred_list = list(predictions)
pred_np = np.squeeze(np.array(pred_list).tolist()[0]) 
#total_predictions.append(pred_np)

#Present emotion distribution for a sequence (7.1 secs).
fig = plt.figure(figsize = (10, 2))
plt.bar(emo_list_names, pred_np, color = "darkturquoise")
plt.ylabel("Probabilty (%)")
plt.show()
st.write("this are your emotions")
st.pyplot(fig)


# if st.button("# rec.."):
#     st.button("# stop...")
#     try:
#while True:
    #st.write("....recording your voice...")
    



#     stream.stop_stream()
#     stream.close()
#     p.terminate()
    
#     max_emo = np.argmax(total_predictions)
#     print("max emotion:", emotions.get(max_emo,-1))
    
#     print(100*"-")
    
#     # Define the last 2 seconds sequence.
#     last_frames = np.array(struct.unpack(str(96 * CHUNK) + "B" , np.stack(( frames[-1], frames[-2], frames[-3], frames[-4],
#                                                                             frames[-5], frames[-6], frames[-7], frames[-8],
#                                                                             frames[-9], frames[-10], frames[-11], frames[-12],
#                                                                             frames[-13], frames[-14], frames[-15], frames[-16],
#                                                                             frames[-17], frames[-18], frames[-19], frames[-20],
#                                                                             frames[-21], frames[-22], frames[-23], frames[-24]),
#                                                                             axis =0)) , dtype = "b")
#     if is_silent(last_frames): # If the last 2 seconds are silent, end the session.
#         break


# # SESSION END        
# toc = time.perf_counter()
# stream.stop_stream()
# stream.close()
# p.terminate()
# wf.close()
# #st.write("the session has ended")
# print("** session ended")

# except:
#     st.error("### try louder")

# total_predictions_np =  np.mean(np.array(total_predictions).tolist(), axis=0)
# fig = plt.figure(figsize = (10, 5))
# plt.bar(emo_list, total_predictions_np, color = 'indigo')
# plt.ylabel("Mean probabilty (%)")
# plt.title("Session Summary")
# plt.show()

# print(f"Emotions analyzed for: {(toc - tic):0.4f} seconds")