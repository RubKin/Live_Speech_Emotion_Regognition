import pandas as pd
import numpy as np
import re
import librosa,warnings
import os
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder, StandardScaler
#
warnings.filterwarnings("ignore")

df = pd.read_csv("csvs/all_data_corr.csv")
df = df.sample(frac=1).reset_index(drop=True)
#df.head(10)
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
def create_new_df(df, name):
    # df = pd.read_csv("csvs/all_data.csv")
    # df = df.sample(frac=1).reset_index(drop=True)
    #print(df.head(10))

    AVAILABLE_EMOTIONS = {"disgust", "fear", "angry","sad","neutral","happy"}
    X_train, y_train = [], []

    for path, emotion in zip(df.iloc[:,0],df.iloc[:,1]):
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        features = export_process(path)

        for element in features:
            X_train.append(element)
            y_train.append(emotion)

    New_Features_Wav = pd.DataFrame(X_train)
    New_Features_Wav["EMOTION"] = y_train

    New_Features_Wav.to_csv(name,index=False)

    New_Features_Wav = pd.read_csv(name)
    
    return New_Features_Wav

df_feat_big= create_new_df(df, "big_audio_feats_corr.csv")
# df_feat_big
# df_feat= create_new_df(df, "test_audio_feats.csv")
# df_feat