import pandas as pd
import numpy as np
import re
import os,warnings,glob
## Paths to the samples
path_tess = "data/tess/tess toronto emotional speech set data/OAF_angry/OAF_back_angry.wav"
path_rav = "data/rav/Actor_01/03-01-01-01-01-01-01.wav"
path_savee = "data/savee/DC_a01.wav"
path_crema = "data/crema/1001_DFA_ANG_XX.wav"
"""
## Ravdess
Dataset link to download: “https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio"
format 03–01–01–01–01–01–01.wav.
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral" emotion.
Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
Repetition (01 = 1st repetition, 02 = 2nd repetition).
03–01–01–01–01–01–01.wav 
03=audio-only, 01=speech, 01=neutral, 01=normal, 01=statement kids and 01=1st

## CREMA-D
Dataset link to download: “https://www.kaggle.com/ejlok1/cremad"

The format of files is 1001_DFA_ANG_XX.wav, where ANG stands for angry emotion.
Similarly different emotion mappings are as follows:
{‘SAD":"sad","ANG":"angry","DIS":"disgust","FEA":"fear","HAP":"happy","NEU":"neutral"}

## TESS
Dataset link to download: “https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess"

OAF_angry, OAF_neural, OAF_disgust, YAF_sad and so on, where name after the underscore of the folder name contains the emotion information, so the name after the underscore of the folder name is taken and files residing insider the folders are labeled accordingly.

4. Surrey Audio Visual Expressed Emotion (Savee) dataset description:
Dataset link to download: “https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee"
Dataset stored on google drive at path: “/content/drive/MyDrive/Audiofiles/ALL/”
The files are in a format DC_a01.wav where a single character contains the emotion information , for example character "a" after underscore in the file name “DC_a01.wav” means emotion is angry.
Similarly different emotion mappings are as follows:
{‘a":"anger","d":"disgust","f":"fear","h":"happiness","n":"neutral","sa":"sadness","su":"surprise"}
---
"""

# Import the libraries and create Datasets

path_tess_sample = "data/tess/tess toronto emotional speech set data/OAF_angry/OAF_back_angry.wav"
path_rav_sample = "data/rav/Actor_01/03-01-01-01-01-01-01.wav"
path_savee_sample = "data/savee/DC_a01.wav"
path_creama_sample = "data/creama/1001_DFA_ANG_XX.wav"

path_tess = "./data/tess"
path_rav = "./data/rav/actors"
path_savee = "./data/savee"
path_crema = "./data/crema"
def load_tess(path):
    wavs = []
    emotions = []
    emotion_tess = {"disgust": "disgust", "ps": "surprised", "fear": "fear", "angry": "angry", "neutral": "neural", "sad": "sad", "happy": "happy"}
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            wavs.append(os.path.join(dirname,filename))
            emotion = filename.split("_")[-1]
            emotion = emotion.split(".")[0]
            emotions.append(emotion.lower())
    df = pd.DataFrame()
    df["wavs"] = wavs
    df["emotion"] = emotions
    df = df.replace({"emotion": emotion_tess})
    
    return df
        

def load_rav(path):
    wavs = []
    emotions = []
    emotion_ravdess = {"01":"neutral","02":"calm","03":"happy","04":"sad","05":"angry","06":"fear","07":"disgust","08":"surprised"}
    for dirname, dirs, filenames in os.walk(path):
        for filename in filenames:
            wavs.append(os.path.join(dirname,filename))
            emotion = filename.split("-")[2]
            emotions.append(emotion.lower())  
    df = pd.DataFrame()
    df["wavs"] = wavs
    df["emotion"] = emotions
    df = df.replace({"emotion": emotion_ravdess})
    
    return df

def load_crema(path):
    wavs = []
    emotions = []
    emotion_crema = {"sad":"sad","ang":"angry","dis":"disgust","fea":"fear","hap":"happy","neu":"neutral"}
    for dirname, dirs, filenames in os.walk(path):
        for filename in filenames:
            wavs.append(os.path.join(dirname,filename))
            emotion = filename.split("_")[2]
            emotions.append(emotion.lower())  
    df = pd.DataFrame()
    df["wavs"] = wavs
    df["emotion"] = emotions
    df = df.replace({"emotion": emotion_crema})
    
    return df

def load_savee(path):
    wavs = []
    emotions = []
    emotion_savee = {"a":"angry","d":"disgust","f":"fear","h":"happy","n":"neutral","sa":"sad","su":"surprised"}
    for dirname, dirs, filenames in os.walk(path):
        for filename in filenames:
            wavs.append(os.path.join(dirname,filename))
            emotion = filename.split("_")[1]
            emotion = re.match(r"([a-z]+)([0-9]+)",emotion)[1]
            emotions.append(emotion.lower())   
    df = pd.DataFrame()
    df["wavs"] = wavs
    df["emotion"] = emotions
    df = df.replace({"emotion": emotion_savee})
    
    return df

def merge_dfs():
    df_ravdess = load_tess(path_tess)
    df_crema = load_rav(path_rav)
    df_tess = load_crema(path_crema)
    df_savee = load_savee(path_savee)

    frames = [df_ravdess,df_crema,df_tess,df_savee]
    df= pd.concat(frames)
    df.reset_index(drop=True,inplace=True)
    
    df.to_csv("all_data_corr.csv",index=False,header=True)
    print("Total length of the dataset is {}".format(len(df)))
    return df

df = merge_dfs()


"""
***
### Links

# ravdess 
https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio

# cremad
https://www.kaggle.com/ejlok1/cremad

# Tess
https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess

# Savee
https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee
"""
