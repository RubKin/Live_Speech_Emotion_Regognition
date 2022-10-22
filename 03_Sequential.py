import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os,time,librosa,warnings,glob, pickle

from sklearn.preprocessing import MinMaxScaler,OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report

from keras.layers import Dense,Input,Add,Flatten,Dropout,Activation,AveragePooling1D,Conv1D, MaxPooling1D
from keras.models import Model,Sequential,load_model, model_from_json
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import callbacks
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler,EarlyStopping,ReduceLROnPlateau,ModelCheckpoint

warnings.filterwarnings("ignore")

df = pd.read_csv("csvs/small_audio_feats.csv")
X  = df.iloc[:, :-1].values
y = df["EMOTION"].values

encoder = OneHotEncoder()
scaler = StandardScaler()

y = encoder.fit_transform(np.array(y).reshape(-1,1)).toarray()
X_scaled = scaler.fit_transform(X)
X_train, X_toval, y_train, y_toval = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_toval, y_toval, test_size=0.20, random_state=42, shuffle=True)

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
X_val = np.expand_dims(X_val, axis=2)

X_train.shape, X_val.shape, X_test.shape,  y_train.shape, y_val.shape, y_test.shape



model = Sequential()
model.add(layers.LSTM(64, return_sequences = True, input_shape=(X_train.shape[1:3])))
model.add(layers.LSTM(64))
model.add(layers.Dense(4, activation = 'softmax'))
print(model.summary())

batch_size = 256

checkpoint_path = 'models_new'


mcp_save = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True,
                           monitor='val_categorical_accuracy',
                           mode='max')

rlrop = callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', 
                                    factor=0.1, patience=100)
                             
 
model.compile(loss='categorical_crossentropy', 
                optimizer='RMSProp', 
                metrics=['categorical_accuracy'])

history = model.fit(X_train, y_train, 
                      epochs=300, batch_size = batch_size, 
                      validation_data = (X_val, y_val), 
                      callbacks = [mcp_save, rlrop])

loss,acc = model.evaluate(X_val, y_val, verbose=2)



tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
plt.savefig("plots/basic_lstm.png")

print(loss, acc)

plt.plot(history.history["loss"], label="Loss (training data)")
plt.plot(history.history["val_loss"], label="Loss (validation data)")
plt.title("Loss for train and validation")
plt.ylabel("Loss value")
plt.xlabel("No. epoch")
plt.legend(loc="upper left")
plt.show()
plt.savefig("plots/Loss.png")

#Plot history: Accuracy
plt.plot(history.history["categorical_accuracy"], label="Acc (training data)")
plt.plot(history.history["val_categorical_accuracy"], label="Acc (validation data)")
plt.title("Model accuracy")
plt.ylabel("Acc %")
plt.xlabel("No. epoch")
plt.legend(loc="upper left")
plt.show()
plt.savefig("plots/Accuracy.png")


y_val_class = np.argmax(y_val, axis=1)
predictions = model.predict(X_val)
y_pred_class = np.argmax(predictions, axis=1)

cm=confusion_matrix(y_val_class, y_pred_class)

enc_labes = {"angry": 0, "happy": 1, "neural": 2, "sad":3}
index = ["angry", "happy", "neutral", "sad" ]  
columns = ["angry", "happy", "neutral", "sad" ]  
 
cm_df = pd.DataFrame(cm,index,columns)                      
plt.figure(figsize=(12,8))
ax = plt.axes()

sns.heatmap(cm_df, ax = ax, cmap = "PuBu", fmt="d", annot=True)
ax.set_ylabel("True emotion")
ax.set_xlabel("Predicted emotion")
values = cm.diagonal()
row_sum = np.sum(cm,axis=1)
acc = values / row_sum
plt.savefig("plots/Confmatrix.png")

print("Validation set predicted emotions accuracy:")
for e in range(0, len(values)):
    print(index[e],':', f"{(acc[e]):0.4f}")
def save_model(model):
    model_json = model.to_json()
    with open("models/model_seq_71.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("models/model_seq_71.h5")

    json_file= open("models/model_seq_71.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("models/model_seq_71.h5")

    return loaded_model

model_json = model.to_json()
with open("models/model_seq_71.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/model_seq_71.h5")
pred = model.predict(X_test)
y_pred_class = np.argmax(predictions, axis=1)
predictions
prediction_test = model.predict(X_test)
y_pred = encoder.inverse_transform(prediction_test)

y_test = encoder.inverse_transform(y_test)
y_new = encoder.fit_transform(y_test)
y_new
print(df_ = pd.DataFrame( y_pred, y_test))

