import tensorflow.keras as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical
import numpy as np
from aa import NNStatFormatter
import pickle


data = pd.read_csv("Data.csv")

data = data.drop(data[data["Waveform"] == "mwc"].index)
data = data.drop(data[data["Waveform"] == "mww"].index)
data = data.drop(data[data["Waveform"] == "mwr"].index)
data = data.drop(data[data["Waveform"] == "mdl"].index)
data = data.drop(data[data["Waveform"] == "m"].index)
data = data.drop(data[data["Waveform"] == "mh"].index)
data = data.drop(data[data["Waveform"] == "mwb"].index)
data = data.drop(data[data["Waveform"] == "mw"].index)
data = data.drop(data[data["Waveform"] == "mblg"].index)
data = data.drop(data[data["Waveform"] == "mw_lg"].index)
data = data.drop(data[data["Waveform"] == "mc"].index)
data = data.drop(data[data["Waveform"] == "mlg"].index)
data = data.drop(data[data["Waveform"] == "ms"].index)
data = data.drop(data[data["Waveform"] == "mwp"].index)
data = data.drop(data[data["Waveform"] == "ma"].index)
data = data.drop(data[data["Waveform"] == "mb_lg"].index)
data = data.drop(data[data["Waveform"] == "ms_20"].index)
data = data.drop(data[data["Waveform"] == "mlr"].index)
count = 0
# for x in data:
#     data["ID"] = count
#     count += 1
data = data.replace("ml", 1)
data = data.replace("mb", 2)
data = data.replace("md", 3)
data["Drop"] = data["Unnamed: 0"]
data.to_csv("RefinedData.csv")



#%%
data = pd.read_csv("RefinedData.csv")
std_mag = []


std_mag = data["Magnitude"]

data["STD Depth"] = data["Depth"].std()
data["STD Magnitude"] = data["Magnitude"].std()
data["STD Distance"] = data["Distance"].std()
data["STD RMS"] = data["Root Mean Square"].std()

data.to_csv("Model Test.csv")

X = data[["Magnitude", "Depth", "Distance","Azimuthal Gap", "Root Mean Square", "STD Magnitude", "STD Depth", "STD Distance", "STD RMS"]]
y = data["Waveform"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1)

X_scaler = MinMaxScaler().fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

#%%

# Step 1: Label-encode data set
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
encoded_y_train = label_encoder.transform(y_train)
encoded_y_test = label_encoder.transform(y_test)

#%%

# Step 2: Convert encoded labels to one-hot-encoding
y_train_categorical = to_categorical(encoded_y_train)
y_test_categorical = to_categorical(encoded_y_test)




# Create model and add layers
model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=9))
model.add(Dense(units=50, activation='relu'))

model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=3, activation='softmax'))


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#%%

model.summary()

#%%

model.fit(
    X_train_scaled,
    y_train_categorical,
    epochs=250,
    shuffle=True,
    verbose=2
)

scores = model.evaluate(X_test_scaled,y_test_categorical, verbose=2)
print(scores)
Model_Stats = pd.DataFrame(scores)
Model_Stats.to_csv("data/NNModelKeyStats.csv")
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# model.save_weights('weight_NN')

# from keras.models import save_model
model.save("NN_model.h5")
NNStatFormatter()
# pkl_filename = "NN_pickle_model_2.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(model, file)

