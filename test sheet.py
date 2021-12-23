import tensorflow.keras as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical
import numpy as np

import pickle
from tensorflow.keras.models import load_model

data = pd.read_csv("Model_Test.csv")



x = data[["Magnitude", "Depth", "Distance","Azimuthal Gap", "Root Mean Square", "STD Magnitude", "STD Depth", "STD Distance", "STD RMS"]]
real = data["Waveform"]

# real["Waveform Clean"] = real.replace(1, "[1.0, 0.0, 0.0]")
# real["Waveform Clean"] = real["Waveform Clean"].replace(2, "[0.0, 1.0, 0.0]")
# real["Waveform Clean"] = real["Waveform Clean"].replace(3, "[0.0, 0.0, 1.0]")
# real = real["Waveform Clean"]



model = load_model('NN_model.h5')

predictions = model.predict(x)

predictions_list = []
for z in predictions:
    predictions_list.append(z)

predictions_DF = pd.DataFrame({"Predictions": predictions_list})


label_encoder = LabelEncoder()
label_encoder.fit(real)
encoded = label_encoder.transform(real)
categorical = to_categorical(encoded)

categorical_list = []
for z in categorical:
    categorical_list.append(z)
categorical_df = pd.DataFrame({"Waveform":categorical_list})



complete = pd.merge(categorical_df, predictions_DF, left_index=True, right_index=True)
complete.to_csv("data/NNwaveformdata.csv")

true = 0
false = 0
i = 0

while i < len(complete):
    # finds the values un compared Dataframe at index i
    value = complete.loc[i]

    value = value.T

    # assigns actual value to com1 and predicted value to com2
    com1 = value["Waveform"]
    com2 = value["Predictions"]
    # com2 = str(com2)

    # compares com to com 2 if same then true if different false then adds 1 to i index
    if all(com1 == com2):
        true += 1
        i += 1

    if any(com1 != com2):
        false += 1
        i += 1




Stats = pd.read_csv("data/NNmodelKeyStats.csv")
Stats["Actual Accuracy"] = (true/i)*100
Stats.to_csv("data/NNmodelKeyStats.csv")
# complete = pd.merge(real, predictions)
