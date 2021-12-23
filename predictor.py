import pickle
import requests
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



data_url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2005-1-01&endtime=2006-1-01&mindepth=.2&eventtype=earthquake&limit=20000&mingap=1" \
           "&maxlatitude=41.9961351764005&minlongitude=-124.21129087870494&minlatitude=32.76271062703306&maxlongitude=-114.51377843459622"


response = requests.get(data_url)
data = response.json()


dmin = []
tsunami = []
mag = []
magType = []
gap = []
rms = []
sig = []
none = []
depth = []

for y in data["features"]:
    if y["properties"]["dmin"] == None:
        none.append((y["properties"]["dmin"]))
    else:
        dmin.append(y["properties"]["dmin"])

    if y["properties"]["mag"] == None:
        none.append((y["properties"]["mag"]))
    else:
        mag.append(y["properties"]["mag"])

    if y["properties"]["magType"] == None:
        none.append((y["properties"]["magType"]))
    else:
        magType.append(y["properties"]["magType"])

    if y["properties"]["gap"] == None:
        none.append((y["properties"]["gap"]))
    else:
        gap.append(y["properties"]["gap"])

    if y["properties"]["rms"] == None:
        none.append((y["properties"]["rms"]))
    else:
        rms.append(y["properties"]["rms"])

    if y["properties"]["sig"] == None:
        none.append((y["properties"]["sig"]))
    else:
        sig.append(y["properties"]["sig"])

    if y["geometry"]["coordinates"][2] == None:
        none.append((y["geometry"]["coordinates"][2]))
    else:
        depth.append((y["geometry"]["coordinates"][2]))


pkl_filename = "pickle_model.pkl"
# loading data
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

clf = pickle_model

case_sensitive_Waveforms = []
for x in magType:
   case_sensitive_Waveforms.append(x.lower())
# creates dataframe from lists
depth_DF = pd.DataFrame({"Depth": depth})
dmin_Df = pd.DataFrame({"Distance": dmin})
mag_df = pd.DataFrame({"Magnitude": mag})
mag_df["Magnitude"] = mag_df["Magnitude"].abs()
magType_df = pd.DataFrame({"Waveform": case_sensitive_Waveforms})
rms_DF = pd.DataFrame({"Root Mean Square": rms})
gap_DF = pd.DataFrame({"Azimuthal Gap": gap})
sig_DF = pd.DataFrame({"Signature": sig})
tsunami_df = pd.DataFrame({"Tsunami": tsunami})

Ses_DF = pd.DataFrame(dmin_Df)
Ses_DF = Ses_DF.merge(mag_df, "inner", right_index=True, left_index=True)
Ses_DF = Ses_DF.merge(magType_df, "inner", right_index=True, left_index=True)
Ses_DF = Ses_DF.merge(rms_DF, "inner", right_index=True, left_index=True)
Ses_DF = Ses_DF.merge(gap_DF, "inner", right_index=True, left_index=True)
Ses_DF = Ses_DF.merge(depth_DF, "inner", right_index=True, left_index=True)
Ses_DF = Ses_DF.merge(sig_DF, "inner", right_index=True, left_index=True)
Ses_DF = Ses_DF.dropna()

print("Dataframe Construction DONE")



X = Ses_DF[["Magnitude", "Depth", "Distance","Azimuthal Gap", "Root Mean Square"]]
Y = Ses_DF["Waveform"]

# test train split
# X_test, Y_test = train_test_split(X, y)




score = clf.score(X, Y)
print(score)
# predicted model
predictedY = clf.predict(X)
predicted = pd.DataFrame({"Predicted_Waveform":predictedY})
print(predicted)
print(Y, X)
print("Prediction DONE")
# merges predicted and actual Waveforms
compared = predicted.merge(Y, "inner", right_index=True, left_index=True)
compared = compared.reset_index(drop=True)
compared.to_csv(path_or_buf="data/PredictionData.csv")
# print(compared)
# compared accuracy list passes a True if the prediction matches the actual waveform, and False if it doesn't
# i is the index for the while loop
# i_list creates an index within the data for later processing

# PROTOTYPE CODE BELOW! DO NOT TOUCH OR UNCOMMENT!!!!!!!!!!
#-------------------------------------------------------------------------
# YOU HAVE BEEN WARNED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#--------------------------------------------------------------------------------------

compared_accuracy = []
i = 0
i_list = []
# while loop for passing correct or incorrect predictions
# while i is less then length, so every element gets covered
while i < len(compared):
    # finds the values un compared Dataframe at index i
    value = compared.loc[i]

    # transposes value
    value = value.T
    # assigns actual value to com1 and predicted value to com2
    com1 = value["Waveform"]
    com2 = value["Predicted_Waveform"]
    # compares com to com 2 if same then true if different false then adds 1 to i index
    if com1 == com2:
       compared_accuracy.append("True")
       i_list.append(i)
       i += 1

    elif com1 != com2:
       compared_accuracy.append("False")
       i_list.append(i)
       i += 1
    else:
        compared_accuracy.append("ERROR")
        i_list.append(i)
        i += 1

#creates datafrome from list for accuracy and index list
compared_accuracyDF = pd.DataFrame({"Match": compared_accuracy})
i_listDF = pd.DataFrame({"Index": i_list})
# merges compared with index,accuracy, and magnitude, depth
compared = compared.merge(i_listDF, "inner", right_index=True, left_index=True)
compared = compared.merge(compared_accuracyDF, "inner", right_index=True, left_index=True)
compared = compared.merge(X, "inner", right_index=True, left_index=True)




# writes compared to csv
compared.to_csv(path_or_buf="data/PredictionData.csv")

# creates a Key stats DF with Accurate Predicted Total

# counts number of true and false values
Accurate_Predicted_Total = compared["Match"].value_counts()
# calculates actual/experimental %
Accurate_Predicted_Total["Actual Percentage"] = (Accurate_Predicted_Total["True"]/compared["Index"].max())*100
# grabs theoretical % from score variable above
Accurate_Predicted_Total["Theoretical Percentage"] = score*100
# calculates percent error (actual-theoretical)/theoretical
# NOTE: any value of absolute value should bre read as positive even if negative
Accurate_Predicted_Total["Percent Error"] = (((score*100)-(Accurate_Predicted_Total["True"]/compared["Index"].max())*100)/(score*100))*100
Accurate_Predicted_Total.to_csv(path_or_buf="data/Keystats.csv")
print("Scoring DONE")
