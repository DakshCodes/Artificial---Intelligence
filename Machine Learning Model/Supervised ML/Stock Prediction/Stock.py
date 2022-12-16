#DATA COLLECTION STRIGHTGY
import pandas_datareader as pdr

#Give Api Key
df=pdr.get_data_tiingo("AAPL",api_key="94ebf734751c9e1cea666099855fb117139f41c5")
#Data____________
df.to_csv("AAPL.csv")

#Read Data From Pandas_____
import pandas as pd
df=pd.read_csv("AAPL.csv")

# print(df.head())

#TAKE CLOSE COLUMNS---------------------------
df1=df.reset_index()["close"]
print(df1)

#FOR PLOTING DATA IN GRAPH-----________------
import matplotlib.pyplot as plt
df1.shape
# plt.plot(df1)

## LSTM  are sensitive to the scale of the data . so we apply MinMax scaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
# print(df1)

##NOW SPLITTING DATA SET INTO TRAIN AND TEST SPLITING----
Training_size=int(len(df1)*0.65)
Testing_size=len(df1)-Training_size

Train_data,Test_data = df1[0:Training_size],df1[Training_size:len(df1),:1]

