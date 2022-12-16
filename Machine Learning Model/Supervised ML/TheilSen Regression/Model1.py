#-----IMPORTING PACKAGES----------------------- 

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import TheilSenRegressor

#-------IMPORTING DATA--------------------------

#---Train Data Import-----------------
Train_data = pd.read_csv("train.csv")
Train = Train_data.dropna()#Training Data

#VARIABLES FOR TRAINING-----------------
X_Train= np.array(Train.iloc[:,:-1].values)
Y_Train= np.array(Train.iloc[:,1].values)

#----Test Data -----------------------
Test_data= pd.read_csv("test.csv")
Test = Test_data.dropna()#Test Data

#VARIABLES FOR TESTING-----------------
X_Test= np.array(Test.iloc[:,:-1].values)
Y_Test= np.array(Test.iloc[:,1].values)



#---------CREATE MODEL----------------------------------------
Model = TheilSenRegressor()
Model.fit(X_Train,Y_Train)
Prediction= Model.predict(X_Test)

# print(Prediction)
#----------ACCURACY----------------------------------
Accuracy = Model.score(X_Test,Y_Test)
print(Accuracy)

#MODEL DONE------------------------------

