import pandas as pd 

# Step 1 :---------- Dataset or Prepare-----
filename="home.dataset.csv"
filepath = pd.read_csv(filename)
filepath.columns

data=filepath.dropna(axis=0)

#Step 2 :------------Labels and Freatures---

Y=data.price #Labels->Dependent
Freatures=["bathroom","rooms","lattitude","longitude"]#Freatures-> Independents

X=data[Freatures]

#Step 3----------------Creating Model----
from sklearn.tree import DecisionTreeClassifier

Model=DecisionTreeClassifier(random_state=1)
Model.fit(X,Y)


#Step 4----------------Testing Model------
print(X.head())


#Step 5----------------Predicting------
print(Model.predict(X.head()))
