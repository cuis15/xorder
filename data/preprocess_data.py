import pandas as pd
import pickle

dataset = "german"

if dataset == "compas":
    numr_data = pd.read_csv("original/propublica-recidivism_numerical-binsensitive.csv", header=0, engine='python')
    numr_data.dropna(how="all", inplace=True)
    numr_data = numr_data.drop("sex-race",1)
elif dataset == "framingham":
    original_data = pd.read_csv("original/framingham.csv", header=0, engine='python')
    original_data.dropna(inplace=True)
    original_data = original_data.drop('id', 1)
    numr_data = pd.get_dummies(original_data)
elif dataset == "german":
    numr_data = pd.read_csv("original/german_numerical-binsensitive.csv", header=0, engine='python')
    numr_data.dropna(how="all", inplace=True)
    numr_data = numr_data.drop("sex-age",1)
elif dataset == "adult":
    numr_data = pd.read_csv("original/adult_numerical-binsensitive.csv", header=0, engine='python')
    numr_data.dropna(how="all", inplace=True)

if dataset == "compas":
    Y = numr_data["two_year_recid"]
    numr_data = numr_data.drop('two_year_recid',1)
    A = numr_data["race"] == 0
    X = numr_data.drop("race",1)
elif dataset == "framingham":
    Y = numr_data["chdfate"]
    numr_data = numr_data.drop('chdfate', 1)
    A = numr_data["sex"] == 1
    X = numr_data
elif dataset == "german":
    Y = numr_data["credit"] == 2
    numr_data = numr_data.drop('credit',1)
    A = numr_data["age"] == 0
    X = numr_data.drop("age",1)
elif dataset == "adult":
    Y = numr_data["income-per-year"]
    numr_data = numr_data.drop('income-per-year',1)
    A = numr_data["race"] == 0
    X = numr_data.drop("race",1)

X = X.values.astype(np.float32)
Y = np.array(Y)
A = np.array(A)

data = {"X": X, "y": Y, "a": A}
fout = open("preprocessed/" + dataset + "_data" + '.pkl', 'wb')
pickle.dump(data, fout)