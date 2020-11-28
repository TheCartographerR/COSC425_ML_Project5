# Ryan Pauly
# CS 425 Project 5
#
# In this project you will apply support vector classification (SVC) to three classification problems. The files
# are in the Project5 folder; the attributes are described in the .name files, also in the folder. You will not
# need to program your own SVC, because you will use off-the-shelf library functions. In this way you will get some
# experience using available machine learning resources.
#
#   PART 1:
#
#   Your first task is a binary classification problem: predicting “good” vs. “bad” interactions of radar
#   signals with electrons in the ionosphere. The last attribute is the label (“g” or “b”) and the other 34 are
#   measurements. There are 351 instances in ionosphere.data in the Project 5 folder
#
#
#######################################################################################################################

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.svm import SVC
from prettytable import PrettyTable
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

if __name__ == "__main__":

    fileName = "ionosphere.data"

    #   Read in the CSV file
    myData = pd.read_csv(fileName, header=None)

    myData = myData.replace(to_replace="g", value="", regex=True)
    myData = myData.replace("", 0.0)

    myData = myData.replace(to_replace="b", value="", regex=True)
    myData = myData.replace("", 1.0)

    # print("myData = \n", myData)
    # print(myData[:][34].dtype)

    myData = myData.to_numpy()
    # print(myData[:, :-1])

    #   STANDARDIZE DATA with sklearn.preprocessing and StandardScalar

    scalar = StandardScaler()
    print("\n", scalar.fit(myData[:, :-1]))

    myData[:, :-1] = scalar.transform(myData[:, :-1])
    # print("standardized ionosphere.data = \n", myData)

    np.random.shuffle(myData)

    #   A good split I found was to separate training to 70%, validation to 10%, and testing for 20%
    #   351 rows of data
    #   training = 70% of 351 == 246
    #   validation = 10% of 351 == 35
    #   testing = 20% of 351 == 70

    # training = myData[:246, :]
    # validation = myData[246:281, :]

    training = myData[:281, :]
    testing = myData[281:, :]

    #   BINARY CLASSIFICATION:

    #   First fit the data:
    X = training[:, :-1]
    Y = training[:, 34]

    # X_v = validation[:, :-1]
    # Y_v = validation[:, 34]

    # print("X = \n", X)
    # print("Y = \n", Y)

    coarse_grid_parameters = {
        "C": [1, 10, 100, 1000],
        "gamma": [0.0001, 0.001, 0.01, 0.1]
    }

    nfolds = 5
    myGridSearch = GridSearchCV(svm.SVC(kernel="rbf"), coarse_grid_parameters, cv=nfolds)
    myGridSearch.fit(X, Y)

    best_coarse_parameters = myGridSearch.best_params_

    print("best_coarse_parameters = ", best_coarse_parameters)

    fine_C_list = []
    if best_coarse_parameters["C"] <= 10:
        fine_C_list.extend(range(1, 50))
    elif 10 < best_coarse_parameters["C"] <= 100:
        fine_C_list.extend(range(10, 200))
    elif 100 < best_coarse_parameters["C"] <= 1000:
        fine_C_list.extend(range(100, 2000))

    fine_gamma = []
    if best_coarse_parameters["gamma"] <= 0.001:
        fine_gamma.extend(np.arange(0.0001, 0.002, 0.00005))
    elif 0.0001 < best_coarse_parameters["gamma"] <= 0.01:
        fine_gamma.extend(np.arange(0.001, 0.02, 0.001))
    elif 0.01 < best_coarse_parameters["gamma"] <= 0.1:
        fine_gamma.extend(np.arange(0.01, 0.2, 0.01))

    fine_grid_parameters = {
        "C": fine_C_list,
        "gamma": fine_gamma
    }

    # print("fine_grid_paramters = ", fine_grid_parameters)

    fineGridSearch = GridSearchCV(svm.SVC(kernel="rbf"), fine_grid_parameters, cv=nfolds)
    fineGridSearch.fit(X, Y)

    best_fine_parameters = fineGridSearch.best_params_

    print("best_fine_parameters = ", best_fine_parameters)

    scale = best_fine_parameters["gamma"]
    penalty = best_fine_parameters["C"]
    mySVC = svm.SVC(gamma=scale, C=penalty)
    mySVC.fit(X, Y)

    testing_true = testing[:, 34]
    myPredictions = mySVC.predict(testing[:, :-1])

    correct = 0
    for i in range(len(myPredictions)):
        if myPredictions[i] == testing_true[i]:
            correct += 1

    accuracy = correct / len(myPredictions)
    print("\nAccuracy = ", accuracy)
