# Ryan Pauly
# CS 425 Project 5
#
# In this project you will apply support vector classification (SVC) to three classification problems. The files
# are in the Project5 folder; the attributes are described in the .name files, also in the folder. You will not
# need to program your own SVC, because you will use off-the-shelf library functions. In this way you will get some
# experience using available machine learning resources.
#
#   PART 3:
#   The third task is another multiclass problem, with 36 features and 7 classes
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

    fileName = "vowel-context.data"


    test_file = "sat.tst"
    train_file = "sat.trn"

    #   Read in the CSV file
    # myData = pd.read_csv(fileName, header=None, delim_whitespace=True)
    training = pd.read_csv(train_file, header=None, delim_whitespace=True)
    testing = pd.read_csv(test_file, header=None, delim_whitespace=True)

    # print(myData[:][34].dtype)

    #   Delete first 3 columns: 0, 1, 2

    #myData = myData.to_numpy()

    testing = testing.to_numpy()
    training = training.to_numpy()

    # print(myData[:, :-1])

    #   STANDARDIZE DATA with sklearn.preprocessing and StandardScalar

    scalar = StandardScaler()
    # print("\n", scalar.fit(myData[:, :-1]))
    #
    # myData[:, :-1] = scalar.transform(myData[:, :-1])
    # # print("standardized ionosphere.data = \n", myData)
    #
    # np.random.shuffle(myData)

    scalar.fit(training[:, :-1])
    training[:, :-1] = scalar.transform(training[:, :-1])

    scalar.fit(testing[:, :-1])
    testing[:, :-1] = scalar.transform(testing[:, :-1])

    #   Multivariate classification

    #   First fit the data:
    X = training[:, :-1]
    Y = training[:, len(training[0])-1]

    # X_v = validation[:, :-1]
    # Y_v = validation[:, 34]

    # print("X = \n", X)
    # print("Y = \n", Y)

    coarse_grid_parameters = {
        "C": [1, 10, 100, 1000],
        "gamma": [0.0001, 0.001, 0.01, 0.1]
    }

    nfolds = 10
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

    #print("fine_grid_paramters = ", fine_grid_parameters)

    myGridSearch = GridSearchCV(svm.SVC(kernel="rbf"), fine_grid_parameters, cv=nfolds)
    myGridSearch.fit(X, Y)

    best_fine_parameters = myGridSearch.best_params_

    print("best_fine_parameters = ", best_fine_parameters)

    scale = best_fine_parameters["gamma"]
    penalty = best_fine_parameters["C"]
    mySVC = svm.SVC(gamma=scale, C=penalty)
    mySVC.fit(X, Y)

    testing_true = testing[:, len(testing[0])-1]
    myPredictions = mySVC.predict(testing[:, :-1])

    correct = 0
    for i in range(len(myPredictions)):
        if myPredictions[i] == testing_true[i]:
            correct += 1

    accuracy = correct / len(myPredictions)
    print("\nAccuracy = ", accuracy)