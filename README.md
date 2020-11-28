# COSC425_ML_Project5
Project 5: Support Vector Classification. In this project we applied support vector classification (SVC) to three classification problems. In this project we were given permission to use scikit-learn, a Python library which provides pre-built machine learning functions.


The first task is a binary classification problem: predicting “good” vs. “bad” interactions of
radar signals with electrons in the ionosphere. The last attribute is the label (“g” or “b”) and the
other 34 are measurements. There are 351 instances in ionosphere.data in the Project
5 folder. You will have to split them into training, validation, and testing subsets. For more
information, see https://archive.ics.uci.edu/ml/datasets/ionosphere. Perform a coarse grid search to find
the best range of hyperparameters (specifically, penalty and kernel scale), and then a fine grid search 
to find the optimal hyperparameters for this problem. Report your results and classification performance for 
these hyperparameters.

The second task is a multiclass problem, with 11 classes (vowel sounds) and 10 features
(https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Vowel+Recognition+-
+Deterding+Data%29). In the file vowel-context.data, the first three attributes are
irrelevant and the last is the vowel label. As in step (3), split the file and perform grid searches
to optimize your hyperparameters.

The third task is also multiclass, with 7 classes and 36 features: determine the type of terrain
from multispectral values of pixels in 3×3 neighborhoods in satellite images [see
https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite) ]. There are 4435 training
instances in sat.trn and 2000 testing (validation) instances in sat.tst. In the files the last
attribute is the label (1–7, but this sample contains no instances of class 6). As in step (3),
perform grid searches to optimize your hyperparameters.
