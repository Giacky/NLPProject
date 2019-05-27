# Sentiment Analysis by Star Rating Prediction of Yelp Reviews
(refering to the dataset provided by Yelp for the "Yelp Dataset Challenge") https://www.yelp.com/dataset/challenge

The Project is about predicting which star rating any unlabeled review represents. In order to achieve that, we are training a Naive Bayes classifier based on BoW model and Support Vector Machine and XGBoost based on a custom Word2vec model. 80% of the data is for training, 20% hold back for testing. 20% of the training set is used as validation to avoid overfitting. 

## Getting Started
Download the <x>.ipynb files from the Repository. 

This project has been done in Python version 3.7.3 using the Jupyter Framework

## Prerequisites

First of all, the review.json file is needed which can be retrieved from https://www.yelp.com/dataset/. 



### Installing
Install Jupyter Notebook on your Computer

The following libraries are required to run the software:
* numpy
* scipy
* pandas
* maplotlib
* seaboarn
* nltk
* sklearn
* xgboost
* gensim
* json
* re
* string


install by typing pip install - <x> into the console or conda install in case the Anaconda environment is used 

```
pip install xgboost
```




## Executing the code
When the review.json file is in the same folder, run 'DatasetPruning.ipynb'  in order to create new .json files containing the desired amount of reviews for the training, validation and testing of the classifiers. The desired amount must be specified in the beginning of both 'main.ipynb' and 'DatasetPruning.ipynb' and must be identical:

```
trainingAmount = 40000
validationAmount = int(trainingAmount * 0.2)
testAmount = 10000

```


After that, go to the main.ipynb file and execute each code cell from top to bottom.

For example run this:

```
import numpy as num
import scipy as sci
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import sklearn as skl
import xgboost
import gensim
import json
import re
import string
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, recall_score

```
And then this:
```
trainingAmount = 40000
validationAmount = int(0.2 * trainingAmount)
testAmount = 10000

# Read JSON files, which are created in 'DatasetPruning' :
df_train = pd.read_json('training' + str(int(trainingAmount-validationAmount))  +'.json', lines=True)
df_validation = pd.read_json('validation' + str(validationAmount) +'.json', lines = True)
df_test = pd.read_json('test' + str(testAmount) +'.json', lines=True)

# Reorder the columns of JSON files:
df_train = df_train.drop("review_id", axis=1).drop("business_id", axis=1).drop("user_id", axis=1).drop("date", axis=1)
df_train = df_train.reindex(['text','stars','useful','funny','cool'], axis=1)

df_validation = df_validation.drop("review_id", axis=1).drop("business_id", axis=1).drop("user_id", axis=1).drop("date", axis=1)
df_validation = df_validation.reindex(['text','stars','useful','funny','cool'], axis=1)

df_test = df_test.drop("review_id", axis=1).drop("business_id", axis=1).drop("user_id", axis=1).drop("date", axis=1)
df_test = df_test.reindex(['text','stars','useful','funny','cool'], axis=1)
```

## Description of implemented classifiers

-Naïve Bayes Classifier is a well known probabilistic classifier. MultinominalNB is imported from the sklearn library as 'classifierNB'
-XGBoost is a regression tree booster. XGBClassifier is imported from the xgboost library as 'xgbclassifier'
-Support Vector Machine is a well known non-probabilistic binary linear classifier making use of multidimensional Kernels.
SVC is imported from the sklearn library as 'svc'

### Running the classifiers

The main part of the code is the running of the classifiers themselves. Below a run of the XGBoost classifier is presented to give an intuition of how it should look when executed correctly. The number in percent refers to the accuracy of the specific classifier over the test set. Multiple runs are performed in XGBoost and SVM to tune hyperparameters and mitigate overfitting. The BoW Naive Bayes is less complicated and should compute in a shorter time than the other two.

![code_execution](https://github.com/Giacky/NLPProject/blob/master/figs/code_execution.png)

#### Troubleshooting
While testing the error message 'The Kernel appears to have died' frequently appeared. It most likely is caused by an overload of memory usage of the RAM. Executing each of the computationally expensive code cells after the previous one has been run solved this problem however. 

## Visualising results 
Interpreting results is best done with having some kind of visualisation of our retrieved results. This is achieved with the libraries matplotlib and seaboarn. Execute the corresponding code segments only after the classifiers. 


The distribution of classes among the dataset:

![figure1](https://github.com/Giacky/NLPProject/blob/master/figs/rating_distribution.png)


When comparing the obtained normalised confusion matrix for the Support Vector Machine and XGBoost, one can clearly see that 
the SVM classifier overfits the training set and its distribution of labels more than XGBoost:

Normalised Confusion Matrix for SVM:


![figure2](https://github.com/Giacky/NLPProject/blob/master/figs/cm_svm_p.png)



Normalised Confusion Matrix for XGBoost:


![figure3](https://github.com/Giacky/NLPProject/blob/master/figs/cm_xgb_p.png)



Another finding was that while the accuracy scores of around 60% (60 % for NB, 62,72 % for XGBoost and 62,93%) do not sound particulary good, when analysing accuracy with a margin of error of one star, all models reach an accuracy of around 90%:

Cumulative Error in Star Prediction of XGBoost:


![figure4](https://github.com/Giacky/NLPProject/blob/master/figs/bar_xgb.png)

(add blue and orange area to obtain the new accuracy with margin of error one)

This implies that the classifiers can detect close sentiments well, but struggle to discretise into classes which represent similar sentiments. (classes 5 and 4 are close in sentiment, while 5 and 1 are at the opposite spectrum, representing 'very bad' vs. 'very good') 



## Authors

* **Hendrik Baacke** - [GitHub](https://github.com/HendrikSimons)
* **Giacomo Anerdi** - [GitHub](https://github.com/Giacky)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details  

## Acknowledgments

* We want to acknowledge nobody, we are alone, the world is evil and only darkness is awaiting us in an infinitely deep abyss  ..and there is no god
