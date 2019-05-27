# Sentiment Analysis by Star Rating Prediction of Yelp Reviews
(refering to the dataset provided by Yelp for the "Yelp Dataset Challenge") https://www.yelp.com/dataset/challenge

The Project is about predicting which star rating any unlabeled review represents. In order to achieve that, we are training a Naive Bayes classifier based on BoW model and Support Vector Machine and XGBoost based on a custom Word2vec model. 80% of the data is for training, 20% hold back for testing. 20% of the training set is used as validation to avoid overfitting. 

## Getting Started
Download the <x>.ipynb files from the Repository. 

This project has been done in Python version 3.7.3 using the Jupyter Framework
### Prerequisites

First of all, the review.json file is needed which can be retrieved from https://www.yelp.com/dataset/. 



### Installing
Install Jupyter Notebook on your Computer

The following libraries are required to run the software:
* numpy
* scipy
* pandas
* maplotlib
* nltk
* sklearn
* xgboost
* gensim
* json
* re
* string
* seaboarn

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


### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
