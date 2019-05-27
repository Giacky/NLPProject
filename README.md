# Sentiment Analysis by Star Rating Prediction of Yelp Reviews
(refering to the dataset provided by Yelp for the "Yelp Dataset Challenge") https://www.yelp.com/dataset/challenge

The Project is about predicting which star rating any unlabeled review represents. In order to achieve that, we are training a Naive Bayes classifier based on BoW model and Support Vector Machine and XGBoost based on a custom Word2vec model. 80% of the data is for training, 20% hold back for testing. 20% of the training set is used as validation to avoid overfitting. 

## Getting Started
Download the <x>.ipynb files from the Repository. 

### Prerequisites

First of all, the review.json file is needed which can be retrieved from https://www.yelp.com/dataset/. 

```
Give examples
```

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





## Executing the code
When the review.json file is in the same folder, run 'DatasetPruning.ipynb'  in order to create new .json files containing the desired amount of reviews for the training, validation and testing of the classifiers. The desired amount can be specified the file. 

After that, go to the main.ipynb file and execute each code cell from top to bottom.

Each code 

```
Give an example
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
