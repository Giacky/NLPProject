# Sentiment Analysis by Star Rating Prediction of Yelp Reviews
(refering to the dataset provided by Yelp for the "Yelp Dataset Challenge") https://www.yelp.com/dataset/challenge

The Project is about predicting which star rating any unlabeled review represents. In order to achieve that, we are training a Naive Bayes classifier based on BoW model and Support Vector Machine and XGBoost based on a custom Word2vec model. 80% of the data is for training, 20% hold back for testing. 20% of the training set is used as validation to avoid overfitting. The dimension of these datasets can be customized in the extra file 'DatasetPruning'. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing
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

Additionally the review.json file is needed which can be retrieved from https://www.yelp.com/dataset/.


And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

DatasetPruning.ipynb is used to create new .json files containing the desired amount of reviews for the training and testing of the classifiers. The Main.ipynb file is instead used to run and test the classifiers.

### Break down into end to end tests

Explain what these tests test and why

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
