import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time


def naive_bayes_bernoulli(xTrain, xTest, yTrain, yTest):
    nbb = BernoulliNB()
    nbb.fit(xTrain, yTrain.ravel())
    yPrediction = nbb.predict(xTest)

    return accuracy_score(yTest, yPrediction)


def naive_bayes_multinomial(xTrain, xTest, yTrain, yTest):
    nbm = MultinomialNB()
    nbm.fit(xTrain, yTrain.ravel())
    yPrediction = nbm.predict(xTest)

    return accuracy_score(yTest, yPrediction)


def logistic_regression(xTrain, xTest, yTrain, yTest):
    lra = LogisticRegression()
    lra.fit(xTrain, yTrain.ravel())
    yPrediction = lra.predict(xTest)

    return accuracy_score(yTest, yPrediction)


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("binary_xtrain",
                        help="filename for features of the training data")
    parser.add_argument("binary_xtest",
                        help="filename for features of the test data")
    parser.add_argument("count_xtrain",
                        help="filename for features of the training data")
    parser.add_argument("count_xtest",
                        help="filename for features of the test data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    binary_xtrain = file_to_numpy(args.binary_xtrain)
    binary_xtest = file_to_numpy(args.binary_xtest)
    count_xtrain = file_to_numpy(args.count_xtrain)
    count_xtest = file_to_numpy(args.count_xtest)
    yTrain = file_to_numpy(args.yTrain)
    yTest = file_to_numpy(args.yTest)

    # Store results
    nbb_result = naive_bayes_bernoulli(binary_xtrain, binary_xtest, yTrain, yTest)
    nbm_result = naive_bayes_multinomial(count_xtrain, count_xtest, yTrain, yTest)
    lrb_result = logistic_regression(binary_xtrain, binary_xtest, yTrain, yTest)
    lrc_result = logistic_regression(count_xtrain, count_xtest, yTrain, yTest)

    # Print Out
    print("Naive Bayes Bernoulli/Binary: " + str(nbb_result))
    print("Naive Bayes Multinomial/Count: " + str(nbm_result))
    print("Logistic Regression/Binary: " + str(lrb_result))
    print("Logistic Regression/Count: " + str(lrc_result))


if __name__ == "__main__":
    main()
