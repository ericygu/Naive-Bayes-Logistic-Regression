import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time


class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None  # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        label : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}
        trainStats = []

        rows, columns = xFeat.shape

        self.w = np.zeros(columns + 1)

        # Iterate through epochs
        for epoch in range(self.mEpoch):
            mistakes = 0
            for row, y_label in zip(xFeat, y):

                yHat = self.predict_simple(row)
                neg_list = y_label - yHat

                self.w[0] += neg_list[0]
                self.w[1:] += row * neg_list

                if y_label == yHat:
                    continue
                else:
                    mistakes += 1

            stats[epoch] = mistakes

            # Stop at 0
            if mistakes == 0:
                break

        return stats

    def predict(self, xFeat):
        yHat = []

        for row in xFeat:
            calc = np.matmul(row, self.w[1:]) + self.w[0]
            if calc < 0:
                yHat.append(0)
            else:
                yHat.append(1)

        return yHat

    def predict_simple(self, xFeat):
        yHat = []

        calc = np.matmul(xFeat, self.w[1:]) + self.w[0]
        if calc < 0:
            yHat.append(0)
        else:
            yHat.append(1)

        return yHat

    def top(self, xTrain, n):
        words = pd.DataFrame(xTrain, columns=['vocab'])
        weight = pd.DataFrame(self.w, columns=['weights'])

        weight = weight.iloc[1:]
        words['weights'] = weight

        top = words.nlargest(n, 'weights')
        return top

    def bot(self, xTrain, n):
        words = pd.DataFrame(xTrain, columns=['vocab'])
        weight = pd.DataFrame(self.w, columns=['weights'])

        weight = weight.iloc[1:]
        words['weights'] = weight

        bot = words.nsmallest(n, 'weights')
        return bot


def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    mistake_count = 0

    for x, y in zip(yHat, yTrue):
        if x == y:
            continue
        else:
            mistake_count = mistake_count + 1

    return mistake_count


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
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)
    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)

    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))

    # Predicted Error
    print("Estimated Predicted Error")
    print(str((1 - accuracy_score(yHat, yTest))*100) + "%")

    # Make sure no pass by value just in case
    xTrain_2 = pd.read_csv(args.xTrain)
    xTrain_2 = list(xTrain_2.head(0))

    # Get top and bottom words
    top_15 = model.top(xTrain_2, 15)
    bot_15 = model.bot(xTrain_2, 15)

    print("Top 15:" + str(top_15))
    print("Bottom 15:" + str(bot_15))


if __name__ == "__main__":
    main()
