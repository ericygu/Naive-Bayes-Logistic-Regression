import argparse
import pandas as pd
from sklearn import model_selection as ms


def model_assessment(filename):
    """
    Given the entire data, decide how
    you want to assess your different models
    to compare perceptron, logistic regression,
    and naive bayes, the different parameters, 
    and the different datasets.
    """
    # Default values for the split, sklearn seems solid
    with open(filename) as myfile:
        lines = myfile.readlines()

    df = pd.DataFrame(lines)

    y = df[0].str.extract('(\d+)').astype(int)
    df1 = df[0].str[1:]
    X = pd.DataFrame(df1)

    X_train, X_test, y_train, y_test = ms.train_test_split(X, y)

    return X_train, X_test, y_train, y_test


def build_vocab_map(X_train):
    df1 = X_train[0].str[1:]
    vocab_map = {}
    # Build complete dictionary
    for row in df1:
        words = row.split()
        words = set(words)
        for word in words:
            if word in vocab_map:
                vocab_map[word] += 1
            else:
                vocab_map[word] = 1
    # Filter all words not in at least 30 emails out
    for key in list(vocab_map.keys()):
        if vocab_map[key] < 30:
            del vocab_map[key]
    return vocab_map


def construct_binary(vocab_map, X_train, X_test):
    """
    Construct the email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    X_train_list = []
    X_train_binary = pd.DataFrame()
    X_test_list = []
    X_test_binary = pd.DataFrame()

    # use string append for speed
    for index, row in X_train.iterrows():
        s = ""
        text = row[0]
        words = text.split()
        for word in vocab_map:
            if word in words:
                s += "1 "
            else:
                s += "0 "
        X_train_list.append(s)

    # list to df to return results
    X_train_binary['binary'] = X_train_list
    X_train_binary = X_train_binary.binary.str.split(expand=True)
    X_train_binary.columns = vocab_map

    for index, row in X_test.iterrows():
        s = ""
        text = row[0]
        words = text.split()
        for word in vocab_map:
            if word in words:
                s += "1 "
            else:
                s += "0 "
        X_test_list.append(s)

    X_test_binary['binary'] = X_test_list
    X_test_binary = X_test_binary.binary.str.split(expand=True)
    X_test_binary.columns = vocab_map

    return X_train_binary, X_test_binary


def construct_count(vocab_map, X_train, X_test):
    """
    Construct the email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    X_train_list = []
    X_train_count = pd.DataFrame()
    X_test_list = []
    X_test_count = pd.DataFrame()

    # use string append for speed
    for index, row in X_train.iterrows():
        s = ""
        text = row[0]
        words = text.split()
        for word in vocab_map:
            if word in words:
                counted = words.count(word)
                s += str(counted) + " "
            else:
                s += "0 "
        X_train_list.append(s)

    # list to df to return results
    X_train_count['counter'] = X_train_list
    X_train_count = X_train_count.counter.str.split(expand=True)
    X_train_count.columns = vocab_map

    for index, row in X_test.iterrows():
        s = ""
        text = row[0]
        words = text.split()
        for word in vocab_map:
            if word in words:
                counted = words.count(word)
                s += str(counted) + " "
            else:
                s += "0 "
        X_test_list.append(s)

    # list to df to return results
    X_test_count['counter'] = X_test_list
    X_test_count = X_test_count.counter.str.split(expand=True)
    X_test_count.columns = vocab_map

    return X_train_count, X_test_count


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()

    # Run methods
    X_train, X_test, y_train, y_test = model_assessment(args.data)

    vocab_map = build_vocab_map(X_train)

    binary_xtrain, binary_xtest = construct_binary(vocab_map, X_train, X_test)

    count_xtrain, count_xtest = construct_count(vocab_map, X_train, X_test)

    # Load into usable Data for other questions
    binary_xtrain.to_csv(r'binary_xtrain.csv', index=False)
    binary_xtest.to_csv(r'binary_xtest.csv', index=False)
    count_xtrain.to_csv(r'count_xtrain.csv', index=False)
    count_xtest.to_csv(r'count_xtest.csv', index=False)

    X_train.to_csv(r'xTrain.csv', index=False)
    X_test.to_csv(r'xTest.csv', index=False)

    y_train.to_csv(r'yTrain.csv', index=False)
    y_test.to_csv(r'yTest.csv', index=False)


if __name__ == "__main__":
    main()
