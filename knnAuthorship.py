import sys
import json
from textVectorizer import Vector
import time
import pandas as pd
import numpy as np

class Item:
    def __init__(self, distance, class_val):
        self.distance = distance
        self.class_val = class_val

    def __lt__(self, other):
        return self.distance > other.distance

def parseKNNAuthorshipArgs(args):
    if len(args) != 3:
        print("Usage: knnAuthorship.py <vector_file> <sim_metric> <k>")
        sys.exit(1)
    if args[1] != 'cos' and args[1] != 'okapi':
        print("sim_metric must be either 'cos' or 'okapi'")
        sys.exit(1)
    try:
        k = int(args[2])
    except ValueError:
        print("k must be an integer")
        sys.exit(1)
    return args[0], args[1], k

def toVector(dict_rep):
    v =  Vector(dict_rep["word_freq"],dict_rep["name"], dict_rep["num_words"])
    v.author = dict_rep["author"]
    v.average_words = dict_rep["average_words"]
    v.word_freq = dict_rep["word_freq"]
    v.weights = dict_rep["weights"]
    return v

def turnToVector(vector_array):
    return [toVector(x) for x in vector_array]

def readVectorFile(vector_file_path):
    f = open(vector_file_path, "r")
    n = int(f.readline())
    doc_appear = json.loads(f.readline())
    doc_freq = turnToVector(json.loads(f.readline()))
    f.close()
    return n, doc_appear, doc_freq

def getClass(distances):
    d = {}
    for x in distances:
        if x.class_val in d:
            d[x.class_val] += 1
        else:
            d[x.class_val] = 1
    return max(d, key=lambda k: d[k])

def predict(vector, doc_freq, n, doc_appear, sim_metric, k):
    distances = []
    for v in doc_freq:
        distances.append(Item(vector.calcDist(v, sim_metric, n, doc_appear), v.author))
    distances.sort()
    return getClass(distances[: k])

def makeMatrix(predictions, actual):
    """
    Create a confusion matrix of prediction vs. truth values.
    Params:
        predictions - a list of predicted class labels, ordered in accordance with data set
        actual - a Series of the ground truth class labels, ordered in accordance with data set
    Return: a confusion matrix, represented as a multi-indexed DataFrame
    """
    # get all possible class labels from truth label set
    class_label_options = actual.unique()   # all possible class labels
    # get actual class labels as a list
    actual_vals = actual.array

    # print(f"Predictions: {predictions}\nActual: {actual_vals}")
    # create an all-zero DataFrame with rows and columns labeled with class labels
    df = pd.DataFrame([[0 * len(class_label_options)] * len(class_label_options)], class_label_options, class_label_options)
    # go through each prediction-truth pair and increment the appropriate cell count
    for x in range(len(predictions)):
        df.loc[actual_vals[x], predictions[x]] += 1 
    df2 = pd.DataFrame(df.values,pd.MultiIndex.from_product([['Actual'], df.index]), pd.MultiIndex.from_product([['Predicted'], df.columns]))
    return df, df2

def authorAccuracy(df):
    for name in df.columns:
        correct = df.loc[name, name]
        print(name + ": " + str(correct/10))

def knn(n, doc_freq, doc_appear, k, sim_metric):
    predictions = []
    actual = []
    corr_predict = 0
    incorr_predict = 0
    tot_predict = 0
    for i in range(len(doc_freq)):
        start = time.time()
        prediction = predict(doc_freq[i], doc_freq[:i] + doc_freq[i + 1:], n, doc_appear, sim_metric, k)
        predictions.append(prediction)
        actual.append(doc_freq[i].author)
        end = time.time()
        print("Time to predict: " + str(end - start) + ' - ' + str(i))
        if prediction == doc_freq[i].author:
            print("Correct")
            corr_predict += 1
        else:
            print("Incorrect")
            incorr_predict += 1
        tot_predict += 1
    print('Accuracy: ' + str(corr_predict/tot_predict) + '%')
    print('Inaccuracy ' + str(incorr_predict/tot_predict) + '%')
    return predictions, actual

if __name__ == '__main__':
    vector_file_path, sim_metric, k = parseKNNAuthorshipArgs(sys.argv[1:])
    n, doc_appear, doc_freq = readVectorFile(vector_file_path)

    predictions, actual = knn(n, doc_freq, doc_appear, k, sim_metric)
    df, matrix = makeMatrix(predictions, pd.Series(actual))
    print(matrix)
    authorAccuracy(df)
    print("Breakpoint")