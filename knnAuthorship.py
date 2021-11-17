import sys
import json
from textVectorizer import Vector
import sklearn.metrics.pairwise as skpair
import time
import pandas as pd
import numpy as np

class Item:
    def __init__(self, distance, class_val, path):
        self.distance = distance
        self.class_val = class_val
        self.path = path

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

def toVector(dict_rep, doc_appear, weight_matrix, sim_metric):
    start = time.time()
    v =  Vector(dict_rep["word_freq"],dict_rep["name"], dict_rep["num_words"])
    v.author = dict_rep["author"]
    v.average_words = dict_rep["average_words"]
    v.word_freq = dict_rep["word_freq"]
    v.tfidf_weights = dict_rep["weights"]

    if sim_metric != "okapi":
        new_weights = []
        for word in doc_appear.keys():
            if word in v.tfidf_weights:
                new_weights.append(v.tfidf_weights[word])
            else:
                new_weights.append(0)
        v.tfidf_weights = new_weights
        weight_matrix.append(new_weights)

    end = time.time()
    print("Time to convert to vector: " + str(end - start))
    return v

def readVectorFile(vector_file_path, sim_metric):
    f = open(vector_file_path, "r")
    n = int(f.readline())
    doc_appear = json.loads(f.readline())
    weight_matrix = []
    doc_freq = [toVector(x, doc_appear, weight_matrix, sim_metric) for x in json.loads(f.readline())]
    f.close()
    
    if sim_metric == 'okapi':
        return n, doc_appear, doc_freq, None
    else:
        return n, doc_appear, doc_freq, np.asarray(weight_matrix)

def makeMatrix(predictions, actual):
    class_label_options = actual.unique()   # all possible class labels
    actual_vals = actual.array

    df = pd.DataFrame([[0 * len(class_label_options)] * len(class_label_options)], class_label_options, class_label_options)
    for x in range(len(predictions)):
        df.loc[actual_vals[x], predictions[x]] += 1 
    df2 = pd.DataFrame(df.values,pd.MultiIndex.from_product([['Actual'], df.index]), pd.MultiIndex.from_product([['Predicted'], df.columns]))
    return df, df2

def authorAccuracy(df):
    for name in df.columns:
        correct = df.loc[name, name]
        print(name + ": " + str(correct/10))

def calcOkapiSims(doc_freq, doc_appear, n):
    distances = []
    for i in range(n):
        start = time.time()
        doc_dist = [doc_freq[i].okapi(doc_freq[j], n, doc_appear) for j in range(n)]
        distances.append(doc_dist)
        end = time.time()
        print("Time to calculate okapi sims: " + str(end - start) + ' - ' + str(i) + '/' + str(n))
    return distances

def predict(doc_freq, distance_arr_v, i, k):
    distance_arr_v[i] = np.NINF
    largest_k_indices = np.argpartition(distance_arr_v, -k)[-k:]
    most_sim_authors = [doc_freq[x].author for x in largest_k_indices]
    # get plurality of authors
    return max(set(most_sim_authors), key=most_sim_authors.count)

def knn(n, doc_freq, doc_appear, k, sim_metric, weight_matrix):
    predictions = []
    actual = []
    corr_predict = 0
    incorr_predict = 0
    tot_predict = 0

    if sim_metric == "cos":
        num = weight_matrix / np.linalg.norm(weight_matrix, 2, axis=1).reshape(-1, 1)
        distances = np.dot(num, num.T)
    elif sim_metric == "okapi":
        distances = calcOkapiSims(doc_freq, doc_appear, n)

    for i in range(len(distances)):
        start = time.time()
        prediction = predict(doc_freq, distances[i], i, k)
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
    n, doc_appear, doc_freq, weight_matrix = readVectorFile(vector_file_path, sim_metric)

    predictions, actual = knn(n, doc_freq, doc_appear, k, sim_metric, weight_matrix)
    
    df, matrix = makeMatrix(predictions, pd.Series(actual))
    print(matrix)
    authorAccuracy(df)