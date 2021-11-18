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
    if len(args) != 4:
        print("Usage: knnAuthorship.py <vector_file> <word_count_file> <sim_metric> <k>")
        sys.exit(1)
    if args[2] != 'cos' and args[2] != 'okapi':
        print("sim_metric must be either 'cos' or 'okapi'")
        sys.exit(1)
    try:
        k = int(args[3])
    except ValueError:
        print("k must be an integer")
        sys.exit(1)
    return args[0], args[1], args[2], k

def toVector(vec_dict, word_list):
    vector = Vector(None, vec_dict['file_name'])
    vector.num_words = vec_dict['num_words']
    vector.average_words = vec_dict['average_words']
    vector = convertToSparseVector(vector, vec_dict, word_list)
    return vector

def convertToSparseVector(vector, vec_dict, word_list):
    start = time.time()
    tfidf_weights = []
    okapi_left = []
    okapi_right = []
    for word in word_list:
        if word in vec_dict['tfidf']:
            tfidf_weights.append(vec_dict['tfidf'][word])
            okapi_left.append(vec_dict['okapi_left'][word])
            okapi_right.append(vec_dict['okapi_right'][word])
        else:
            tfidf_weights.append(0.0)
            okapi_left.append(0.0)
            okapi_right.append(0.0)
    vector.tfidf_weights = np.asarray(tfidf_weights)
    vector.okapi_left = np.asarray(okapi_left)
    vector.okapi_right = np.asarray(okapi_right)
    end = time.time()
    # print("Time to convert to sparse vector: " + str(end - start))
    return vector

def readVectorFile(vector_file_path, word_counts_path):
    vector_file = open(vector_file_path, 'r')
    word_counts_file = open(word_counts_path, 'r')
    word_list = json.loads(word_counts_file.readline())
    word_counts = json.loads(word_counts_file.readline())
    doc_vectors_dicts = json.loads(vector_file.readline())
    doc_vectors = [toVector(vec_dict, word_list) for vec_dict in doc_vectors_dicts]
    return doc_vectors, word_list, word_counts

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

def predict(doc_freq, distance_arr_v, i, k):
    distance_arr_v[i] = np.NINF
    largest_k_indices = np.argpartition(distance_arr_v, -k)[-k:]
    most_sim_authors = [doc_freq[x].author for x in largest_k_indices]
    # get plurality of authors
    return max(set(most_sim_authors), key=most_sim_authors.count)

def calcCosSims(doc_vectors):
    weight_matrix = np.asarray([vector.tfidf_weights for vector in doc_vectors])
    num = weight_matrix / np.linalg.norm(weight_matrix, 2, axis=1).reshape(-1, 1)
    return np.dot(num, num.T)

def calcOkapiSims(doc_vectors):
    okapi_left_matrix = np.asarray([vector.okapi_left for vector in doc_vectors])
    okapi_right_matrix = np.asarray([vector.okapi_right for vector in doc_vectors])
    return np.dot(okapi_left_matrix, okapi_right_matrix.T)

def knn(doc_vectors, k, sim_metric):
    predictions = []
    actual = []
    corr_predict = 0
    incorr_predict = 0
    tot_predict = 0

    if sim_metric == "cos":
        similarities = calcCosSims(doc_vectors)
    elif sim_metric == "okapi":
        similarities = calcOkapiSims(doc_vectors)

    for i in range(len(similarities)):
        start = time.time()
        prediction = predict(doc_vectors, similarities[i], i, k)
        predictions.append(prediction)
        actual.append(doc_vectors[i].author)
        end = time.time()
        print("Time to predict: " + str(end - start) + ' - ' + str(i))
        if prediction == doc_vectors[i].author:
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
    vector_file_path, word_counts_path, sim_metric, k = parseKNNAuthorshipArgs(sys.argv[1:])
    doc_vectors, word_list, word_counts = readVectorFile(vector_file_path, word_counts_path)
    predictions, actual = knn(doc_vectors, k, sim_metric)

    print('breakpoint')
    # predictions, actual = knn(n, doc_freq, doc_appear, k, sim_metric, weight_matrix)
    
    # df, matrix = makeMatrix(predictions, pd.Series(actual))
    # print(matrix)
    # authorAccuracy(df)