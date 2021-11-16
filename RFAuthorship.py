import sys
import numpy as np
import pandas as pd
import knnAuthorship

def parseRFAuthorshipArgs(args):
    if len(args) != 5:
        print("Usage: RFAuthorship.py <vector_file> <sim_metric> <num_trees> <num_attr> <num_data_points> <thres>")
        sys.exit(1)
    return args[0], int(args[1]), int(args[2]), int(args[3]), int(args[4])

def toVector(dict_rep, doc_appear):
    v =  Vector(dict_rep["word_freq"],dict_rep["name"], dict_rep["num_words"])
    v.author = dict_rep["author"]
    v.average_words = dict_rep["average_words"]
    v.word_freq = dict_rep["word_freq"]
    v.weights = dict_rep["weights"]
    return v

def readVectorFile(vector_file_path):
    f = open(vector_file_path, "r")
    n = int(f.readline())
    doc_appear = json.loads(f.readline())
    doc_freq = [toVector(x, doc_appear) for x in json.loads(f.readline())]
    f.close()
    return n, doc_appear, doc_freq


def create_data_frame(doc_freq):
    df = pd.DataFrame(doc_freq[0].word_freq)
    class_col = [doc_freq[0].author]
    for vector in doc_freq[1:]:
        df = pd.concat[df, pd.DataFrame(doc_freq[0].word_freq)]
        class_col.append(vector.author)
    df = df.replace('NaN', 0)
    return df, class_col


if __name__ == '__main__':
    vector_file, num_trees, num_attr, num_data_points, thres = parseRFAuthorshipArgs(sys.argv[1:])
    n, doc_appear, doc_freq = knnAuthorship.readVectorFile(vector_file_path)
    df = create_data_frame(doc_freq)
