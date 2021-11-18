import sys
import numpy as np
import pandas as pd
import knnAuthorship

def parseRFAuthorshipArgs(args):
    if len(args) != 5:
        print("Usage: RFAuthorship.py <vector_file> <sim_metric> <num_trees> <num_attr> <num_data_points>")
        sys.exit(1)
    return args[0], args[1], int(args[2]), int(args[3]), int(args[4])

def createDF(doc_vectors, word_list):
    vector_weights = []
    vector_authors = []
    vector_file_names = []
    for vector in doc_vectors:
        print('Getting vector info')
        vector_weights.append(vector.tfidf_weights)
        vector_authors.append(vector.author)
        vector_file_names.append(vector.name.split('/')[-1])
    print("Creating dataframe...")
    dataframe = pd.DataFrame(np.asarray(vector_weights), columns=word_list)
    dataframe['AUTHOR'] = np.asarray(vector_authors)
    dataframe['FILE_NAME'] = np.asarray(vector_file_names)
    print("Dataframe created.")
    return dataframe

if __name__ == '__main__':
    TREE_THRES = 0.1
    vector_file_path, word_counts_path, num_trees, num_attr, num_data_points = parseRFAuthorshipArgs(sys.argv[1:])
    doc_vectors, word_list, word_counts = knnAuthorship.readVectorFile(vector_file_path, word_counts_path)
    dataframe = createDF(doc_vectors, word_list)
    print(dataframe.head(10))
