import sys
import numpy as np
import pandas as pd
import knnAuthorship
import time
import randomForest

def parseRFAuthorshipArgs(args):
    if len(args) != 5:
        print("Usage: RFAuthorship.py <vector_file> <sim_metric> <num_trees> <num_attr> <num_data_points>")
        sys.exit(1)
    return args[0], args[1], int(args[2]), int(args[3]), int(args[4])

def createDF(doc_vectors, word_list):
    vector_weights = []
    vector_authors = []
    vector_file_ids = []
    for vector in doc_vectors:
        # print('Getting vector info')
        vector_weights.append(vector.tfidf_weights)
        vector_authors.append(vector.author)
        vector_file_ids.append(vector.name.split('/')[-1])
    print("Creating dataframe...")
    dataframe = pd.DataFrame(np.asarray(vector_weights), columns=word_list)
    dataframe['AUTHOR'] = np.asarray(vector_authors)
    print("Dataframe created.")
    return dataframe, 'AUTHOR', vector_file_ids

def outputResults(file_path, predictions, vector_file_ids):
    # write to filepath
    file_path = 'RFOutput/' + file_path
    with open(file_path, 'w') as f:
        f.write("file_name,author\n")
        for idx, file_id in enumerate(vector_file_ids):
            f.write(str(file_id) + ',' + str(predictions[idx]) + '\n')
        f.close()

if __name__ == '__main__':
    # NOTE: Important, maintain row indices of dataframe in order to get correct prediction-actual pairs
    TREE_THRES = 0

    vector_file_path, word_counts_path, num_trees, num_attr, num_data_points = parseRFAuthorshipArgs(sys.argv[1:])

    doc_vectors, word_list, word_counts = knnAuthorship.readVectorFile(vector_file_path, word_counts_path)
    dataframe, class_col, vector_file_ids = createDF(doc_vectors, word_list)

    start = time.time()
    trees = randomForest.createRandomForest(dataframe, word_list, class_col, num_trees, num_attr, num_data_points, TREE_THRES)
    print("Trees Created")
    actual, predictions = randomForest.classifyRandomForest(trees, dataframe, class_col)
    print("Classified")
    end = time.time()
    print("Time taken to build trees and classify: " + str(end - start))

    outputResults('classified_' + str(num_trees) + '_' + str(num_attr) + '_' + str(num_data_points) + '.csv', predictions, vector_file_ids)
