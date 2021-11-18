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
    vector_file_names = []
    for vector in doc_vectors:
        # print('Getting vector info')
        vector_weights.append(vector.tfidf_weights)
        vector_authors.append(vector.author)
        vector_file_names.append(vector.name.split('/')[-1])
    print("Creating dataframe...")
    dataframe = pd.DataFrame(np.asarray(vector_weights), columns=word_list)
    dataframe['AUTHOR'] = np.asarray(vector_authors)
    print("Dataframe created.")
    return dataframe, 'AUTHOR'

if __name__ == '__main__':
    # NOTE: Important, maintain row indices of dataframe in order to get correct prediction-actual pairs

    TREE_THRES = 0.8

    vector_file_path, word_counts_path, num_trees, num_attr, num_data_points = parseRFAuthorshipArgs(sys.argv[1:])

    doc_vectors, word_list, word_counts = knnAuthorship.readVectorFile(vector_file_path, word_counts_path)
    dataframe, class_col = createDF(doc_vectors, word_list)
    attr_dict = {word:1 for word in word_list}
    # tree = InduceC45New.c45(attr_dict, dataframe, TREE_THRES)
    vals_per_attr = pd.Series({word:0 for word in word_list})
    attr_info = {A: dataframe[A].unique() for A in word_list}
    # trees = randomForest.build_random_forest(TREE_THRES, num_trees, num_attr, num_data_points, dataframe, attr_info, class_col, vals_per_attr)

    # # Get Predictions
    # predictions = randomForest.forest_classify(dataframe, trees, vals_per_attr)
    # matrix = knnAuthorship.makeMatrix(predictions, dataframe["AUTHOR"])
    
    #print(classify_author(dataframe, 'AlanCrosby'))
    # Get the list of Authors
    author_list = {}
    for vector in doc_vectors:
        if vector.author not in author_list:
            author_list[vector.author] = 1
    # Save the actual authors
    actual_authors = dataframe["AUTHOR"]
    # Begin Loop
    for author_name in author_list.keys():
        start = time.time()
        # Change to Boolean Value depending on Author
        dataframe["AUTHOR"] = dataframe["AUTHOR"] == author_name
        # Create Trees
        trees = randomForest.build_random_forest(TREE_THRES, num_trees, num_attr, num_data_points, dataframe, attr_info, class_col, vals_per_attr)
        # print("Built Trees")
        # Get Predictions
        predictions = randomForest.forest_classify(dataframe, trees, vals_per_attr)
        # print("Finished Predicting")
        print("\nPredictions for " + author_name)
        matrix = knnAuthorship.makeMatrix(predictions, dataframe["AUTHOR"])
        print(matrix[1])
        # Reset DataFrame
        dataframe["AUTHOR"] = actual_authors
        #print("Time to complete one classification:", time.time() - start)

    '''print(predictions)

    print(dataframe.head(10))
    print(vals_per_attr)
    print(vals_per_attr.at['senat'])'''
