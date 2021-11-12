import sys
import os
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text as sktext

# TODO: further input validation and error handling??
def parseVectorizerArgs(args):
    # if there are not 2 arguments then error
    if len(args) != 2:
        print("Error:")
        print("Usage: python3 textVectorizer.py <directory_path> <output_name>")
        sys.exit(1)
    return args[0], args[1]

# TODO: Examine whether this is how we want to do this
def createGroundTruth(dataset_path, output_name):
    # list of document paths
    documents = []
    output_path = output_name + ".csv"
    # iterate over dataset_path directory and create ground truth .csv file
    with open(output_path, 'w') as output_file:
        output_file.write('file_name,author\n')
        for dir in os.listdir(dataset_path):
            for author in os.listdir(dataset_path + '/' + dir):
                for file in os.listdir(dataset_path + '/' + dir + '/' + author):
                    output_file.write(file + ',' + author + '\n')
                    documents.append(dataset_path + '/' + dir + '/' + author + '/' + file)
        output_file.close()
    return documents, output_path

if __name__ == '__main__':
    # dataset_path: name of directory
    # output_path: name of output file (without .csv extension)
    dataset_path, output_name = parseVectorizerArgs(sys.argv[1:])
    # documents: list of document paths
    # output_path: <output_name>.csv
    documents, output_path = createGroundTruth(dataset_path, output_name)
    ground_truth_df = pd.read_csv(output_path)

    # TODO: REMOVE THESE BEFORE FINAL SUBMISSION
    # --- SKLEARN VECTORIZERS (FOR OUR COMPARISON) ---
    # Sk Word Count Matrix
    sk_count_vectorizer = sktext.CountVectorizer(analyzer='word', input='filename')
    sk_count_wm = sk_count_vectorizer.fit_transform(documents)
    sk_count_tokens = sk_count_vectorizer.get_feature_names_out()
    sk_wm_df = pd.DataFrame(sk_count_wm.toarray(), index=documents, columns=sk_count_tokens)

    # Sk tf-idf Matrix
    sk_tfidf_vectorizer = sktext.TfidfVectorizer(analyzer='word', input='filename')
    sk_tfidf_wm = sk_tfidf_vectorizer.fit_transform(documents)
    sk_tfidf_tokens = sk_tfidf_vectorizer.get_feature_names_out()
    sk_tfidf_df = pd.DataFrame(sk_tfidf_wm.toarray(), index=documents, columns=sk_tfidf_tokens)
    # ------
