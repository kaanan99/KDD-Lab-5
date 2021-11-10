import sys
import os
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text as sktext

def parseVectorizerArgs(args):
    # if there are not 2 arguments then error
    if len(args) != 2:
        print("Error:")
        print("Usage: python3 textVectorizer.py <directory_path> <output_name>")
        sys.exit(1)
    return args[0], args[1]

def createGroundTruth(dataset_path, output_path):
    documents = []
    with open(output_path + '.csv', 'w') as output_file:
        output_file.write('file_name,author\n')
        for dir in os.listdir(dataset_path):
            for author in os.listdir(dataset_path + '/' + dir):
                for file in os.listdir(dataset_path + '/' + dir + '/' + author):
                    output_file.write(file + ',' + author + '\n')
                    documents.append(dataset_path + '/' + dir + '/' + author + '/' + file)
        output_file.close()
    return documents

if __name__ == '__main__':
    dataset_path, output_path = parseVectorizerArgs(sys.argv[1:])
    documents = createGroundTruth(dataset_path, output_path)


    # --- SKLEARN VECTORIZERS (FOR COMPARISON) ---
    sk_count_vectorizer = sktext.CountVectorizer(analyzer='word', input='filename')
    sk_count_wm = sk_count_vectorizer.fit_transform(documents)
    sk_count_tokens = sk_count_vectorizer.get_feature_names_out()
    sk_wm_df = pd.DataFrame(sk_count_wm.toarray(), index=documents, columns=sk_count_tokens)

    sk_tfidf_vectorizer = sktext.TfidfVectorizer(analyzer='word', input='filename')
    sk_tfidf_wm = sk_tfidf_vectorizer.fit_transform(documents)
    sk_tfidf_tokens = sk_tfidf_vectorizer.get_feature_names_out()
    sk_tfidf_df = pd.DataFrame(sk_tfidf_wm.toarray(), index=documents, columns=sk_tfidf_tokens)
    # ------

    print(sk_tfidf_df)
