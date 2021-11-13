import sys
import os
import numpy as np
import pandas as pd
import string
import nltk
import time
import math
import sklearn.feature_extraction.text as sktext

class Vector:

    def __init__(self, word_freq, name, num_words):
        self.word_freq = word_freq
        self.name = name
        self.num_words = num_words
        self.average_words = None
        self.weights = {}

    def tf_idf(self, n, num_docs):
        max_freq = max(self.word_freq.values())
        for word, freq in self.word_freq.items():
            tf = freq/max_freq
            idf = math.log2(n/num_docs[word])
            self.weights[word] = tf * idf

    # TODO: double check this
    # TODO: MAYBE CONVERT TO LIST COMPREHENSION/NP ARRAYS
    def cosSim(self, other):
        similarity = 0
        numerator = 0
        denom1 = 0
        denom2 = 0
        for word, weight in self.weights.items():
            if word in other.weights:
                numerator += weight * other.weights[word]
                denom1 += weight ** 2
                denom2 += other.weights[word] ** 2
        denominator = math.sqrt(denom1 * denom2)
        similarity = numerator / denominator
        return similarity
    
    # TODO: MAYBE CONVERT TO LIST COMPREHENSION/NP ARRAYS
    # TODO: recheck this
    def okapi(self, other, n, num_docs):
        k1 = 1.5
        b = .75
        k2 = 500
        ok_sim = 0
        for word, freq in self.word_freq.items():
            if word in other.word_freq:
                alt_idf_d = math.log((n - num_docs[word] + .5) / (num_docs[word] + .5))
                alt_tf = ((k1 + 1) * freq) / (k1 * (1 - b + b * (self.num_words / self.average_words)) + freq)
                alt_idf_q = ((k2 + 1) * other.word_freq[word]) / (k2 + other.word_freq[word]) 
                ok_sim += (alt_idf_d * alt_tf * alt_idf_q)
        return ok_sim

# TODO: further input validation and error handling??
def parseVectorizerArgs(args):
    # if there are not 2 arguments then error
    if len(args) != 3:
        print("Error:")
        print("Usage: python3 textVectorizer.py <directory_path> <output_name>")
        sys.exit(1)
    return args[0], args[1], args[2]

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

def processStopWords(stop_word_path):
    f = open(stop_word_path, "r")
    words = f.read().split()
    f.close()
    # TODO: replace punctuation with spaces instead of empty string?
    return {w.lower().translate(str.maketrans('', '', string.punctuation)):1 for w in words}

def getFrequencies(documents, stop_words):
    total_freq = {}
    doc_appear = {}
    doc_freq = []
    total_words = 0
    ps = nltk.PorterStemmer()
    for file_path in documents:
        start = time.time()
        # ------------------
        f = open(file_path, 'r')
        words = f.read().split()
        f.close()
        num_words = 0
        ind_freq = {}
        for word in words:
            num_words += 1
            total_words += 1
            # Remove puncuation and set to lower case
            # TODO: replace punctuation with spaces instead of empty string?
            word = word.lower().translate(str.maketrans('', '', string.punctuation))
            #Check if stop word
            if word in stop_words:
                continue
            else:
                # Stem word 
                word = ps.stem(word)
                # Add word to total frequency
                if word in total_freq:
                    total_freq[word] += 1
                else:
                    total_freq[word] = 1
                # Add word to individual frequency
                if word in ind_freq:
                    ind_freq[word] += 1
                else:
                    ind_freq[word] = 1
                    # Add word tot number of documents in which it appears
                    if word in doc_appear:
                        doc_appear[word] += 1
                    else:
                        doc_appear[word] = 1
        doc_freq.append(Vector(ind_freq, file_path, num_words))
        # -------stop and print timer---------
        end = time.time()
        print("Time to process file: " + str(end - start))

    avg_words = total_words / len(documents)
    for vector in doc_freq:
        vector.tf_idf(len(documents), doc_appear)
        vector.average_words = avg_words
        
    return total_freq, doc_freq, doc_appear

def filterSingles(total_freq):
    new_dict = {}
    for word in total_freq.keys():
        if total_freq[word] > 1:
            new_dict[word] = total_freq[word]
    return new_dict

if __name__ == '__main__':
    # dataset_path: name of directory
    # output_path: name of output file (without .csv extension)
    # stop_words_path: path to stop words file
    dataset_path, output_name, stop_words_path = parseVectorizerArgs(sys.argv[1:])
    # documents: list of document paths
    # output_path: <output_name>.csv
    # stop_words: dictionary of stop words
    documents, output_path = createGroundTruth(dataset_path, output_name)
    stop_words = processStopWords(stop_words_path)
    ground_truth_df = pd.read_csv(output_path)

    total_freq, doc_freq, doc_appear = getFrequencies(documents[:3], stop_words)
    total_freq = filterSingles(total_freq)

    cossim1 = doc_freq[0].cosSim(doc_freq[0])
    print(cossim1)
    cossim2 = doc_freq[0].cosSim(doc_freq[1])
    print(cossim2)
    oksim1 = doc_freq[0].okapi(doc_freq[0], 3, doc_appear)
    print(oksim1)
    oksim2 = doc_freq[0].okapi(doc_freq[1], 3, doc_appear)
    print(oksim2)
    print("BREAKPOINT")
    # # TODO: REMOVE THESE BEFORE FINAL SUBMISSION
    # # --- SKLEARN VECTORIZERS (FOR OUR COMPARISON) ---
    # # Sk Word Count Matrix
    # sk_count_vectorizer = sktext.CountVectorizer(analyzer='word', input='filename')
    # sk_count_wm = sk_count_vectorizer.fit_transform(documents)
    # sk_count_tokens = sk_count_vectorizer.get_feature_names_out()
    # sk_wm_df = pd.DataFrame(sk_count_wm.toarray(), index=documents, columns=sk_count_tokens)

    # # Sk tf-idf Matrix
    # sk_tfidf_vectorizer = sktext.TfidfVectorizer(analyzer='word', input='filename')
    # sk_tfidf_wm = sk_tfidf_vectorizer.fit_transform(documents)
    # sk_tfidf_tokens = sk_tfidf_vectorizer.get_feature_names_out()
    # sk_tfidf_df = pd.DataFrame(sk_tfidf_wm.toarray(), index=documents, columns=sk_tfidf_tokens)
    # # ------
