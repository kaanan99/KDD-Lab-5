import sys
import os
import numpy as np
import pandas as pd
import string
import nltk
import time
import math
import sklearn.feature_extraction.text as sktext
import json

class Vector:

    def __init__(self, word_freq, name, num_words):
        self.name = name
        self.author = name.split('/')[-2]
        self.num_words = num_words
        self.average_words = None
        self.word_freq = word_freq
        self.weights = {}

    def to_json(self):
        return {"name": self.name, "author": self.author,"num_words": self.num_words, 
                "average_words": self.average_words, "word_freq": self.word_freq, 
                "weights": self.weights}

    def tf_idf(self, n, doc_appear):
        max_freq = max(self.word_freq.values())
        for word, freq in self.word_freq.items():
            tf = freq/max_freq
            idf = math.log2(n/doc_appear[word])
            self.weights[word] = tf * idf

    def calcDist(self, other, metric, n, doc_appear):
        if metric == 'cos':
            return self.cosSim(other)
        elif metric == 'okapi':
            return self.okapi(other, n, doc_appear)

    # TODO: double check this
    # TODO: MAYBE CONVERT TO LIST COMPREHENSION/NP ARRAYS
    def cosSim(self, other):
        #Version 3
        return np.dot(self.weights, other.T)/(np.linalg.norm(self.weights)*np.linalg.norm(other, axis=1).T)
        # similarity = 0
        # numerator = 0
        # denom1 = 0
        # denom2 = 0
        # for word in totalWords:
        #     selfWeight = 0
        #     otherWeight = 0
        #     if word in self.weights:
        #         selfWeight = self.weights[word]
        #     if word in other.weights:
        #         otherWeight = other.weights[word]
        #     numerator += selfWeight * otherWeight
        #     denom1 += selfWeight ** 2
        #     denom2 += otherWeight ** 2
        # denominator = math.sqrt(denom1 * denom2)
        # similarity = numerator / denominator
        # return similarity
        
        # Original
        # similarity = 0
        # numerator = 0
        # denom1 = 0
        # denom2 = 0
        # match_words = set(self.weights.keys()).union(set(other.weights.keys()))
        # for word in match_words:
        #     numerator += self.weights[word] * other.weights[word]
        #     denom1 += self.weights[word] ** 2
        #     denom2 += other.weights[word] ** 2
        # denominator = math.sqrt(denom1 * denom2)
        # similarity = numerator / denominator
        # return similarity
    
    # TODO: MAYBE CONVERT TO LIST COMPREHENSION/NP ARRAYS
    # TODO: recheck this
    def okapi(self, other, n, doc_appear):
        k1 = 1.5
        b = .75
        k2 = 500
        ok_sim = 0
        match_words = self.weights.keys() & other.weights.keys()
        if not match_words:
            return np.NINF
        for word in match_words:
            alt_idf_d = math.log((n - doc_appear[word] + .5) / (doc_appear[word] + .5))
            alt_tf = ((k1 + 1) * self.word_freq[word]) / (k1 * (1 - b + b * (self.num_words / self.average_words)) + self.word_freq[word])
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
                for file in os.listdir(dataset_path + '/' + dir + '/' + author): # TODO: remove this [:5]
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
    total_freq = {} # word freq across document set
    doc_appear = {} # how many documents a word appears in
    doc_freq = [] # list of document vectors
    num_docs = len(documents)
    total_words = 0
    ps = nltk.PorterStemmer()

    i = 1
    for file_path in documents:
        start = time.time()
        num_words = 0
        ind_freq = {}

        f = open(file_path, 'r')
        words = f.read().split()
        f.close()

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
        end = time.time()
        print("Time to process file: " + str(end - start) + ' - ' + str(i) + '/' + str(num_docs))
        i += 1

    doc_freq, doc_appear = removeSinglets(doc_freq, doc_appear)
    avg_words = total_words / num_docs
    for vector in doc_freq:
        vector.tf_idf(num_docs, doc_appear)
        vector.average_words = avg_words
    return total_freq, doc_freq, doc_appear

def write_json(doc_freq, doc_appear, n):
    json_vectors = [x.to_json() for x in doc_freq]
    f = open("vectors.json", "w")
    f.write(str(n))
    f.write("\n")
    json.dump(doc_appear, f)
    f.write("\n")
    json.dump(json_vectors, f)
    f.close()

def removeSinglets(doc_freq, doc_appear):
    # Remove all words that appear in only one document
    new_doc_appear = {}
    num_words = len(doc_appear)
    i = 1
    for word in doc_appear:
        if doc_appear[word] == 1:
            for vector in doc_freq:
                if word in vector.word_freq:
                    del vector.word_freq[word]
                    break
        else:
            new_doc_appear[word] = doc_appear[word]
        print('Checking Words ' + str(i) + '/' + str(num_words))
        i += 1
    return doc_freq, new_doc_appear


# TODO: FIX PROBLEM OF EMPTY STRINGS BEING ADDED TO VECTORIZATION
if __name__ == '__main__':
    dataset_path, output_name, stop_words_path = parseVectorizerArgs(sys.argv[1:])
    documents, output_path = createGroundTruth(dataset_path, output_name)
    stop_words = processStopWords(stop_words_path)
    ground_truth_df = pd.read_csv(output_path)

    total_freq, doc_freq, doc_appear = getFrequencies(documents, stop_words)
    write_json(doc_freq, doc_appear, len(documents))

    print("Breakpoint")

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
