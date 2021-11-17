import sys
import os
import numpy as np
import pandas as pd
import string
import nltk
import time
import math
import json

class Vector:

    def __init__(self, word_freq, name, num_words):
        self.name = name
        self.author = name.split('/')[-2]
        self.num_words = num_words
        self.average_words = None
        self.word_freq = word_freq
        self.weights = {}
        self.word_vector = None

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
    
    # TODO: recheck this
    def okapi(self, other, n, doc_appear, k1=1.5, b=0.75, k2=500):
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
        print("Usage: python3 textVectorizer.py <directory_path> <ground_path> <stop_words_path>")
        sys.exit(1)
    #check if args[0] is a directory
    if not os.path.isdir(args[0]):
        print("Error:")
        print("Usage: python3 textVectorizer.py <directory_path> <ground_path> <stop_words_path>")
        print("Directory path does not exist")
        sys.exit(1)
    # check if args[1] is a csv
    if not args[1].endswith('.csv'):
        print("Error:")
        print("Usage: python3 textVectorizer.py <directory_path> <ground_path> <stop_words_path>")
        print("Output file must be a csv")
        sys.exit(1)
    # check if args[2] is a txt file
    if not args[2].endswith('.txt'):
        print("Error:")
        print("Usage: python3 textVectorizer.py <directory_path> <ground_path> <stop_words_path>")
        print("Output file must be a txt")
        sys.exit(1)
    return args[0], args[1], args[2]

# TODO: Examine whether this is how we want to do this
def createGroundTruth(dataset_path, output_path):
    # list of document paths
    documents = []
    # iterate over dataset_path directory and create ground truth .csv file
    with open(output_path, 'w') as output_file:
        output_file.write('file_name,author\n')
        for dir in os.listdir(dataset_path):
            for author in os.listdir(dataset_path + '/' + dir):
                for file in os.listdir(dataset_path + '/' + dir + '/' + author): # TODO: remove this [:5]
                    output_file.write(file + ',' + author + '\n')
                    documents.append(dataset_path + '/' + dir + '/' + author + '/' + file)
        output_file.close()
    return documents

def processStopWords(stop_word_path):
    f = open(stop_word_path, "r")
    words = f.read().split()
    f.close()
    # TODO: replace punctuation with spaces instead of empty string?
    return {w.lower().translate(str.maketrans('', '', string.punctuation)):1 for w in words}

def createVectors(documents, stop_words):
    doc_appear = {} # how many documents a word appears in
    doc_vectors = [] # list of document vectors
    num_docs = len(documents) # num documents in dataset
    total_words = 0 # num words in document dataset
    stemmer = nltk.PorterStemmer()

    i = 1
    for file_path in documents:
        start = time.time() # start timer

        f = open(file_path, 'r')
        words = f.read().split()
        f.close()

        num_words = 0 # num words in document
        doc_freqs = {} # term frequencies for given document

        for word in words:
            word = word.lower().translate(str.maketrans('', '', string.punctuation)) # lower case and remove punctuation

            if word in stop_words or word == '':
                continue
            else:
                num_words += 1 # word is considered in document comparison
                word = stemmer.stem(word) # stem word

                # Add word to individual frequency
                if word in doc_freqs:
                    doc_freqs[word] += 1
                else:
                    doc_freqs[word] = 1
                    if word in doc_appear:
                        doc_appear[word] += 1
                    else:
                        doc_appear[word] = 1
        total_words += num_words
        doc_vectors.append(Vector(doc_freqs, file_path, num_words))

        end = time.time()
        print("Time to process file: " + str(end - start) + ' - ' + str(i) + '/' + str(num_docs))
        i += 1

    avg_words = total_words / num_docs
    # compute weights for each document term
    for vector in doc_vectors:
        vector.tf_idf(num_docs, doc_appear)
        vector.average_words = avg_words
    
    doc_vectors = convertToSparse(doc_vectors, doc_appear)
    doc_vectors, doc_appear = removeSinglets(doc_vectors, doc_appear)
    return doc_vectors, doc_appear

def convertToSparse(doc_vectors, doc_appear):
    for word in doc_appear:
        for vector in doc_vectors:
            new_word_freq = []
            new_word_weights = []
            if word in vector.word_freq:
                new_word_freq.append(vector.word_freq[word])
                new_word_weights[word] = vector.word_weights[word]
            else:
                new_word_freq[word] = 0
                new_word_weights[word] = 0.0
    return doc_vectors


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

def writeJson(doc_freq, doc_appear, n):
    json_vectors = [x.to_json() for x in doc_freq]
    f = open("vectors.json", "w")
    f.write(str(n))
    f.write("\n")
    json.dump(doc_appear, f)
    f.write("\n")
    json.dump(json_vectors, f)
    f.close()

# TODO: FIX PROBLEM OF EMPTY STRINGS BEING ADDED TO VECTORIZATION
if __name__ == '__main__':
    dataset_path, output_path, stop_words_path = parseVectorizerArgs(sys.argv[1:])
    documents = createGroundTruth(dataset_path, output_path)
    stop_words = processStopWords(stop_words_path)
    ground_truth_df = pd.read_csv(output_path)

    doc_vectors, doc_appear = createVectors(documents, stop_words)
    writeJson(doc_vectors, doc_appear, len(documents))

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
