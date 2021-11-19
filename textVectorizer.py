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

    def __init__(self, word_freq, name):
        self.name = name # path to document
        self.author = name.split('/')[-2] # author of the document
        self.num_words = None # num words in the document
        self.average_words = None # average words in document collection
        self.word_freq = word_freq # frequency of each word in the document
        self.tfidf_weights = None # tf-idf weights for each word in the document
        self.okapi_left = None
        self.okapi_right = None

    def tf_idf(self, n, word_doc_counts):
        self.tfidf_weights = np.multiply(self.word_freq/np.max(self.word_freq), np.log2(n/word_doc_counts))
    
    def calc_okapi(self, n, word_doc_counts, k1=1.5, b=0.75, k2=500):
        self.okapi_left = np.multiply(np.log((n-word_doc_counts + 0.5)/(word_doc_counts+0.5)), 
                                    ((k1+1)*self.word_freq)/(k1 * (1-b+b*self.num_words/self.average_words) + self.word_freq))
        self.okapi_right = ((k2+1) * self.word_freq)/(k2+self.word_freq)

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
        print("Usage: python3 textVectorizer.py <directory_path> <ground_path> <stop_words_dir>")
        print("Directory path does not exist")
        sys.exit(1)
    # check if args[1] is a csv
    if not args[1].endswith('.csv'):
        print("Error:")
        print("Usage: python3 textVectorizer.py <directory_path> <ground_path> <stop_words_dir>")
        print("Output file must be a csv")
        sys.exit(1)
    # check if args[2] is a directory
    if not os.path.isdir(args[2]):
        print("Error:")
        print("Usage: python3 textVectorizer.py <directory_path> <ground_path> <stop_words_dir>")
        print("Stop word directory path does not exist")
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

def processStopWords(stop_words_dir):
    stop_words = {}
    for file in os.listdir(stop_words_dir):
        with open(stop_words_dir + '/' + file, 'r') as stop_file:
            words = stop_file.read().split()
            f_stop_words = {w.lower().translate(str.maketrans('', '', string.punctuation)):1 for w in words}
            stop_words = {**stop_words, **f_stop_words}
            stop_file.close()
    stop_words[''] = 1
    return stop_words

def createVectors(documents, stop_words):
    doc_appear = {} # how many documents a word appears in
    doc_vectors = [] # list of document vectors
    num_docs = len(documents) # num documents in dataset
    stemmer = nltk.SnowballStemmer(language='english')
    i = 1
    for file_path in documents:
        start = time.time() # start timer
        f = open(file_path, 'r')
        words = f.read().split()
        f.close()

        word_freqs = {} # term frequencies for given document

        for word in words:
            word = word.lower().translate(str.maketrans('', '', string.punctuation)) # lower case and remove punctuation
            if word in stop_words or word.isnumeric(): # if word is a stop word or a number
                continue
            else:
                word = stemmer.stem(word) # stem word
                # Add word to individual frequency
                if word in word_freqs:
                    word_freqs[word] += 1
                else:
                    word_freqs[word] = 1
                    if word in doc_appear:
                        doc_appear[word] += 1
                    else:
                        doc_appear[word] = 1
        doc_vectors.append(Vector(word_freqs, file_path))
        end = time.time()
        print("Time to process file: " + str(end - start) + ' - ' + str(i) + '/' + str(num_docs))
        i += 1
    
    doc_appear = {k:v for k,v in doc_appear.items() if v > 1} # remove words that appear in only one document
    word_list = list(doc_appear.keys()) # list of words in the dataset
    word_doc_counts = np.asarray(list(doc_appear.values())) # list of how many documents each word appears in

    doc_vectors, total_Words = convertToSparse(doc_vectors, word_list) # align and convert vectors to sparse
    
    avg_words = total_Words/num_docs
    for vector in doc_vectors:
        start = time.time()
        vector.average_words = avg_words
        vector.tf_idf(num_docs, word_doc_counts)
        vector.calc_okapi(num_docs, word_doc_counts)
        end = time.time()
        print("Time to process tf-idf and okapi: " + str(end - start))
    return doc_vectors, word_list, word_doc_counts

# NOTE: this could be unnecessary?
def convertToSparse(doc_vectors, word_list):
    total_words = 0
    for vector in doc_vectors:
        start = time.time()
        new_freqs = np.asarray([vector.word_freq[word] if word in vector.word_freq else 0 for word in word_list], dtype=np.int32)
        vector.word_freq = new_freqs
        vector.num_words = np.sum(new_freqs)
        total_words += vector.num_words
        end = time.time()
        print("Time to convert to sparse: " + str(end - start))
    return doc_vectors, total_words

def writeToFiles(doc_vectors, word_list, word_doc_counts):
    print("WRITING TO FILES....")
    word_counts_file = open('vectorized/word_counts.txt', 'w')
    json.dump(word_list, word_counts_file)
    word_counts_file.write('\n')
    json.dump(word_doc_counts.tolist(), word_counts_file)
    word_counts_file.close()

    doc_vec = open('vectorized/doc_vectors.txt', 'w')

    doc_vectors_arr = [{'author': str(vector.author), 'file_name': str(vector.name), 'num_words': int(vector.num_words), 'average_words': float(vector.average_words),
                            'tfidf': vector.tfidf_weights, 'okapi_left': vector.okapi_left, 'okapi_right': vector.okapi_right} 
                            for vector in doc_vectors]
    json.dump(doc_vectors_arr, doc_vec)
    doc_vec.close()

# prunes down representations for output to file
def pruneToDict(doc_vectors, word_list):
    for vector in doc_vectors:
        start = time.time()
        new_tfidf_weights = {}
        new_okapi_left = {}
        new_okapi_right = {}
        for idx, word in enumerate(word_list):
            if vector.word_freq[idx] != 0:
                new_tfidf_weights[word] = vector.tfidf_weights[idx]
                new_okapi_left[word] = vector.okapi_left[idx]
                new_okapi_right[word] = vector.okapi_right[idx]
        vector.tfidf_weights = new_tfidf_weights
        vector.okapi_left = new_okapi_left
        vector.okapi_right = new_okapi_right
        end = time.time()
        print("Time to prune to dict: " + str(end - start))
    return doc_vectors

if __name__ == '__main__':
    dataset_path, output_path, stop_words_dir = parseVectorizerArgs(sys.argv[1:])
    documents = createGroundTruth(dataset_path, output_path)
    stop_words = processStopWords(stop_words_dir)
    ground_truth_df = pd.read_csv(output_path)

    doc_vectors, word_list, word_doc_counts = createVectors(documents, stop_words)
    doc_vectors = pruneToDict(doc_vectors, word_list)
    writeToFiles(doc_vectors, word_list, word_doc_counts)

