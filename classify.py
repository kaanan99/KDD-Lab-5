import math
import pandas as pd
import numpy as np
import argparse
import json
import os
from tree import *

def get_data(csv_path):
    """
    Parse the data and metadata.
    Params:
        csv_path - the full path to the data csv file: first row is column names; second row is number of possible values for each column; third row's first cell is name of class column; all following rows are data pts
    Returns:
        D - whole dataset (without metadata rows) as DataFrame
        class_col - the column name of the class label column, None if none specified
        vals_per_attrb - a Series containing the number of unique values per attribute;
            an attribute's value count is 0 if the attribute is continuous
    """
    D = pd.read_csv(csv_path)
    #print(D)

    # get the number of attribute values per attribute
    vals_per_attrb = pd.to_numeric(D.iloc[0], downcast="integer")

    # get the class column name from the second row
    class_col = D.iloc[1, 0]

    # remove the first two metadata rows from D
    D = D.iloc[2:]

    # return None class_col if none given in second row
    if pd.isnull(class_col):
        class_col = None

    # convert continuous data columns to numerical
    for A in D.columns:
        if int(vals_per_attrb.at[A]) == 0:
            D[A] = pd.to_numeric(D[A], downcast="float")

    # randomize D
    D = D.sample(frac=1, replace=False)

    return D, class_col, vals_per_attrb


def parse_dt(dt_json):
    with open(dt_json) as f:
        return json.load(f)


def print_record(record):
    return " ".join([str(x) for x in record])


def classify_one(record, T, vals_per_attrb):
    """
    Classify one record using the decision tree.
    Params:
        record - a single data point (i.e. one row in the dataset DataFrame)
        T - the decision tree represented as a per-spec dictionary
        vals_per_attrb - a Series containing the number of unique values per attribute;
            an attribute's value count is 0 if the attribute is continuous
    Return: the predicted class label
    """
    if "leaf" in T.keys():
       return T["leaf"]["decision"]

    current = T["node"]
    # run until a leaf node is found
    while True:
        #print(f"Current keys: {current.keys()}")
        attrb = current["var"]      # attrb name
        val = record[attrb]         # the value of this data point

        # if attribute is numerical, parse and check alpha value
        if vals_per_attrb[attrb] == 0:
            a_edge = current["edges"][0]["edge"]            # get one of the two edges; both have same alpha
            alpha = a_edge["value"]                         # the alpha value
            cdtn = a_edge["direction"]                      # the condition string: 'le' or 'gt'
            index_of_lte_edge = 0 if cdtn == "le" else 1    # index (0 or 1) of edge that is <= alpha

            # set i to be the index of the edge this data point falls under, based on alpha comparison
            if val <= alpha:
                i = index_of_lte_edge
            else:
                i = 1 - index_of_lte_edge

        # otherwise search through categorical edges for matching attribute value
        else:
            i = 0
            #print(f"Looking for edge matching '{attrb}' value {record[attrb]} in {[e['edge']['value'] for e in current['edges']]}")
            while val != current["edges"][i]["edge"]["value"]:
                #print(f"  Current edge is {current['edges'][i]['edge']['value']}")
                i += 1

        # if the correct edge leads to a leaf, return the decision of that leaf, else keep looping
        if "leaf" in current["edges"][i]["edge"].keys():
#            print("Record:", print_record(record))
#            print("Prediction:", current["edges"][i]["edge"]["leaf"]["decision"],  "\n")
            return current["edges"][i]["edge"]["leaf"]["decision"]
        current = current["edges"][i]["edge"]["node"]


def classify_all(D, T, vals_per_attrb):
    """
    Classify all data points in a data set (using only selected attributes) based on a 
    decision tree.
    Params:
        D - the dataset (as a DataFrame)
        T - the decision tree represented as a per-spec dictionary
        vals_per_attrb - a Series containing the number of unique values per attribute;
            an attribute's value count is 0 if the attribute is continuous
    Return: a list of predicted class labels, one for each data point in the same order as data points in D
    """
    predictions = []
    for i in range(D.shape[0]):
        record = D.iloc[i]
        predictions.append(classify_one(record, T, vals_per_attrb))
    return predictions


def make_matrix(predictions, actual):
    """
    Create a confusion matrix of prediction vs. truth values.
    Params:
        predictions - a list of predicted class labels, ordered in accordance with data set
        actual - a Series of the ground truth class labels, ordered in accordance with data set
    Return: a confusion matrix, represented as a multi-indexed DataFrame
    """
    # get all possible class labels from truth label set
    class_label_options = actual.unique()   # all possible class labels

    # get actual class labels as a list
    actual_vals = actual.array

    # print(f"Predictions: {predictions}\nActual: {actual_vals}")

    # create an all-zero DataFrame with rows and columns labeled with class labels
    df = pd.DataFrame([[0 * len(class_label_options)] * len(class_label_options)], class_label_options, class_label_options)

    # go through each prediction-truth pair and increment the appropriate cell count
    for x in range(len(predictions)):
        df.loc[actual_vals[x], predictions[x]] += 1 

    df = pd.DataFrame(df.values,pd.MultiIndex.from_product([['Predicted'], df.index]), pd.MultiIndex.from_product([['Actual'], df.columns]))
    return df


def calc_accuracies(predictions, actual, k):
    """
    Calculate the overall and average (of per fold) accuracies by comparing the predictions to the actual class labels.
    """
    actual = actual.array

    if k == -1:
        k = len(predictions)
    # prevent divide by 0; k=0 and k=1 mean the same thing: no folds
    if k == 0:
        k == 1
    fold_len = len(predictions) // k  # num data pts per fold

    # get a list of lists: [[1, 0, 0, 1, 0, ...], [...], ...] where each item is a list of yes/no (1/0) matches of the value in that fold
    folds = []
    for i in range(k-1):      # for each fold but the last
        fold_start = i*fold_len     # start of data index
        fold_end = (i+1)*fold_len   # EOD index
        fold_accs = []
        for j in range(fold_start, fold_end):
            fold_accs.append(1 if predictions[j] == actual[j] else 0)
        folds.append(fold_accs)

    # calc accuracies for last fold
    last_fold_start = (k-1) * fold_len  # last fold goes until EOD
    fold_accs = []
    for j in range(last_fold_start, len(predictions)):
        fold_accs.append(1 if predictions[j] == actual[j] else 0)
    folds.append(fold_accs)

    overall = 0
    avg = 0
    for i, fold in enumerate(folds):
        overall += sum(fold)    # add to the numerator of the overall accuracy
        #print(f"Added {sum(fold)} correct predictions. Overall numer now {overall}")
        avg += sum(fold) / len(fold)    # add to the numerator of the avg accuracy

    # divide accuracies by the correct denom
    overall /= len(predictions)
    avg /= len(folds)
    
    return overall, avg


def main():
    parser = argparse.ArgumentParser(description="Classify a test data set based on a given decision tree")
    parser.add_argument("csv_path", help="The filepath to the training data csv file")
    parser.add_argument("json_path", help="The filepath to the json file representing the decision tree")

    args = parser.parse_args()

    D, class_col, vals_per_attrb = get_data(args.csv_path)
    T = parse_dt(args.json_path)
    predictions = classify_all(D, T, vals_per_attrb)
    
    # evaluate predictions only if the class column is given
    if class_col is not None:
        matrix = make_matrix(predictions, D[class_col])
        print(matrix)
        overall, avg = calc_accuracies(predictions, D[class_col], 1)    # this classifier runs with no k-folds, so k=1
        print(f"Overall accuracy: {overall:.4f}")
        print(f"Average accuracy: {avg:.4f}")
        #print(predictions)


if __name__ == "__main__":
    main()
