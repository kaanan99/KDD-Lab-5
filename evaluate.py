import math
import pandas as pd
import numpy as np
import argparse
import json
import os
from tree import *
from InduceC45 import *
import classify


def print_record(record):
    return " ".join([x for x in record])


def get_k_folds(D, k):
    """
    Split data D into k folds, returning a list of k DataFrames.
    """
    if k == -1:
        k = D.shape[0]

    # prevent divide by 0; k=0 and k=1 men the same thing: no folds
    if k == 0:
        k == 1

    fold_len = D.shape[0] // k  # num data pts per fold

    # split the data into folds; use min of calculated fold end index and dataset length to prevent index OOB error (i.e. in case k doesn't divide evenly)
    folds = [D.iloc[i*fold_len:(i+1)*fold_len] for i in range(k-1)]  # 1 to k-1 folds
    last_fold_start = (k-1)*fold_len    # last fold goes until EOD
    folds.append(D.iloc[last_fold_start:])
    # for i in range(len(folds)):
    #     print(f"len(fold[{i}]) = {len(folds[i])}")
    return folds


def get_train_test(folds, i):
    """
    Return the train and test datasets for the ith fold, given i and a list of all folds.
    """
    D_test = folds[i]
    D_trains = folds[:i] + folds[i+1:]
    D_train = D_trains[0]
    for fold in D_trains[1:]:
        D_train = D_train.append(fold)
    #print(f"D_test: {D_test}\nD_train: {D_train}")

    return D_train, D_test


def cross_validate(D, attrbs, class_col, thr, thr_is_ratio, k, vals_per_attrb):
    """
    Perform cross-validation on D using k folds.
    Return: a list of predicted class labels, one for each data point in the same order as data points in D
    """
    folds = get_k_folds(D, k)

    predictions = []
    
    i = 0
    while i < k:
        #print(f"i = {i}")

        # get the train and test data subsets for this ith fold
        D_train, D_test = get_train_test(folds, i)

        # build a decision tree based on the training set
        T = c45(D_train, attrbs, thr, thr_is_ratio, class_col, vals_per_attrb)

        node_type = 'leaf' if T.leaf else 'node'
        T = {node_type: T.get_dict()}  # tree as wrapped dict
        #print(f"Tree for {i}th fold: {T}")
        
        # append predictions of ith fold
        predictions.extend(classify.classify_all(D_test, T, vals_per_attrb))

        i += 1

    return predictions


def output_predictions(csv_path, actual, predictions, res_str):
    """
    Output pt-by-pt predictions to a .csv file.
    Params:
        csv_path - the path to the datafile (used to name results csv)
        actual - an iterable of true class labels
        predictions - an iterable of predicted class labels (in same order as actual)
        res_str - a string describing the algorithm and hyperparameters used
    """
    name = os.path.basename(csv_path)
    name = os.path.splitext(name)[0]
    df = pd.DataFrame({'Truth': actual, 'Prediction': predictions})
    df.to_csv(f'{name}_results_{res_str}.csv')
    return name


def main():
    parser = argparse.ArgumentParser(description="Perform k-fold cross-validation")
    parser.add_argument("csv_path", help="The filepath to the data csv file")
    parser.add_argument('-r', '--restrictions', required=False, action='store', default=None, help='An optional attribute restriction/selector text file')
    parser.add_argument('-t', '--threshold', required=True, action='store', type=float, help="The information gain threshold. Can be absolute or ratio. If ratio, must also select --ratio.")
    parser.add_argument('--ratio', required=False, action='store_true', help="Select if the gain threshold given is a threshold on the information gain RATIO")
    parser.add_argument("-k", required=True, type=int, help="Create k folds for cross-validation (integer k > 1)")

    args = parser.parse_args()

    D, attrbs, class_col, vals_per_attrb  = get_data(args.csv_path, args.restrictions)

    # skip cross-validation if k is 0 or 1
    if args.k == 0 or args.k == 1:
        T = c45(D, attrbs, args.threshold, args.ratio, class_col, vals_per_attrb)
        node_type = 'leaf' if T.leaf else 'node'
        T = {node_type: T.get_dict()}  # tree as wrapped dict
        predictions = classify.classify_all(D, T, vals_per_attrb)
    else:
        predictions = cross_validate(D, attrbs, class_col, args.threshold, args.ratio, args.k, vals_per_attrb)

    # output pt-by-pt predictions to results.txt
    res_str = f"c45-thr{args.threshold}-k{args.k}"
    name = output_predictions(args.csv_path, D[class_col], predictions, res_str)

    matrix = classify.make_matrix(predictions, D[class_col])
    print(matrix)

    overall, avg = classify.calc_accuracies(predictions, D[class_col], args.k)
    print(f"Overall accuracy: {overall:.4f}")
    print(f"Average accuracy: {avg:.4f}")

    with open(f"{name}_eval_{res_str}.txt", 'w') as file:
        file.write(matrix.to_string())
        file.write(f"\nOverall accuracy: {overall:.4f}")
        file.write(f"\nAverage accuracy: {avg:.4f}")

if __name__ == "__main__":
    main()
