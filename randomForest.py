import time
import math
import pandas as pd
import numpy as np
import argparse
import json
import os
import random
from collections import Counter
from InduceC45 import *
import classify
from evaluate import *


DEBUG = False

def get_data_selection(D, attrbs, class_col, numAttrbs, numDataPts):
    """
    Select a random subset of data.
    Params:
        D - the dataset (a DataFrame)
        attrbs - a list of all attribute column names in D
        class_col - the name of the class column (so not removed from D_subset)
        numAttrbs - the number of attributes to randomly select (without replacement)
        numDataPts - the number of data points to randomly select (with replacement)
    Returns:
        a subset of D with randomly selected data points consisting of only the given number of (randomly selected) attributes
        the list of selected attributes
    """
    rand_attrbs = random.sample(attrbs, numAttrbs)
    # need to include class col
    if class_col is not None:
        select_attrbs = rand_attrbs.copy()
        select_attrbs.append(class_col)
    return D[select_attrbs].sample(n=numDataPts, replace=True), rand_attrbs


def build_random_forest(TREE_THRES, num_trees, num_attr, num_data_points, D, attrbs_info, class_col, vals_per_attrb):
    trees = []
    select_times = 0
    c45_times = 0

    for i in range(num_trees):
        start = time.time()
        Di, attrbsi = get_data_selection(D, attrbs_info.keys(), class_col, num_attr, num_data_points)    # get subset of data
        attrbsi_info = {A: attrbs_info[A] for A in attrbsi}                                         # get subset of attrb info
        end = time.time()

        Ti = c45(Di, attrbsi_info, TREE_THRES, True, class_col, vals_per_attrb)               # gen tree
        end2 = time.time()
        trees.append(Ti)

        select_times += end - start
        c45_times += end2 - end
        # print(f"  Tree {i} took {end2 - start}s to build")

    select_times /= num_trees
    c45_times /= num_trees
    # print(f"  On avg, data select ({args.numAttrbs}/{len(attrbs_info.keys())} attrbs, {args.numDataPts}/{D.shape[0]} pts) takes {select_times}s")
    # print(f"  On avg, c45 ({args.numAttrbs}/{len(attrbs_info.keys())} attrbs, {args.numDataPts}/{D.shape[0]} pts) takes {c45_times}s")

    return trees


def forest_classify(D_test, trees, vals_per_attrb):
    """
    Return the list of class predictions for the data in D_test based on the plurality of predicted classes from the tree in trees.
    """
    final_preds = []    # an array for the final plurality class of each data point
    classify_times = 0

    # have each tree predict a list of predictions for each test data point
    for i in range(D_test.shape[0]):
        record = D_test.iloc[i]     # the ith data point
        i_preds = []                # a list of predictions for this ith data point (plurality to be taken)
        for T in trees:
            # wrap the tree (of nodes) in a dict (expected by classifier)
            node_type = 'leaf' if T.leaf else 'node'
            T = {node_type: T.get_dict()}  # tree as wrapped dict

            start = time.time()
            pred = classify.classify_one(record, T, vals_per_attrb)
            end = time.time()
            classify_times += end - start

            i_preds.append(pred)
        
        # store the plurality class for each data point
        c = Counter(i_preds)
        final_preds.append(c.most_common(1)[0][0])

    classify_times /= (D_test.shape[0] * len(trees))
    # print(f"  On avg, classify_one takes {classify_times}s ({D_test.shape[0]} points, {len(trees)} trees)")

    return final_preds


def main():
    parser = argparse.ArgumentParser(description="A random forest classifier.")
    parser.add_argument("csv_path", help="The filepath to the data csv file")
    parser.add_argument('-r', '--restrictions', required=False, action='store', default=None, help='An optional attribute restriction/selector text file')
    parser.add_argument('-t', '--threshold', required=True, action='store', type=float, help="The information gain ratio threshold.")
    parser.add_argument('-m', '--numAttrbs', required=True, action='store', type=int, help='the number of attributes each built decision tree should contain')
    parser.add_argument('-k', '--numDataPts', required=True, action='store', type=int, help='the number of data pts selected randomly with relpacement to form each decision tree')
    parser.add_argument('-N', '--numTrees', required=True, action='store', type=int, help='the number of decision trees to build')

    args = parser.parse_args()

    D, attrbs_info, class_col, vals_per_attrb = get_data(args.csv_path, args.restrictions)   # TODO: restrictions = None?

    # break the dataset into folds
    k_fold = 10
    folds = get_k_folds(D, k_fold)

    # build a forest on the training set for each ith fold, then classify that fold to get predictions
    predictions = []    # a list for all the data pt predictions
    for i, fold in enumerate(folds):
        # get the train and test sets
        D_train, D_test = get_train_test(folds, i)

        # build a forest using the training set
        start = time.time()
        trees = build_random_forest(args, D_train, attrbs_info, class_col, vals_per_attrb)
        end = time.time()
        # print(f"Built fold {i} forest in {end - start} seconds")

        # classify the test set
        start = time.time()
        i_preds = forest_classify(D_test, trees, vals_per_attrb)
        end = time.time()
        # print(f"  Classified fold {i} in {end - start} seconds")

        # add the predictions for this fold to the overall list
        predictions.extend(i_preds)

    # print_time_avgs()

    # output pt-by-pt predictions to results.txt
    res_str = f"rt-thr{args.threshold}-m{args.numAttrbs}-k{args.numDataPts}-N{args.numTrees}"
    name = output_predictions(args.csv_path, D[class_col], predictions, res_str)


    # output confusion matrix and accuracies
    matrix = classify.make_matrix(predictions, D[class_col])
    print(matrix)
    overall, avg = classify.calc_accuracies(predictions, D[class_col], k_fold)
    print(f"Overall accuracy: {overall:.4f}")
    print(f"Average accuracy: {avg:.4f}")
    with open(f"{name}_eval_{res_str}.txt", 'w') as file:
        file.write(matrix.to_string())
        file.write(f"\nOverall accuracy: {overall:.4f}")
        file.write(f"\nAverage accuracy: {avg:.4f}")


if __name__ == "__main__":
    main()
