import concurrent.futures
import InduceC45
import numpy as np
import pandas as pd
import random
import time
import sys
import itertools as it

def createForestTree(dataframe, attributes, class_col, num_attr, num_data_points, tree_thres, i):
    print(f"Creating Tree {i}...")
    start = time.time()
    sampled_df = dataframe.sample(n=num_data_points, replace=True)
    sampled_attributes = random.sample(attributes, num_attr)
    sampled_df = sampled_df[sampled_attributes + [class_col]]
    tree = InduceC45.c45(sampled_attributes, sampled_df, class_col, tree_thres)
    end = time.time()
    print(f"Tree {i} created - took {end - start} sec")
    return tree

def predictRow(trees, row):
    print(f'Making Prediction {row[1].name}')
    start = time.time()
    predictions = [InduceC45.classifyItem(tree, row[1]) for tree in trees]
    prediction_maj = pd.Series(predictions).value_counts().index[0]
    end = time.time()
    print(f'Finished Prediction {row[1].name} in {end - start}')
    return prediction_maj

def classifyRandomForest(trees, dataframe, class_col):
    actual = list(dataframe[class_col])
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(predictRow, it.repeat(trees), dataframe.iterrows())
        predictions = list(results)
    return actual, predictions

def createRandomForest(dataframe, attributes, class_col, num_trees, num_attr, num_data_points, tree_thres):
    if num_attr > len(attributes):
        sys.exit('ERROR: num_attr > len(attributes)')
    if num_data_points > len(dataframe):
        sys.exit('ERROR: num_data_points > len(dataframe)')

    trees = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(createForestTree, dataframe, attributes, class_col, num_attr, num_data_points, tree_thres, i) for i in range(num_trees)]
        for result in concurrent.futures.as_completed(results):
            trees.append(result.result())
    return trees

