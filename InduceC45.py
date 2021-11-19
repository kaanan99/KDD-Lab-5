from json import encoder
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class Node:
    def __init__(self, attr, leaf):
        self.leaf = leaf
        self.attr = attr
        self.edges = {}

class Edge:
    def __init__(self, value):
        self.value = value
        self.child = None

def classifyItem(tree, item):
    if tree.leaf:
        return tree.attr

    split_val = tree.edges['le'].value
    if item[tree.attr] <= split_val:
        return classifyItem(tree.edges['le'].child, item)
    else:
        return classifyItem(tree.edges['gt'].child, item)

def calcEntropyDF(dataframe, class_col):
    values, counts = np.unique(dataframe[class_col], return_counts=True)
    counts = counts / len(dataframe)
    entropy = -(np.sum(counts * np.log2(counts)))
    return entropy

def findBestSplit(dataframe, attr, class_col, entropy_parent):
    # start = time.time()
    alphas = sorted(dataframe[attr].unique())
    class_totals = dataframe[class_col].value_counts()
    class_vals = class_totals.index
    class_vals_counts = np.asarray(class_totals.values)
    dataframe_size = len(dataframe)

    # bulk of computation
    le_dfs = [dataframe[dataframe[attr] <= value] for value in alphas]
    le_dfs_sizes = np.array([[len(df)] for df in le_dfs])
    le_dfs_class_totals = np.asarray([[(df[class_col].values == class_val).sum() for class_val in class_vals] for df in le_dfs])

    gt_dfs_sizes = dataframe_size - le_dfs_sizes
    gt_dfs_class_totals = class_vals_counts - le_dfs_class_totals

    le_probs = (le_dfs_sizes / dataframe_size).ravel()
    gt_probs = (gt_dfs_sizes / dataframe_size).ravel()

    le_consts = np.divide(le_dfs_class_totals, le_dfs_sizes)
    gt_consts = np.divide(gt_dfs_class_totals, gt_dfs_sizes)
    # replace na with 0 in le_consts and gt_consts

    entropies_le = -np.sum((np.nan_to_num(le_consts * np.log2(le_consts))), axis=1)
    entropies_gt = -np.sum((np.nan_to_num(gt_consts * np.log2(gt_consts))), axis=1)
    info_gains = entropy_parent - ((le_probs * entropies_le) + (gt_probs * entropies_gt))

    # get index of max info gain
    max_info_gain_index = np.argmax(info_gains)
    # end = time.time()
    # print("Time to find best split: " + str(end - start))
    return (attr, alphas[max_info_gain_index], info_gains[max_info_gain_index], (entropies_le[max_info_gain_index], entropies_gt[max_info_gain_index]))

def selectSplitAttribute(attributes, dataframe, class_col, thres, entropy_parent):
    max_info_gain_tuple = None
    # compute info gain of each attribute
    start = time.time()
    for attr in attributes:
        alpha_info = findBestSplit(dataframe, attr, class_col, entropy_parent)
        if max_info_gain_tuple is None or alpha_info[2] > max_info_gain_tuple[2]:
            max_info_gain_tuple = alpha_info
    # if max info gain is less than threshold, return None
    # print("Split Attribute " + str(max_info_gain_tuple[0]) + " with info gain " + str(max_info_gain_tuple[2]))
    # end = time.time()
    # print("Time to find split attr: " + str(end - start))
    if max_info_gain_tuple[2] <= thres:
        # print("Create Leaf")
        return None, None, None
    return max_info_gain_tuple[0], max_info_gain_tuple[1], max_info_gain_tuple[3]

# attributes is list of attributes
def c45(attributes, dataframe, class_col, thres, entropy_parent=None):
    class_col_values = dataframe[class_col].unique()
    if entropy_parent is None:
        entropy_parent = calcEntropyDF(dataframe, class_col)
        # print(f"Number of unique class vals {len(class_col_values)}")

    if len(class_col_values) == 1:
        return Node(class_col_values[0], True)
    else:
        # split entropies = tuple(le entropy, gt_entropy)
        split_attr, alpha, split_entropies = selectSplitAttribute(attributes, dataframe, class_col, thres, entropy_parent)
        if split_attr == None:
            return Node(dataframe[class_col].mode()[0], True)
        else:
            new_node = Node(split_attr, False)

            le_df = dataframe[dataframe[split_attr] <= alpha]
            gt_df = dataframe[dataframe[split_attr] > alpha]
            new_edge_le = Edge(alpha)
            new_edge_gt = Edge(alpha)

            if len(le_df) == 0:
                new_edge_le.child = Node(dataframe[class_col].mode()[0], True)
                new_node.edges['le'] = new_edge_le
            else:
                new_edge_le.child = c45(attributes, le_df, class_col, thres, split_entropies[0])
                new_node.edges['le'] = new_edge_le
            if len(gt_df) == 0:
                new_edge_gt.child = Node(dataframe[class_col].mode()[0], True)
                new_node.edges['gt'] = new_edge_gt
            else:
                new_edge_gt.child = c45(attributes, gt_df, class_col, thres, split_entropies[1])
                new_node.edges['gt'] = new_edge_gt
            return new_node
