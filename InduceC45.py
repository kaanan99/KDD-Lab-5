import math
import pandas as pd
import numpy as np
import argparse
import json
import os
import time
from tree import *

DEBUG = False

FIND_MOST_FREQ_TIME = 0
FIND_MOST_FREQ_NUM = 0
SEL_SPLIT_TIME = 0
SEL_SPLIT_NUM = 0
FIND_BEST_SPLIT_TIME = 0
FIND_BEST_SPLIT_NUM = 0
ENT_TIME = 0
ENT_NUM = 0
ENT_AI_TIME = 0
ENT_AI_NUM = 0
ENT_NUMER_TIME = 0
ENT_NUMER_NUM = 0


def get_data(csv_path, restrictions):
    """
    Parse the data and metadata.
    Params:
        csv_path - the full path to the data csv file: first row is column names; second row is number of possible values for each column; third row's first cell is name of class column; all following rows are data pts
        restrictions - the full path to the restrictions text file containing a single line of whitespace-separated 1's and 0's marking 'use' or 'don't use' this attribute; must be ordered the same as columns in the data file, considering the class column removed
    Returns:
        D - whole dataset (without metadata rows) as DataFrame
        attrb_info - a dict of attribute names to it's list of possible values
        class_col - the column name of the class label column
        vals_per_attrb - a Series containing the number of unique values per attribute;
            an attribute's value count is 0 if the attribute is continuous
    """
    D = pd.read_csv(csv_path)
    #print(D)

    # get the class column name from the second row
    class_col = D.iloc[1, 0]

    # get the number of attribute values per attribute
    vals_per_attrb = pd.to_numeric(D.iloc[0], downcast="integer")

    # remove the first two metadata rows from D
    D = D.iloc[2:]

    # get a list of only attribute column names
    attrbs = list(D.columns)
    if not pd.isnull(class_col):
        attrbs.remove(class_col)
    else:
        class_col = None

    # parse restrictions txt file
    if restrictions is not None:
        f = open(restrictions, 'r')
        restrxns = f.readline().strip().split()
        #print(restrxns)
        f.close()

        # select attribute columns to use based on restrictions
        attrbs = [A for i,A in enumerate(attrbs) if restrxns[i] == '1']

    # get list of all possible values for needed attributes
    attrb_info = {A: D[A].unique() for A in attrbs}

    # delete records with missing values before trying to convert anything
    D = D.replace('?', np.nan)
    D = D.dropna()

    # convert continuous data columns to numerical
    for A in attrbs:
        if vals_per_attrb.at[A] == 0:
            D[A] = pd.to_numeric(D[A], downcast="float")

    #print(attrbs)

    # randomize D
    D = D.sample(frac=1, replace=False)

    return D, attrb_info, class_col, vals_per_attrb


def count_data_with_value(data, col_name, value):
    """
    Return the number of data points with the given value in the given column.
    """
    return data.loc[data[col_name] == value].shape[0]


def find_most_freq_label(D, class_col):
    """
    Find the plurality class.
    Params:
        D - the dataset; a DataFrame
        class_col - the name of the class column
    Returns:
        the most frequent class label
        the frequency of the most frequent class label
    """
    global FIND_MOST_FREQ_TIME, FIND_MOST_FREQ_NUM
    start = time.time()
    # class_vals = D[class_col].unique()
    max_count = -1
    max_class = None
    for c in D[class_col].unique():
        count = count_data_with_value(D, class_col, c)
        if count > max_count:
            max_count = count
            max_class = c

    end = time.time()
    FIND_MOST_FREQ_TIME += end - start
    FIND_MOST_FREQ_NUM += 1

    return max_class, max_count


def entropy(D, col):
    """
    Calculate the entropy of a labeled dataset.
    Params:
        D - a pandas dataframe where each row entry is a the data point plus its label in the class column
        col - the column name to calc entropy for
    Return: the entropy (> 0) of the dataset given the class col
    """
    global ENT_TIME, ENT_NUM
    start = time.time()

    size_D = D.shape[0]

    # count the number of data points with each ith column value
    Di_sizes = np.array(list(map(lambda class_value: count_data_with_value(D, col, class_value), D[col].unique())))

    # calculate the probability of each ith column value by dividing by the total dataset size
    Ci_probs = Di_sizes / size_D
    Ci_probs = np.delete(Ci_probs, np.where(Ci_probs == 0))

    # calculate the entropy of each ith column value
    # Ci_ents = np.array(list(map(lambda prob_Ci: prob_Ci * math.log2(prob_Ci) if prob_Ci else 0, Ci_probs)))
    Ci_ents = np.multiply(Ci_probs, np.log2(Ci_probs))

    end = time.time()
    ENT_TIME += end - start
    ENT_NUM += 1
    # return the total entropy
    return -np.sum(Ci_ents)


def entropy_Ai(D, class_col, Ai):
    """
    Calculate the entropy that would result from splitting the data by a given attribute.
    Params:
        D - a pandas dataframe where each row entry is a the data point plus its label in the class column
        class_col - the name of the class label column
        Ai - the name of the attribute to consider splitting by
    Return: the entropy (> 0) that would result from splitting the data by the given attribute.
    """
    global ENT_AI_TIME, ENT_AI_NUM
    start = time.time()

    size_D = D.shape[0]

    # get the subsets of data with each jth attribute value
    Djs = list(map(lambda jth_value: D.loc[D[Ai] == jth_value], D[Ai].unique()))

    # count the number of data points with each jth attribute value
    Dj_sizes = np.array(list(map(lambda Dj: Dj.shape[0], Djs)))

    # calculate the probability of each jth value by dividing by the total dataset size
    # Cj_probs = Dj_sizes / size_D

    # calculate the entropy of each jth value
    # Cj_ents = np.array(list(map(lambda prob_Cj, Dj: prob_Cj * entropy(Dj, class_col), Cj_probs, Djs)))
    ent_Djs = np.array(list(map(lambda Dj: entropy(Dj, class_col), Djs)))
    Cj_ents = np.multiply(Dj_sizes / size_D, ent_Djs)

    end = time.time()
    ENT_AI_TIME += end - start
    ENT_AI_NUM += 1

    # return the total entropy
    return np.sum(Cj_ents)


def entropy_numer_split(D, attrb, left_counts, class_totals):
    """
    Calculate the CLASS LABEL entropy that would result from splitting data D by the given
        continuous attribute if selecting an alpha resulting in the provided class distribution.
    Params:
        D - the (subset) of data; a DataFrame
        attrb - the name of the continuous attribute
        left_counts - a NumPy array (length = # unique class labels in D) of counts;
            the ith count is the number of data points in D with attrb-val <= alpha
            (externally chosen) AND that are labeled with class label i
        class_totals - a NumPy array (length = # unique class labels in D) of counts;
            the ith count is the total number of data points in D labeled with class label i;
            MUST be in the same order (by class label) as left_counts!
    Return: the entropy (> 0) of splitting D by attrb given the class distribution represented by the counts
    """
    global ENT_NUMER_TIME, ENT_NUMER_NUM
    start = time.time()

    D_size = D.shape[0] # total number of data points in D
    left_count_total = np.sum(left_counts)   # total num data pts w/ attrb-val <= alpha
    right_count_total = D_size - left_count_total   # total num data pts w/ attrb-val > alpha

    # get the probabilities of value being left/right of each alpha
    # left_probs = np.array(list(map(lambda left_count: left_count / left_count_total, left_counts)))
    left_probs = left_counts / left_count_total
    left_probs = np.delete(left_probs, np.where(left_probs == 0))

    # right_probs = np.array(list(map(lambda left_count, class_total:
    #                                (class_total - left_count) / right_count_total,
    #                                left_counts, class_totals)))
    right_probs = (class_totals - left_counts) / right_count_total
    right_probs = np.delete(right_probs, np.where(right_probs == 0))

    # get the entropies of value being left/right of each alpha
    # left_entropies = np.array(list(map(lambda left_prob: left_prob * math.log2(left_prob) if left_prob else 0,
    #                                    left_probs)))
    left_entropies = np.multiply(left_probs, np.log2(left_probs))
    # right_entropies = np.array(list(map(lambda right_prob: right_prob * math.log2(right_prob) if right_prob else 0,
    #                                    right_probs)))
    right_entropies = np.multiply(right_probs, np.log2(right_probs))

    # mult each entropy half by the proportion of the dataset they compose, then return the negative sum
    res = (-1) * (np.sum(left_entropies) * left_count_total / D_size +
                   np.sum(right_entropies) * right_count_total / D_size)
    # print(f"Entropy numer split = {res}")

    end = time.time()
    ENT_NUMER_TIME += end - start
    ENT_NUMER_NUM += 1

    return res


def find_best_split(attrb, D, class_col, D_ent):
    """
    Find the best alpha value by which to split the given continuous attribute.
    Params:
        attrb - the name of the continuous attribute
        D - the (subset) of data; a DataFrame
        class_col - the class column name
        D_ent - the entropy of D (before doing any splitting)
    Returns:
        best_alpha - the best alpha value for splitting the given attribute
        max_gain - the information gain resulting from splitting the given attribute by the best alpha
        denom_entropy - the entropy of this attribute itself (used as the denominator for info gain ratio)
    """
    global FIND_BEST_SPLIT_TIME, FIND_BEST_SPLIT_NUM
    start = time.time()

    D_size = D.shape[0]
    class_labels = D[class_col].unique()    # all unique class labels appearing in this data subset
    
    # array of counts of data points with each class label
    class_totals = np.array(list(map(lambda label: count_data_with_value(D, class_col, label), class_labels)))

    alphas = sorted(D[attrb].unique())      # sorted list (asc) of unique attrb (alpha) values

    # return early if homogenous set, else rm last element (pointless)
    if len(alphas) == 1:
        return alphas[0], 1, 0
    alphas = alphas[:-1]

    # get the subsets of data with <= each ith alpha
    Dis = list(map(lambda alpha: D.loc[D[attrb] <= alpha], alphas))

    # count the number of data points with value <= each ith alpha
    Di_sizes = np.array(list(map(lambda Di: Di.shape[0], Dis)))

    # calculate the probability of <= each ith alpha by dividing by the total dataset size
    Ci_probs = Di_sizes / D_size
    Ci_probs = np.delete(Ci_probs, np.where(Ci_probs == 0))

    # calculate the entropy of <= each ith alpha
    # Ci_ents = np.array(list(map(lambda prob_Ci: prob_Ci * math.log2(prob_Ci) if prob_Ci else 0, Ci_probs)))
    Ci_ents = np.multiply(Ci_probs, np.log2(Ci_probs))

    # entropy of the attribute itself (denom for info gain ratio)
    entropy_alphas = -np.sum(Ci_ents)

    # for each class label, count how many data points have that class label AND have Ai-val <= current alpha
    counts = list(map(lambda alpha, Di: np.array(list(map(lambda label: Di.loc[Di[class_col] == label].shape[0],
                                            class_labels))),
                      alphas, Dis))

    # calculate the information gain for each ith alpha split
    gains = np.array(list(map(lambda per_alpha_class_counts:
                              D_ent - entropy_numer_split(D, attrb, per_alpha_class_counts, class_totals),
                              counts)))
    # print(f"Find best split of attrb '{attrb}'\n  alphas: {alphas}\n  left counts: {counts}\n  gains: {gains}")
    # find the max gain and the alpha that causes it; return
    max_idx = np.argmax(gains)
    max_gain = gains[max_idx]
    best_alpha = alphas[max_idx]

    end = time.time()
    FIND_BEST_SPLIT_TIME += end - start
    FIND_BEST_SPLIT_NUM += 1

    return best_alpha, max_gain, entropy_alphas


def select_splitting_attrb(D, class_col, attrbs, ratio_thr, vals_per_attrb):
    """
    Select the next attribute to split by.
    Params:
        D - a pandas dataframe where each row entry is a the data point plus its label in the class column
        class_col - the name of the class label column
        attrbs - a dict of attribute names to it's list of possible values
        ratio_thr - the information gain threshold
        vals_per_attrb - a Series containing the number of unique values per attribute;
            an attribute's value count is 0 if the attribute is continuous
    Return:
        None, None if no attribute meets the threshold for splitting
        attrb_name, best_alpha if the best splitting attribute is continuous
        attrb_name, None if the best splitting attribute is categorical
    """
    global SEL_SPLIT_TIME, SEL_SPLIT_NUM
    # print("***SELECT SPLIT CALLED***")
    start = time.time()
    ent_D = entropy(D, class_col)   # entropy before splitting by anything

    ratios = {}
    best_alphas = {}    # a dictionary of {continuous-attrb-name: best-alpha-val}
    #print("\n Looking for best attribute")
    for i, Ai in enumerate(attrbs):
        # calc info gain and the denominator for calculating info gain ratio
        if vals_per_attrb.at[Ai] == 0:  # if continuous
            alpha, gain_Ai, denom = find_best_split(Ai, D, class_col, ent_D)    # find best alpha to split by
            best_alphas[Ai] = alpha
        else:    # if categorical
            gain_Ai = ent_D - entropy_Ai(D, class_col, Ai)
            denom = entropy(D, Ai)  # denom (to get gain ratio) is the entropy of the attrb rather than class column

        # prevent dividing by 0
        if denom == 0:
            ratios[Ai] = -1
            continue

        # store the information gain ratio for this attribute
        ratios[Ai] = gain_Ai / denom
        #print("Attrb:", Ai, "ratio:", ratios[Ai])
    # find the attribute with the max info gain ratio
    max_ratio_attrb = max(ratios, key=ratios.get)
    #print(f"Attrb {max_ratio_attrb} found w/ max in {ratios}")

    end = time.time()
    SEL_SPLIT_TIME += end - start
    SEL_SPLIT_NUM += 1

    # return variables accordingly based on what kind of attribute was selected (if any)
    if ratios[max_ratio_attrb] < ratio_thr:             # done splitting
        return None, None
    if DEBUG:
        print(f"SELECT SPLIT '{max_ratio_attrb}' out of {ratios}")
    if vals_per_attrb.at[max_ratio_attrb] == 0:    # continuous
        return max_ratio_attrb, best_alphas[max_ratio_attrb]
    return max_ratio_attrb, None                        # categorical


def c45(D, attrbs, thr, thr_is_ratio, class_col, vals_per_attrb):
    """
    Run the C45 algorithm to build a decision tree on the given data.
    Params:
        D - the dataset; a DataFrame
        attrbs - a dictionary of attribute info {attrb_name: list of unique values}
        thr - the information gain / ratio threshold below which to longer split data
        thr_is_ratio - use info gain ratio if True, else use absolute info gain
        class_col - the name of the class column
        vals_per_attrb - a Series containing the number of unique values per attribute;
            an attribute's value count is 0 if the attribute is continuous
    """
    # TODO: implement check based on abs info gain? i.e. use thr_is_ratio
    # check termination condition: end if homogenous
    classes = D[class_col].unique()
    if len(classes) == 1:
        leaf = Node(True, classes[0], 1.0)
        # print(f"Made leaf as a result of CASE 1.")
        return leaf
        #T.set_node(leaf)

    # check if attribute list is empty
    elif len(attrbs.keys()) == 0:
        max_class, max_count = find_most_freq_label(D, class_col)
        leaf = Node(True, max_class, max_count/(D.shape[0]))
        # print(f"Made leaf as a result of CASE 2.")
        return leaf

    # select splitting attribute
    else:
        Ag, alpha = select_splitting_attrb(D, class_col, attrbs.keys(), thr, vals_per_attrb)
        #print(Ag)
        if Ag is None:  # no splitting attrb
            max_class, max_count = find_most_freq_label(D, class_col)
            leaf = Node(True, max_class, max_count/(D.shape[0]))
            # print(f"Made leaf as a result of CASE 3.")
            return leaf
        elif alpha is None: # categorical splitting attrb
            node = Node(False, Ag)
            #print(f"Node {Ag} is a leaf: {node.leaf}")
            #print(D)
            #print(f"Selected '{Ag}' as splitting attrb with unique vals {D[Ag].unique()}")

            # need an edge from current T to a new node for each val of Ag
            for val in attrbs[Ag]:
                Dv = D.loc[D[Ag] == val]
                if Dv.shape[0] != 0:
                    A = attrbs.copy()
                    A.pop(Ag)
                    Tv = c45(Dv, A, thr, thr_is_ratio, class_col, vals_per_attrb)
                    edge = Edge(val)
                    edge.set_node(Tv)
                    node.add_edge(edge)
                else:
                    max_class, max_count = find_most_freq_label(D, class_col)
                    leaf = Node(True, max_class, max_count/(D.shape[0]))
                    edge = Edge(val)
                    edge.set_node(leaf)
                    node.add_edge(edge)
                    # print(f"Made leaf as a result of CASE 4.")

            return node
        else:   # numerical splitting attrb
            node = Node(False, Ag)

            # left edge (<= alpha)
            Dv_left = D.loc[D[Ag] <= alpha]
            Tv_left = c45(Dv_left, attrbs, thr, thr_is_ratio, class_col, vals_per_attrb)
            edge_left = Edge(alpha)
            edge_left.set_numer_direction("le")
            edge_left.set_node(Tv_left)
            node.add_edge(edge_left)

            # right edge (> alpha)
            Dv_right = D.loc[D[Ag] > alpha]
            Tv_right = c45(Dv_right, attrbs, thr, thr_is_ratio, class_col, vals_per_attrb)
            edge_right = Edge(alpha)
            edge_right.set_numer_direction("gt")
            edge_right.set_node(Tv_right)
            node.add_edge(edge_right)
            # print(f"Made leaf as a result of CASE 5.")

            return node


def print_time_avg(func_name, time_sum, func_call_num):
    avg = 0 if func_call_num == 0 else time_sum / func_call_num
    print(f"  On avg, {func_name}() takes {avg:.4f}s and on the whole {time_sum}s")

def print_time_avgs():
    print_time_avg('find_most_freq_label', FIND_MOST_FREQ_TIME, FIND_MOST_FREQ_NUM)
    print_time_avg('entropy', ENT_TIME, ENT_NUM)
    print_time_avg('entropy_Ai', ENT_AI_TIME, ENT_AI_NUM)
    print_time_avg('entropy_numer_split', ENT_NUMER_TIME, ENT_NUMER_NUM)
    print_time_avg('find_best_split', FIND_BEST_SPLIT_TIME, FIND_BEST_SPLIT_NUM)
    print_time_avg('select_splitting_attrb', SEL_SPLIT_TIME, SEL_SPLIT_NUM)


def to_json(T, dataset_name, outpath):
    with open(outpath, 'w') as f:
        node_type = 'leaf' if T.leaf else 'node'
        json.dump({'dataset': dataset_name, node_type: T.get_dict()}, f)


def main():
    parser = argparse.ArgumentParser(description="Run C45 on a given set of data")
    parser.add_argument("csv_path")
    parser.add_argument('-r', '--restrictions', required=False, action='store', default=None, help='An optional attribute restriction/selector text file')
    parser.add_argument('-t', '--threshold', required=True, action='store', type=float, help="The information gain threshold. Can be absolute or ratio. If ratio, must also select --ratio.")
    parser.add_argument('--ratio', required=False, action='store_true', help="Select if the gain threshold given is a threshold on the information gain RATIO")

    args = parser.parse_args()

    D, attrbs, class_col, vals_per_attrb = get_data(args.csv_path, args.restrictions)

    T = c45(D, attrbs, args.threshold, args.ratio, class_col, vals_per_attrb)
    # print_time_avgs()
    #print(T.get_dict())
    csv = os.path.basename(args.csv_path)
    to_json(T, csv, os.path.splitext(csv)[0] + '.json')


if __name__ == "__main__":
    main()
