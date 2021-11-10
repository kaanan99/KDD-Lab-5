import pandas as pd
import numpy as np
import heapq
import math
import argparse
import classify
import evaluate
import time

class Item:

    def __init__(self, distance, class_val):
        self.distance = distance
        self.class_val = class_val

    def __lt__(self, other):
        return self.distance < other.distance


def normalize(x, max_val, min_val):
    '''
    Parameters:
       x - the value which is being normalized
       max_val - the maximum value in the column
       min_val - the minimum value in the column
    Returns:
       A numerical value which represents the normalized x
    '''
    return (x - min_val) / (max_val - min_val)


def get_data(csv_path):
    '''
    Parameters:
       csv_path - full path to the csv data file
    Returns:
       D - The data frame with numerical values normalized
       num_attrbs - A list containing names of numerical attributes
       cat_attrbs - A list containing ganmes of categorical attributes
       class_col - The column containing the class column
    '''
    # Read in Data
    D = pd.read_csv(csv_path)

    # Initialize attribute colums
    num_attrbs = []
    cat_attrbs = []

    # Adding column label to appropriate place
    class_col = D.iloc[1, 0]
    for col in D.columns:
        if col != class_col:
            if int(D.iloc[0][col]) == 0:
                num_attrbs.append(col)
            else:
                cat_attrbs.append(col)

    # Get arid of meta-data
    D = D.iloc[2:]

    # Normalize numerical attributes:
    for num_at in num_attrbs:
        D[num_at] = pd.to_numeric(D[num_at])
        max_val = max(D[num_at])
        min_val = min(D[num_at])
        D[num_at] = D[num_at].apply(normalize, args=[max_val, min_val])

    return D, num_attrbs, cat_attrbs, class_col


def eucledian_distance(val1, val2):
    '''
    Parameters:
       val1- A numerical value
       val2- A numerical value
    Returns:
       The distance between both values
    '''
    return math.sqrt((val1 - val2) ** 2)


def anti_dice(val1, val2, cat_attrbs):
    '''
    Parameters:
       val1 - A Dataframe of one row which will be used to find distance
       val2 - A Dataframe of one row which will be used to find distance
       cat_attrbs - A list of categorical attributes
    Returns:
       A ratio of total mismatches over sum of length of values
    '''
    mismatch = 0
    for category in cat_attrbs:
        if val1[category] != val2[category]:
            mismatch += 1
    return mismatch / (len(val1) + len(val2))


def calculate_distance(val1, val2, cat_attrbs, num_attrbs):
    '''
    Parameters:
       val1 - A Dataframe of one row which will be used to find distance
       val2 - A Dataframe of one row which will be used to find distance
       cat_attrbs - A list of categorical attributes
       num_attrbs - A list of numerical attributes
    Returns:
       The distance between the two points
    '''
    num_len = len(num_attrbs)
    cat_len = len(cat_attrbs)
    total_len = num_len + cat_len
    numeric_distance = 0
    for num_var in num_attrbs:
        numeric_distance += eucledian_distance(val1[num_var], val2[num_var])
    categorical_distance = anti_dice(val1, val2, cat_attrbs)
    return ((num_len / total_len) * numeric_distance) + ((cat_len / total_len) * categorical_distance)


def predict_one(val1, D, cat_attrbs, num_attrbs, k, class_col):
    '''
    Parameters:
       val1: base value from which distances are being calculated
       D: Dataframe not including val1 of values to be used to calcualte distance
       cat_attrbs: List of categorical attributes
       num_attrbs: List of numerical attributes
       k: Number of nearest neighbors being compared
       class_col: Name of the class column
    Return:
       Prediction for what class val1 belongs to

    '''
    heap = []
    for i in range(D.shape[0]):
        distance = calculate_distance(val1, D.iloc[i], cat_attrbs, num_attrbs)
        heapq.heappush(heap, Item(distance, D.iloc[i][class_col]))
    j = k
    common_attrbs = {}
    while j >= 0:
        popped = heapq.heappop(heap)
        if popped.class_val in common_attrbs:
            common_attrbs[popped.class_val] += 1
        else:
            common_attrbs[popped.class_val] = 1
        j -= 1
    max_class = None
    max_val = -1
    for x in common_attrbs.keys():
        if common_attrbs[x] > max_val:
            max_val = common_attrbs[x]
            max_class = x
    return max_class


def knn(D, cat_attrbs, num_attrbs, k, class_col):
    predictions = []
    for i in range(D.shape[0]):
        val1 = D.iloc[i]
        df = pd.concat([D.iloc[:i], D.iloc[i:]])
        prediction = predict_one(val1, df, cat_attrbs, num_attrbs, k, class_col)
        predictions.append(prediction)
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Predict observations using K nearest neighbors with given K value and Dataset")
    parser.add_argument("csv_path", help="The filepath to the training data csv file")
    parser.add_argument("-k", required=True, type=int,
                        help="The filepath to the json file representing the decision tree")

    args = parser.parse_args()
    
    D, num_attrbs, cat_attrbs, class_col = get_data(args.csv_path)
    start = time.time()
    predictions = knn(D, cat_attrbs, num_attrbs, args.k, class_col)
    print(time.time()- start)

    # output pt-by-pt predictions to results.txt
    evaluate.output_predictions(args.csv_path, D[class_col], predictions,
                       f"knn-k{args.k}")

    matrix = classify.make_matrix(predictions, D[class_col])
    print(matrix)
    the_sum = 0
    for x in range(len(predictions)):
       if predictions[x] == D[class_col].iloc[x]:
          the_sum += 1
    print("Average accuracy:", the_sum / len(predictions))


if __name__ == "__main__":
    main()
