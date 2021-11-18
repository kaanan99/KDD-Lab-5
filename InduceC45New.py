import time
import numpy as np
import pandas as pd

# attributes is list of attributes
def c45(attributes, dataframe, class_col, thres):
    # if all values in dataframe with class_col are the same, return that value
    class_col_values = dataframe[class_col].unique()
    if len(class_col_values) == 1:
        # create leaf node with value of the class_col
    else:
        split_attr = selectSplitAttribute(attributes, dataframe, class_col, thres)
        if split_attr == None:
            # get most common value in class_col
            class_col_values = dataframe[class_col].unique()
            

