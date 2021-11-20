import sys
import pandas as pd

def parseClassEvalArgs(args):
    # check if length args = 2
    if len(args) != 2:
        print("Usage: classEvaluation.py <vector_file> <sim_metric> <classify_type>")
        sys.exit(1)
    if not args[0].endswith('.csv') or not args[1].endswith('.csv'):
        print('Error: Input files must be csv files')
        print('Usage: python3 classiferEvaluation.py <predictions_path> <ground_truth_path>, <classify_type>')
        sys.exit(1)
    return args[0], args[1]

def outputStats(matrix, author_stats, overall_correct, overall_incorrect, overall_accuracy, classify_type, input_file_path):
    if classify_type == 'knn':
        # remove the .csv from the end of the input file path
        file_strings = input_file_path.split('.')[0].split('/')[-1].split('_')[1:]
        stats_file_path = 'eval_outputs/knn_stats_' + str(file_strings[0]) + '_' + str(file_strings[1]) + '.txt'
        matrix_file_path = 'eval_outputs/knn_matrix_' + str(file_strings[0]) + '_' + str(file_strings[1]) + '.csv'

        file = open(stats_file_path, 'w')
        matrix_file = open(matrix_file_path, 'w')
    elif classify_type == 'rf':
        file_strings = input_file_path.split('.')[0].split('/')[-1].split('_')[1:]
        stats_file_path = 'eval_outputs/rf_stats_' + str(file_strings[0]) + '_' + str(file_strings[1]) + '_' + str(file_strings[2]) + '.txt'
        matrix_file_path = 'eval_outputs/rf_matrix_' + str(file_strings[0]) + '_' + str(file_strings[1]) + '_' + str(file_strings[2]) + '.csv'

        file = open(stats_file_path, 'w')
        matrix_file = open(matrix_file_path, 'w')
    else:
        print('Error: classify_type must be knn or rf')
        sys.exit(1)
    
    file.write("===AUTHOR INFORMATION===\n")
    for author, author_info in author_stats.items():
        file.write('Author: ' + str(author) + '\n')
        file.write('-----' + '\n')
        file.write('Hits: ' + str(author_info['hits']) + '\n')
        file.write('Misses: ' + str(author_info['misses']) + '\n')
        file.write('Strikes: ' + str(author_info['strikes']) + '\n')
        file.write('Precision: ' + str(author_info['precision']) + '\n')
        file.write('Recall: ' + str(author_info['recall']) + '\n')
        file.write('F1: ' + str(author_info['f1']) + '\n')
        file.write('\n')
    
    file.write("===CLASSIFIER STATISTICS===\n")
    file.write('Overall Accuracy: ' + str(overall_accuracy) + '\n')
    file.write('Overall Correct: ' + str(overall_correct) + '\n')
    file.write('Overall Incorrect: ' + str(overall_incorrect) + '\n')

    file.close()

    # ROWS ARE ACTUAL COLUMNS ARE PREDICTED
    matrix.to_csv(matrix_file, index=True, index_label='Author')

def calculateStats(dataframe, classify_type, input_file_path):
    overall_correct = 0
    overall_incorrect = 0
    overall_accuracy = 0

    matrix = pd.crosstab(dataframe['actual'], dataframe['predicted'])
    # , rownames=['ACTUAL'], colnames=['PREDICTED']
    # print(matrix)

    author_stats = {}
    authors = dataframe['actual'].unique()

    for author in authors:
        actual_author_df = dataframe[dataframe['actual'] == author]
        predicted_author_df = dataframe[dataframe['predicted'] == author]

        author_info = {}
        author_info['hits'] = len(actual_author_df[actual_author_df['actual'] == actual_author_df['predicted']])
        author_info['misses'] = len(actual_author_df[actual_author_df['actual'] != actual_author_df['predicted']])
        author_info['strikes'] = len(predicted_author_df[predicted_author_df['actual'] != predicted_author_df['predicted']])

        if author_info['hits'] + author_info['strikes'] == 0:
            author_info['precision'] = 0
        else:
            author_info['precision'] = author_info['hits'] / (author_info['hits'] + author_info['strikes'])
        if author_info['hits'] + author_info['misses'] == 0:
            author_info['recall'] = 0
        else:
            author_info['recall'] = author_info['hits'] / (author_info['hits'] + author_info['misses'])
        if author_info['precision'] + author_info['recall'] == 0:
            author_info['f1'] = 0
        else:
            author_info['f1'] = 2 * (author_info['precision'] * author_info['recall'] / (author_info['precision'] + author_info['recall']))

        overall_correct += author_info['hits']
        overall_incorrect += author_info['misses']

        author_stats[author] = author_info
    
    overall_accuracy = overall_correct / (overall_correct + overall_incorrect)

    outputStats(matrix, author_stats, overall_correct, overall_incorrect, overall_accuracy, classify_type, input_file_path)


if __name__ == '__main__':
    input_file_path, ground_truth_path = parseClassEvalArgs(sys.argv[1:])

    file_path_split = input_file_path.split('/')
    if file_path_split[0] == 'KNNOutput':
        classify_type = 'knn'
    elif file_path_split[0] == 'RFOutput':
        classify_type = 'rf'
    else:
        print("Error: Input file must be in directory KNNOutput/ or RFOutput/")
        sys.exit(1)

    pred_df = pd.read_csv(input_file_path)
    dataframe = pd.read_csv(ground_truth_path)

    predictions_extract = pred_df['author']
    dataframe.insert(1, 'predicted', predictions_extract)
    dataframe = dataframe.rename(columns={'author': 'actual'})

    calculateStats(dataframe, classify_type, input_file_path)
