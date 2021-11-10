import sys
import os

def parseVectorizerArgs(args):
    # if there are not 2 arguments then error
    if len(args) != 2:
        print("Error:")
        print("Usage: python3 textVectorizer.py <directory_path> <output_name>")
        sys.exit(1)
    return args[0], args[1]

def createGroundTruth(dataset_path, output_path):
    with open(output_path + '.csv', 'w') as output_file:
        output_file.write('file_name,author\n')
        for dir in os.listdir(dataset_path):
            for author in os.listdir(dataset_path + '/' + dir):
                for file in os.listdir(dataset_path + '/' + dir + '/' + author):
                    output_file.write(file + ',' + author + '\n')
        output_file.close()

if __name__ == '__main__':
    dataset_path, output_path = parseVectorizerArgs(sys.argv[1:])
    createGroundTruth(dataset_path, output_path)


    print(dataset_path, output_path)