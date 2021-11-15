import sys

def parseRFAuthorshipArgs(args):
    if len(args) != 6:
        print("Usage: RFAuthorship.py <vector_file> <sim_metric> <num_trees> <num_attr> <num_data_points> <thres>")
        sys.exit(1)
    return args[0], args[1], int(args[2]), int(args[3]), int(args[4]), float(args[5])

if __name__ == '__main__':
    vector_file, sim_metric, num_trees, num_attr, num_data_points, thres = parseRFAuthorshipArgs(sys.argv[1:])
