import argparse
import csv
import numpy as np


def main():
    args = parser.parse_args()
    file, eta, threshold = args.data, float(args.eta), float(
        args.threshold)  # save respective command line inputs into variables

    # read csv file and the last column is the target output and is separated from the input (X) as Y
    with open(file) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        X = []
        Y = []
        for row in reader:
            X.append([1.0] + row[:-1])
            Y.append([row[-1]])

    # Convert data points into float and initialise weight vector with 0s.
    n = len(X)
    X = np.array(X).astype(float)
    Y = np.array(Y).astype(float)
    W = np.zeros(X.shape[1]).astype(float)

    # this matrix is transposed to match the necessary matrix dimensions for calculating dot product
    W = W.reshape(X.shape[1], 1).round(4)

    # Calculate the predicted output value
    f_x = calculatePredicatedValue(X, W)

    # Calculate the initial SSE
    sse_old = calculateSSE(Y, f_x)

    # ----------------- python student.py --data random1.csv --eta 0.0001 --threshold 0.0001 ------------------

    print(*[0], *["{0:.4f}".format(val) for val in W.T[0]], *["{0:.4f}".format(sse_old)], sep=',')

    gradient, W = calculateGradient(W, X, Y, f_x, eta)

    iteration = 1
    while True:
        f_x = calculatePredicatedValue(X, W)
        sse_new = calculateSSE(Y, f_x)

        if abs(sse_new - sse_old) > threshold:
            print(*[iteration], *["{0:.4f}".format(val) for val in W.T[0]], *["{0:.4f}".format(sse_new)], sep=',')
            gradient, W = calculateGradient(W, X, Y, f_x, eta)
            iteration += 1
            sse_old = sse_new
        else:
            break
    print(*[iteration], *["{0:.4f}".format(val) for val in W.T[0]], *["{0:.4f}".format(sse_new)], sep=',')


def calculateGradient(W, X, Y, f_x, eta):
    gradient = (Y - f_x) * X
    gradient = np.sum(gradient, axis=0)
    # gradient = np.array([float("{0:.4f}".format(val)) for val in gradient])
    temp = np.array(eta * gradient).reshape(W.shape)
    W = W + temp
    return gradient, W


def calculateSSE(Y, f_x):
    sse = np.sum(np.square(f_x - Y))
    return sse


def calculatePredicatedValue(X, W):
    f_x = np.dot(X, W)
    return f_x


# initialise argument parser and read arguments from command line with the respective flags
# and then call the main() function

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Data File")
    parser.add_argument("-l", "--eta", help="Learning Rate")
    parser.add_argument("-t", "--threshold", help="Threshold")
    main()
