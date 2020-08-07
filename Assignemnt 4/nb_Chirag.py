import csv
import math
import sys
import argparse
import numpy as np

ap = argparse.ArgumentParser()
# Add the arguments to the parser
ap.add_argument("-d", "--data", required=True, help="data")
ap.add_argument("-o", "--output", required=True, help="output")
args = vars(ap.parse_args())
filename = args['data']
outputfile = args['output']

with open(filename) as tsv_file:
    data_frame = list(csv.reader(tsv_file, delimiter='\t'))

data_arr = np.array(data_frame)
x = data_arr[:,1:3]
y = data_arr[:,0]

y_unique, count = np.unique(y, return_counts=True)
prob = np.divide(count, sum(count))

x_shape = np.shape(x)
att = x_shape[1]
class_size = np.shape(y_unique)

m = np.array(np.zeros([class_size[0], att]))
sig = np.array(np.zeros([class_size[0], att]))

x = x.astype(float)

#calculating the value of Mu
for i in range(0, class_size[0]):
    for j in range(0, att):
        if (i == 0):
            for k in range(0, count[0]):
                m[i, j] = m[i, j] + x[k, j]
            m[i, j] = m[i, j] / count[0]
        if (i == 1):
            for k in range(count[0], sum(count)):
                m[i, j] = m[i, j] + x[k, j]
            m[i, j] = m[i, j] / count[1]

#calculating the value of Sigma
for i in range(0, class_size[0]):
    for j in range(0, att):
        if (i == 0):
            for k in range(0, count[0]):
                sig[i, j] = sig[i, j] + ((x[k, j] - m[i, j]) * (x[k, j] - m[i, j]))
            sig[i, j] = sig[i, j] / (count[0] - 1)
        if (i == 1):
            for k in range(count[0], sum(count)):
                sig[i, j] = sig[i, j] + ((x[k, j] - m[i, j]) * (x[k, j] - m[i, j]))
            sig[i, j] = sig[i, j] / (count[0] - 1)


#calculating the various probabilities
probability = np.zeros([att, 1])
probability_Unique = np.zeros([y_unique.shape[0], 1])
classified = np.zeros([x_shape[0], 1])
for i in range(0, x_shape[0]):
    for c in range(0, (y_unique.shape)[0]):
        for a in range(0, att):
            b = x[i, :]
            probability[a] = (1 / math.sqrt(2 * math.pi * sig[c, a])) * (
                math.exp(-np.square(b[a] - m[c, a]) / (2 * sig[c, a])))
        probability_Unique[c] = np.prod(probability)
    classified[i] = np.argmax(probability_Unique)

Class = np.zeros([x_shape[0], 1])
for i in range(0, x_shape[0]):
    if y[i] == 'A':
        Class[i] = 0
    else:
        Class[i] = 1

misclassified, n = np.unique(np.equal(Class, classified), return_counts=True)

temp = np.equal(Class, classified)
temp = temp.astype(int)
misclassified_new = x_shape[0] - sum(temp)

with open(outputfile, 'w', newline='')as tsv_file:
    writerObject = csv.writer(tsv_file, delimiter='\t')
    writerObject.writerow((m[0, 0], sig[0, 0], m[0, 1], sig[0, 1], prob[0]))
    writerObject.writerow((m[1, 0], sig[1, 0], m[1, 1], sig[1, 1], prob[1]))
    writerObject.writerow(misclassified_new)