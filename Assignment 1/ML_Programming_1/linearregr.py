import argparse
import numpy as np
import pandas as pd

def add_row(W, SSE, df):
	W = np.round(W, 4)
	row = []
	weights = W.tolist()[0]
	for weight in weights:
		row.append(weight)

	row.append(SSE)
	df.loc[len(solution)] = row

	return df


ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-d", "--data", required=True, help="data")
ap.add_argument("-l", "--learningRate", required=True, help="Learning Rate")
ap.add_argument("-t", "--threshold", required=True, help="Threshold")
args = vars(ap.parse_args())

print(args)

data = args['data']
LR = float(args['learningRate'])
threshold = float(args['threshold'])

input_data = pd.read_csv(data, header=None)

n = input_data.shape[0]     # number of rows
ones = np.ones([n,1])
npdata = np.concatenate((ones,input_data),axis=1)

X = npdata[:, :-1]   # Extracting the features

m = X.shape[1]

Y = npdata[:,-1][np.newaxis].T   # Extracting the target output

W = np.zeros([1,m])   # Initializing the weights

solution = pd.DataFrame(columns=[i for i in range(m+1)])

Y_pred = X @ W.T
SSE_curr = round(np.sum(np.power(Y-Y_pred, 2)),4)    # Calculation sum of squared errors
solution = add_row(W, SSE_curr, solution)

step = SSE_curr

while(step>=threshold):
	SSE_prev = SSE_curr
	P=X * (Y - X @ W.T)
	W = W + LR * np.sum(P, axis=0)    # Updating the weights
	Y_pred = X @ W.T
	SSE_curr = round(np.sum(np.power(Y - Y_pred, 2)),4)
	solution = add_row(W, SSE_curr, solution)
	SSE_curr = np.sum(np.power(Y - Y_pred, 2))
	step = SSE_prev - SSE_curr

solution.to_csv('solution_{}_lr{}_thres{}.csv'.format(data.split('.')[0],LR,threshold), header = None)
