from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
from keras.datasets import mnist
import argparse

clauses = 2000
T = 50*100
s = 10.0

def default_args(**kwargs):
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default=250, type=int)
	parser.add_argument("--runs", default=100, type=int)
	parser.add_argument("--number-of-clauses", default=2000, type=int)
	parser.add_argument("--number-of-state-bits", default=8, type=int)
	parser.add_argument("--T", default=5000, type=int)
	parser.add_argument("--s", default=10.0, type=float)
	parser.add_argument("--q", default=1.0, type=float)
	parser.add_argument("--clause-weight-update-ratio", default=0.01, type=float)
	parser.add_argument("--constant-update-p", action='store_true')
	parser.add_argument('--binary-inference', action='store_true')
	parser.add_argument('--no-clipping', action='store_true')
	parser.add_argument('--literal-budget', default=0, type=int)

	args = parser.parse_args()
	for key, value in kwargs.items():
		if key in args.__dict__:
			setattr(args, key, value)
	return args

args = default_args()

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

tm = MultiClassTsetlinMachine(clauses,
	args.T,
	args.s,
	weighted_clauses=True,
	clause_weight_update_ratio=args.clause_weight_update_ratio,
	literal_budget=args.literal_budget,
	q=args.q,
	hierarchy_structure=((tm.AND_GROUP, 28*28), (tm.AND_GROUP, 1))
)

print("\nAccuracy over 500 epochs:\n")
for e in range(500):
	start_training = time()
	tm.fit(X_train, Y_train)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	clause_precision = tm.tms[0].clause_precision(X_test, Y_test)
	transformed_X = tm.tms[0].transform(X_test)
	high_precision_matches = 0
	max_precision = np.zeros(transformed_X.shape[0])
	for i in range(transformed_X.shape[0]):
		high_precision_match = False
		for j in range(tm.tms[0].number_of_clauses):
			if not high_precision_match and transformed_X[i, j] > 0 and clause_precision[j] >= 1.0:
				high_precision_matches += 1
				high_precision_match = True

			if transformed_X[i, j] > 0 and clause_precision[j] >= max_precision[i]:
				max_precision[i] = clause_precision[j]

	print("#%d Test Accuracy: %.2f%% (%.2f%%/%.2f%%/%.2f%%/%.2f%%/%.2f%% %.2f%%) Training Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" %
			(e+1, result_test, 100.0 * np.percentile(max_precision, 1), 100.0 * np.percentile(max_precision, 10), 100.0 * np.median(max_precision), 100.0 * np.percentile(max_precision, 90), 100.0 * np.percentile(max_precision, 99), 100.0 * high_precision_matches / transformed_X.shape[0], result_train, stop_training-start_training, stop_testing-start_testing))
