from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
import argparse

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--clauses", default=32, type=int)
    parser.add_argument("--T", default=250, type=int)
    parser.add_argument("--s", default=25.0, type=float)
    parser.add_argument("--q", default=1.0, type=float)
    parser.add_argument("--boost", default=0, type=int)
    parser.add_argument("--number_of_state_bits", default=8, type=int)
    parser.add_argument("--or_alternatives", default=3, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

train_data = np.loadtxt("./examples/NoisyParityTrainingData.txt").astype(np.uint32)
X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

test_data = np.loadtxt("./examples/NoisyParityTestingData.txt").astype(np.uint32)
X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

tm = TsetlinMachine(args.clauses, args.T, args.s, number_of_state_bits=args.number_of_state_bits, boost_true_positive_feedback=args.boost, hierarchy_structure=((tm.AND_GROUP, 3), (tm.OR_ALTERNATIVES, args.or_alternatives), (tm.AND_GROUP, 2), (tm.OR_ALTERNATIVES, args.or_alternatives), (tm.AND_GROUP, 2)))

print("\nAccuracy over %d epochs:\n" % (args.epochs))
for e in range(args.epochs):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=10, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	tm.print_hierarchy()

	print("\n#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (e+1, result, stop_training-start_training, stop_testing-start_testing))
