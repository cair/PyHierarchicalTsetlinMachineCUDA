from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
import argparse

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--clauses", default=2000, type=int)
    parser.add_argument("--T", default=6400, type=int)
    parser.add_argument("--s", default=50.0, type=float)
    parser.add_argument("--board_dim", default=10, type=int)
    parser.add_argument("--boost", default=1, type=int)
    parser.add_argument("--or_alternatives", default=40, type=int)
    parser.add_argument("--or_alternatives_1", default=8, type=int)
    parser.add_argument("--or_alternatives_2", default=8, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


args = default_args()

data = np.loadtxt("./examples/hex_data.txt").astype(np.uint32)
X_train = data[:int(len(data)*0.8),0:-1]
Y_train = data[:int(len(data)*0.8),-1]

X_test = data[int(len(data)*0.8):,0:-1]
Y_test = data[int(len(data)*0.8):,-1]

tsetlin_machine = TsetlinMachine(args.clauses, args.T, args.s, number_of_state_bits=8, boost_true_positive_feedback=args.boost, hierarchy_structure=((tm.AND_GROUP, (args.board_dim/2)**2 * 2), (tm.OR_ALTERNATIVES, args.or_alternatives), (tm.AND_GROUP, 4)))
#tsetlin_machine = TsetlinMachine(args.clauses, args.T, args.s, number_of_state_bits=8, boost_true_positive_feedback=args.boost, hierarchy_structure=((tm.AND_GROUP, args.board_dim * args.board_dim * 2), (tm.AND_GROUP, 1)))
#tsetlin_machine = TsetlinMachine(args.clauses, args.T, args.s, weighted_clauses=True, number_of_state_bits=8, boost_true_positive_feedback=args.boost, hierarchy_structure=((tm.AND_GROUP, 8), (tm.OR_ALTERNATIVES, args.or_alternatives), (tm.AND_GROUP, 9), (tm.OR_ALTERNATIVES, args.or_alternatives), (tm.AND_GROUP, 4)))
#tsetlin_machine = TsetlinMachine(args.clauses, args.T, args.s, number_of_state_bits=8, boost_true_positive_feedback=args.boost, hierarchy_structure=((tm.AND_GROUP, 8), (tm.AND_GROUP, 9), (tm.OR_ALTERNATIVES, args.or_alternatives), (tm.AND_GROUP, 4)))
#tsetlin_machine = TsetlinMachine(args.clauses, args.T, args.s, number_of_state_bits=8, boost_true_positive_feedback=args.boost, hierarchy_structure=((tm.AND_GROUP, 8), (tm.OR_ALTERNATIVES, args.or_alternatives), (tm.AND_GROUP, 9), (tm.AND_GROUP, 4)))
tsetlin_machine = TsetlinMachine(args.clauses, args.T, args.s, weighted_clauses=False, number_of_state_bits=8, boost_true_positive_feedback=args.boost, hierarchy_structure=((tm.AND_GROUP, 18), (tm.OR_ALTERNATIVES, args.or_alternatives_1), (tm.AND_GROUP, 4), (tm.OR_ALTERNATIVES, args.or_alternatives_2), (tm.AND_GROUP, 4)))

print("\nAccuracy over 1000 epochs:\n")
for e in range(args.epochs):
	start_training = time()
	tsetlin_machine.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tsetlin_machine.predict(X_test) == Y_test).mean()
	stop_testing = time()

	#tm.print_hierarchy()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (e+1, result, stop_training-start_training, stop_testing-start_testing))
