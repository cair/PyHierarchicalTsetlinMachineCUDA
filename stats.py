import numpy as np
import sys

m = np.loadtxt(sys.argv[1])

runs = int(m[:,0].max())+1

max_test_accuracies = np.empty(int(runs))
epochs_to_max_test_accuracies = np.empty(int(runs))
held_out_test_accuracies = np.empty(int(runs))

for run in range(int(runs)):
	i = m[m[:,0]==run][:,2].argmax()
	max_test_accuracies[run] = m[m[:,0]==run][i,2]
	held_out_test_accuracies[run] = m[m[:,0]==run][i,3]
	epochs_to_max_test_accuracies[run] = m[m[:,0]==run][i,1]

print("Average max test accuracy %.2f +/- %.2f" % (max_test_accuracies.mean(), 1.96 * max_test_accuracies.std() / np.sqrt(runs)))
print("Average held out test accuracy at max test accuracy %.2f +/- %.2f" % (held_out_test_accuracies.mean(), 1.96 * held_out_test_accuracies.std() / np.sqrt(runs)))
print("Average epochs to max test accuracy %.2f +/- %.2f" % (epochs_to_max_test_accuracies.mean(), 1.96 * epochs_to_max_test_accuracies.std() / np.sqrt(runs)))