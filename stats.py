import numpy as np
import sys

m = np.loadtxt(sys.argv[1])

runs = int(m[:,0].max())+1

max_accuracies = np.empty(int(runs))
epochs_to_max_accuracies = np.empty(int(runs))
for run in range(int(runs)):
	i = m[m[:,0]==run][:,3].argmax()
	max_test_accuracies[run] = m[m[:,0]==run][i,3]
	epochs_to_max_test_accuracies[run] = m[m[:,0]==run][i,1]

	i = m[m[:,0]==run][:,4].argmax()
	max_held_out_test_accuracies[run] = m[m[:,0]==run][i,4]
	epochs_to_max_held_out_test_accuracies[run] = m[m[:,0]==run][i,1]

print("Average max test accuracy %.2f +/- %.2f" % (max_test_accuracies.mean(), 1.96 * max_test_accuracies.std() / np.sqrt(runs)))
print("Average epochs to max test accuracy %.2f +/- %.2f" % (epochs_to_max_test_accuracies.mean(), 1.96 * epochs_to_max_test_accuracies.std() / np.sqrt(runs)))

print("Average max held out test accuracy %.2f +/- %.2f" % (max_held_out_test_accuracies.mean(), 1.96 * max_held_out_test_accuracies.std() / np.sqrt(runs)))
print("Average epochs to max held out test accuracy %.2f +/- %.2f" % (epochs_to_max_held_out_test_accuracies.mean(), 1.96 * epochs_to_max_held_out_test_accuracies.std() / np.sqrt(runs)))
