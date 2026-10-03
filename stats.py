import numpy as np
import sys

m = np.loadtxt(sys.argv[1])

runs = int(m[:,0].max())+1

max_accuracies = np.empty(int(runs))
epochs_to_max_accuracies = np.empty(int(runs))
for run in range(int(runs)):
	i = m[m[:,0]==run][:,2].argmax()
	max_accuracies[run] = m[m[:,0]==run][i,2]
	epochs_to_max_accuracies[run] = m[m[:,0]==run][i,1]

	#print(m[m[:,0]==run][i])

print("Average max accuracy %.2f +/- %.2f" % (max_accuracies.mean(), 1.96 * max_accuracies.std() / np.sqrt(runs)))
print("Average epochs to max accuracy %.2f +/- %.2f" % (epochs_to_max_accuracies.mean(), 1.96 * max_accuracies.std() / np.sqrt(runs)))
