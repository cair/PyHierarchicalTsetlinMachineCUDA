import numpy as np
import sys

m = np.loadtxt(sys.argv[1])

runs = int(m[:,0].max())+1

average_max_accuracy = 0.0
average_epochs_to_max_accuracy = 0.0
for run in range(int(runs)):
	i = m[m[:,0]==run][:,2].argmax()
	average_max_accuracy += m[m[:,0]==run][i,2] / float(runs)
	average_epochs_to_max_accuracy += m[m[:,0]==run][i,1] / float(runs)
	print(m[m[:,0]==run][i])

print("Average max accuracy %.2f" % (average_max_accuracy))
print("Average epochs to max accuracy %.2f" % (average_epochs_to_max_accuracy))
