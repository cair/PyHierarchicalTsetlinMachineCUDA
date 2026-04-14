from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
#from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassCoalescedTsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
from keras.datasets import mnist
from skimage.util import view_as_windows

clauses = 2000
T = 5000
s = 10.0

patch_size = 10

(X_org_train, Y_train), (X_org_test, Y_test) = mnist.load_data()

X_org_train = np.where(X_org_train > 75, 1, 0).astype(np.uint32)
X_org_test = np.where(X_org_test > 75, 1, 0).astype(np.uint32)

number_of_patches = int((X_org_train.shape[1] - patch_size + 1) * (X_org_train.shape[2] - patch_size + 1))

X_train = np.zeros((X_org_train.shape[0], number_of_patches, patch_size * patch_size + (patch_size-1)*2))
for i in range(X_train.shape[0]):
	X_train[i,:,:] = view_as_windows(X_org_train[i,:,:patch_size*patch_size], (patch_size, patch_size)).reshape((number_of_patches, patch_size, patch_size))
X_train = X_train.reshape((X_org_train.shape[0], -1))

X_test = np.zeros((X_org_test.shape[0], number_of_patches, patch_size * patch_size + (patch_size-1)*2))
for i in range(X_test.shape[0]):
	X_test[i,:,:] = view_as_windows(X_org_test[i,:,:patch_size*patch_size], (patch_size, patch_size)).reshape((number_of_patches, patch_size*patch_size))
X_test = X_test.reshape((X_org_test.shape[0], -1))

tm = MultiClassTsetlinMachine(clauses, T, s, weighted_clauses=True, hierarchy_structure=((tm.AND_GROUP, patch_size**2), (tm.OR_GROUP, number_of_patches)))

print("\nAccuracy over 500 epochs:\n")
for i in range(500):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
