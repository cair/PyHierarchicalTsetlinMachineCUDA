from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassCoalescedTsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
from skimage.util import view_as_windows
import cv2
from keras.datasets import fashion_mnist


clauses = 40000
T = 5000
s = 10.0

patch_size = 3
padding = 1

(X_org_train, Y_train), (X_org_test, Y_test) = fashion_mnist.load_data()
X_org_train = np.copy(X_org_train)
X_org_test = np.copy(X_org_test)

for i in range(X_org_train.shape[0]):
	X_org_train[i,:] = cv2.adaptiveThreshold(X_org_train[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

for i in range(X_org_test.shape[0]):
	X_org_test[i,:] = cv2.adaptiveThreshold(X_org_test[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

number_of_patches_x = X_org_train.shape[1] - patch_size + 1
number_of_patches_y = X_org_train.shape[2] - patch_size + 1
number_of_patches = number_of_patches_x * number_of_patches_y

number_of_patches_x_with_padding = X_org_train.shape[1] - patch_size + 1 + padding
number_of_patches_y_with_padding = X_org_train.shape[2] - patch_size + 1 + padding
number_of_patches_with_padding = number_of_patches_x_with_padding * number_of_patches_y_with_padding

X_train = np.zeros((X_org_train.shape[0], number_of_patches_with_padding, patch_size * patch_size + (number_of_patches_x_with_padding - 1) + (number_of_patches_y_with_padding - 1)))
for i in range(X_train.shape[0]):
	all_views = view_as_windows(X_org_train[i,:,:], (patch_size, patch_size)).reshape((number_of_patches, patch_size*patch_size))
	
	patch_counter = 0

	
	
	for x in range(number_of_patches_x_with_padding // 2):
		for y in range(number_of_patches_y_with_padding // 2):
			if x * number_of_patches_y + y < number_of_patches:
				X_train[i, patch_counter,:patch_size*patch_size] = all_views[x * number_of_patches_y + y]

			for z in range(number_of_patches_x_with_padding):
				if z < x:
					X_train[i, patch_counter, patch_size * patch_size + z - 1] = 1

			for z in range(number_of_patches_y_with_padding):
				if z < y:
					X_train[i, patch_counter, patch_size * patch_size + (number_of_patches_x_with_padding - 1) + z - 1] = 1

			patch_counter += 1

	for x in range(number_of_patches_x_with_padding // 2, number_of_patches_x_with_padding):
		for y in range(number_of_patches_y_with_padding // 2):
			if x * number_of_patches_y + y < number_of_patches:
				X_train[i, patch_counter,:patch_size*patch_size] = all_views[x * number_of_patches_y + y]

			for z in range(number_of_patches_x_with_padding):
				if z < x:
					X_train[i, patch_counter, patch_size * patch_size + z - 1] = 1

			for z in range(number_of_patches_y_with_padding):
				if z < y:
					X_train[i, patch_counter, patch_size * patch_size + (number_of_patches_x_with_padding - 1) + z - 1] = 1

			patch_counter += 1

	for x in range(number_of_patches_x_with_padding // 2):
		for y in range(number_of_patches_y_with_padding // 2, number_of_patches_y_with_padding):
			if x * number_of_patches_y + y < number_of_patches:
				X_train[i, patch_counter,:patch_size*patch_size] = all_views[x * number_of_patches_y + y]

			for z in range(number_of_patches_x_with_padding):
				if z < x:
					X_train[i, patch_counter, patch_size * patch_size + z - 1] = 1

			for z in range(number_of_patches_y_with_padding):
				if z < y:
					X_train[i, patch_counter, patch_size * patch_size + (number_of_patches_x_with_padding - 1) + z - 1] = 1

			patch_counter += 1

	for x in range(number_of_patches_x_with_padding // 2, number_of_patches_x_with_padding):
		for y in range(number_of_patches_y_with_padding // 2, number_of_patches_y_with_padding):
			if x * number_of_patches_y + y < number_of_patches:
				X_train[i, patch_counter,:patch_size*patch_size] = all_views[x * number_of_patches_y + y]

			for z in range(number_of_patches_x_with_padding):
				if z < x:
					X_train[i, patch_counter, patch_size * patch_size + z - 1] = 1

			for z in range(number_of_patches_y_with_padding):
				if z < y:
					X_train[i, patch_counter, patch_size * patch_size + (number_of_patches_x_with_padding - 1) + z - 1] = 1

			patch_counter += 1

X_train = X_train.reshape((X_org_train.shape[0], -1))

X_test = np.zeros((X_org_test.shape[0], number_of_patches_with_padding, patch_size * patch_size + (number_of_patches_x_with_padding - 1) + (number_of_patches_y_with_padding - 1)))
for i in range(X_test.shape[0]):
	all_views = view_as_windows(X_org_test[i,:,:], (patch_size, patch_size)).reshape((number_of_patches, patch_size*patch_size))
	
	patch_counter = 0
	for x in range(number_of_patches_x_with_padding // 2):
		for y in range(number_of_patches_y_with_padding // 2):
			if x * number_of_patches_y + y < number_of_patches:
				X_test[i, patch_counter,:patch_size*patch_size] = all_views[x * number_of_patches_y + y]

			for z in range(number_of_patches_x_with_padding):
				if z < x:
					X_test[i, patch_counter, patch_size * patch_size + z - 1] = 1

			for z in range(number_of_patches_y_with_padding):
				if z < y:
					X_test[i, patch_counter, patch_size * patch_size + (number_of_patches_x_with_padding - 1) + z - 1] = 1

			patch_counter += 1

	for x in range(number_of_patches_x_with_padding // 2, number_of_patches_x_with_padding):
		for y in range(number_of_patches_y_with_padding // 2):
			if x * number_of_patches_y + y < number_of_patches:
				X_test[i, patch_counter,:patch_size*patch_size] = all_views[x * number_of_patches_y + y]

			for z in range(number_of_patches_x_with_padding):
				if z < x:
					X_test[i, patch_counter, patch_size * patch_size + z - 1] = 1

			for z in range(number_of_patches_y_with_padding):
				if z < y:
					X_test[i, patch_counter, patch_size * patch_size + (number_of_patches_x_with_padding - 1) + z - 1] = 1

			patch_counter += 1

	for x in range(number_of_patches_x_with_padding // 2):
		for y in range(number_of_patches_y_with_padding // 2, number_of_patches_y_with_padding):
			if x * number_of_patches_y + y < number_of_patches:
				X_test[i, patch_counter,:patch_size*patch_size] = all_views[x * number_of_patches_y + y]

			for z in range(number_of_patches_x_with_padding):
				if z < x:
					X_test[i, patch_counter, patch_size * patch_size + z - 1] = 1

			for z in range(number_of_patches_y_with_padding):
				if z < y:
					X_test[i, patch_counter, patch_size * patch_size + (number_of_patches_x_with_padding - 1) + z - 1] = 1

			patch_counter += 1

	for x in range(number_of_patches_x_with_padding // 2, number_of_patches_x_with_padding):
		for y in range(number_of_patches_y_with_padding // 2, number_of_patches_y_with_padding):
			if x * number_of_patches_y + y < number_of_patches:
				X_test[i, patch_counter,:patch_size*patch_size] = all_views[x * number_of_patches_y + y]

			for z in range(number_of_patches_x_with_padding):
				if z < x:
					X_test[i, patch_counter, patch_size * patch_size + z - 1] = 1

			for z in range(number_of_patches_y_with_padding):
				if z < y:
					X_test[i, patch_counter, patch_size * patch_size + (number_of_patches_x_with_padding - 1) + z - 1] = 1

			patch_counter += 1

X_test = X_test.reshape((X_org_test.shape[0], -1))

tm = MultiClassCoalescedTsetlinMachine(clauses, T, s, hierarchy_structure=((tm.AND_GROUP, patch_size**2 + (number_of_patches_x_with_padding - 1) + (number_of_patches_y_with_padding - 1)), (tm.OR_GROUP, number_of_patches_with_padding // 4), (tm.AND_GROUP, 4)))

print("\nAccuracy over 500 epochs:\n")
for i in range(500):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
