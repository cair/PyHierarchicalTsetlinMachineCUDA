import pickle
from lzma import LZMAFile
import numpy as np
import PyHierarchicalTsetlinMachineCUDA.tm as tm_module
from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine


def train(model, X_train, Y_train, X_test, Y_test, epochs=50):
	print("Training...")
	for epoch in range(epochs):
		model.fit(X_train, Y_train, epochs=1, incremental=True)
		acc = 100 * (model.predict(X_test) == Y_test).mean()
		print("Epoch %4d  Accuracy: %.2f%%" % (epoch + 1, acc))
	return model


def save_model(model, path):
	state_dict = model.save()
	with LZMAFile(path, "wb") as f:
		pickle.dump(state_dict, f)
	print("\nModel saved to %s" % path)


def load_model(path):
	with LZMAFile(path, "rb") as f:
		state_dict = pickle.load(f)

	params = state_dict["params"]
	model = TsetlinMachine(
		number_of_clauses=params["number_of_clauses"],
		T=params["T"],
		s=params["s"],
		q=params["q"],
		hierarchy_structure=params["hierarchy_structure"],
		boost_true_positive_feedback=params["boost_true_positive_feedback"],
		number_of_state_bits=params["number_of_state_bits"],
		append_negated=params["append_negated"],
		weighted_clauses=state_dict["weighted_clauses"] or False,
	)
	model.load(state_dict)
	print("Model loaded from %s" % path)
	return model


if __name__ == "__main__":
	train_data = np.loadtxt("./examples/NoisyParityTrainingData.txt").astype(np.uint32)
	X_train = train_data[:, 0:-1]
	Y_train = train_data[:, -1]

	test_data = np.loadtxt("./examples/NoisyParityTestingData.txt").astype(np.uint32)
	X_test = test_data[:, 0:-1]
	Y_test = test_data[:, -1]

	model = TsetlinMachine(
		number_of_clauses=16,
		T=250,
		s=25.0,
		number_of_state_bits=8,
		boost_true_positive_feedback=0,
		hierarchy_structure=(
			(tm_module.AND_GROUP, 3),
			(tm_module.OR_ALTERNATIVES, 3),
			(tm_module.AND_GROUP, 2),
			(tm_module.OR_ALTERNATIVES, 3),
			(tm_module.AND_GROUP, 2),
		),
		seed=10,
	)

	model = train(model, X_train, Y_train, X_test, Y_test)

	save_model(model, "./examples/noisyparitymodel.tm")
	loaded_model = load_model("./examples/noisyparitymodel.tm")

	original_pred = model.predict(X_test)
	loaded_pred = loaded_model.predict(X_test)

	print("\nOriginal model accuracy : %.2f%%" % (100 * (original_pred == Y_test).mean()))
	print("Loaded model accuracy   : %.2f%%" % (100 * (loaded_pred == Y_test).mean()))
	print("Predictions match       : %s" % np.array_equal(original_pred, loaded_pred))

	print("\nContinuing training from loaded model...")
	loaded_model = train(loaded_model, X_train, Y_train, X_test, Y_test)
	continued_pred = loaded_model.predict(X_test)
	print("Accuracy after continued training: %.2f%%" % (100 * (continued_pred == Y_test).mean()))
