import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
import numpy as np

import PyHierarchicalTsetlinMachineCUDA.tm as tm
from PyHierarchicalTsetlinMachineCUDA.utils import clause_bank_to_nx, clause_to_nx


def load_data():
	train_data = np.loadtxt('./examples/NoisyParityTrainingData.txt').astype(np.uint32)
	X_train = train_data[:, 0:-1]
	Y_train = train_data[:, -1]

	test_data = np.loadtxt('./examples/NoisyParityTestingData.txt').astype(np.uint32)
	X_test = test_data[:, 0:-1]
	Y_test = test_data[:, -1]

	return X_train, Y_train, X_test, Y_test


def train(tm, X_train, Y_train, X_test, Y_test, epochs=1):
	for epoch in range(epochs):
		tm.fit(X_train, Y_train, epochs=1, incremental=True)
		result = 100 * (tm.predict(X_test) == Y_test).mean()
		print('Epoch %d: Accuracy: %.2f%%' % (epoch + 1, result))


def get_node_colors(G, op_colors, feat_colors):
	node_colors = []
	for node in G.nodes():
		data = G.nodes[node]
		op = data.get('op', None)
		if op in op_colors:
			node_colors.append(op_colors[op])
		else:
			# Literal node — extract feature index from label (x3 or ¬x3)
			label = data.get('label', '')
			feat_str = label.lstrip('¬~')
			if feat_str.startswith('x'):
				feat_idx = int(feat_str[1:])
				node_colors.append(feat_colors[feat_idx])
			else:
				node_colors.append('lightgray')
	return node_colors


def visualize_clause(model, clause_idx):
	G = clause_to_nx(model, clause_idx)
	labels = nx.get_node_attributes(G, 'label')
	pos = nx.nx_agraph.pygraphviz_layout(G, prog='twopi')

	# Coloring the nodes.
	op_colors = {
		tm.AND_GROUP: 'lightblue',
		tm.OR_ALTERNATIVES: 'lightyellow',
	}
	cmap = mpl.colormaps["tab20"].resampled(model.number_of_features_hierarchy)
	feat_colors = [mpl.colors.to_hex(cmap(i)) for i in range(model.number_of_features_hierarchy)]
	node_colors = get_node_colors(G, op_colors, feat_colors)

	fig, ax = plt.subplots(figsize=(16, 16), dpi=150, layout="compressed")
	nx.draw(
		G,
		pos,
		with_labels=True,
		labels=labels,
		node_size=100,
		node_color=node_colors,
		font_size=4,
		ax=ax,
	)
	ax.axis('off')
	fig.suptitle(f'Clause {clause_idx}')

	# Add legend
	handles = [Patch(color=color, label=f"x{i}") for i, color in enumerate(feat_colors)]
	ax.legend(handles=handles, title="Features", loc='center right', bbox_to_anchor=(1.01, 0.5), fontsize=6)
	return fig, ax


def visualize_clause_bank(model):
	G = clause_bank_to_nx(model)
	labels = nx.get_node_attributes(G, 'label')
	pos = nx.nx_agraph.pygraphviz_layout(G, prog='twopi')

	op_colors = {
		tm.AND_GROUP: 'lightblue',
		tm.OR_ALTERNATIVES: 'lightyellow',
	}
	cmap = mpl.colormaps["tab20"].resampled(model.number_of_features_hierarchy)
	feat_colors = [mpl.colors.to_hex(cmap(i)) for i in range(model.number_of_features_hierarchy)]
	node_colors = get_node_colors(G, op_colors, feat_colors)

	fig, ax = plt.subplots(figsize=(16, 16), dpi=150, layout="compressed")
	nx.draw(
		G,
		pos,
		with_labels=True,
		labels=labels,
		node_size=100,
		node_color=node_colors,
		font_size=4,
		ax=ax,
	)
	ax.axis('off')
	fig.suptitle('Clause Bank')
	return fig, ax


if __name__ == '__main__':
	X_train, Y_train, X_test, Y_test = load_data()

	# Training the model
	model = tm.TsetlinMachine(
		number_of_clauses=16,
		T=100,
		s=32.1,
		number_of_state_bits=8,
		boost_true_positive_feedback=0,
		hierarchy_structure=(
			(tm.AND_GROUP, 3),
			(tm.OR_ALTERNATIVES, 3),
			(tm.AND_GROUP, 2),
			(tm.OR_ALTERNATIVES, 3),
			(tm.AND_GROUP, 2),
		),
		seed=10,
	)
	train(model, X_train, Y_train, X_test, Y_test, epochs=1)

	# Visualize a single clause
	fig_clause, _ = visualize_clause(model, clause_idx=0)

	# Visualize the entire clause bank
	fig_all, _ = visualize_clause_bank(model)

	plt.show()
