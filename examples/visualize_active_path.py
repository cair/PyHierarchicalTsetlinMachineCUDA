import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Patch

import PyHierarchicalTsetlinMachineCUDA.tm as tm
from PyHierarchicalTsetlinMachineCUDA.utils import active_path_graph


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

def plot_single_clause(G):
	labels = nx.get_node_attributes(G, 'label')
	pos = nx.nx_agraph.pygraphviz_layout(G, prog='twopi')

	# Coloring the nodes.
	op_colors = {
		tm.AND_GROUP: 'lightgreen',
		tm.OR_ALTERNATIVES: 'limegreen',
	}
	cmap = mpl.colormaps['tab20'].resampled(model.number_of_features_hierarchy)
	feat_colors = [
		mpl.colors.to_hex(cmap(i)) for i in range(model.number_of_features_hierarchy)
	]

	default_color = '#e0e0e0'
	active_edge = 'green'
	root_color = '#FF6B35'

	node_colors = []
	nodes = G.nodes()
	depth = max(node[0] for node in nodes)
	for node in nodes:
		if node[0] == depth:
			node_colors.append(root_color)
			continue
		active = nodes[node].get('active', False)
		if active:
			op = nodes[node].get('op', None)
			if op in op_colors:
				node_colors.append(op_colors[op])
			else:
				label = nodes[node].get('label', '')
				feat_str = label.lstrip('¬~')
				feat_idx = int(feat_str[1:])
				node_colors.append(feat_colors[feat_idx])
		else:
			node_colors.append(default_color)

	edge_colors = []
	edge_labels = {}
	edges = G.edges()
	for u, v in edges:
		active = edges[u, v].get('active', False)
		edge_colors.append(active_edge if active else default_color)

		if active:
			vote = nodes[u if u[0] < v[0] else v]['vote']
			if vote is not None:
				edge_labels[(u, v)] = vote

	fig, ax = plt.subplots(figsize=(16, 16), dpi=150, layout='compressed')
	nx.draw(
		G,
		pos,
		with_labels=True,
		labels=labels,
		edge_color=edge_colors,
		node_color=node_colors,
		node_size=100,
		font_size=4,
		ax=ax,
	)
	nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=4)
	ax.axis('off')

	# Add legend
	handles = [Patch(color=color, label=f'x{i}') for i, color in enumerate(feat_colors)]
	ax.legend(
		handles=handles, loc='center right', bbox_to_anchor=(1.01, 0.5), fontsize=6
	)

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

	G, class_sums = active_path_graph(model, X_test[1])

	# Plot first clause
	# for ci in range(model.number_of_clauses):
	for ci in range(1):
		g_clause = G[ci]
		fig, ax = plot_single_clause(g_clause)
		fig.suptitle(f'Clause {ci} - Activation', fontsize='small')
		ax.set_title(
			f'Clause vote: {g_clause.nodes[(model.depth, 0)]["vote"]}',
			fontsize='x-small',
		)
	plt.show()
