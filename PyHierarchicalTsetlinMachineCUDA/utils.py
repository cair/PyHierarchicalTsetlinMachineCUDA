from collections import deque

import networkx as nx
import numpy as np

from .tm import AND_ALTERNATIVES, AND_GROUP, OR_ALTERNATIVES, OR_GROUP, CommonTsetlinMachine

DEFAULT_NODE_TYPE_LABELS: dict[int, str] = {
	AND_GROUP: ' ∧* ',
	AND_ALTERNATIVES: ' ∧ ',
	OR_GROUP: ' ∨* ',
	OR_ALTERNATIVES: ' ∨ ',
}


def make_hierarchy_graph(
	G: nx.Graph,
	hier: list[tuple[int, int]],
	node_type_labels: dict[int, str] | None = None,
):
	"""Create Graph from the hierarchy."""
	node_type_labels = {**DEFAULT_NODE_TYPE_LABELS, **(node_type_labels or {})}

	# BFS
	q = deque()
	q.append((len(hier), 0))  # (level, idx)
	while q:
		level, idx = q.popleft()
		op = hier[level - 1][0]
		G.add_node((level, idx), label=node_type_labels.get(op, op), op=op)
		if level == 1:
			continue
		branching = hier[level - 1][1]
		for cid in range(idx * branching, (idx + 1) * branching):
			q.append((level - 1, cid))
			G.add_edge((level, idx), (level - 1, cid))


def clause_to_nx(
	tm: CommonTsetlinMachine,
	clause_idx: int,
	feature_names: list[str] | None = None,
	negation_prefix: str = '¬',
	clause_literals: np.ndarray | None = None,
	ta_to_fid_mapping=None,
	node_type_labels: dict[int, str] | None = None,
):
	"""
	Create a networkx Graph for a single clause.
	Args:
		`tm`: The model.
		`clause_idx`: Index of the clause to visualize.
		`feature_names`: Optional list of feature names for labeling. If None, defaults to 'x0', 'x1', ...
		`negation_prefix`: Prefix for negated features (default: '¬').
		`clause_literals`: Optional pre-extracted literals for the clause, must have shape (n_components, literals_per_component). If None, literals will be extracted from the model.
		`ta_to_fid_mapping`: Optional mapping from (component_id, literal_id) to feature_id. If None, it will be obtained from the model.
		`node_type_labels`: Optional dict overriding the display symbol for one or more hierarchy node types (`AND_GROUP`/`AND_ALTERNATIVES`/`OR_GROUP`/`OR_ALTERNATIVES`). Merged over `DEFAULT_NODE_TYPE_LABELS`; unset keys fall back to the default.
	"""

	assert tm.hierarchy_structure is not None, (
		'Hierarchy structure not defined in the model. Are you sure tm belongs to CommonTsetlinMachine or its subclasses?'
	)

	if feature_names is None:
		feature_names = [f'x{i}' for i in range(tm.number_of_features_hierarchy)]

	literals = (
		clause_literals
		if clause_literals is not None
		else tm.get_literals()[clause_idx]
	)
	ta_to_fid = (
		ta_to_fid_mapping
		if ta_to_fid_mapping is not None
		else tm.map_ta_id_to_feature_id()
	)
	n_comp = tm.hierarchy_size[1]
	lits_per_comp = tm.number_of_literals_per_leaf

	G = nx.Graph()
	make_hierarchy_graph(G, tm.hierarchy_structure, node_type_labels)

	feat_per_comp = lits_per_comp // 2 if tm.append_negated else lits_per_comp

	# Add included literals to leaf components
	for comp_id in range(n_comp):
		for fid in range(feat_per_comp):
			pos_lit = literals[comp_id, fid]
			if pos_lit:
				G.add_node(
					(0, comp_id * lits_per_comp + fid),
					label=feature_names[ta_to_fid[comp_id, fid]],
				)
				G.add_edge((1, comp_id), (0, comp_id * lits_per_comp + fid))

			if tm.append_negated:
				neg_lit = literals[comp_id, feat_per_comp + fid]
				if neg_lit:
					lab = f'{negation_prefix}{feature_names[ta_to_fid[comp_id, fid]]}'
					G.add_node(
						(0, comp_id * lits_per_comp + feat_per_comp + fid), label=lab
					)
					G.add_edge(
						(1, comp_id), (0, comp_id * lits_per_comp + feat_per_comp + fid)
					)

	return G


def clause_bank_to_nx(
	tm: CommonTsetlinMachine,
	feature_names: list[str] | None = None,
	negation_prefix: str = '¬',
	node_type_labels: dict[int, str] | None = None,
):
	"""
	Create a networkx Graph for the entire clause bank.
	Args:
		`tm`: The model.
		`feature_names`: Optional list of feature names for labeling. If None, defaults to 'x0', 'x1', ...
		`negation_prefix`: Prefix for negated features (default: '¬').
		`node_type_labels`: Optional dict overriding the display symbol for one or more hierarchy node types. Merged over `DEFAULT_NODE_TYPE_LABELS`; unset keys fall back to the default.
	"""
	node_type_labels = {**DEFAULT_NODE_TYPE_LABELS, **(node_type_labels or {})}

	literals = tm.get_literals()
	ta_to_fid = tm.map_ta_id_to_feature_id()
	G = nx.Graph()
	cb_root = 'CB_ROOT'
	G.add_node(cb_root, label=node_type_labels.get(OR_ALTERNATIVES, OR_ALTERNATIVES), op=OR_ALTERNATIVES)
	for ci in range(tm.number_of_clauses):
		clause_G = clause_to_nx(
			tm, ci, feature_names, negation_prefix, literals[ci], ta_to_fid, node_type_labels
		)
		mapping = {node: (ci, node) for node in clause_G.nodes}
		G = nx.compose(G, nx.relabel_nodes(clause_G, mapping))
		G.add_edge(cb_root, (ci, (tm.depth, 0)))

	return G


def active_path_graph(
	tm: CommonTsetlinMachine,
	X: np.ndarray,
	feature_names: list[str] | None = None,
	negation_prefix: str = '¬',
	node_type_labels: dict[int, str] | None = None,
):
	"""
	Show the activated path, and vote calculation for a single sample `X`.
	Args:
		`tm`: The model.
		`X`: A single sample with shape (n_features,).
		`feature_names`: Optional list of feature names for labeling. If None, defaults to 'x0', 'x1', ...
		`negation_prefix`: Prefix for negated features (default: '¬').
		`node_type_labels`: Optional dict overriding the display symbol for one or more hierarchy node types. Merged over `DEFAULT_NODE_TYPE_LABELS`; unset keys fall back to the default.
	Returns:
		A list of networkx Graphs for each clause, with 'active' and 'vote' attributes on nodes and edges, and class sums for each class.
	"""
	assert X.ndim == 1, 'X must be a single sample with shape (n_features,)'

	hierarchy_votes, class_sums = tm.calc_hierarchy_votes(X.reshape(1, -1))
	hierarchy_votes = hierarchy_votes[0]
	class_sums = class_sums[:, 0]

	literals = tm.get_literals()
	ta_to_fid = tm.map_ta_id_to_feature_id()
	lits_per_comp = tm.number_of_literals_per_leaf
	feat_per_comp = lits_per_comp // 2 if tm.append_negated else lits_per_comp

	G = [
		clause_to_nx(tm, ci, feature_names, negation_prefix, literals[ci], ta_to_fid, node_type_labels)
		for ci in range(tm.number_of_clauses)
	]
	for ci in range(tm.number_of_clauses):
		g = G[ci]

		for node in g.nodes():
			level, idx = node
			if level == 0:
				comp_id = idx // lits_per_comp
				local_lit = idx % lits_per_comp
				fid = ta_to_fid[comp_id, local_lit]
				is_neg = (local_lit >= feat_per_comp) if tm.append_negated else False
				g.nodes[node]['active'] = (X[fid] == 0) if is_neg else (X[fid] == 1)
				g.nodes[node]['vote'] = 1 if g.nodes[node]['active'] else None
			else:
				vote = hierarchy_votes[level - 1][ci, idx]
				g.nodes[node]['active'] = vote > 0
				g.nodes[node]['vote'] = vote

		for u, v in g.edges():
			g.edges[u, v]['active'] = g.nodes[u].get('active', False) and g.nodes[
				v
			].get('active', False)

	return G, class_sums
