import numpy as np
import config

class Node():
	"""
	Represents a node in the Monte Carlo Tree Search (MCTS) algorithm.

	Attributes:
		state: The state of the game at this node.
		playerTurn: The player whose turn it is at this node.
		id: The unique identifier of this node.
		edges: The edges connecting this node to its child nodes.
	"""

	def __init__(self, state):
		self.state = state
		self.playerTurn = state.turn
		self.id = state.id
		self.edges = []

	def isLeaf(self):
		"""
		Checks if the node is a leaf node.

		Returns:
			True if the node is a leaf node, False otherwise.
		"""
		if len(self.edges) > 0:
			return False
		else:
			return True

class Edge():
	"""
	Represents an edge connecting two nodes in a Monte Carlo Tree Search (MCTS) algorithm.

	Attributes:
		id (str): The unique identifier of the edge.
		inNode (Node): The input node of the edge.
		outNode (Node): The output node of the edge.
		playerTurn (int): The player's turn when the edge is traversed.
		action: The action taken to traverse the edge.
		stats (dict): The statistics associated with the edge, including:
			- 'N': The number of times the edge has been visited.
			- 'Q': The total action value accumulated from the edge visits.
			- 'P': The prior probability of selecting the edge.
	"""

	def __init__(self, inNode, outNode, prior, action):
		self.id = inNode.state.id + '|' + outNode.state.id
		self.inNode = inNode
		self.outNode = outNode
		self.playerTurn = inNode.state.turn
		self.action = action
		self.stats = {
			'N': 0,
			'Q': 0,
			'P': prior,
		}

class MCTS():
	"""
	Monte Carlo Tree Search (MCTS) algorithm implementation.

	Args:
		root (Node): The root node of the tree.

	Attributes:
		root (Node): The root node of the tree.
		tree (dict): A dictionary representing the tree structure.
		cpuct (float): The exploration constant for MCTS.
	"""

	def __init__(self, root):
		self.root = root
		self.tree = {}
		self.cpuct = config.MCTS_CPUCT
		self.addNode(root)

	def __len__(self):
		"""
		Get the number of nodes in the tree.

		Returns:
			int: The number of nodes in the tree.
		"""
		return len(self.tree)

	def moveToLeaf(self):
		"""
		Traverse the tree from the root to a leaf node.

		Returns:
			tuple: A tuple containing the current leaf node, the value of the leaf node, a flag indicating if the game is done, and the breadcrumbs (edges traversed).
		"""
		breadcrumbs = []
		currentNode = self.root

		done = 0
		value = 0

		while not currentNode.isLeaf():
			maxQU = -99999

			Nb = 0
			for action, edge in currentNode.edges:
				Nb = Nb + edge.stats['N']

			simulationAction = None
			for idx, (action, edge) in enumerate(currentNode.edges):

				U = self.cpuct * edge.stats['P'] * (np.sqrt(Nb) / (1 + edge.stats['N']))
				Q = edge.stats['Q']

				if Q + U > maxQU:
					maxQU = Q + U
					simulationAction = action
					simulationEdge = edge

			if simulationAction is not None:
				newState, value, done = currentNode.state.takeAction(simulationAction) #the value of the newState from the POV of the new playerTurn
				currentNode = simulationEdge.outNode
				breadcrumbs.append(simulationEdge)

		return currentNode, value, done, breadcrumbs

	def backFill(self, leaf, value, breadcrumbs):
		"""
		Backpropagate the value from a leaf node to the root node.

		Args:
			leaf (Node): The leaf node.
			value (float): The value to backpropagate.
			breadcrumbs (list): The edges traversed from the root to the leaf node.
		"""
		currentPlayer = leaf.state.turn

		for edge in breadcrumbs:
			playerTurn = edge.playerTurn
			if playerTurn == currentPlayer:
				direction = 1
			else:
				direction = -1

			edge.stats['Q'] = (edge.stats['N'] * edge.stats['Q'] + (value * direction)) / (edge.stats['N'] + 1)
			edge.stats['N'] = edge.stats['N'] + 1

	def addNode(self, node):
		"""
		Add a node to the tree.

		Args:
			node (Node): The node to add.
		"""
		self.tree[node.id] = node

