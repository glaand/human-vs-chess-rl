import numpy as np
import config

class Node():
	def __init__(self, state):
		self.state = state
		self.playerTurn = state.turn
		self.id = state.id
		self.edges = []

	def isLeaf(self):
		if len(self.edges) > 0:
			return False
		else:
			return True

class Edge():
	def __init__(self, inNode, outNode, prior, action):
		self.id = inNode.state.id + '|' + outNode.state.id
		self.inNode = inNode
		self.outNode = outNode
		self.playerTurn = inNode.state.turn
		self.action = action
		self.stats =  {
			'N': 0,
			'Q': 0,
			'P': prior,
		}			

class MCTS():
	def __init__(self, root):
		self.root = root
		self.tree = {}
		self.cpuct = config.MCTS_CPUCT
		self.addNode(root)
	
	def __len__(self):
		return len(self.tree)

	def moveToLeaf(self):
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
		self.tree[node.id] = node

