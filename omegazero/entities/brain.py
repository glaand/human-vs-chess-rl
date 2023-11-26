import numpy as np
import random

import config
from algorithms.mcts import MCTS, Node, Edge
from algorithms.nn import NNManager
from entities.memory import Memory

class Brain:
    def __init__(self, episode, is_play_stage=False):
        self.action_size = 4096
        self.MCTSsimulations = config.MCTS_SIMULATIONS
        self.mcts = None
        self.nn_manager = NNManager(episode)
        self.memory = Memory()
        self.stockfish_player = None
        self.is_play_stage = is_play_stage

    def learn(self):
        """
        Trains the neural network using the memory buffer.
        """
        self.nn_manager.learn(self.memory)

    def treeHasAction(self, node, stockfish_action):
        """
        Checks if the given node has the specified stockfish_action.

        Args:
            node (Node): The node to check.
            stockfish_action (str): The stockfish action to check.

        Returns:
            bool: True if the node has the move, False otherwise.
        """
        for action, edge in node.edges:
            if action == stockfish_action and edge.stats['N'] > 0:
                return True
        return False
    
    def simulate(self):
        """
        Simulates a game by moving the leaf node, evaluating the leaf node, and backfilling the value through the tree.
        """
        ##### MOVE THE LEAF NODE
        leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()

        ##### EVALUATE THE LEAF NODE
        value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

        ##### BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.backFill(leaf, value, breadcrumbs)

        return done

    def act(self, state, tau):
        """
        Selects an action based on the given state and temperature parameter.

        Args:
            state (State): The current state of the game.
            tau (float): The temperature parameter for action selection.

        Returns:
            tuple: A tuple containing the selected action, action probabilities, value estimate, and neural network value estimate.
        """
        if self.mcts == None or state.id not in self.mcts.tree:
            self.buildMCTS(state)
        else:
            self.changeRootMCTS(state)

        #### run the simulation until one done=1 is found or the number of simulations is reached
        doneFound = False
        for _ in range(self.MCTSsimulations):
            done = self.simulate()
            if done == 1:
                doneFound = True
                break

        #### get action values
        pi, values = self.getAV()

        ####pick the action
        action, value = self.chooseAction(pi, values, tau)
        nextState, _, _ = state.takeAction(action)
        NN_value = self.get_preds(nextState)[0]

        return (action, pi, value, NN_value, doneFound)
    
    def stockfish_act(self, state, stockfish_move):
        """
        Executes the stockfish_act action in the brain.

        Args:
            state (State): The current state of the game.
            stockfish_move (str): The move made by Stockfish.

        Returns:
            tuple: A tuple containing the action, action probabilities, value, and neural network value.
        """
        if self.mcts == None or state.id not in self.mcts.tree:
            self.buildMCTS(state)
        else:
            self.changeRootMCTS(state)

        #### run the simulation until the stockfish_move is reached
        action = state.getIndexOfAllowedMove(stockfish_move) # debug
        doneFound = False
        while self.treeHasAction(self.mcts.root, action) == False:
            done = self.simulate()
            if done == 1:
                doneFound = True
                break

        #### get action values
        pi, values = self.getAV()

        ####pick the action
        action = state.getIndexOfAllowedMove(stockfish_move)
        value = values[action]

        nextState, _, _ = state.takeAction(action)
        NN_value = self.get_preds(nextState)[0]

        return (action, pi, value, NN_value, doneFound)
    
    def get_value_from_stockfish(self, state):
        value = self.stockfish_player.evaluate_position(state)
        return value

    def get_preds(self, state):
        """
        Get the predictions for a given game state.

        Args:
            state: The game state.

        Returns:
            A tuple containing the predicted value, probabilities, and allowed actions.
        """

        game_state_tensor = state.as_tensor()
        # add batch dimension
        game_state_tensor = np.expand_dims(game_state_tensor, axis=0)
        preds = self.nn_manager.predict(game_state_tensor)
        value_array = preds[0].cpu().detach().numpy()
        logits_array = preds[1].cpu().detach().numpy()

        # Since i dont know if at the moment the reason why is not working is the NN, i will try the score from stockfish
        if self.is_play_stage and False:
            value = np.array([self.get_value_from_stockfish(state)])
        else:
            value = value_array[0] # minus or not????????? asshole

        logits = logits_array[0]

        mask = np.ones(logits.shape,dtype=bool)
        allowedActions = state.allowedActionsIndexes

        mask[allowedActions] = False
        logits[mask] = -100

        #SOFTMAX
        odds = np.exp(logits)
        probs = odds / np.sum(odds)

        return ((value, probs, allowedActions))

    def evaluateLeaf(self, leaf, value, done, breadcrumbs):
        """
        Evaluates a leaf node in the MCTS tree.

        Args:
            leaf (Node): The leaf node to evaluate.
            value (float): The value associated with the leaf node.
            done (int): Flag indicating if the game is done or not.
            breadcrumbs (list): The sequence of actions taken to reach the leaf node.

        Returns:
            tuple: A tuple containing the value of the leaf node and the breadcrumbs.
        """
        if done == 0:
            value, probs, allowedActions = self.get_preds(leaf.state)
            probs = probs[allowedActions]
            for idx, action in enumerate(allowedActions):
                newState, _, _ = leaf.state.takeAction(action)
                if newState.id not in self.mcts.tree:
                    node = Node(newState)
                    self.mcts.addNode(node)
                else:
                    node = self.mcts.tree[newState.id]
                newEdge = Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, newEdge))
        return ((value, breadcrumbs))
        
    def getAV(self):
        """
        Get the action probabilities and values for the root node of the MCTS.

        Returns:
        pi (numpy.ndarray): The action probabilities.
        values (numpy.ndarray): The action values.
        """
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)
        
        for action, edge in edges:
            pi[action] = edge.stats['N']
            values[action] = edge.stats['Q']

        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    def chooseAction(self, pi, values, tau):
        """
        Selects an action based on the given policy probability distribution and values.

        Args:
            pi (numpy.ndarray): The policy probability distribution.
            values (numpy.ndarray): The values associated with each action.
            tau (int): The temperature parameter for exploration.

        Returns:
            tuple: A tuple containing the selected action and its corresponding value.
        """
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx==1)[0][0]

        value = values[action]
        return action, value

    def buildMCTS(self, state):
        """
        Builds the Monte Carlo Tree Search (MCTS) for the given state.

        Args:
            state: The initial state of the game.

        Returns:
            None
        """
        self.root = Node(state)
        self.mcts = MCTS(self.root)

    def changeRootMCTS(self, state):
        """
        Changes the root node of the Monte Carlo Tree Search (MCTS) to the specified state.

        Args:
            state: The new root state for the MCTS.

        Returns:
            None
        """
        self.mcts.root = self.mcts.tree[state.id]