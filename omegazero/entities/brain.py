import numpy as np
import random

import config
from algorithms.mcts import MCTS, Node, Edge
from algorithms.nn import NNManager
from entities.memory import Memory

class Brain:
    def __init__(self, episode):
        self.action_size = 4096
        self.MCTSsimulations = config.MCTS_SIMULATIONS
        self.mcts = None
        self.nn_manager = NNManager(episode)
        self.memory = Memory()

    def learn(self):
        self.nn_manager.learn(self.memory)
    
    def simulate(self):
        ##### MOVE THE LEAF NODE
        leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()

        ##### EVALUATE THE LEAF NODE
        value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

        ##### BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.backFill(leaf, value, breadcrumbs)

    def act(self, state, tau):
        if self.mcts == None or state.id not in self.mcts.tree:
            self.buildMCTS(state)
        else:
            self.changeRootMCTS(state)

        #### run the simulation
        for _ in range(self.MCTSsimulations):
            self.simulate()

        #### get action values
        pi, values = self.getAV(tau)

        ####pick the action
        action, value = self.chooseAction(pi, values, tau)
        nextState, _, _ = state.takeAction(action)
        NN_value = self.get_preds(nextState)[0]

        return (action, pi, value, NN_value)
    
    def stockfish_act(self, state, stockfish_move, tau):
        if self.mcts == None or state.id not in self.mcts.tree:
            self.buildMCTS(state)
        else:
            self.changeRootMCTS(state)

        #### run the simulation
        for _ in range(self.MCTSsimulations):
            self.simulate()

        #### get action values
        pi, values = self.getAV(tau)

        ####pick the action
        action = state.getIndexOfAllowedMove(stockfish_move)
        value = values[action]

        nextState, _, _ = state.takeAction(action)
        NN_value = self.get_preds(nextState)[0]

        return (action, pi, value, NN_value)

    def get_preds(self, state):
        game_state_tensor = state.as_tensor()
        # add batch dimension
        game_state_tensor = np.expand_dims(game_state_tensor, axis=0)
        preds = self.nn_manager.predict(game_state_tensor)
        value_array = preds[0].cpu().detach().numpy()
        logits_array = preds[1].cpu().detach().numpy()
        value = -value_array[0] # minus or not????????? asshole

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
        
    def getAV(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)
        
        for action, edge in edges:
            if tau == 0:
                pi[action] = 99999
            else:
                pi[action] = edge.stats['N']
            values[action] = edge.stats['Q']

        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    def chooseAction(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx==1)[0][0]

        value = values[action]
        return action, value

    def buildMCTS(self, state):
        self.root = Node(state)
        self.mcts = MCTS(self.root)

    def changeRootMCTS(self, state):
        self.mcts.root = self.mcts.tree[state.id]