from entities.player import LearningPlayer
import os

artifacts_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "artifacts")

class LearnStage:
    def __init__(self):
        print("")
        print("=====================")
        print("=    LEARN STAGE    =")
        print("=====================")
        print("► Input: Game State Tensor")
        print("► Output: New Player")
        print("---------------------")

    def learn(self):
        self.brain.learn()
    
    def setInput(self, brain):
        self.brain = brain

    def getOutput(self):
        return self.brain
    
