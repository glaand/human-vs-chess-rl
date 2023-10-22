from entities.player import LearningPlayer
import os

artifacts_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "artifacts")

class LearnStage:
    def __init__(self, episode):
        self.episode = episode
        print("")
        print("=====================")
        print("=    LEARN STAGE    =")
        print("=====================")
        print("► Input: Game State Tensor")
        print("► Output: New Player")
        print("---------------------")

    def learn(self):
        self.new_player = LearningPlayer()
        self.new_player.brain.learn(self.memory, self.episode)
    
    def setInput(self, memory):
        self.memory = memory

    def getOutput(self):
        return self.new_player
    
