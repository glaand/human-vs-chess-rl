from entities.player import LearningPlayer
import os

artifacts_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "artifacts")

class LearnStage:
    """
    Represents the learn stage of the game.
    
    This stage is responsible for learning from the game state tensor and generating a new player.
    """

    def __init__(self):
        print("")
        print("=====================")
        print("=    LEARN STAGE    =")
        print("=====================")
        print("► Input: Game State Tensor")
        print("► Output: New Player")
        print("---------------------")

    def learn(self):
        """
        Performs the learning process.
        """
        self.brain.learn()
    
    def setInput(self, brain):
        """
        Sets the input brain for the learning stage.
        
        Parameters:
            brain (object): The brain object representing the game state tensor.
        """
        self.brain = brain

    def getOutput(self):
        """
        Returns the output brain representing the new player.
        
        Returns:
            object: The brain object representing the new player.
        """
        return self.brain
    
