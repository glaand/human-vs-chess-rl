#!/bin/bash

EPISODES=10  # Set the number of episodes here
INITIAL_EXPLORATION=0.85  # Set the initial exploration probability here
DECAY_FACTOR=0.6 # Set the decay factor here

exploration_prob=$INITIAL_EXPLORATION

for ((episode=1; episode<=EPISODES; episode++)); do
    python omegazero.py "$episode" "$exploration_prob"

    # Calculate exploration decay for the next episode
    exploration_prob=$(python -c "from math import exp; print($exploration_prob * exp(-$DECAY_FACTOR * $episode / $EPISODES))")
done

python evaluation.py