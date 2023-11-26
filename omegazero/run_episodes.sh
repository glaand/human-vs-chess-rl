#!/bin/bash

EPISODES=100  # Set the number of episodes here

for ((episode=1; episode<=EPISODES; episode++)); do
    python omegazero.py "$episode"
done

python evaluation.py