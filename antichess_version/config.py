from collections import deque


learning_rate = 0.01
discount_factor = 0.85
state_space_size = (8, 8, 12)
action_space_size = 4096
experience_replay_buffer = deque(maxlen=10000)