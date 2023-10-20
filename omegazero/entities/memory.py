import config
import numpy as np
from collections import deque

class Memory:
	def __init__(self):
		self.MEMORY_SIZE = config.MEMORY_SIZE
		self.ltmemory = deque(maxlen=config.MEMORY_SIZE)
		self.stmemory = deque(maxlen=config.MEMORY_SIZE)

	@property
	def ltmemory_nparray(self):
		ltmemory_list = list(self.ltmemory)
		ltmemory_states = [entry['state'] for entry in ltmemory_list]
		ltmemory_policies = [entry['policy'] for entry in ltmemory_list]
		ltmemory_values = [entry['value'] for entry in ltmemory_list]

		ltmemory_states_tensor = np.stack(ltmemory_states, axis=0)
		ltmemory_policies_tensor = np.stack(ltmemory_policies, axis=0)
		ltmemory_values_tensor = np.stack(ltmemory_values, axis=0)

		return (ltmemory_states_tensor, ltmemory_policies_tensor, ltmemory_values_tensor)

	def commit_stmemory(self, identities, state, policy, value):
		for r in identities(state, policy):
			self.stmemory.append({
				'state': r[0].as_tensor(), 
				'policy': r[1].reshape(4096),
				'value': [value]
			})

	def commit_ltmemory(self):
		for i in self.stmemory:
			self.ltmemory.append(i)
		self.clear_stmemory()

	def clear_stmemory(self):
		self.stmemory = deque(maxlen=config.MEMORY_SIZE)
		