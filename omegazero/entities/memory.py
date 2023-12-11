import config
import numpy as np
from collections import deque

class Memory:
	"""
	Represents a memory buffer for storing game states, policies, and values.

	Attributes:
		MEMORY_SIZE (int): The maximum size of the memory buffer.
		ltmemory (deque): A deque object for storing long-term memory entries.
		stmemory (deque): A deque object for storing short-term memory entries.
	"""

	def __init__(self):
		self.MEMORY_SIZE = config.MEMORY_SIZE
		self.ltmemory = deque(maxlen=config.MEMORY_SIZE)
		self.stmemory = deque(maxlen=config.MEMORY_SIZE)

	@property
	def ltmemory_nparray(self):
		"""
		Convert the long-term memory entries into NumPy arrays.

		Returns:
			tuple: A tuple containing the NumPy arrays for states, policies, and values.
		"""
		ltmemory_list = list(self.ltmemory)
		ltmemory_states = [entry['state'] for entry in ltmemory_list]
		ltmemory_policies = [entry['policy'] for entry in ltmemory_list]
		ltmemory_values = [entry['value'] for entry in ltmemory_list]

		ltmemory_states_tensor = np.stack(ltmemory_states, axis=0)
		ltmemory_policies_tensor = np.stack(ltmemory_policies, axis=0)
		ltmemory_values_tensor = np.stack(ltmemory_values, axis=0)

		return (ltmemory_states_tensor, ltmemory_policies_tensor, ltmemory_values_tensor)

	def commit_stmemory(self, identities, state, policy, value):
		"""
		Commit short-term memory entries to the memory buffer.

		Args:
			identities: A function that generates identities for the given state and policy.
			state: The game state.
			policy: The policy for the given state.
			value: The value associated with the state and policy.
		"""
		for r in identities(state, policy):
			self.stmemory.append({
				'state': r[0].as_tensor(),
				'policy': r[1].reshape(4096),
				'value': [value]
			})

	def commit_ltmemory(self):
		"""
		Commit short-term memory entries to the long-term memory buffer and clear the short-term memory.
		"""
		for i in self.stmemory:
			self.ltmemory.append(i)
		self.clear_stmemory()

	def clear_stmemory(self):
		"""
		Clear the short-term memory buffer.
		"""
		self.stmemory = deque(maxlen=config.MEMORY_SIZE)
		