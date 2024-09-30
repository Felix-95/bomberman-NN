import os

from collections import deque
import random
import numpy as np

class Memory:
    def __init__(self):
        self.current_reward = 0
        self.filename_reward = "rewards.txt"
        self.filename_loss = "loss.txt"
        self.script_dir = os.path.dirname(__file__)  # Directory of the script
        self.directory = os.path.join(os.getcwd(), "plotting")
    def save_rewards(self, sum_reward):
        # Saves the loss to a file for later plotting
        # Make sure the directory exists
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.current_reward = sum_reward
        # Full path to file
        file_path = os.path.join(self.directory, self.filename_reward)
        
        # Save loss to file
        with open(file_path, "a") as f:
            f.write(f"{sum_reward}\n")
    def get_rewards(self):
        return self.current_reward
    
    def save_loss(self, loss):
        # Saves the loss to a file for later plotting
        
        # Make sure the directory exists
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        # Full path to file
        file_path = os.path.join(self.directory, self.filename_loss)
        
        # Save loss to file
        with open(file_path, "a") as f:
            f.write(f"{loss}\n")

    def reset_memory(self, logger):
        """
        Reset the memory by deleting the files.
        """   
        file_path = os.path.join(self.directory, self.filename_loss)  # Absolut path to file
        def delete_file(file_name):
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Memory reset: {file_path} deleted.")
            else:
                logger.info(f"No memory file to delete: {file_path}")

        delete_file(self.filename_loss)
        delete_file(self.filename_reward)


class ReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=1.2e-5):
        self.alpha = alpha  # How much prioritization is used (0 = uniform sampling, 1 = prioritization)
        self.beta = beta  # Degree of importance-sampling corrections
        self.beta_increment_per_sampling = beta_increment_per_sampling  # Increment for beta per sampling
        self.capacity = capacity
        self.memory = deque(maxlen=capacity) # Experience replay buffer
        self.priorities = deque(maxlen=capacity) # Priorities for sampling
    
    def push(self, state, action, next_state, reward):
        """
        Pushes a new experience into the replay buffer with the highest priority.
        """
        max_priority = max(self.priorities) if self.priorities else 1.0  # Maximum priority of the memory
        self.memory.append((state, action, next_state, reward))
        self.priorities.append(max_priority)  # New experiences are considered as important
    
    def sample(self, batch_size):
        """
        Samples a batch of transitions based on their priorities.
        """
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)  # Increases beta each time

        # Ensure priorities are valid and handle any NaN or invalid values
        priorities_np = np.array(self.priorities, dtype=np.float32)

        # Replace any NaN, inf, or zero values with a small constant epsilon
        if np.any(np.isnan(priorities_np)) or np.any(priorities_np <= 0):
            priorities_np = np.nan_to_num(priorities_np, nan=1e-6, posinf=1.0, neginf=1e-6)
            priorities_np[priorities_np <= 0] = 1e-6  # Set any zero or negative priorities to epsilon

        # Calculate sampling probabilities from priorities
        probabilities = priorities_np ** self.alpha
        probabilities /= probabilities.sum()  # Normalize probabilities

        # Ensure no NaN values after normalization
        if np.any(np.isnan(probabilities)):
            raise ValueError("NaN found in probabilities after normalization")

        # Sample based on probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        # Calculate importance sampling weights to compensate for bias
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        return samples, weights, indices

    def update_priorities(self, indices, priorities): # Update priorities after calculating the loss
        epsilon = 1e-6  # Small epsilon to ensure positive priorities
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(priority, epsilon)
    
    def get_length(self):
        return len(self.memory)
    
    def sum_rewards(self, last_n):
        """Sums up the rewards from the last `last_n` transitions."""
        memory_list = list(self.memory)[-last_n:]  # Newest 'last_n' transitions
        total_reward = sum(transition[3] for transition in memory_list)
        return total_reward