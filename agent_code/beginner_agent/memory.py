import os

from collections import deque
import random
import numpy as np

class Memory:
    """
    Class for saving the rewards and losses during training.

    The rewards and losses are saved to a file for later plotting.
    """
    def __init__(self):
        self.current_reward = 0
        self.filename_reward = "rewards.txt"
        self.filename_loss = "loss.txt"
        self.script_dir = os.path.dirname(__file__)  # Directory of the script
        self.directory = os.path.join(os.getcwd(), "plotting")
    def save_rewards(self, sum_reward):
        """
        Saves the reward to a file for later plotting
        
        :param sum_reward: The total reward from the episode
        """
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
        """
        Saves the loss to a file for later plotting
        
        :param loss: The loss from the training step
        """
        
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
        
        :param logger: Logger object
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
    """
    Replay memory buffer for saving the experiences of the agent.
    
    The replay memory is used to store the experiences of the agent.
    The agent can then sample a batch of experiences from the replay memory
    to train the neural network.
    
    :param capacity: The maximum number of experiences the replay memory can store
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity) # Experience replay buffer
    
    def push(self, state, action, next_state, reward):
        """
        Pushes a new experience into the replay buffer
        """
        self.memory.append((state, action, next_state, reward))
    
 
    def sample(self, batch_size):
        """
        Samples a batch of transitions randomly from the memory.
        """
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        samples = [self.memory[idx] for idx in indices]
        
        return samples
    
    def get_length(self):
        """
        Returns the number of experiences currently stored in the replay memory.
        """
        return len(self.memory)
    
    def sum_rewards(self, last_n):
        """
        Sums up the rewards from the last `last_n` transitions.
        'last_n' represent the steps of the last episode.
        
        :param last_n: The number of last transitions to sum up
        """
        memory_list = list(self.memory)[-last_n:]  # Newest 'last_n' transitions
        total_reward = sum(transition[3] for transition in memory_list)
        return total_reward