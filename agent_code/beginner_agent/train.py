from collections import namedtuple, deque
import torch
import torch.nn as nn 
import numpy as np
from .model import MLPQNetwork

from typing import List

import events as e
from .callbacks import ACTIONS, state_to_features
from .memory import Memory, ReplayMemory
from .plotting_actions import plot_action_distribution
from .utils import device
from .custom_events import check_custom_events

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
RECORD_ENEMY_TRANSITIONS = 1.0  # Record enemy transitions with a probability
TARGET_UPDATE_FREQUENCY = 30  # Update the target network every ... steps
POLICY_UPDATE_FREQUENCY = 15  # Update the policy network every ... steps

def setup_training(self):
    """
    Initialize the agent for training purposes.

    This method is called after `setup` in callbacks.py.
    
    :param self: This object is passed to all callbacks and allows setting arbitrary values.
    """
    self.step_counter = 0
    self.round_step_counter = 0
    self.official_rewards = 0

    self.NO_ESCAPE = False

    self.direction_history = deque(maxlen=5)
    self.position_history = deque(maxlen=10)
    self.no_progress_steps = 0

    transition_history_size = self.config["memory_history_size"]  # Buffer size for storing past transitions
    self.gamma = self.config["GAMMA"]  # Discount factor for future rewards (0.95-0.99 is common)
    self.batch_size = self.config["BATCH_SIZE"]  # Batch size for training (32-64 is common)
    
    # Custom events for rewards
    self.old_agent_pos = [0, 0]
  
    # Setup replay memory to store transitions (state, action, reward, next_state)
    self.replay_memory = ReplayMemory(transition_history_size)

    # Store the rewards per episode
    self.memory = Memory()
    self.memory.reset_memory(self.logger)  # Reset the memory for a new file to store rewards and losses
    
    # Define two networks: one for training and one for making predictions
    self.target_net = MLPQNetwork(self.input_size, self.hidden_size1, self.hidden_size2, self.n_actions, self.dropout_prob, train=self.train).to(device)
    self.target_net.load_state_dict(self.policy_net.state_dict())  # Copy initial weights from the policy network
    self.target_net.eval()  # Set target network to evaluation mode
        
    
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    This method is called after every step to allow the agent to process intermediate rewards based on game events.

    :param self: The object passed to all callbacks.
    :param old_game_state: The state of the game passed to the previous call of `act`.
    :param self_action: The action the agent took.
    :param new_game_state: The current state of the game.
    :param events: A list of all game events that occurred during the transition.
    """    
    self.step_counter += 1
    self.round_step_counter += 1
    if(self.debug):
        self.logger.debug(f'Encountered game event(s): {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Track circular movements (prevents the agent from looping in circles)
    old_position = old_game_state['self'][3]
    new_position = new_game_state['self'][3]
    direction = (new_position[0] - old_position[0], new_position[1] - old_position[1])
    self.direction_history.append(direction)

    self.position_history.append(new_game_state['self'][3])  # Add current position to position history
    if len(self.position_history) == self.position_history.maxlen:
        unique_positions = len(set(self.position_history))  # Count unique positions in history
        if unique_positions < 5:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        
        if detect_circular_movement(self.position_history):  # Detect cycles in movement
            self.no_progress_steps += 1
    else:
        self.no_progress_steps = 0

    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)
     
    # Check if the agent can now plant a bomb after previously being restricted
    if old_game_state["self"][2] == 0 and new_game_state["self"][2] == 1:
        agent_newly_allowed_to_plant_bomb = True
    else:
        agent_newly_allowed_to_plant_bomb = False
        
    check_custom_events(self, events, old_features, new_features, old_game_state, new_game_state, self_action, agent_newly_allowed_to_plant_bomb)

    reward = reward_from_events(self, events)

    # Add the new transition to the replay memory
    self.replay_memory.push(old_features, self_action, new_features, reward)
    
    # Update the policy network every POLICY_UPDATE_FREQUENCY steps
    if self.step_counter % POLICY_UPDATE_FREQUENCY == 0:
        optimize_model(self)

    # Update the target network every TARGET_UPDATE_FREQUENCY steps
    if self.step_counter % TARGET_UPDATE_FREQUENCY == 0:
        self.logger.info("Updating target network")
        update_target_network(self)

def update_target_network(self):
    """
    Copy the weights from the policy network to the target network.
    """
    self.target_net.load_state_dict(self.policy_net.state_dict())



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent dies to calculate final rewards.
    This replaces game_events_occurred in this round.

    :param self: The object passed to all callbacks.
    :param last_game_state: The final state of the game.
    :param last_action: The last action taken by the agent.
    :param events: The list of events that occurred during the final step.
    """
    # Calculate final rewards 
    old_features = state_to_features(self, last_game_state)
    check_custom_events(self, events, old_features, None, last_game_state, None, last_action, False)

    if(self.debug):
        self.logger.debug(f'Updated epsilon: {self.epsilon}')
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    if(self.debug):
        self.logger.debug(f'Encountered event(s): {", ".join(map(repr, events))} in the final step')

    # Add the final transition to replay memory
    self.replay_memory.push(old_features, last_action, None, reward_from_events(self, events))

    # Store rewards in memory
    rewards = self.replay_memory.sum_rewards(self.round_step_counter)
    self.round_step_counter = 0
    self.memory.save_rewards(rewards)
    
    self.official_rewards = 0
    self.NO_ESCAPE = False

    # Store actions in memory and plot action distribution every 100 episodes
    self.action_counts.append(self.current_episode_actions)
    self.current_episode_actions = np.zeros(len(ACTIONS))  # Reset action counter

    if len(self.action_counts) % 100 == 0:  # Plot action distribution every 100 episodes
        plot_action_distribution(self.action_counts)

    # Save the model
    torch.save(self.policy_net.state_dict(), "my-saved-model.pt")
    if(self.debug):
        self.logger.info(f'Model saved to {"my-saved-model.pt"}')


def reward_from_events(self, events: List[str]) -> int:
    """
    Adjust rewards based on the events and return the total reward.

    :param events: A list of game events that occurred.
    :return: The total reward based on the events.
    """
    game_rewards = {
            # Coin-related rewards
            e.MOVED_TO_COIN: 7,  # Updated from configuration
            e.COIN_COLLECTED: 40,  # Updated from configuration
            # Crate-related rewards
            e.CRATE_DESTROYED_NOT_KILLED_SELF: 9,  # Updated from configuration
            # Safe/Danger tiles
            e.MOVED_ONTO_SAFE_TILE: 15,  # Updated from configuration
            e.MOVED_INTO_DANGER: -35,  # Updated from configuration
            e.MOVED_INTO_EXPLOSION: -90,  # Updated from configuration
            e.NO_MOVE_WHILE_ON_DANGER_TILE: -20,  # Updated from configuration
            e.MOVED_TOWARDS_SAFE_TILE: 8,  # Updated from configuration
            e.MOVED_AWAY_FROM_SAFE_TILE: -12,  # Updated from configuration
            e.DID_NOT_CHOOSE_SAVING_MOVE: -60,  # Updated from configuration
            # Bomb-related rewards and penalties
            e.BOMB_DROPPED_WITH_NO_ESCAPE: -60,  # Updated from configuration
            e.CRATE_IN_POSSIBLE_EXPLOSION: 12,  # Updated from configuration
            e.SURVIVED_BOMB: 35,  # Updated from configuration
            e.KILLED_SELF: -75,  # Updated from configuration
            # Movement penalties to discourage wandering
            e.WAITED: -3,  # Updated from configuration
            e.MOVED_UP: -1,  # Updated from configuration
            e.MOVED_DOWN: -1,  # Updated from configuration
            e.MOVED_LEFT: -1,  # Updated from configuration
            e.MOVED_RIGHT: -1,  # Updated from configuration
            e.INVALID_ACTION: -10,  # Updated from configuration
            e.WALKING_IN_CIRCLES: -45,  # Updated from configuration
            # Enemy-related rewards
            e.KILLED_OPPONENT: 300,
            e.GOT_KILLED_BY_OPPONENT: -150,
            # Long-term rewards
            e.SURVIVED_ROUND: 100  # Updated from configuration
        }
    reward_sum = 0
    for event in events:
        if event == e.COIN_COLLECTED:
            self.official_rewards += 1
        elif event == e.KILLED_OPPONENT:
            self.official_rewards += 5

        if event in game_rewards:
            reward_sum += game_rewards[event]

    if(self.debug):
        self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    
    return reward_sum

def optimize_model(self):
    """
    Optimize the model by sampling a batch from the replay memory and updating the policy network weights.

    The Bellman equation is used to compute the target Q-values, which are compared with the predicted Q-values.
    The loss is minimized to update the network.

    :param self: The agent instance.
    :return: None
    """
    # If there are not enough transitions in the replay memory, skip optimization
    if self.replay_memory.get_length() < self.batch_size: 
        return 
    if(self.debug):
        self.logger.info("Optimizing model")
    
    # Sample a batch of transitions from the replay memory
    transitions = self.replay_memory.sample(self.batch_size)
    batch = Transition(*zip(*transitions))

    # Convert the batch to arrays and tensors for efficiency
    non_final_mask = np.array([s is not None for s in batch.next_state])
    non_final_mask = torch.from_numpy(non_final_mask).bool().to(device)

    non_final_next_states = np.array([s for s in batch.next_state if s is not None])
    non_final_next_states = torch.from_numpy(non_final_next_states).float().to(device)

    state_batch = np.array(batch.state)
    state_batch = torch.from_numpy(state_batch).float().to(device)

    action_batch = np.array([ACTIONS.index(action) for action in batch.action])
    action_batch = torch.from_numpy(action_batch).long().unsqueeze(1).to(device)

    reward_batch = np.array(batch.reward)
    reward_batch = torch.from_numpy(reward_batch).float().to(device)

    # Compute Q-values for all possible actions at each state
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    # Calculate the maximum Q-values for the next states
    next_state_values = torch.zeros(self.batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

    # Compute the temporal difference target (ground truth for training the network)
    expected_state_action_values = (next_state_values * self.gamma) + reward_batch
    
    # Compute the loss between predicted and target Q-values
    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    self.scheduler.step(loss)  # Adjust the learning rate based on the loss

    # Save the loss for tracking purposes
    self.memory.save_loss(loss.item())
    
    # Optimize the network by performing backpropagation and updating the weights
    self.optimizer.zero_grad()  # Reset gradients
    loss.backward()  # Backpropagation

    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
    self.optimizer.step()  # Apply the weight updates

def optimize_model_single_transition(self, state, action, reward, next_state):
    """
    Optimize the model using the current transition (state, action, reward, next_state).
    This method allows the agent to learn immediately from the current experience.
    """
    if state is None: 
        return

    # Convert to tensors
    state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
    action_tensor = torch.tensor([[ACTIONS.index(action)]], device=device)
    reward_tensor = torch.tensor([reward], device=device)

    # Compute Q(s_t, a)
    state_action_values = self.policy_net(state_tensor).gather(1, action_tensor)

    if next_state is not None:
        next_state_tensor = torch.from_numpy(np.array(next_state)).float().unsqueeze(0).to(device)
        # Compute V(s_{t+1}) for the non-terminal next state
        with torch.no_grad():
            next_state_values = self.target_net(next_state_tensor).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_tensor
    else:
        expected_state_action_values = reward_tensor

    # Compute the loss
    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model by performing backpropagation
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
    self.optimizer.step()

def detect_circular_movement(position_history):
    """
    Check if the agent is performing circular movements, such as moving back and forth between two or three positions.

    :param position_history: A history of the agent's recent positions.
    :return: True if circular movement is detected, False otherwise.
    """
    # Check if the last movements form a cyclic sequence
    if len(position_history) < 4:
        return False  # Not enough data to detect a cycle

    # Example: If positions follow the pattern A -> B -> A -> B (back and forth)
    last_pos = position_history[-1]
    second_last_pos = position_history[-2]
    third_last_pos = position_history[-3]
    fourth_last_pos = position_history[-4]

    # Check for cyclic patterns like A -> B -> A -> B
    if last_pos == third_last_pos and second_last_pos == fourth_last_pos:
        return True

    # Check for longer cyclic patterns like A -> B -> C -> A -> B -> C
    if len(position_history) >= 6:
        if (position_history[-1] == position_history[-3] and 
            position_history[-2] == position_history[-4] and 
            position_history[-3] == position_history[-5]):
            return True

    return False
