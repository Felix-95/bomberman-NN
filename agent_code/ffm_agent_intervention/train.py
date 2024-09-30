import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple, deque
from typing import List
import wandb

from .model import MLPQNetwork
from .memory import Memory, ReplayMemory
from .plotting_actions import plot_action_distribution
from .WandbTracking import WandBLogger
from .callbacks import ACTIONS, state_to_features
from .custom_events import check_custom_events
import events as e

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
RECORD_ENEMY_TRANSITIONS = 1.0  # Probability to record enemy transitions
TARGET_UPDATE_FREQUENCY = 300    # Frequency to update target network
POLICY_UPDATE_FREQUENCY = 15     # Frequency to update policy network
SOFT_UPDATE_TAU = 0.005          # Soft update parameter for target network

def setup_training(self):
    """
    Initialize training parameters and models.

    This is called after `setup` in callbacks.py.
    """
    # Initialize counters and flags
    self.step_counter = 0
    self.round_step_counter = 0
    self.official_rewards = 0
    self.NO_ESCAPE = False # Flag indicating if the agent has no escape route

    # Initialize position tracking and counters for movement analysis
    self.visited_tiles = None

    self.position_history = deque(maxlen=20)
    self.no_progress_steps = 0

    # Compute distance matrix to the center of the map
    field_shape = (17, 17)
    center = np.array([field_shape[0] // 2, field_shape[1] // 2])
    self.distance_matrix = np.zeros(field_shape)
    for x in range(field_shape[0]):
        for y in range(field_shape[1]):
            self.distance_matrix[x, y] = np.linalg.norm(np.array([x, y]) - center)

    max_distance = np.max(self.distance_matrix)
    self.distance_matrix = 1 - self.distance_matrix / max_distance
    
    # Check if WandB hyperparameter search is active
    self.wandb_hyperparameter_search = bool(dict(wandb.config))
    try:
        if self.wandb_hyperparameter_search:
            _initialize_wandb_parameters(self)
        else:
            _initialize_local_parameters(self)
    except Exception as error:
        print(f"Error during parameter initialization: {error}")

    # Initialize replay memory and model networks(self)
    try:
        _initialize_replay_memory_and_networks(self)
    except Exception as error:
        self.logger.error(f"Error during replay memory and network initialization: {error}")

def _initialize_wandb_parameters(self):
    """
    Initialize training parameters when WandB hyperparameter search is active.
    """
    self.transition_history_size = wandb.config["memory_history_size"] 
    self.gamma = wandb.config["GAMMA"] 
    self.batch_size = wandb.config["BATCH_SIZE"]

    # Initialize WandB logger
    wandb.init(project="bomberman", settings=wandb.Settings(console="off"))
    self.wandb_logger = WandBLogger(project_name="bomberman", config={
        "learning_rate": wandb.config["lr"],
        "batch_size": wandb.config["BATCH_SIZE"],
        "gamma": wandb.config["GAMMA"]
    })

def _initialize_local_parameters(self):
    """
    Initialize training parameters from local config if WandB hyperparameter search is not active.
    """
    self.transition_history_size = self.config["memory_history_size"]
    self.gamma = self.config["GAMMA"]
    self.batch_size = self.config["BATCH_SIZE"]

    # Initialize WandB logger
    wandb.init(project="bomberman", settings=wandb.Settings(console="off"))
    self.wandb_logger = WandBLogger(project_name="bomberman", config={
        "learning_rate": self.config["lr"],
        "batch_size": self.batch_size,
        "gamma": self.gamma
    })

def _initialize_replay_memory_and_networks(self):
    """
    Initialize replay memory and policy/target networks.
    """
    # Initialize replay memory
    self.replay_memory = ReplayMemory(self.transition_history_size)

    # Initialize memory for storing rewards per episode
    self.memory = Memory()
    self.memory.reset_memory(self.logger)

    # Initialize target network and synchronize with policy network
    self.target_net = MLPQNetwork(
        self.input_size, self.hidden_size1, self.hidden_size2,
        self.n_actions, dropout_prob=self.dropout, train=self.train
    ).to(device)
    self.target_net.load_state_dict(self.policy_net.state_dict())  # Sync with policy net
    self.target_net.eval()  # Set to evaluation mode

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """    
    self.step_counter += 1
    self.round_step_counter += 1

    if self.debug:
        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Handle movement history and detect "WALKING_IN_CIRCLES" event
    _handle_position_history(self, new_game_state)

    # Extract features from old and new game states
    try:
        old_features = state_to_features(self, old_game_state)
        new_features = state_to_features(self, new_game_state)
    except Exception as error:
        self.logger.error(f"Error during feature extraction: {error}")
        return

    # Initialize visited tiles if it's a new round or not yet done
    _initialize_visited_tiles(self, new_game_state)

    # Check if the agent is newly allowed to plant a bomb
    try:
        agent_newly_allowed_to_plant_bomb = _check_bomb_planting_permission(self, old_game_state, new_game_state)
    except Exception as error:
        self.logger.error(f"Error during bomb planting permission check: {error}")
        agent_newly_allowed_to_plant_bomb = False

    # Check for custom events
    check_custom_events(self, events, old_features, new_features, old_game_state, new_game_state, self_action, agent_newly_allowed_to_plant_bomb)

    # Calculate reward from events
    try:
        reward = reward_from_events(self, events)
    except Exception as error:
        self.logger.error(f"Error during reward calculationa: {error}")
        reward = 0

    # Push transition to replay memory and optimize model
    self.replay_memory.push(old_features, self_action, new_features, reward)

    if not self.used_action_intervention:
        optimize_model_single_transition(self, old_features, self_action, reward, new_features)
   
    optimize_model_single_transition(self, old_features, self_action, reward, new_features)

    # Periodically update the policy and target networks
    _update_policy_and_target_network(self)

def _initialize_visited_tiles(self, new_game_state):
    """Initialize the visited tiles grid if it hasn't been set or if it's a new round."""
    if self.visited_tiles is None or new_game_state["step"] == 1:
        field_shape = new_game_state["field"].shape
        self.visited_tiles = np.zeros(field_shape, dtype=bool)
        if self.debug:
            self.logger.info("Visited tiles map initialized")

def _handle_position_history(self, new_game_state):
    """Update position history and detect if the agent is walking in circles."""
    self.position_history.append(new_game_state['self'][3])  # Append current position

    # Check for movement patterns if history is full
    if len(self.position_history) == self.position_history.maxlen:
        positions_array = np.array(self.position_history)
        std_x, std_y = np.std(positions_array[:, 0]), np.std(positions_array[:, 1])
        std_threshold = 1.0  # Adjustable threshold for standard deviation

        if std_x < std_threshold and std_y < std_threshold:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        unique_positions = len(set(self.position_history))  # Count unique positions
        if unique_positions < 5:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

def _check_bomb_planting_permission(self, old_game_state, new_game_state):
    """Check if the agent is now allowed to plant bombs."""
    return old_game_state["self"][2] == 0 and new_game_state["self"][2] == 1
        
def _update_policy_and_target_network(self):
    """Periodically update the policy and target networks based on step counters."""
    # Update policy network
    if self.step_counter % POLICY_UPDATE_FREQUENCY == 0:
        optimize_model(self)

    # Soft update the target network
    if self.step_counter % TARGET_UPDATE_FREQUENCY == 0:
        self.logger.info("Soft updating target network")
        soft_update_target_network(self)

def soft_update_target_network(self):
    """
    Perform a soft update of the target network's parameters using the current policy network's parameters.
    """
    for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
        target_param.data.copy_(
            SOFT_UPDATE_TAU * local_param.data + (1.0 - SOFT_UPDATE_TAU) * target_param.data
        )


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # Extract the final state features and handle custom events
    old_features = state_to_features(self, last_game_state)
    check_custom_events(self, events, old_features, None, last_game_state, None, last_action, False)
    if(self.debug):
        self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    # Reset variables for the next round
    _reset_for_new_round(self)

    # Update epsilon for exploration-exploitation trade-off
    _update_epsilon(self)

    # Log events and calculate the final reward for this round
    try:
        last_reward = reward_from_events(self, events)
    except Exception as error:
        self.logger.error(f"Error during reward calculation: {error}")
        last_reward = 0

    # Store the final transition and optimize the model
    self.replay_memory.push(old_features, last_action, None, last_reward)
    optimize_model_single_transition(self, old_features, last_action, last_reward, None)

    # Store rewards and actions
    _store_rewards_and_actions(self)

    # Save the model after each round
    _save_model(self)

def _reset_for_new_round(self):
    """Reset visited tiles, position history, and other variables for the next round."""
    self.visited_tiles = None  # Reset visited tiles map
    self.position_history.clear()  # Clear position history
    self.no_progress_steps = 0  # Reset no progress steps
    self.NO_ESCAPE = False  # Reset escape flag

    if self.debug:
        self.logger.info("Reset variables for the next round")

def _update_epsilon(self):
    """Update the epsilon value for exploration."""
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    self.wandb_logger.log({"epsilon": self.epsilon})
    if self.debug:
        self.logger.debug(f'Epsilon updated to {self.epsilon}')

def _store_rewards_and_actions(self):
    """Store rewards and action counts, and log relevant data."""
    rewards = self.replay_memory.sum_rewards(self.round_step_counter)
    self.round_step_counter = 0
    self.memory.save_rewards(rewards)
    self.wandb_logger.log({"rewards": self.memory.get_rewards()})
    self.wandb_logger.log({"official_rewards": self.official_rewards})
    self.official_rewards = 0

    # Store actions in memory
    self.action_counts.append(self.current_episode_actions)
    self.current_episode_actions = np.zeros(len(ACTIONS)) # Reset the action counter
    if len(self.action_counts) % 100 == 0 and not self.wandb_hyperparameter_search:
        plot_action_distribution(self.action_counts)
    
def _save_model(self):
    """Save the model to a file."""
    torch.save(self.policy_net.state_dict(), "my-saved-model.pt")
    if self.debug:
        self.logger.info('Model saved to "my-saved-model.pt"')


def reward_from_events(self, events: List[str]) -> int:
    """
    Adjusts the rewards based on the events and clips the final reward.

    :param events: List of events that occurred during the game step.
    :return: The total reward for the step, possibly clipped.
    """
    # Check if hyperparameter search is active and use the appropriate reward structure
    if not self.wandb_hyperparameter_search:
        game_rewards = _get_default_game_rewards(self)
    else:
        game_rewards = _get_wandb_game_rewards(self)

    # Update official rewards and calculate the reward sum for the current step
    reward_sum = _calculate_event_rewards(self, events, game_rewards)

    # Log the reward and the events that occurred
    if self.debug:
        self.logger.info(f"Awarded {reward_sum} for events {', '.join(map(repr, events))}")

    return reward_sum

def _get_default_game_rewards(self) -> dict:
    """
    Returns the default game rewards that are used when hyperparameter search is not active.
    """
    return {
        # Coin-related rewards
        e.MOVED_TO_COIN: 7, 
        e.COIN_COLLECTED: 40, 
        e.EXPLORED_NEW_TILE: 2,
        # Crate-related rewards
        e.CRATE_DESTROYED_NOT_KILLED_SELF: 7,
        # Safe/Danger tile rewards
        e.MOVED_ONTO_SAFE_TILE: 10,
        e.MOVED_INTO_DANGER: -20,
        e.MOVED_INTO_EXPLOSION: -50,
        e.MOVED_TOWARDS_SAFE_TILE: 5,
        e.MOVED_AWAY_FROM_SAFE_TILE: -8,
        e.NO_MOVE_WHILE_ON_DANGER_TILE: -6,
        e.DID_NOT_CHOOSE_SAVING_MOVE: -60,
        # Bomb-related rewards and penalties
        e.BOMB_DROPPED_WITH_NO_ESCAPE: -60,
        e.CRATE_IN_POSSIBLE_EXPLOSION: 4,
        e.KILLED_SELF: -50,
        # Movement-related penalties
        e.WAITED: -3,
        e.INVALID_ACTION: -10,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.WALKING_IN_CIRCLES: -40,
        e.INVALID_BOMB_DROP: -10,
        # Enemy actions
        e.KILLED_OPPONENT: 140,
        e.GOT_KILLED_BY_OPPONENT: -150,
        # Long-term goal reward
        e.SURVIVED_ROUND: 150,
        e.EXPLORED_NEW_TILE: 3,
    }

def _get_wandb_game_rewards(self) -> dict:
    """
    Setup custom rewards using WandB configuration.
    This method assigns reward values to different game events based on WandB configuration.
    """
    return {
        # Coin-related rewards
            e.MOVED_TO_COIN: wandb.config.MOVED_TO_COIN,
            e.COIN_COLLECTED: wandb.config.COIN_COLLECTED,
        # Crates-related rewards
            e.CRATE_DESTROYED_NOT_KILLED_SELF: wandb.config.CRATE_DESTROYED_NOT_KILLED_SELF,
        # Safe/Danger tile rewards
            e.MOVED_ONTO_SAFE_TILE: wandb.config.MOVED_ONTO_SAFE_TILE,
            e.MOVED_INTO_DANGER: wandb.config.MOVED_INTO_DANGER,
            e.MOVED_INTO_EXPLOSION: wandb.config.MOVED_INTO_EXPLOSION,
            e.NO_MOVE_WHILE_ON_DANGER_TILE: wandb.config.NO_MOVE_WHILE_ON_DANGER_TILE,
            e.MOVED_TOWARDS_SAFE_TILE: wandb.config.MOVED_TOWARDS_SAFE_TILE, 
            e.MOVED_AWAY_FROM_SAFE_TILE: wandb.config.MOVED_AWAY_FROM_SAFE_TILE,
            e.DID_NOT_CHOOSE_SAVING_MOVE: wandb.config.DID_NOT_CHOOSE_SAVING_MOVE,
        # Bomb-related rewards
            e.BOMB_DROPPED_WITH_NO_ESCAPE: wandb.config.BOMB_DROPPED_WITH_NO_ESCAPE,
            e.CRATE_IN_POSSIBLE_EXPLOSION: wandb.config.CRATE_IN_POSSIBLE_EXPLOSION,
            e.KILLED_SELF: wandb.config.KILLED_SELF,
        # Self-action rewards
            e.WAITED: wandb.config.WAITED,
            e.INVALID_ACTION: wandb.config.INVALID_ACTION,
            e.MOVED_UP: wandb.config.MOVED_UP,
            e.MOVED_DOWN: wandb.config.MOVED_DOWN,
            e.MOVED_LEFT: wandb.config.MOVED_LEFT,
            e.MOVED_RIGHT: wandb.config.MOVED_RIGHT,
            e.WALKING_IN_CIRCLES: wandb.config.WALKING_IN_CIRCLES, 
            e.EXPLORED_NEW_TILE: wandb.config.EXPLORED_NEW_TILE,
            e.INVALID_BOMB_DROP: wandb.config.INVALID_BOMB_DROP,
        # Enemy-related rewards
            e.KILLED_OPPONENT: wandb.config.KILLED_OPPONENT,
            e.GOT_KILLED_BY_OPPONENT: wandb.config.GOT_KILLED_BY_OPPONENT,
        # End of round rewards
            e.SURVIVED_ROUND: wandb.config.SURVIVED_ROUND,
            e.EXPLORED_NEW_TILE: wandb.config.EXPLORED_NEW_TILE,
    }
def _calculate_event_rewards(self, events: List[str], game_rewards: dict) -> int:
    """
    Calculates the total reward based on the occurred events and updates the official rewards.
    
    :param events: List of occurred events.
    :param game_rewards: Dictionary mapping events to their corresponding rewards.
    :return: The total reward for the step.
    """
    reward_sum = 0

    for event in events:
        # Update official rewards for important events
        if event == e.COIN_COLLECTED:
            self.official_rewards += 1
        elif event == e.KILLED_OPPONENT:
            self.official_rewards += 5

        # Add the reward for the event
        if event in game_rewards:
            reward_sum += game_rewards[event]

    return reward_sum

def optimize_model(self):
    """
    Optimize the model by sampling a batch from the replay memory and updating the weights.
    """
    # Ensure there are enough samples in replay memory for optimization
    if self.replay_memory.get_length() < self.batch_size:
        return

    if self.debug:
        self.logger.info("Optimizing model")

    # Sample a batch of transitions from replay memory
    transitions, weights, indices = self.replay_memory.sample(self.batch_size)
    batch = Transition(*zip(*transitions))
    
    # Convert states, actions, rewards, and next states to tensors
    state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask = _prepare_batch_tensors(self, batch)

    # Compute Q-values for actions taken in current states
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    # Compute target Q-values for non-terminal next states using the target network
    next_state_values = _compute_next_state_values(self, non_final_next_states, non_final_mask)

    # Calculate expected Q-values for current state-action pairs using Bellman equation
    expected_state_action_values = (next_state_values * self.gamma) + reward_batch

    # Compute the loss, incorporating importance-sampling weights for prioritized replay
    loss = _compute_loss(self, state_action_values, expected_state_action_values, weights)

    # Backpropagation and optimization
    _optimize_model_weights(self, loss)

    # Update priorities in replay memory based on TD error
    _update_replay_priorities(self, state_action_values, expected_state_action_values, indices)


def _prepare_batch_tensors(self, batch):
    """
    Prepare and convert states, actions, rewards, and next states to tensors.
    
    :param batch: A batch of transitions from replay memory.
    :return: Tensors for states, actions, rewards, non-final next states, and a mask for non-final states.
    """
    # non_final_mask - marks the states that are not terminal
    non_final_mask = np.array([s is not None for s in batch.next_state])
    non_final_mask = torch.from_numpy(non_final_mask).bool().to(device)

    # non_final_next_states - holds the states that are not None
    non_final_next_states = np.array([s for s in batch.next_state if s is not None])
    non_final_next_states = torch.from_numpy(non_final_next_states).float().to(device)

    # Convert state, action, and reward batches to tensors
    state_batch = torch.from_numpy(np.array(batch.state)).float().to(device)
    action_batch = torch.from_numpy(np.array([ACTIONS.index(action) for action in batch.action])).long().unsqueeze(1).to(device)
    reward_batch = torch.from_numpy(np.array(batch.reward)).float().to(device)

    return state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask

def _compute_next_state_values(self, non_final_next_states, non_final_mask):
    """
    Compute the Q-values for the next states using the target network for non-terminal states.
    
    :param non_final_next_states: Tensors of non-terminal next states.
    :param non_final_mask: Mask indicating non-terminal states.
    :return: Q-values for the next states.
    """
    # Initialize next state values to zero
    next_state_values = torch.zeros(self.batch_size, device=device)
    
    # For non-terminal states, calculate the maximum Q-value predicted by the target network
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
    
    return next_state_values

def _compute_loss(self, state_action_values, expected_state_action_values, weights):
    """
    Compute the smooth L1 loss for the given Q-values, incorporating importance-sampling weights.
    
    :param state_action_values: Predicted Q-values for the state-action pairs.
    :param expected_state_action_values: Target Q-values.
    :param weights: Importance-sampling weights for prioritized replay.
    :return: The calculated loss.
    """
    weights = torch.FloatTensor(weights).unsqueeze(1).to(device)
    loss = (weights * nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')).mean()

    self.wandb_logger.log({"loss": loss.item()})
    self.scheduler.step(loss)
    self.wandb_logger.log({"learning_rate": self.optimizer.param_groups[0]['lr']})

    # Save the loss for plotting
    self.memory.save_loss(loss.item())

    return loss

def _optimize_model_weights(self, loss):
    """
    Perform backpropagation and update model weights.
    
    :param loss: The computed loss from the Bellman equation.
    """
    # Reset gradients
    self.optimizer.zero_grad()

    # Perform backpropagation
    loss.backward()

    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)

    # Update model weights
    self.optimizer.step()

def _update_replay_priorities(self, state_action_values, expected_state_action_values, indices):
    """
    Update priorities in the replay memory based on the TD error.
    
    :param state_action_values: Predicted Q-values for the state-action pairs.
    :param expected_state_action_values: Target Q-values from Bellman equation.
    :param indices: Indices of sampled transitions from the replay memory.
    """
    # Calculate the TD error (absolute difference between predicted and expected Q-values)
    with torch.no_grad():
        td_errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1)).detach().cpu().numpy()

    # Update the priorities in the replay memory
    self.replay_memory.update_priorities(indices, td_errors.squeeze())



def optimize_model_single_transition(self, state, action, reward, next_state):
    """
    Optimize the model using the current transition (state, action, reward, next_state).
    This allows the agent to learn immediately from the current experience.
    """
    #if state is None or next_state is None:
    # updatet to handel tranistions in where the agent dies or the game ends
    if state is None: 
        return

    # Convert to tensors
    state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
    action_tensor = torch.tensor([[ACTIONS.index(action)]], device=device)
    reward_tensor = torch.tensor([reward], device=device)

    # Compute Q(s_t, a) using the policy network
    state_action_values = self.policy_net(state_tensor).gather(1, action_tensor)

    # Compute the expected Q-values depending on whether the next state exists
    expected_state_action_values = _compute_expected_q_values(self, next_state, reward_tensor)

# Perform backpropagation and optimization step on the model based on the computed loss.
    # Compute the loss using smooth L1 (Huber loss)
    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad() # Reset gradients
    loss.backward() # Backpropagation
    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100) # Gradient clipping
    self.optimizer.step() # Update the model weights

    # Log the loss
    self.wandb_logger.log({"loss_single": loss.item()})

def _compute_expected_q_values(self, next_state, reward_tensor):
    """
    Compute the expected Q-values for the next state using the target network.
    If next_state is None (terminal), use the reward directly.
    """
    if next_state is not None:
        next_state_tensor = torch.from_numpy(np.array(next_state)).float().unsqueeze(0).to(device)
        # Compute max Q value for next state (V(s_{t+1})) using the target network
        with torch.no_grad():
            next_state_values = self.target_net(next_state_tensor).max(1)[0]
        # Calculate expected Q-values using the Bellman equation
        return (next_state_values * self.gamma) + reward_tensor
    else:
        # If next_state is None, return just the reward
        return reward_tensor