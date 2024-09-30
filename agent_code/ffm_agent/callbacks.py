import os
import random

from .model import MLPQNetwork

import torch
import torch.optim as optim

from ..rule_based_agent import callbacks as rule_based_callbacks

import json
import numpy as np
from collections import deque
import wandb
import time
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants for the game elements
CRATES = 2
COINS = 3
ENEMY = 4
OBSTACLE = -1
GOAL = 1
FREE_TILE = 0
BOMB_TIME = 4
BOMB_RANGE = 3

NOT_FOUND = 135

NORMALISED_DISTANCE = 135
ENEMY_SEARCH_RADIUS = 6

# Action definitions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
AGENT_MOVES_WORD = ["UP", "DOWN", "LEFT", "RIGHT"]
AGENT_MOVES_VEC = np.array([(0,-1),(0,1),(-1,0),(1,0)]) # up,down,left,right

def setup(self):
    """
    When in training mode, the separate `setup_training` in train.py is called
    after this method

    :param self: FEHLT
    """
    
    self.input_size = 39 # Number of features (First layer of MLP)
    self.n_actions = len(ACTIONS)  # Number of possible actions
    self.current_round = 0
    
    use_wandb = False
    
    self.rng = np.random.default_rng(int(time.time()))
    # self.rng = np.random.default_rng(12)  # seed

    if use_wandb:
        wandb.init(project="bomberman_sweep")
        self.wandb_hyperparameter_search = bool(dict(wandb.config))
    else:
        self.wandb_hyperparameter_search = False
    
    # parameters passed through wandb config
    if self.wandb_hyperparameter_search and use_wandb:
        self.hidden_size1 = wandb.config.hidden_size1 
        self.hidden_size2 = wandb.config.hidden_size2 
        self.epsilon = wandb.config.EPS_START  
        self.epsilon_min = wandb.config.EPS_END  
        self.epsilon_decay = wandb.config.EPS_DECAY  
        self.load_existing_model = wandb.config.load_existing_model 
        self.dropout = wandb.config.dropout
        learning_rate = wandb.config.lr
        lr_patience = wandb.config.lr_patience
        min_lr = wandb.config.min_lr
    # parameter are read from JSON file
    else:    
        with open("parameters.json", "r") as f:
            self.config = json.load(f)
            
        self.load_existing_model = True
        self.hidden_size1 = self.config["hidden_size1"]
        self.hidden_size2 = self.config["hidden_size2"]
        self.epsilon = self.config["EPS_START"]
        self.epsilon_min = self.config["EPS_END"]
        self.epsilon_decay = self.config["EPS_DECAY"]
        self.dropout = self.config["dropout"]
        learning_rate = self.config["lr"]
        lr_patience = self.config["lr_patience"]
        min_lr = self.config["min_lr"]

    # Debug flag - set to False to disable logging
    self.debug = True

    # Initialize the action logger
    self.action_counts = []
    self.current_episode_actions = np.zeros(len(ACTIONS))

    model_path = "my-saved-model.pt"
    if self.load_existing_model and os.path.isfile(model_path):
        print("++ Loading model from saved state.")
        self.logger.info("++ Loading model from saved state.")
        self.policy_net = MLPQNetwork(
            self.input_size, self.hidden_size1, self.hidden_size2,
            self.n_actions, self.dropout, train=self.train
        ).to(device)
        self.policy_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print("++ Setting up model from scratch.")
        self.logger.info("++ Setting up model from scratch.")
        self.policy_net = MLPQNetwork(
            self.input_size, self.hidden_size1, self.hidden_size2,
            self.n_actions, self.dropout, train=self.train
        ).to(device)
        
    # Initialize optimizer and scheduler
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer, mode='min', factor=0.95,
        patience=lr_patience, min_lr=min_lr
    )


def act(self, game_state: dict) -> str:
    """
    Decide on an action based on the current game state.

    :param game_state: The current game state.
    :return: The chosen action as a string.
    """
    
    features = None
    self.used_action_intervention = False

    rule_based_enabled = False
    if self.wandb_hyperparameter_search:
        rule_based_enabled = wandb.config.activate_rule_based_agent

    # Exploration vs. Exploitation trade-off
    if self.train and self.rng.random() < self.epsilon:
        if not rule_based_enabled:
            self.logger.debug("Choosing action purely at random.")
            chosen_action = self.rng.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        else:
            # probability of choosing a random action instead of the one by the rule based agent
            if self.wandb_hyperparameter_search:
                p = wandb.config.randomnes_rule_based_agent
            else:
                p = 0.3

            rule_based_action = rule_based_callbacks.act(self, game_state)

            # rule_based_agent is choosen
            if rule_based_action is not None and self.rng.random() < p:
                self.logger.debug("Using rule-based agent's action.")
                chosen_action = rule_based_action
            else:
                self.logger.debug("Rule based agent is enabled, but choosing action purely at random.")
                chosen_action = self.rng.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
          
    else:
        if self.debug:
            self.logger.debug("Querying model for action.")
        
        # Compute features from game_state
        features = state_to_features(self, game_state)
        self.logger.debug(f"Features: {features}")

        # Query model for Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor.to(device))
            action = torch.argmax(q_values).item()
            # Store action in memory
            self.current_episode_actions[action] += 1
            
            model_chosen_action = ACTIONS[action]  
            final_action = avoid_self_destruction(features, model_chosen_action, game_state)
            if final_action != model_chosen_action:
                self.used_action_intervention = True
            chosen_action = final_action

    if features is None:
        features = state_to_features(self, game_state)
            
    # Adjust action to avoid self-destruction
    if not self.train:
        final_action = avoid_self_destruction(features, chosen_action, game_state)
        if final_action != chosen_action:
                self.used_action_intervention = True
    else:
        final_action = chosen_action
    
    if self.debug:
        self.logger.debug(f"Chose action: {chosen_action}")
        if final_action != chosen_action:
            self.logger.debug(f"To avoid self destrucion switched from {chosen_action} to {final_action}")
            # print(f"***** must choose other action {chosen_action} => {final_action}")
            
    return final_action

# IMPORTANT: if you change the features, you need to adapt the input_size in the setup function as well
def state_to_features(self, game_state):
    """
    Converts the game state into a feature vector.

    :param self: The agent instance.
    :param game_state: A dictionary representing the state of the game.
    :return: A feature vector as a NumPy array.
    """
    if game_state is None:
        return None

    features = []
    feature_dict = {}

    # Agent's position
    _, _, _, (agent_x, agent_y) = game_state["self"]  
    agent_pos = (agent_x, agent_y)
    
    if(self.debug):
        self.logger.debug(f"### start feature computation")
    

    coin_vision, exist_coin_bool = get_vision_map(game_state, COINS)
    crate_vision, exist_crate_bool = get_vision_map(game_state, CRATES)
    enemy_vision, exist_enemy_bool = get_vision_map(game_state, ENEMY)

    # Calculate potentials for coins and crates
    if exist_coin_bool or exist_crate_bool:
        coin_distance_vec, crate_distance_vec = get_potentials_coin_crate(agent_pos, coin_vision, crate_vision, exist_crate_bool, exist_coin_bool)
    else:
        coin_distance_vec = np.full(4, NOT_FOUND)
        crate_distance_vec = np.full(4, NOT_FOUND)
        
    if exist_enemy_bool:
        enemy_distance_vec = get_potentials_enemy(agent_pos, enemy_vision) 
    else:
        enemy_distance_vec = np.full(4, NOT_FOUND)
        
    current_index = 0
    
    # Coin features *0, 1, 2, 3, 4, 5, 6, 7
    try:
        coin_features = extract_coin_features(agent_pos, coin_vision, coin_distance_vec)
        features.extend(coin_features)
        for feature_value in coin_features:
            feature_dict[f"{current_index}_coin_features"] = feature_value
            current_index += 1
    except Exception as e:
        self.logger.error(f"Error extracting coin features: {e}")
        features.extend([0] * 8)
        current_index += 8
    
    # Crate features *8, 9, 10, 11, 12, 13, 14, 15
    try:
        crate_features = extract_crate_features(agent_pos, crate_vision, exist_crate_bool, crate_distance_vec)
        features.extend(crate_features)
        for feature_value in crate_features:
            feature_dict[f"{current_index}_crate_vision"] = feature_value
            current_index += 1
    except Exception as e:
        self.logger.error(f"Error extracting crate features: {e}")
        features.extend([0] * 8)
        current_index += 8
    
    # Wall features *16, 17, 18, 19
    try:
        wall_features = extract_wall_features(game_state, agent_pos)
        features.extend(wall_features)
        for feature_value in wall_features:
            feature_dict[f"{current_index}_wall_vision"] = feature_value
            current_index += 1
    except Exception as e:
        self.logger.error(f"Error extracting wall features: {e}")
        features.extend([0] * 4)
        current_index += 4

    # Danger Feature *20, 21, 22, 23, 24
    try:
        danger_vision = get_danger_map(game_state)
        danger_features = extract_danger_features(agent_pos, danger_vision)
        features.extend(danger_features)
        for feature_value in danger_features:
            feature_dict[f"{current_index}_danger"] = feature_value
            current_index += 1
    except Exception as e:
        self.logger.error(f"Error extracting danger features: {e}")
        features.extend([0] * 5)
        current_index += 5
        
    
    # Safe tile features *25, 26, 27, 28
    try:
        safe_tile_features = extract_safe_tile_features(agent_pos, danger_vision, danger_features, enemy_distance_vec)
        features.extend(safe_tile_features)
        for feature_value in safe_tile_features:
            feature_dict[f"{current_index}_safe_tile"] = feature_value
            current_index += 1
    except Exception as e:
        self.logger.error(f"Error extracting safe tile features: {e}")
        features.extend([0] * 4)
        current_index += 4
    
    # Crates in bomb range *29
    try:
        crates_in_bomb_range = [count_possible_crates_in_bomb_range(game_state, agent_x, agent_y)]
        features.extend(crates_in_bomb_range)
        for feature_value in crates_in_bomb_range:
            feature_dict[f"{current_index}_crates_in_bomb_range"] = feature_value
            current_index += 1
    except Exception as e:
        self.logger.error(f"Error extracting crates in bomb range: {e}")
        features.extend([0])
        current_index += 1
    
    # Can plant bomb *30
    try:
        can_plant_bomb = can_agent_plant_bomb(game_state)
        features.extend(can_plant_bomb)
        for feature_value in can_plant_bomb:
            feature_dict[f"{current_index}_can_plant_bomb"] = feature_value
            current_index += 1
    except Exception as e:
        self.logger.error(f"Error checking can plant bomb: {e}")
        features.extend([0])
        current_index += 1
    
    # Enemy rel features *31, 32, 33, 34
    # Distance in direction up, down, left, right
    try:
        enemy_features = extract_enemy_features(game_state, agent_pos)
        features.extend(enemy_features)
        for feature_value in enemy_features:
            feature_dict[f"{current_index}_enemys_rel"] = feature_value
            current_index += 1
    except Exception as e:
        self.logger.error(f"Error extracting enemy features: {e}")
        features.extend([0] * 4)
        current_index += 4
      
    # Enemy bfs features *35, 36, 37, 38  
    try:
        enemy_features_bfs = extract_enemy_bfs_features(enemy_distance_vec)
        features.extend(enemy_features_bfs)
        for feature_value in enemy_features_bfs:
            feature_dict[f"{current_index}_enemys"] = feature_value
            current_index += 1
    except Exception as e:
        self.logger.error(f"Error extracting enemy_features_bfs: {e}")
        features.extend([0] * 4)
        current_index += 4
        
    if self.debug:
        self.logger.debug(f"Feature dict: {feature_dict}")

    return np.array(features)

def extract_enemy_bfs_features(enemy_distance_vec):
    enemy_distance_vec[enemy_distance_vec == NOT_FOUND] = 0
    biggest_enemy_distance = max(enemy_distance_vec)
    # inverse to that shortest distance takes largest value
    relative_enemy_distance = (biggest_enemy_distance + 1) - enemy_distance_vec
    relative_enemy_distance[relative_enemy_distance  == (biggest_enemy_distance + 1)] = 0
    
    if max(relative_enemy_distance) != 0:
        max_dist = max(relative_enemy_distance)
        inverse_relative_enemy_distance = np.where(
            relative_enemy_distance != 0,
            max_dist - relative_enemy_distance + np.min(relative_enemy_distance[relative_enemy_distance != 0]),
            0)
        normalized_relative_inverse_enemy_distance = inverse_relative_enemy_distance / max(inverse_relative_enemy_distance)
    else:
        normalized_relative_inverse_enemy_distance = relative_enemy_distance
        
    return normalized_relative_inverse_enemy_distance
    

def safe_tile_features_if_bomb(game_state):
    """
    Simulate dropping a bomb and extract safe tile features.
    """
    simulated_game_state = copy.deepcopy(game_state)
    
    _, _, _, agent_pos = simulated_game_state["self"]  
    simulated_game_state["bombs"].append((agent_pos, 3))
    
    danger_vision = get_danger_map(simulated_game_state)
    danger_features = np.full(5, 0.75)
    
    enemy_distance_vec = np.full(4, NOT_FOUND)
    
    safe_tile_features = extract_safe_tile_features(agent_pos, danger_vision, danger_features, enemy_distance_vec)
    return safe_tile_features
    
def avoid_self_destruction(features, action, game_state):
    """
    Adjust the chosen action to avoid self-destruction.
    
    :param features: The feature vector.
    :param action: The originally chosen action.
    :return: The adjusted action.
    """
    # should this be active
    deterministically_evade_death = False
    
    must_choose_other_move = False
    
    if deterministically_evade_death:
        for idx, move in enumerate(AGENT_MOVES_WORD): # all possible moves
            if move == action:
                if not move_is_valid(idx, features):
                    must_choose_other_move = True
                    
                #     action = move
                
                # # 16,17,18,19 wall_vision
                # if (features[25 + idx] == 0 # cetain death in this direction
                #     or features[16 + idx] == 0 # there is a wall
                #     or features[8 + idx] == 1 # there is a crate
                #     or features[31 + idx] == 1/ENEMY_SEARCH_RADIUS): # there is no enemy
                    
                #     must_choose_other_move = True
                    
        if action == "WAIT" or action == "BOMB":
            # *20,21,22,23,24 (up, down, left, right, at agent's position) danger at that tile
            if features[24] < 0.3: # will die this in next iteration if stays here
                must_choose_other_move = True
                
            if action == "BOMB":
                _, _, bomb_allowed, _ = game_state["self"]
                if not bomb_allowed: # agent is not allowed to drop bomb => invalid move
                    must_choose_other_move = True
                    
                # simulate bomb drop at current position and calculate if its safe
                safe_tile_features_if_bomb_array = safe_tile_features_if_bomb(game_state) 
                
                if np.all(safe_tile_features_if_bomb_array == 0):
                    must_choose_other_move = True
                # check if escape exists, if not choose other move
            
        if must_choose_other_move:
            original_move = action
            for idx, move in enumerate(AGENT_MOVES_WORD):
                if move_is_valid(idx, features):
                    action = move
                    
                # if (features[25 + idx] != 0 # move does not lead to death
                #     and features[16 + idx] != 0 # there is no wall
                #     and features[8 + idx] != 1 # there is no crate
                #     and features[31 + idx] != 1/ENEMY_SEARCH_RADIUS): # there is no enemy in the way
                #     action = move
                    
            # action can not be saved by moving => must wait
            if action == original_move: 
                action = "WAIT"
            
    return action

def move_is_valid(idx, features):
    if (features[25 + idx] != 0 # move does not lead to death
        and features[16 + idx] != 0 # there is no wall
        and features[8 + idx] != 1): # there is no crate
        return True
    return False

def extract_coin_features(agent_pos, coin_vision, coin_distance_vec):

    coin_features = np.zeros(8)
    # Coins adjacent to the agent
    for idx, move in enumerate(AGENT_MOVES_VEC):
        new_pos = tuple(np.add(agent_pos, move))
        if coin_vision[new_pos] == 1:
            coin_features[idx] = 1

    # Nearest coin distance
    if coin_distance_vec is not None:
        coin_distance_vec[coin_distance_vec == NOT_FOUND] = 0
        biggest_coin_distance = max(coin_distance_vec)
        # inverse to that shortest distance takes largest value
        inverse_coin_distance = (biggest_coin_distance + 1) - coin_distance_vec
        
        inverse_coin_distance[inverse_coin_distance  == (biggest_coin_distance+1)] = 0
        if max(inverse_coin_distance) != 0:
            normalized_inverse_coin_distance = inverse_coin_distance / max(inverse_coin_distance)
        else:
            normalized_inverse_coin_distance = inverse_coin_distance

        coin_features[4:8] = normalized_inverse_coin_distance

    return coin_features
    
def extract_crate_features(agent_pos, crate_vision, exist_crate_bool, crate_distance_vec):
    crate_features = np.zeros(8)
    # crates next to move - ajdacent crates
    for idx, move in enumerate(AGENT_MOVES_VEC):
        if crate_vision[tuple(np.add(move, agent_pos))] == 1:
            crate_features[idx] = 1
    
    # Nearest crate distance
    if exist_crate_bool:
        crate_distance_vec[crate_distance_vec == NOT_FOUND] = 0
        biggest_crate_distance = max(crate_distance_vec)
        inverse_crate_distance = (biggest_crate_distance + 1) - crate_distance_vec
        
        inverse_crate_distance[inverse_crate_distance  == (biggest_crate_distance+1)] = 0
        if max(inverse_crate_distance) != 0:
            normalized_inverse_crate_distance = inverse_crate_distance / max(inverse_crate_distance)
        else:
            normalized_inverse_crate_distance = inverse_crate_distance
            
        crate_features[4:8] = normalized_inverse_crate_distance

    return crate_features

def extract_wall_features(game_state, agent_pos):
    # check if there are walls in the neighborhood of the player *16,17,18,19
    
        
    # 0 = wall/bomb (bad)
    # 1 = No wall (good)
    wall_vision = np.ones(4) # up, down, left, right
    for idx, move in enumerate(AGENT_MOVES_VEC):
        move_tile_pos = tuple(np.add(move, agent_pos))
        move_tile_value = game_state["field"][move_tile_pos]
        
        if move_tile_value == -1:
            wall_vision[idx] = 0
            
        # bomb is considered a wall because you can't walk through it
        for bomb_pos, _ in game_state["bombs"]:
            if bomb_pos == move_tile_pos:
                wall_vision[idx] = 0
                
        # player is considered wall because you can't walk through it
        for _, _,_ , other_pos  in game_state["others"]:
            if other_pos == move_tile_pos:
                wall_vision[idx] = 0
                
    return wall_vision

def extract_danger_features(agent_pos, danger_vision):
    danger_feature = np.zeros(5)
    # actual danger next to agent *20,21,22,23,24 (up, down, left, right, at agent's position)
    min_danger_value_in_neighborhood = 1
    for idx, move in enumerate(AGENT_MOVES_VEC):
        danger_at_tile = 1 - danger_vision[tuple(np.add(move, agent_pos))][0]
        min_danger_value_in_neighborhood = min(min_danger_value_in_neighborhood, danger_at_tile)
        
        danger_feature[idx] = danger_at_tile
        # Check for walls or crates
        
        if danger_vision[tuple(np.add(move, agent_pos))][1] == True:
            danger_feature[idx] = 0
        elif danger_vision[tuple(np.add(move, agent_pos))][2] == True:
            danger_feature[idx] = 0
    if danger_vision[agent_pos][2]:
        danger_feature[4] = min_danger_value_in_neighborhood
    else:
        danger_feature[4] = 1 - danger_vision[agent_pos][0]
        
    return danger_feature
        
def extract_safe_tile_features(agent_pos, danger_vision, danger_features, enemy_distance_vec):
    # nearest safe tile distance *25,26,27,28
    nearest_safe_tile_feature = np.ones(4)
    
    # if current tile is not safe (danger != 1)
    if danger_features[4] != 1:
        # max distance which is escapable is 3
        # therefor we can normalize with 3
        # best escape route (shortest path = 1, good)
        # not escapable in this dir (= 0, bad)
        distance_to_safe_tile = get_distance_to_safe_tile(agent_pos, danger_vision)
        distance_to_safe_tile[distance_to_safe_tile == NOT_FOUND] = 0
        
        for idx, safe_tile_distance_in_direction in enumerate(distance_to_safe_tile):
            danger_level = danger_features[idx]
            
            maximum_distance_which_can_be_reached = (danger_level * 4 ) + 1 # calulate how many steps there are until explosion
                
            if safe_tile_distance_in_direction > maximum_distance_which_can_be_reached: # tile can not be reached in time
                distance_to_safe_tile[idx] = 0 # tile is not safe

        biggest_distance_to_safe_tile = max(distance_to_safe_tile)
        # inverse to that shortest distance takes largest value
        inverse_distance_to_safe_tile = (biggest_distance_to_safe_tile + 1) - distance_to_safe_tile
        
        inverse_distance_to_safe_tile[inverse_distance_to_safe_tile  == (biggest_distance_to_safe_tile+1)] = 0
        
        if max(inverse_distance_to_safe_tile) != 0:
            normalized_inverse_distance_to_safe_tile = inverse_distance_to_safe_tile / max(inverse_distance_to_safe_tile)
        else:
            normalized_inverse_distance_to_safe_tile = inverse_distance_to_safe_tile
        
        nearest_safe_tile_feature = normalized_inverse_distance_to_safe_tile
    else:
        # if agent is on safe tile, he should still not get the message to move away from it
        for idx, move in enumerate(AGENT_MOVES_VEC):
            danger_at_tile = 1 - danger_vision[tuple(np.add(move, agent_pos))][0]
            if danger_at_tile != 1: # if tile is not safe
                nearest_safe_tile_feature[idx] = 0
                
    # update safe tile distance base on position of enemies if there are multiple save ways
    if (np.count_nonzero(nearest_safe_tile_feature == 1) > 1 
        and np.count_nonzero(nearest_safe_tile_feature == 1) != 4
        and not np.all(enemy_distance_vec == NOT_FOUND) # a enemy was found
        ):
        # Enemy features (up, down, left, right)
        safe_tile_indices = np.where(nearest_safe_tile_feature == 1)[0]
        
        min_distance = float('inf') 
        nearest_safe_tile_index = None
        
        # Iterate over the indices of safe tiles
        for index in safe_tile_indices:
            if enemy_distance_vec[index] < min_distance:
                min_distance = enemy_distance_vec[index]
                nearest_safe_tile_index = index
                
        # smallest value != 0 and 1
        possible_values = [val for val in nearest_safe_tile_feature if 0 < val < 1]
        
        if possible_values:
            smallest_current_non_zero_save_tile_distance = min(possible_values)
        else:
            smallest_current_non_zero_save_tile_distance = 0.5

        # Set the nearest safe tile to the min_value
        if nearest_safe_tile_index is not None:
            nearest_safe_tile_feature[nearest_safe_tile_index] = smallest_current_non_zero_save_tile_distance
        
    return nearest_safe_tile_feature

def get_potentials_enemy(position, enemy_vision):
    distance_to_enemy = []
    for move in AGENT_MOVES_VEC:
        new_position = position + move
        distance_enemy = bfs_to_target_enemy(position, new_position, enemy_vision)
        distance_to_enemy.append(distance_enemy)
    distance_to_enemy = np.array(distance_to_enemy) 
    return distance_to_enemy

def can_agent_plant_bomb(game_state):
    _, _, _, agent_pos = game_state["self"]  
    
    can_plant_bomb = []
    if game_state['self'][2]:
        path_to_safe_tile = existing_path_to_safe_tile(agent_pos, get_future_danger_map(game_state))
        path_to_safe_tile = np.array(path_to_safe_tile)
        # certain death into all directions

        if np.all(path_to_safe_tile == 0):
            can_plant_bomb.append(0)
        else:
            can_plant_bomb.append(1)
    else:
        can_plant_bomb.append(0)
        
    return can_plant_bomb

def extract_enemy_features(game_state, agent_pos):
    enemy_features = np.zeros(4) 
    # enemy next to move *31,32,33,34
    enemy_distance_per_direction = get_beeline_enemy_distance_per_direction(game_state, agent_pos)
    # 1: no enemy in this direction
    # close to 1: enemy far in this direction
    # close 0: enemy close in this direction
    normalized_enemy_distance_per_direction = (enemy_distance_per_direction / ENEMY_SEARCH_RADIUS)
    enemy_features = normalized_enemy_distance_per_direction
    
    return enemy_features


def get_beeline_enemy_distance_per_direction(game_state, agent_pos):
    """
    Calculate beeline distances to enemies in each direction.
    """
    enemy_distance_array = np.full(4, np.inf)  # [up, down, left, right]
    
    for other in game_state["others"]:
        _, _, _, enemy_pos = other  
        delta_x = enemy_pos[0] - agent_pos[0]
        delta_y = enemy_pos[1] - agent_pos[1]
        
        if abs(delta_x) < ENEMY_SEARCH_RADIUS and abs(delta_y) < ENEMY_SEARCH_RADIUS:
            if delta_y < 0:  # Enemy is up
                enemy_distance_array[0] = min(enemy_distance_array[0], abs(delta_y))
            elif delta_y > 0:  # Enemy is down
                enemy_distance_array[1] = min(enemy_distance_array[1], abs(delta_y))
            if delta_x < 0:  # Enemy is left
                enemy_distance_array[2] = min(enemy_distance_array[2], abs(delta_x))
            elif delta_x > 0:  # Enemy is right
                enemy_distance_array[3] = min(enemy_distance_array[3], abs(delta_x))

    enemy_distance_array[enemy_distance_array == np.inf] = ENEMY_SEARCH_RADIUS
    return enemy_distance_array

def get_vision_map(game_state, element):
    """
    Creates a vision map based on the specified element (ENEMY, CRATES, COINS).

    :param element: The element type for which the vision map should be created.
    :return: A tuple containing the vision map and a boolean indicating the presence of the element.
    """
    vision_map = game_state['field'].copy()
    has_goal = False

    # Mark bombs as obstacles
    for (bomb_x, bomb_y), _ in game_state['bombs']:
        vision_map[bomb_x, bomb_y] = OBSTACLE

    # Handle enemy vision map
    if element == ENEMY:
        # Set crates as obstacles
        vision_map[vision_map == 1] = OBSTACLE
        for _, _, _, (enemy_x, enemy_y) in game_state['others']:
            vision_map[enemy_x, enemy_y] = GOAL
            has_goal = True  # Mark that at least one enemy was found
        return vision_map, has_goal

    # Mark enemies as obstacles
    for _, _, _, (enemy_x, enemy_y) in game_state['others']:
        vision_map[enemy_x, enemy_y] = OBSTACLE

    if element == CRATES:
        # Crates vision map
        if np.any(vision_map == GOAL):  # Check if there's at least one crate
            has_goal = True
        return vision_map, has_goal
    elif element == COINS:
        # Set crates as obstacles
        vision_map[vision_map == 1] = OBSTACLE
        # Mark coin positions as goals
        for coin_x, coin_y in game_state['coins']:
            vision_map[coin_x, coin_y] = GOAL
            has_goal = True  # Mark that at least one coin was found
        return vision_map, has_goal
    else:
        raise ValueError(f"Invalid element type '{element}' for vision map. Use 'ENEMY', 'CRATES', or 'COINS'.")

def get_danger_map(game_state):
    """
    Create a vision map representing the danger levels from bombs.

    :return: A 2D numpy array where each element is a tuple (danger_value, is_crate, is_wall).
    """
    rows, cols = game_state['field'].shape
    
    # Initialize danger map with tuples (danger_value, is_crate, is_wall)
    danger_map = np.empty((rows, cols), dtype=object)
    
    CRATE = 1
    WALL = -1
    # Initialize the map: Set default values (0.0 for danger, False for crate, False for wall)
    for x in range(rows):
        for y in range(cols):
            is_crate = game_state['field'][x, y] == CRATE
            is_wall = game_state['field'][x, y] == WALL
            danger_map[x, y] = (0.0, is_crate, is_wall)
    
    BOMB_TIME = 4
    BOMB_RANGE = 3

    # Mark all explosion sites with the highest danger (1.0)
    for x, y in np.argwhere(game_state['explosion_map'] > 0):
        current_danger, is_crate, is_wall = danger_map[x, y]
        danger_map[x, y] = (1.0, is_crate, is_wall)  # Set danger to maximum

    # Calculate bomb danger zones
    for (bomb_x, bomb_y), timer in game_state['bombs']:
        # Danger value at the bomb's position itself
        current_danger, is_crate, is_wall = danger_map[bomb_x, bomb_y]
        # Since the agent can't go through the bomb, we define it as a wall
        danger_map[bomb_x, bomb_y] = (0.0, False, True) 

        # Spread danger along the bomb's explosion range
        for direction in AGENT_MOVES_VEC:
            for step_in_range in range(1, BOMB_RANGE + 1):
                bomb_zone = direction * step_in_range + np.array([bomb_x, bomb_y])

                # Check if bomb_zone is within bounds
                if not (0 <= bomb_zone[0] < rows and 0 <= bomb_zone[1] < cols):
                    break  # If out of bounds, stop checking this direction

                # Get current values at the bomb zone
                current_danger, is_crate, is_wall = danger_map[bomb_zone[0], bomb_zone[1]]

                # Check for walls
                if game_state['field'][bomb_zone[0], bomb_zone[1]] == WALL:
                    break  # Bomb's explosion is blocked by a wall
                
                # Calculate relative danger based on the bomb's timer
                relative_danger_timer = (BOMB_TIME - timer) / BOMB_TIME

                # Update the danger map if the current danger is higher than the existing value
                if current_danger < relative_danger_timer:
                    danger_map[bomb_zone[0], bomb_zone[1]] = (relative_danger_timer, is_crate, is_wall)

    return danger_map

def get_potentials_coin_crate(position, coin_field, crate_field, exist_crate_bool, exist_coin_bool):
    distance_to_coin = []
    distance_to_crate = []
    for move in AGENT_MOVES_VEC:
        new_position = position + move
        distance_coin, distance_crate = bfs_to_target_coin_crate(position, new_position, coin_field, crate_field, exist_crate_bool, exist_coin_bool)
        distance_to_coin.append(distance_coin)
        distance_to_crate.append(distance_crate)

    return np.array(distance_to_coin), np.array(distance_to_crate)

def existing_path_to_safe_tile(position, danger_field):
    """
    Calculates the distances to the nearest safe tiles in all four directions.
    :param position: Tuple (x, y) representing the agent's current position.
    :param danger_field: 2D danger map representing the game's danger zones.
    :return: Array with distances to the nearest safe tiles for each possible move direction.
    """
    distance_to_safe_tile = []

    for move in AGENT_MOVES_VEC:
        new_position = (position[0] + move[0], position[1] + move[1])  # Move in the current direction
        distance_bomb = NOT_FOUND  # Initialize the distance to NOT_FOUND
        if not danger_field[new_position[0], new_position[1]][1] and not danger_field[new_position[0], new_position[1]][2]:  # Check if the new position is not a wall or crate
            distance_bomb = bfs_to_target_with_danger(position, new_position, AGENT_MOVES_VEC, danger_field)  # Calculate distance to nearest safe tile
        distance_to_safe_tile.append(distance_bomb)  # Append the result for the current move
    # Convert to a numpy array for vectorized operations
    distance_to_safe_tile = np.array(distance_to_safe_tile)
    distance_to_safe_tile = distance_to_safe_tile / NORMALISED_DISTANCE

    # Since 0 means good and 1 means bad, we need to invert the values
    distance_to_safe_tile = 1 - distance_to_safe_tile
    return distance_to_safe_tile

def get_distance_to_safe_tile(position, danger_field):
    """
    Calculates the distances to the nearest safe tiles in all four directions.
    :param position: Tuple (x, y) representing the agent's current position.
    :param danger_field: 2D danger map representing the game's danger zones.
    :return: Array with distances to the nearest safe tiles for each possible move direction.
    """
    distance_to_safe_tile = []

    for move in AGENT_MOVES_VEC:
        new_position = (position[0] + move[0], position[1] + move[1])  # Move in the current direction
        distance_bomb = NOT_FOUND  # Initialize the distance to NOT_FOUND
        
        # Check if the new position is not a wall or crate
        if not danger_field[new_position[0], new_position[1]][1] and not danger_field[new_position[0], new_position[1]][2]:  
            distance_bomb = bfs_to_target_with_danger(position, new_position, AGENT_MOVES_VEC, danger_field)  # Calculate distance to nearest safe tile
        distance_to_safe_tile.append(distance_bomb)  # Append the result for the current move
    # Convert to a numpy array for vectorized operations
    distance_to_safe_tile = np.array(distance_to_safe_tile)

    return distance_to_safe_tile

def bfs_to_target_coin_crate(start_pos, target_pos, coin_field, crate_field, exist_crate_bool, exist_coin_bool):
    """
    Performs a BFS to find the shortest path from start_pos to a GOAL on the field (COIN and CRATE).

    :param start_pos: Tuple (x, y) representing the starting position of the agent.
    :param target_pos: Tuple (x, y) representing the new target position (after the move).
    :param coin_field: 2D array representing the vision map for coins.
    :param crate_field: 2D array representing the vision map for crates.
    :param exist_crate_bool: Boolean indicating if crates exist.
    :param exist_coin_bool: Boolean indicating if coins exist.
    :return: The distance to the nearest coin and crate, or (0, 0) if no target is found.
    """
    rows, cols = coin_field.shape

    # Check if the target is out of bounds or an obstacle
    if (target_pos[0] < 0 or target_pos[0] >= rows or 
        target_pos[1] < 0 or target_pos[1] >= cols or 
        crate_field[target_pos[0], target_pos[1]] == OBSTACLE):
        return float(NOT_FOUND), float(NOT_FOUND)
    if (coin_field[target_pos[0], target_pos[1]] == OBSTACLE):
        exist_coin_bool = False
    # BFS setup
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    queue = deque([(target_pos, 1)])  # Initialize queue with target position and distance 1
    visited[start_pos[0]][start_pos[1]] = True

    distance_coin = float(NOT_FOUND)  # Initialize default values
    distance_crate = float(NOT_FOUND)

    coin_bool = False
    crate_bool = False

    while queue:
        (x, y), distance = queue.popleft()
        
        # If we reach the target (coin/crate), save the distance
        if coin_field[x][y] == GOAL and coin_bool == False: 
            distance_coin = distance
            coin_bool = True
        if coin_bool == True and exist_crate_bool == False:
            return distance_coin, float(NOT_FOUND)
        
        if crate_field[x][y] == GOAL and crate_bool == False: 
            distance_crate = distance 
            crate_bool = True
        if crate_bool == True and exist_coin_bool == False:
            return float(NOT_FOUND), distance_crate
        
        # If we have found both targets, return the distances
        if coin_bool == True and crate_bool == True:
            return distance_coin, distance_crate
        
        
        # Explore neighboring positions
        for dx, dy in AGENT_MOVES_VEC:
            new_x, new_y = x + dx, y + dy

            if 0 <= new_x < rows and 0 <= new_y < cols and not visited[new_x][new_y]:
                # We use the coin_field, since crates are markes as obstacles in the coin_field 
                # and coins aren't marked in crate field at all
                if crate_field[new_x][new_y] == GOAL and crate_bool == False:  # Move to target tile
                    distance_crate = distance + 1
                    crate_bool = True
                if coin_field[new_x][new_y] == FREE_TILE or coin_field[new_x][new_y] == GOAL:  # Move to target tile
                    visited[new_x][new_y] = True
                    queue.append(((new_x, new_y), distance + 1))
        
    return distance_coin, distance_crate  
  
def bfs_to_target_enemy(start_pos, target_pos, field):
    """
    Performs a BFS to find the shortest path from start_pos to a GOAL on the field.

    :param start_pos: Tuple (x, y) representing the starting position of the agent.
    :param target_pos: Tuple (x, y) representing the new target position (after the move).
    :param field: 2D array representing the game field (vision map).
                - 1 represents the GOAL.
                - 0 represents free tiles.
                - -1 represents obstacles (like walls, bombs, players, if coin_map also crates).
    :return: The distance to the nearest GOAL, or a value (float(0)) if no GOAL is found.
    """
    rows, cols = field.shape

    # Check if the target is out of bounds or an obstacle
    if (target_pos[0] < 0 or target_pos[0] >= rows or 
        target_pos[1] < 0 or target_pos[1] >= cols or 
        field[target_pos[0], target_pos[1]] == OBSTACLE):
        return float(NOT_FOUND)

    # BFS setup
    visited = np.zeros((rows, cols), dtype=bool)
    queue = deque([(target_pos, 1)])  # Initialize queue with target position and distance 1
    visited[start_pos[0]][start_pos[1]] = True

    while queue:
        (x, y), distance = queue.popleft()
        
        # If we reach the target (coin/crate), return the distance
        if field[x][y] == GOAL:
            return distance
        
        # Explore neighboring positions
        for dx, dy in AGENT_MOVES_VEC:
            new_x, new_y = x + dx, y + dy

            if 0 <= new_x < rows and 0 <= new_y < cols and not visited[new_x][new_y]:
                if field[new_x][new_y] == FREE_TILE or field[new_x][new_y] == GOAL:  # Move to target tile
                    visited[new_x][new_y] = True
                    queue.append(((new_x, new_y), distance + 1))

    return float(NOT_FOUND)  # If no target is found

def bfs_to_target_with_danger(start_pos, target_pos, agent_moves, danger_map):
    """
    Performs a BFS to find the nearest safe tile (free of obstacles and low danger) from the agent's current position.

    :param start_pos: Tuple (x, y) representing the starting position of the agent.
    :param agent_moves: List of possible agent moves [(dx, dy), ...].
    :param danger_map: 2D array where:
                       -1 represents obstacles (walls, crates),
                       0 represents free tiles,
                       0.25 - 1 represents varying danger levels (the higher the value, the more dangerous the tile).
    :return: The distance to the nearest free and safe tile or float('inf') if none is found.
    """
    rows, cols = danger_map.shape

    # BFS setup with queue
    visited = np.zeros((rows, cols), dtype=bool)
    queue = deque([(target_pos, 1)])  # (position, distance)
    visited[start_pos[0]][start_pos[1]] = True

    while queue:
        (x, y), distance = queue.popleft()

        if danger_map[x][y][0] == 0:
            return distance

        # Explore neighboring positions
        for move in agent_moves:
            new_x, new_y = x + move[0], y + move[1]

            # Check if the new position is within bounds and not visited
            if 0 <= new_x < rows and 0 <= new_y < cols and not visited[new_x][new_y]:
                # Only consider tiles that are not obstacles (-1) and have low danger (< 1.0)
                if not danger_map[new_x][new_y][1] and not danger_map[new_x][new_y][2] and danger_map[new_x][new_y][0] < 1.0:
                    visited[new_x][new_y] = True
                    queue.append(((new_x, new_y), distance + 1))

    return NOT_FOUND
    
def get_future_danger_map(game_state):
    """
    Create a danger map simulating a bomb dropped by the agent.
    """
    rows, cols = game_state['field'].shape
    agent_pos = game_state['self'][3]
    # Initialize danger map with tuples (danger_value, is_crate, is_wall)
    danger_map = np.empty((rows, cols), dtype=object)
    
    CRATE = 1
    WALL = -1
    # Initialize the map: Set default values (0.0 for danger, False for crate, False for wall)
    for x in range(rows):
        for y in range(cols):
            is_crate = game_state['field'][x, y] == CRATE
            is_wall = game_state['field'][x, y] == WALL
            danger_map[x, y] = (0.0, is_crate, is_wall)

    # Mark all explosion sites with the highest danger (1.0)
    for x, y in np.argwhere(game_state['explosion_map'] > 0):
        current_danger, is_crate, is_wall = danger_map[x, y]
        danger_map[x, y] = (1.0, is_crate, is_wall)  # Set danger to maximum

    # Calculate bomb danger zones, at the position of the agent with BOMB_TIME
    # Danger value at the bomb's position itself

    current_danger, is_crate, is_wall = danger_map[agent_pos[0], agent_pos[1]]
    danger_map[agent_pos[0], agent_pos[1]] = (0.0, False, True) # Since the agent can't go through the bomb, we define it as a wall

        # Spread danger along the bomb's explosion range
    for direction in AGENT_MOVES_VEC:
        for step_in_range in range(1, BOMB_RANGE + 1):
            bomb_zone = direction * step_in_range + np.array([agent_pos[0], agent_pos[1]])

            # Check if bomb_zone is within bounds
            if not (0 <= bomb_zone[0] < rows and 0 <= bomb_zone[1] < cols):
                break  # If out of bounds, stop checking this direction

                # Get current values at the bomb zone
            current_danger, is_crate, is_wall = danger_map[bomb_zone[0], bomb_zone[1]]

                # Check for walls
            if game_state['field'][bomb_zone[0], bomb_zone[1]] == WALL:
                  break  # Bomb's explosion is blocked by a wall
                
                # Calculate relative danger based on the bomb's timer
            relative_danger_timer = 1 / BOMB_TIME

                # Update the danger map if the current danger is higher than the existing value
            if current_danger < relative_danger_timer:
                danger_map[bomb_zone[0], bomb_zone[1]] = (relative_danger_timer, is_crate, is_wall)

    return danger_map

def count_possible_crates_in_bomb_range(game_state, bomb_x, bomb_y):
    """
    Count the number of crates within the bomb's explosion range.
    """
    explosion_range = 3  
    
    field = game_state['field']

    crates = 0
    field_width, field_height = field.shape

    # check up
    for dy in range(1, explosion_range + 1):
        if bomb_y - dy >= 0 and field[bomb_x, bomb_y - dy] != -1:  # -1 for wall so we skip them
            if field[bomb_x, bomb_y - dy] == 1:  # 1 for crate
                crates += 1
        else:
            break

    # check down
    for dy in range(1, explosion_range + 1):
        if bomb_y + dy < field_height and field[bomb_x, bomb_y + dy] != -1:
            if field[bomb_x, bomb_y + dy] == 1:
                crates += 1
        else:
            break

    # check left
    for dx in range(1, explosion_range + 1):
        if bomb_x - dx >= 0 and field[bomb_x - dx, bomb_y] != -1:
            if field[bomb_x - dx, bomb_y] == 1:
                crates += 1
        else:
            break

    # check right
    for dx in range(1, explosion_range + 1):
        if bomb_x + dx < field_width and field[bomb_x + dx, bomb_y] != -1:
            if field[bomb_x + dx, bomb_y] == 1:
                crates += 1
        else:
            break

    crates = crates / 10  # Normalize by max possible amount of crates
    return crates
