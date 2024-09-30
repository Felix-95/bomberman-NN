import os
import pickle
import random

from .model import MLPQNetwork

import torch
import torch.optim as optim

from .utils import device, rotate_game_state, determine_rotation, should_mirror_along_main_diagonal, mirror_along_main_diagonal
from .utils import denormalize_action_mirroring, denormalize_action_rotation

import numpy as np
import json
from collections import deque
import time

 


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup up the agent. Load necessary data and initialize the model.
    """

    with open("parameters.json", "r") as file:
        self.config = json.load(file)

    # Set the parameters of the model, through the parameters.json file
    self.input_size = self.config["input_size"]
    self.hidden_size1 = self.config["hidden_size1"]
    self.hidden_size2 = self.config["hidden_size2"]
    self.n_actions = self.config["n_actions"]
    self.dropout_prob = self.config["dropout"]

    self.epsilon = self.config["EPS_START"]  # Exploration rate 
    self.epsilon_min = self.config["EPS_END"]  # Minimum exploration rate
    self.epsilon_decay = self.config["EPS_DECAY"]  # Decay rate for exploration rate

    # Initialize the model
    self.policy_net = MLPQNetwork(self.input_size, self.hidden_size1, self.hidden_size2, self.n_actions, self.dropout_prob, train=self.train).to(device)
    
    # Load the model if it exists
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
    else:
        self.logger.info("Loading model from saved state.")
        self.policy_net.load_state_dict(torch.load("my-saved-model.pt", map_location=device, weights_only=True))

    # Initialize the optimizer
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config["lr"])

    # Debug flag to enable/disable logging of debug messages
    self.debug = True

    # Initialize the action logger
    self.action_counts = []
    self.current_episode_actions = np.zeros(len(ACTIONS))

    self.optimizer = optim.Adam(
        self.policy_net.parameters(), lr=self.config["lr"]  # Learning rate, can be adjusted default: 0.0005
    )
    # Define the learning rate scheduler, for more learning at the beginning (bin nicht so zufrieden damit)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.95, patience=75, min_lr=self.config["min_lr"])


def act(self, game_state: dict) -> str:
    """
    Perform the next action given the current game state.

    :param game_state: The dictionary that describes everything on the board.
    """
    # E-greedy action selection
    if self.train and random.random() < self.epsilon:
        if(self.debug):
            self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        if self.debug:
            self.logger.debug("Querying model for action.")

        # Use model to get a prediction
        feautures = state_to_features(self, game_state)
        if feautures is None:
            return np.random.choice(ACTIONS)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(feautures).unsqueeze(0).to(device) # unsqueeze adds a batch dimension

            q_values = self.policy_net(state_tensor).to(device) 
            action = torch.argmax(q_values).item() # get the action with the highest q-value
            self.current_episode_actions[action] += 1
            return ACTIONS[action]


# action definitions:
agent_moves = np.array([(0,-1),(0,1),(-1,0),(1,0)]) # up,down,left,right
move_names = ['UP','DOWN','LEFT','RIGHT']

# IMPORTANT: if you change the features, you need to adapt the input_size in the setup function as well
def state_to_features(self, game_state):
    """
    Converts the game state into a feature vector that can be processed by the MLP.
    
    :param normalized_game_state: A dictionary representing the state of the game.
    :return: A feature vector as a numpy array.
    """
    if game_state is None:
        return None

    features = []
    if(self.debug):
        self.logger.debug(f"### start feature computation")

    def get_coins_vision():
        coin_vision, exist_coin_bool = get_vision_map(game_state, COINS)
        return coin_vision, exist_coin_bool
    def get_crates_vision():
        crates_vision, exist_crate_bool = get_vision_map(game_state, CRATES)
        return crates_vision, exist_crate_bool
    def get_danger_vision():
        danger_vision = get_danger_map(game_state)
        return danger_vision
    
    ### Vision maps:
    # Coin Vision:
    coin_vision, exist_coin_bool = get_coins_vision()
    # Crates Vision: 2D ARRAY representing crates marked as X and obstacles marked as Y
    crates_vision, exist_crate_bool = get_crates_vision()
    # Danger Vision:
    danger_vision = get_danger_vision()
    # Enemy Vision:

    # Agent's position
    _, _, _, (agent_x, agent_y) = game_state["self"]  # Extract the (x, y) coordinates of the agent
    agent_pos = game_state["self"][3]

    if(self.debug):
        self.logger.debug(f"Normalized - agent_x: {agent_x}, agent_y: {agent_y}")

    ### Coins   
    coin_features = np.zeros(8)
    # moves:
    # coins next to move - coins in the immediate vicinity 
    # *0,1,2,3
    for idx, move in enumerate(agent_moves):
        if coin_vision[tuple(np.add(move, agent_pos))] == 1:
            coin_features[idx] = 1
    
    # Calculating the potential of the coins and crates
    if exist_coin_bool == True or exist_crate_bool == True:
        coin_distance, crate_distance = get_coin_crate_potential(agent_pos, coin_vision, crates_vision, exist_crate_bool, exist_coin_bool)
    
    # nearest(potential) coin distance 
    # *4,5,6,7
    if exist_coin_bool == True:
        coin_distance[coin_distance == NOT_FOUND] = 0
        biggest_coin_distance = max(coin_distance)
        # inverse to that shortest distance takes largest value
        inverse_coin_distance = (biggest_coin_distance + 1) - coin_distance
        
        inverse_coin_distance[inverse_coin_distance  == (biggest_coin_distance+1)] = 0
        if max(inverse_coin_distance) != 0:
            normalized_inverse_coin_distance = inverse_coin_distance / max(inverse_coin_distance)
        else:
            normalized_inverse_coin_distance = inverse_coin_distance
        
        coin_features[4] = normalized_inverse_coin_distance[0] # up
        coin_features[5] = normalized_inverse_coin_distance[1] # down
        coin_features[6] = normalized_inverse_coin_distance[2] # left
        coin_features[7] = normalized_inverse_coin_distance[3] # right

    features.extend(coin_features)
        
    ### Crates
    crate_features = np.zeros(8)
    # moves:
    # crates next to move - crates in the immediate vicinity
    # *8,9,10,11
    for idx, move in enumerate(agent_moves):
        if crates_vision[tuple(np.add(move, agent_pos))] == 1:
            crate_features[idx] = 1 
    # nearest(potential) crate distance
    # *12,13,14,15
    if exist_crate_bool == True:
        crate_distance[crate_distance == NOT_FOUND] = 0
        biggest_crate_distance = max(crate_distance)
        # inverse to that shortest distance takes largest value
        inverse_crate_distance = (biggest_crate_distance + 1) - crate_distance
        
        inverse_crate_distance[inverse_crate_distance  == (biggest_crate_distance+1)] = 0
        if max(inverse_crate_distance) != 0:
            normalized_inverse_crate_distance = inverse_crate_distance / max(inverse_crate_distance)
        else:
            normalized_inverse_crate_distance = inverse_crate_distance
        
        crate_features[4] = normalized_inverse_crate_distance[0] # up
        crate_features[5] = normalized_inverse_crate_distance[1] # down
        crate_features[6] = normalized_inverse_crate_distance[2] # left
        crate_features[7] = normalized_inverse_crate_distance[3] # right

    features.extend(crate_features)

    ### Walls
    # check if there are walls in the neighborhood of the player
    # *16,17,18,19
    
    # 0 = wall/bomb (bad)
    # 1 = No wall (good)
    wall_vision = np.ones(4) # up, down, left, right
    for idx, move in enumerate(agent_moves):
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
        

    features.extend(wall_vision)

    ### Bombs
    bomb_neighbor_features = np.zeros(5)
    # moves:
    # actual danger next to agent 
    # *20,21,22,23,24
    min_danger_value_in_neighborhood = 1
    for idx, move in enumerate(agent_moves):
        danger_at_tile = 1 - danger_vision[tuple(np.add(move, agent_pos))][0]
        min_danger_value_in_neighborhood = min(min_danger_value_in_neighborhood, danger_at_tile)
        
        bomb_neighbor_features[idx] = danger_at_tile
        if danger_vision[tuple(np.add(move, agent_pos))][1] == True:
            bomb_neighbor_features[idx] = 0
        elif danger_vision[tuple(np.add(move, agent_pos))][2] == True:
            bomb_neighbor_features[idx] = 0
    
    # actual danger at agent position
    if danger_vision[agent_pos][2] == True:
        bomb_neighbor_features[4] = min_danger_value_in_neighborhood
    else:
        bomb_neighbor_features[4] = 1 - danger_vision[agent_pos][0] # not on bomb

    # nearest safe tile distance
    # *25,26,27,28
    features.extend(bomb_neighbor_features) 
    
    nearest_safe_tile_feature = np.ones(5)
    
    # if current tile is not safe (danger != 1)
    if bomb_neighbor_features[4] != 1:
        # max distance which is escapable is 3 (i guess)
        # therefor we can normalize with 3
        # best escape route (shortest path = 1, good)
        # not escapable in this dir (= 0, bad)
        
        distance_to_safe_tile = get_distance_to_safe_tile(agent_pos, danger_vision)
        
        distance_to_safe_tile[distance_to_safe_tile == NOT_FOUND] = 0
        biggest_distance_to_safe_tile = max(distance_to_safe_tile)
        # inverse to that shortest distance takes largest value
        inverse_distance_to_safe_tile = (biggest_distance_to_safe_tile + 1) - distance_to_safe_tile
        
        inverse_distance_to_safe_tile[inverse_distance_to_safe_tile  == (biggest_distance_to_safe_tile+1)] = 0
        if max(inverse_distance_to_safe_tile) != 0:
            normalized_inverse_distance_to_safe_tile = inverse_distance_to_safe_tile / max(inverse_distance_to_safe_tile)
        else:
            normalized_inverse_distance_to_safe_tile = inverse_distance_to_safe_tile
        
        nearest_safe_tile_feature[0] = normalized_inverse_distance_to_safe_tile[0] # up
        nearest_safe_tile_feature[1] = normalized_inverse_distance_to_safe_tile[1] # down
        nearest_safe_tile_feature[2] = normalized_inverse_distance_to_safe_tile[2] # left
        nearest_safe_tile_feature[3] = normalized_inverse_distance_to_safe_tile[3] # right

    # number of crates in possible bomb range
    # *29
    crates_in_possible_bomb_range = count_possible_crates_in_bomb_range(game_state, agent_x, agent_y, game_state["field"].shape[0], game_state["field"].shape[1], self)
    nearest_safe_tile_feature[4] = crates_in_possible_bomb_range

    features.extend(nearest_safe_tile_feature) 

    # check if agent can drop a bomb
    # *30
    """
    Here we check if the agent can plant a bomb. If the agent can plant a bomb, the feature is set to 1, otherwise to 0.
    We check if the agent has planted a bomb before.
    Furthermore we check, if the agent would plant a bomb, if he can, would he be in danger. If so the feature is set to 0.
    """
    can_plant_bomb = []
    if game_state['self'][2]:
        path_to_safe_tile = existing_path_to_safe_tile(agent_pos, get_future_danger_map(game_state))
        path_to_safe_tile = np.array(path_to_safe_tile)
        # certain death into all directions
        if np.all(path_to_safe_tile == 0):
            can_plant_bomb.append(0)
        # there exists a path to a safe tile => can plant bomb
        else:
            can_plant_bomb.append(1)
    # agent hat noch cooldown
    else:
        can_plant_bomb.append(0)
    features.extend(can_plant_bomb)

    #### Enemy
    enemy_features = np.zeros(4)

    enemy_distance_per_direction = get_beeline_enemy_distance_per_direction(game_state, agent_pos)
    # 1: no enemy in this direction
    # close to 1: enemy far in this direction
    # close 0: enemy close in this direction
    normalized_enemy_distance_per_direction = (enemy_distance_per_direction / ENEMY_SEARCH_RADIUS)
    enemy_features[0] = normalized_enemy_distance_per_direction[0] # up
    enemy_features[1] = normalized_enemy_distance_per_direction[1] # down
    enemy_features[2] = normalized_enemy_distance_per_direction[2] # left
    enemy_features[3] = normalized_enemy_distance_per_direction[3] # right
    
    features.extend(enemy_features)

    return np.array(features)


def get_beeline_enemy_distance_per_direction(game_state, agent_pos):
    # Initialize the enemy distance array
    enemy_distance_array = np.full(4, np.inf)  # [up, down, left, right]
    
    for other in game_state["others"]:
        _, _, _, enemy_pos = other  # Extract the (x, y) coordinates of the enemy
        # Calculate the distance between the agent and the enemy in both directions
        delta_x = enemy_pos[0] - agent_pos[0]
        delta_y = enemy_pos[1] - agent_pos[1]
        
        if abs(delta_x) < ENEMY_SEARCH_RADIUS and abs(delta_y) < ENEMY_SEARCH_RADIUS:
            # Vertical direction
            if delta_y < 0:  # Enemy is above
                enemy_distance_array[0] = min(enemy_distance_array[0], abs(delta_y))
            elif delta_y > 0:  # Enemy is below
                enemy_distance_array[1] = min(enemy_distance_array[1], abs(delta_y))
            # Horizontal direction
            if delta_x < 0:  # Enemy is left
                enemy_distance_array[2] = min(enemy_distance_array[2], abs(delta_x))
            elif delta_x > 0:  # Enemy is right
                enemy_distance_array[3] = min(enemy_distance_array[3], abs(delta_x))

    # Replace all inf values with the maximum search radius
    # This is necessary to normalize the values
    enemy_distance_array[enemy_distance_array == np.inf] = ENEMY_SEARCH_RADIUS
    
    return enemy_distance_array

# Define global variables for general use
CRATES = 2
COINS = 3
ENEMY = 4
OBSTACLE = -1
GOAL = 1
FREE_TILE = 0

NOT_FOUND = 135

NORMALISED_DISTANCE = 135
ENEMY_SEARCH_RADIUS = 6

def get_vision_map(game_state, element):
    """
    Creates a vision map based on the specified element (ENEMY, CRATES, COINS).
    
    :param game_state: The current game state.
    :param element: The element type for which the vision map should be created.
    :return: A tuple containing:
        - A 2D numpy array representing the vision map.
        - A boolean indicating whether at least one GOAL (enemy, crate, coin) was present.
    """
    # Initialize the vision map as a copy of the game field to avoid modifying the original
    vision_map = game_state['field'].copy()
    has_goal = False  # Boolean to check if we have at least one GOAL

    # Mark bombs as obstacles
    for (bomb_x, bomb_y), _ in game_state['bombs']:
        vision_map[bomb_x, bomb_y] = OBSTACLE

    # Handle enemy vision map
    if element == ENEMY:
        # Set crates as obstacles
        vision_map[vision_map == 1] = OBSTACLE
        # Mark enemy positions
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
    Create a vision map for the danger of the bombs and include information about crates and walls.
    :param game_state: The current game state.
    :return: A 2D numpy array where each element is a tuple (danger_value, is_crate, is_wall).
    """
    rows, cols = game_state['field'].shape
    
    # Initialize danger map with tuples (danger_value, is_crate, is_wall)
    danger_map = np.empty((rows, cols), dtype=object)
    
    # Overwrite global variable for local use
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
        danger_map[bomb_x, bomb_y] = (0.0, False, True) # Since the agent can't go through the bomb, we define it as a wall

        # Spread danger along the bomb's explosion range
        for direction in agent_moves:
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

def get_coin_crate_potential(position, coin_field, crate_field, exist_crate_bool, exist_coin_bool): 
    distance_to_coin, distance_to_crate = get_potentials_coin_crate(position, coin_field, crate_field, exist_crate_bool, exist_coin_bool)
    return distance_to_coin, distance_to_crate
####
def get_potentials_coin_crate(position, coin_field, crate_field, exist_crate_bool, exist_coin_bool):
    distance_to_coin = []
    distance_to_crate = []
    for move in agent_moves:
        new_position = position + move
        distance_coin, distance_crate = bfs_to_target_coin_crate(position, new_position, coin_field, crate_field, exist_crate_bool, exist_coin_bool)
        distance_to_coin.append(distance_coin)
        distance_to_crate.append(distance_crate)
    distance_to_coin = np.array(distance_to_coin) # convert in np.array for vectorised operations
    distance_to_crate = np.array(distance_to_crate) # convert in np.array for vectorised operations

    return distance_to_coin, distance_to_crate
def existing_path_to_safe_tile(position, danger_field):
    """
    Calculates the distances to the nearest safe tiles in all four directions.
    :param position: Tuple (x, y) representing the agent's current position.
    :param danger_field: 2D danger map representing the game's danger zones.
    :return: Array with distances to the nearest safe tiles for each possible move direction.
    """
    distance_to_safe_tile_bomb = []

    for move in agent_moves:
        new_position = (position[0] + move[0], position[1] + move[1])  # Move in the current direction
        distance_bomb = NOT_FOUND  # Initialize the distance to NOT_FOUND
        if not danger_field[new_position[0], new_position[1]][1] and not danger_field[new_position[0], new_position[1]][2]:  # Check if the new position is not a wall or crate
            distance_bomb = bfs_to_target_with_danger(position, new_position, agent_moves, danger_field)  # Calculate distance to nearest safe tile
        distance_to_safe_tile_bomb.append(distance_bomb)  # Append the result for the current move
    # Convert to a numpy array for vectorized operations
    distance_to_safe_tile_bomb = np.array(distance_to_safe_tile_bomb)
    distance_to_safe_tile_bomb = distance_to_safe_tile_bomb / NORMALISED_DISTANCE

    # Since 0 means good and 1 means bad, we need to invert the values
    distance_to_safe_tile_bomb = 1 - distance_to_safe_tile_bomb
    return distance_to_safe_tile_bomb

def get_distance_to_safe_tile(position, danger_field):
    """
    Calculates the distances to the nearest safe tiles in all four directions.
    :param position: Tuple (x, y) representing the agent's current position.
    :param danger_field: 2D danger map representing the game's danger zones.
    :return: Array with distances to the nearest safe tiles for each possible move direction.
    """
    distance_to_safe_tile_bomb = []

    for move in agent_moves:
        new_position = (position[0] + move[0], position[1] + move[1])  # Move in the current direction
        distance_bomb = NOT_FOUND  # Initialize the distance to NOT_FOUND
        
        # Check if the new position is not a wall or crate
        if not danger_field[new_position[0], new_position[1]][1] and not danger_field[new_position[0], new_position[1]][2]:  
            distance_bomb = bfs_to_target_with_danger(position, new_position, agent_moves, danger_field)  # Calculate distance to nearest safe tile
        distance_to_safe_tile_bomb.append(distance_bomb)  # Append the result for the current move
    # Convert to a numpy array for vectorized operations
    distance_to_safe_tile_bomb = np.array(distance_to_safe_tile_bomb)

    return distance_to_safe_tile_bomb

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
    rows, cols = coin_field.shape  # same size for coin_field and crate_field

    # Check if the target is out of bounds or an obstacle
    if (target_pos[0] < 0 or target_pos[0] >= rows or 
        target_pos[1] < 0 or target_pos[1] >= cols or 
        crate_field[target_pos[0], target_pos[1]] == OBSTACLE): # same for crate_field
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
        for dx, dy in agent_moves:
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
        for dx, dy in agent_moves:
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

        # If we reach a free tile (danger level == 0), return the distance
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

    # Return float(1000) if no safe tile is found
    return NOT_FOUND

def bomb_with_escape_route(game_state):
    """
    Check if the agent has already planted a bomb and if there is a safe tile to escape to.
    :param game_state: The current game state.
    :return: Boolean indicating if the agent has planted a bomb and there is a safe tile to escape to.
    """
    # Get the danger map
    danger_map = get_future_danger_map(game_state)

    return danger_map
    # Check if there is a safe tile to escape to
    
def get_future_danger_map(game_state):
    """
    Create a vision map for the danger of the bombs and include information about crates and walls.
    It's being used to calculate the danger map, if the agent would plant a bomb.
    :param game_state: The current game state.
    :return: A 2D numpy array where each element is a tuple (danger_value, is_crate, is_wall).
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
    
    BOMB_TIME = 4
    BOMB_RANGE = 3

    # Mark all explosion sites with the highest danger (1.0)
    for x, y in np.argwhere(game_state['explosion_map'] > 0):
        current_danger, is_crate, is_wall = danger_map[x, y]
        danger_map[x, y] = (1.0, is_crate, is_wall)  # Set danger to maximum

    # Calculate bomb danger zones, at the position of the agent with BOMB_TIME
    # Danger value at the bomb's position itself

    current_danger, is_crate, is_wall = danger_map[agent_pos[0], agent_pos[1]]
    danger_map[agent_pos[0], agent_pos[1]] = (0.0, False, True) # Since the agent can't go through the bomb, we define it as a wall

        # Spread danger along the bomb's explosion range
    for direction in agent_moves:
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

def enemy_range_check(vision_map, agent_pos):
    """
    Check if an enemy is in range of the agent.
    :param vision_map: 2D array representing the vision map for the enemies.
    :param agent_pos: Tuple (x, y) representing the agent's current position.
    :return: Boolean indicating if an enemy is in range.
    """

    width, height = vision_map.shape
    agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])

    # Radius of 3 tiles around the agent
    radius = 3
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            new_x, new_y = agent_x + dx, agent_y + dy

            # Ensure the new position is within the bounds of the map
            if 0 <= new_x < width and 0 <= new_y < height:
                # Check if there's an enemy (represented by 1) in the vision map
                if vision_map[new_x, new_y] == 1:
                    return 1

    return 0

def count_possible_crates_in_bomb_range(game_state, bomb_x, bomb_y, field_width, field_height, self):
    """
    Determines if there is a crate within the bomb's explosion range.

    :param game_state: The current state of the game.
    :param bomb_x: The x-coordinate of the bomb.
    :param bomb_y: The y-coordinate of the bomb.
    :param field_width: The width of the game field.
    :param field_height: The height of the game field.
    :return: True if a crate is within the explosion range, False otherwise.
    """
    explosion_range = 3  
    
    field = game_state['field']

    crates = 0

    # check up
    for dy in range(1, explosion_range + 1):
        if self.debug:
            self.logger.info(f"bomb_x: {bomb_x}, bomb_y: {bomb_y}, dy: {dy}")
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
    crates = crates/10
    return crates