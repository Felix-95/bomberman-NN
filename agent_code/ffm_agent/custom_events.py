import numpy as np
import events as e

# Custom event definitions

# coins:
MOVED_TO_COIN = "MOVED_TO_COIN"
# Crates:
CRATE_IN_POSSIBLE_EXPLOSION = "CRATE_IN_POSSIBLE_EXPLOSION"
BOMB_PLACED_NEXT_TO_CRATE = "BOMB_PLACED_NEXT_TO_CRATE"
CRATE_DESTROYED_NOT_KILLED_SELF = "CRATE_DESTROYED_NOT_KILLED_SELF"

# Safe/Danger events:
MOVED_ONTO_SAFE_TILE = "MOVED_ONTO_SAFE_TILE"
MOVED_INTO_DANGER = "MOVED_INTO_DANGER"
MOVED_INTO_EXPLOSION = "MOVED_INTO_EXPLOSION"
MOVED_TOWARDS_SAFE_TILE = "MOVED_TOWARDS_SAFE_TILE"
MOVED_AWAY_FROM_SAFE_TILE = "MOVED_AWAY_FROM_SAFE_TILE"
NO_MOVE_WHILE_ON_DANGER_TILE = "NO_MOVE_WHILE_ON_DANGER_TILE"
DID_NOT_CHOOSE_SAVING_MOVE = "DID_NOT_CHOOSE_SAVING_MOVE"

# Bomb related events:
CRATE_IN_POSSIBLE_EXPLOSION = "CRATE_IN_POSSIBLE_EXPLOSION"
BOMB_DROPPED_WITH_NO_ESCAPE = "BOMB_DROPPED_WITH_NO_ESCAPE"

# Self-actions:
WALKING_IN_CIRCLES = "WALKING_IN_CIRCLES"
EXPLORED_NEW_TILE = "EXPLORED_NEW_TILE"
MOVED_AWAY_FROM_DEAD_END = "MOVED_AWAY_FROM_DEAD_END"
INVALID_BOMB_DROP = "INVALID_BOMB_DROP" 

#enemys
GOT_KILLED_BY_OPPONENT = "GOT_KILLED_BY_OPPONENT"


def check_custom_events(self, events, old_features, new_features, old_game_state, new_game_state, self_action, agent_newly_allowed_to_plant_bomb):
    """
    Defines custom events to be used to train the agent.

    :param self: agent
    :param events: List of events to append custom events to
    :param old_features: Feature vector of the old game state
    :param new_features: Feature vector of the new game state
    :param old_game_state: The previous game state
    :param new_game_state: The current game state
    :param self_action: The action taken by the agent
    :param agent_newly_allowed_to_plant_bomb: Flag if agent can theoretically plant bomb
    """
    
    old_agent_danger = old_features[24]  # Danger level at current position (1 = safe, <1 = dangerous)
    
    max_old_safe_tile_distance = max(old_features[25:29])  # Max safe tile distance in old state
    
    # INVALID_BOMB_DROP:  Attempting to drop a bomb when not allowed
    allowed_to_drop_bomb = old_game_state["self"][2]
    if self_action == "BOMB" and allowed_to_drop_bomb == False:
        events.append(e.INVALID_BOMB_DROP)

    if new_features is not None: # all cases execept for agent dies/time is up (end_of_round)
        

        # MOVED_TO_COIN: Agent moved towards a coin
        max_new_safe_tile_distance = max(new_features[25:29])  # Max safe tile distance in new state

        old_coin_direction = old_features[4:8]  # Old state coin potentials (up, down, left, right)
        moved_to_coin = False
        
        if self_action == "UP" and old_coin_direction[0] == 1:
            moved_to_coin = True
        elif self_action == "DOWN" and old_coin_direction[1] == 1:
            moved_to_coin = True
        elif self_action == "LEFT" and old_coin_direction[2] == 1:  
            moved_to_coin = True
        elif self_action == "RIGHT" and old_coin_direction[3] == 1:
            moved_to_coin = True
        
        if moved_to_coin:
            events.append("MOVED_TO_COIN")
            if self.debug:
                self.logger.info("Agent moved closer to a coin")

        # EXPLORED_NEW_TILE: Check if the agent explored a new tile
        new_agent_pos = new_game_state["self"][3]  # Agent's new position (x, y)

        if not self.visited_tiles[new_agent_pos]:
            self.visited_tiles[new_agent_pos] = True
            events.append("EXPLORED_NEW_TILE") 
            if self.debug:
                self.logger.info(f"New tile explored at position {new_agent_pos}")
        
    #--BOMB_DROPPED_WITH_NO_ESCAPE: Check if the agent dropped a bomb with no escape route
        if self_action == "BOMB" and e.INVALID_BOMB_DROP not in events: # agent dropped bomb
            bombed_dropped_and_escape_possible = False
            
            nearest_safe_tile_distances = new_features[25:29]  # Nearest safe tile distances (up, down, left, right)
            for nearest_safe_tile_distance in nearest_safe_tile_distances:
                if nearest_safe_tile_distance > 0:
                    bombed_dropped_and_escape_possible = True
                    break
            if not bombed_dropped_and_escape_possible: # no escape possible
                events.append(BOMB_DROPPED_WITH_NO_ESCAPE)
                self.NO_ESCAPE = True
                if self.debug:
                    self.logger.info("Agent dropped bomb with no escape route")
                
        
            # CRATE_IN_POSSIBLE_EXPLOSION: Agent destroyed crates without killing itself
            elif new_features[29] > 0 and e.INVALID_BOMB_DROP not in events: # number of crates in bomb range (normalized / 10)
                for _ in range(int(new_features[29]*10)): # Number of crates in bomb range
                    events.append(CRATE_IN_POSSIBLE_EXPLOSION)
                if self.debug:
                    self.logger.info("Agent dropped bomb with crate in range")

        #CRATE_DESTROYED_NOT_KILLED_SELF Only Reward Agent for destroying crate, when not killed
        if e.CRATE_DESTROYED in events and e.KILLED_SELF not in events:
            count_crate_destroyed = events.count(e.CRATE_DESTROYED)
            for _ in range(count_crate_destroyed):
                events.append("CRATE_DESTROYED_NOT_KILLED_SELF")
        
        new_agent_danger = new_features[24] # danger at current position
        
        # DID_NOT_CHOOSE_SAVING_MOVE: Agent moved into a tile with a coming bomb explosion and no escape. Except if he planted the bomb and has no escape 
        if (old_agent_danger < 1 # agent was in danger in old game_state
            and max_old_safe_tile_distance == 1 # agent had a possible escape in old game_state
            and max_new_safe_tile_distance != 1 # agent has no escape in new game_state
            and self.NO_ESCAPE == False): # should not be punished when no escape already possible
                
                events.append("DID_NOT_CHOOSE_SAVING_MOVE")
                self.NO_ESCAPE = True
                if self.debug:
                    self.logger.info("Agent moved into a tile with a bomb explosion!")
                    
        # MOVED_ONTO_SAFE_TILE:  Agent moved from danger to safety
        if old_agent_danger < 1 and new_agent_danger == 1:
            events.append("MOVED_ONTO_SAFE_TILE")
            self.NO_ESCAPE = False
            if self.debug:
                self.logger.info("Agent moved into a danger zone!")    

    old_safe_tile_distance = old_features[25:29]  # tile distance in the old state

    # MOVED_INTO_EXPLOSION: Agent moved into an explosion and died
    if old_agent_danger == 1 and e.KILLED_SELF in events:
        old_moves_danger = old_features[20:24]  # Danger levels for possible moves
        moved_to_explosion = False
        
        if self_action == "UP" and old_moves_danger[0] == 0:
            moved_to_explosion = True
        elif self_action == "DOWN" and old_moves_danger[1] == 0:
            moved_to_explosion = True
        elif self_action == "LEFT" and old_moves_danger[2] == 0:
            moved_to_explosion = True
        elif self_action == "RIGHT" and old_moves_danger[3] == 0:
            moved_to_explosion = True

        if moved_to_explosion:
            events.append("MOVED_INTO_EXPLOSION")
            if self.debug:
                self.logger.info("Agent moved into danger and killed")
    
    # MOVED_INTO_DANGER: Agent moved from a safe tile (old safe distance == 0) to a danger zone (new safe distance > 0), except dropping bomb
    new_position_is_worse = False
    
    if new_features is not None:
        new_position_is_worse = 1 > new_features[24]
    elif e.KILLED_SELF in events:
        new_position_is_worse = True
    

    if (old_agent_danger == 1 
        and e.MOVED_INTO_EXPLOSION not in events 
        and new_position_is_worse 
        and e.BOMB_DROPPED not in events): # no negative reward for bomb drop
            
        events.append("MOVED_INTO_DANGER")
        if self.debug:
            self.logger.info("Agent moved into danger")
    
    # Agent is on danger tile and did not move towards safety
    elif (old_agent_danger < 1 # agent is not on a safe tile 
         and MOVED_ONTO_SAFE_TILE not in events # agent did not decide to move onto a safe tile
         and max_old_safe_tile_distance == 1 # and there is a possible safe tile
         and self.NO_ESCAPE == False # and there is escape possible
         and e.KILLED_SELF not in events 
         and e.GOT_KILLED not in events
         ): 
        
        moved_to_safe_tile = False
        
        #--MOVED_TOWARDS_SAFE_TILE: Agent moved closer to the nearest safe tile
        if self_action == "UP" and old_safe_tile_distance[0] == 1:
            moved_to_safe_tile = True
        elif self_action == "DOWN" and old_safe_tile_distance[1] == 1:
            moved_to_safe_tile = True
        elif self_action == "LEFT" and old_safe_tile_distance[2] == 1:
            moved_to_safe_tile = True
        elif self_action == "RIGHT" and old_safe_tile_distance[3] == 1:
            moved_to_safe_tile = True
            
        if moved_to_safe_tile:  
            events.append("MOVED_TOWARDS_SAFE_TILE")
            if self.debug:
                self.logger.info("Agent moved closer to a safe tile")
                
        # NO_MOVE_WHILE_ON_DANGER_TILE: Agent stayed on a danger tile without moving while escape would be possible
        elif (e.INVALID_ACTION in events or e.WAITED in events or self_action == "BOMB") and self.NO_ESCAPE == False:
            events.append("NO_MOVE_WHILE_ON_DANGER_TILE")
            if self.debug:
                self.logger.info("Agent stayed on a danger tile although escape would be possible")
        # MOVED_AWAY_FROM_SAFE_TILE: Agent moved further away from the nearest safe tile (new safe distance increased)
        else: 
            events.append("MOVED_AWAY_FROM_SAFE_TILE")
            if self.debug:
                self.logger.info("Agent moved further away from the nearest safe tile!")

    #ENEMY GOT_KILLED_BY_OPPONENT: Agent was killed by an opponent
    if (e.KILLED_SELF not in events and e.GOT_KILLED in events):
        events.append(e.GOT_KILLED_BY_OPPONENT)
        
    # WALKING_IN_CIRCLES: Agent is stuck in a loop
    if self.no_progress_steps >= 3:
        events.append(WALKING_IN_CIRCLES)
        if self.debug:
            self.logger.info("Agent is walking in circles!")
        self.no_progress_steps = 0

def explosion_next_to_agent(game_state, agent_x, agent_y):
    """
    Determines if there is an explosion within a radius of 3 around the agent

    :param game_state: The current state of the game
    :param agent_x: The x-coordinate of the agent
    :param agent_y: The y-coordinate of the agent
    :return: True if an explosion is nearby, False otherwise
    """
    explosion_map = game_state['explosion_map']
    width, height = explosion_map.shape

    agent_x, agent_y = int(agent_x), int(agent_y)

    # Check each tile in a 3-tile radius around the agent
    radius = 3
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            new_x, new_y = agent_x + dx, agent_y + dy
            # Ensure the new position is within the bounds of the map
            if 0 <= new_x < width and 0 <= new_y < height:
                 if explosion_map[int(new_x), int(new_y)] != 0: # explosion happens
                    return True

    return False
