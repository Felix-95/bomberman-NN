import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def determine_rotation(game_state):
    """
    Determines the rotation angle based on the agent's position.

    :param game_state: The current game state
    """
    # Extract the field and the agent's coordinates
    field = game_state['field']
    height, width = field.shape
    
    # Extract the agent's coordinates
    _, _, _, (agent_x, agent_y) = game_state['self']
    
    # Determine the upper left corner of the field
    half_width = width // 2
    half_height = height // 2
    
    # Is the agent in the upper left quarter? If yes, no rotation is needed
    if agent_x < half_width and agent_y < half_height:
        return 0  # No rotation needed
    
    # Is the agent in the upper right quarter? If yes, 90° rotation
    elif agent_x >= half_width and agent_y < half_height:
        return 90  # Rotation by 90°
    
    # Is the agent in the lower right quarter? If yes, 180° rotation
    elif agent_x >= half_width and agent_y >= half_height:
        return 180  # Rotation by 180°
    
    # Is the agent in the lower left quarter? If yes, 270° rotation
    else:
        return 270  # Rotation by 270°
    
def rotate_point(x, y, width, height, angle):
    """
    Rotates a point (x, y) by a given angle around the center of the field.

    :param x: x-coordinate of the point
    :param y: y-coordinate of the point
    :param width: Width of the field
    :param height: Height of the field
    :param angle: Angle by which to rotate the point
    """
    if angle == 90:
        return y, width - 1 - x
    elif angle == 180:
        return width - 1 - x, height - 1 - y
    elif angle == 270:
        return height - 1 - y, x
    else:
        return x, y  # No rotation

def rotate_game_state(game_state, angle):
    # Extract the field and the explosion map
    field = game_state['field']
    explosion_map = game_state['explosion_map']
    
    # Rotate the field and the explosion map
    rotated_field = np.rot90(field, k=- (angle // 90))
    rotated_explosion_map = np.rot90(explosion_map, k=- (angle // 90))
    
    # Rotate the coordinates of the bombs
    width, height = field.shape
    rotated_bombs = [((rotate_point(x, y, width, height, angle)), t) for (x, y), t in game_state['bombs']]
    
    # Rotate the coordinates of the coins
    rotated_coins = [rotate_point(x, y, width, height, angle) for (x, y) in game_state['coins']]
    
    # Rotate the coordinates of the others
    rotated_others = [(n, s, b, rotate_point(x, y, width, height, angle)) for n, s, b, (x, y) in game_state['others']]
    
    # Rotate the coordinates of the agent
    name, score, can_bomb, (agent_x, agent_y) = game_state['self']
    rotated_self = (name, score, can_bomb, rotate_point(agent_x, agent_y, width, height, angle))
    
    # Create the rotated game state
    rotated_game_state = {
        'round': game_state['round'],
        'step': game_state['step'],
        'field': rotated_field,
        'bombs': rotated_bombs,
        'explosion_map': rotated_explosion_map,
        'coins': rotated_coins,
        'self': rotated_self,
        'others': rotated_others,
        'user_input': game_state['user_input']
    }
    
    return rotated_game_state
def denormalize_action_rotation(action, rotation_angle):
    """
    Reverts the effect of rotation on the directional action.
    
    :param action: The action to denormalize
    :param rotation_angle: The angle by which the game state was rotated
    """
    if action in ["BOMB",  "WAIT"]:
        return action
    
    # Mapping for 90° counterclockwise rotations because we want to reverse the clockwise rotation 
    # from earlier
    if rotation_angle == 90:
        action_mapping = {
            'UP': 'LEFT',
            'LEFT': 'DOWN',
            'DOWN': 'RIGHT',
            'RIGHT': 'UP'
        }
    elif rotation_angle == 180:
        action_mapping = {
            'UP': 'DOWN',
            'DOWN': 'UP',
            'LEFT': 'RIGHT',
            'RIGHT': 'LEFT'
        }
    elif rotation_angle == 270:
        action_mapping = {
            'UP': 'RIGHT',
            'RIGHT': 'DOWN',
            'DOWN': 'LEFT',
            'LEFT': 'UP'
        }
    else:
        # No rotation, return the action as-is
        return action
    
    # Return the denormalized action
    return action_mapping[action]



def mirror_along_main_diagonal(game_state):
    """
    mirrors all values of game_state along their main diagonal axis
    
    2d array: transpose
    points: change x and y position
    """
    # mirror among main diagonal (transpose)
    mirrored_field = np.transpose(game_state["field"])
    mirrored_explosion_map = np.transpose(game_state["explosion_map"])
    
    # mirror coordinates of bombs
    mirrored_bombs = [((y, x), t) for (x, y), t in game_state['bombs']]
    
    mirrored_coins = [(y, x) for (x, y) in game_state['coins']]
    
    name, score, can_bomb, (agent_x, agent_y) = game_state['self']
    mirrored_self = (name, score, can_bomb, (agent_y, agent_x))
    
    mirrored_others = [(n, s, b, (y, x)) for n, s, b, (x, y) in game_state['others']]
    
    # update game state
    mirrored_game_state = game_state.copy() 
    mirrored_game_state['field'] = mirrored_field
    mirrored_game_state['explosion_map'] = mirrored_explosion_map
    mirrored_game_state['bombs'] = mirrored_bombs
    mirrored_game_state['coins'] = mirrored_coins
    mirrored_game_state['self'] = mirrored_self
    mirrored_game_state['others'] = mirrored_others
    
    return mirrored_game_state


def should_mirror_along_main_diagonal(game_state):
    """
    Determines if the game state should be mirrored along the main diagonal axis.

    :param game_state: The current game state
    """
    _, _, _, (agent_x, agent_y) = game_state['self']
    
    # if y > x then the agents is in the top right half of the main diagonal if we think of 
    # the field as a 2d array
    if agent_y > agent_x:
        return True 
    return False

def denormalize_action_mirroring(action):
    """
    Reverts the effect of mirroring on the directional action.

    :param action: The action to denormalize
    """
    # Mapping of mirrored actions (swapping UP/DOWN and LEFT/RIGHT)
    action_mapping = {
        'UP': 'LEFT',
        'LEFT': 'UP',
        'DOWN': 'RIGHT',
        'RIGHT': 'DOWN'
    }
    
    return action_mapping.get(action, action)  # Return the mirrored action or the original if it doesn't need mirroring
