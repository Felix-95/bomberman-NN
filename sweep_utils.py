from environment import BombeRLeWorld

import os
from argparse import ArgumentParser
from tqdm import tqdm

import settings as s
from environment import BombeRLeWorld

from argparse import ArgumentParser

import os
from utils import copy_ground_model_if_exists

import wandb
from utils import rename_model

def world_controller(world, n_rounds, *,
                     gui, every_step, turn_based, make_video, update_interval):
    if make_video and not gui.screenshot_dir.exists():
        gui.screenshot_dir.mkdir()
        
    user_input = None
    for _ in tqdm(range(n_rounds)):
        world.new_round()
        while world.running:
            # Advances step (for turn based: only if user input is available)
            if world.running and not (turn_based and user_input is None):
                world.do_step(user_input)
                # total_reward += world.get_total_reward()  # Füge hier das Sammeln der Rewards hinzu
                user_input = None

    print("Sweep finished")

    world.end()


def create_args(agents):
    # Erstelle Argumente mit dynamisch übergebenen Agenten
    argv = [
        'play',  # Sub-Befehl
        '--no-gui',  # GUI deaktivieren
        '--agents', *agents,  # Agenten werden als Argument übergeben
        '--train', '1',  # Training für den ersten Agenten oder mehr
        # '--seed', '123',  # Training für den ersten Agenten oder mehr
    ]

    # ArgumentParser initialisieren
    parser = ArgumentParser()

    # Subparser initialisieren
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    # Run arguments für 'play'
    play_parser = subparsers.add_parser("play")
    agent_group = play_parser.add_mutually_exclusive_group()
    agent_group.add_argument("--my-agent", type=str, help="Play agent of name ... against three rule_based_agents")
    agent_group.add_argument("--agents", type=str, nargs="+", default=["rule_based_agent"] * s.MAX_AGENTS,
                             help="Explicitly set the agent names in the game")
    play_parser.add_argument("--train", default=0, type=int, choices=[0, 1, 2, 3, 4],
                             help="First … agents should be set to training mode")
    play_parser.add_argument("--continue-without-training", default=False, action="store_true")

    play_parser.add_argument("--scenario", default="classic", choices=s.SCENARIOS)

    play_parser.add_argument("--seed", type=int, help="Reset the world's random number generator to a known number for reproducibility")

    play_parser.add_argument("--n-rounds", type=int, default=10, help="How many rounds to play")
    play_parser.add_argument("--save-replay", const=True, default=False, action='store', nargs='?', help='Store the game as .pt for a replay')
    play_parser.add_argument("--match-name", help="Give the match a name")

    play_parser.add_argument("--silence-errors", default=False, action="store_true", help="Ignore errors from agents")

    # Custom arguments
    play_parser.add_argument("--learning-rate", type=float, help="Set the learning rate for the agent")
    play_parser.add_argument("--gamma", type=float, help="Set the gamma for the agent")
    play_parser.add_argument("--batch-size", type=float, help="Set the batch size for the agent")

    group = play_parser.add_mutually_exclusive_group()
    group.add_argument("--skip-frames", default=False, action="store_true", help="Play several steps per GUI render.")
    group.add_argument("--no-gui", default=False, action="store_true", help="Deactivate the user interface and play as fast as possible.")

    # Replay arguments
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("replay", help="File to load replay from")

    # Interaction arguments
    for sub in [play_parser, replay_parser]:
        sub.add_argument("--turn-based", default=False, action="store_true", help="Wait for key press until next movement")
        sub.add_argument("--update-interval", type=float, default=0.1, help="How often agents take steps (ignored without GUI)")
        sub.add_argument("--log-dir", default=os.path.dirname(os.path.abspath(__file__)) + "/logs")
        sub.add_argument("--save-stats", const=True, default=False, action='store', nargs='?', help='Store the game results as .json for evaluation')

        # Video?
        sub.add_argument("--make-video", const=True, default=False, action='store', nargs='?', help="Make a video from the game")

    # Parsen der simulierten Argumente
    args = parser.parse_args(argv)

    return args

def execute_sweep_iteration(map_name, n_rounds, rename_model_bool=True, remove_old=False, wandb_agents = ["ffm_agent"],):
    """
    Führt eine Sweep-Iteration aus.

    :param map_name: Name der Map, auf der das Training durchgeführt wird.
    :param n_rounds: Anzahl der Runden im Sweep.
    :param agents: Liste der Agenten, die am Spiel teilnehmen sollen.
    :param rename_model_bool: Ob das Modell nach dem Sweep umbenannt werden soll (Standard: True).
    :param remove_old: Ob das alte Modell gelöscht werden soll (Standard: False).
    """
    print("The execute_sweep_iteration is now called")
    
    copy_ground_model_if_exists()
    
    wandb.init(project="bomberman-jeff")

    # Agenten aus der Sweep-Konfiguration extrahieren
    wandb_agents = []
    agent_1 = wandb.config.agent_1
    agent_2 = wandb.config.agent_2
    agent_3 = wandb.config.agent_3
    agent_4 = wandb.config.agent_4
    
    if agent_1:
        wandb_agents.append(agent_1)
    if agent_2:
        wandb_agents.append(agent_2)
    if agent_3:
        wandb_agents.append(agent_3)
    if agent_4:
        wandb_agents.append(agent_4)    
        
    # W&B-Konfigurationen auslesen und anwenden
    args = create_args(agents=wandb_agents)

    # Map und Anzahl der Runden anpassen
    args.scenario = map_name
    args.n_rounds = n_rounds
    args.agents = wandb_agents
    
    # Agents für das Spiel initialisieren
    agents = []
    for agent_name in args.agents:
        agents.append((agent_name, len(agents) < args.train))
    
    world = BombeRLeWorld(args, agents)

    # Führe das Training durch
    world_controller(world, args.n_rounds,
                     gui=None, every_step=False, turn_based=False,
                     make_video=False, update_interval=None)
    
    # Modell umbenennen, falls gewünscht
    if rename_model_bool:
        rename_model()
        print("Model renamed")