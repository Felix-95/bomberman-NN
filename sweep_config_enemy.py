sweep_config = {
    'method': 'grid',  # Bayesian Optimization
    'metric': {
        'name': 'rewards',
        'goal': 'maximize'
    },
    # 'early_terminate': {
    #     'type': 'hyperband',  # Use Hyperband for early stopping
    #     'min_iter': 10,       # Minimum iterations before stopping underperforming runs
    #     's': 3             # Specifies the aggressiveness of early stopping
    # },
    'parameters': {
        # run parameters
        'map': {
            'value': 'classic'  # Fixierte Map für diesen Sweep
        },
        'n_rounds': {
            'value': 100  # Anzahl der Runden je Run
            # 2500 
        },
        'activate_rule_based_agent': {
            'values': [1]  # Ob Rule Based Agent aktiviert ist
        },
        'randomnes_rule_based_agent': {
            'values': [0.2]  # Wahrscheinlichkeit, dass RBA Aktion ausgewählt wird und nicht random
        },
        'run_quantity': {
            'value': 1 # Anzahl der Runs pro Sweep
        },
        'load_existing_model': {
            'value': 0  # Wahrscheinlichkeit, dass RBA Aktion ausgewählt wird und nicht random
        },
        'agent_1': {
            'value': 'ffm_agent'  # Agent 1, muss gesetzt sein
        },
        'agent_2': {
            'value': ''  # Optionaler Agent 2
            # 'value': ''  # Optionaler Agent 2
        },
        'agent_3': {
            'value': ''  # Optionaler Agent 3 (leer, falls nicht gesetzt)
            # 'value': ''  # Optionaler Agent 2
        },
        'agent_4': {
            'value': ''  # Optionaler Agent 4 (leer, falls nicht gesetzt)
            # 'value': ''  # Optionaler Agent 2
        },
        
## rewards 
    # coins
    'MOVED_TO_COIN': {
    'values': [7]
    },
    'COIN_COLLECTED': {
        'values': [40]
    },
    'EXPLORED_NEW_TILE': {
        'values': [3]
    },
    'CRATE_DESTROYED_NOT_KILLED_SELF': {
        'values': [7]
    },
    # safe/danger
    'MOVED_ONTO_SAFE_TILE': {
        'values': [10]
    },
    'MOVED_INTO_DANGER': {
        'values': [-20]
    },
    'MOVED_INTO_EXPLOSION': {
        'values': [-50] 
    },
    'MOVED_TOWARDS_SAFE_TILE': {
        'values': [4]
    },
    'MOVED_AWAY_FROM_SAFE_TILE': {
        'values': [-8]
    },
    'NO_MOVE_WHILE_ON_DANGER_TILE': {
        'values': [-6]  
    },
    'DID_NOT_CHOOSE_SAVING_MOVE': {
        'values': [-60] 
    },
    # bomb related
    'BOMB_DROPPED_WITH_NO_ESCAPE': {
        'values': [-60] 
    },
    'CRATE_IN_POSSIBLE_EXPLOSION': {
        'values': [4]
    },
    'KILLED_SELF': {
        'values': [-100]
    },
    'WAITED': {
        'values': [-3]
    },
    'INVALID_ACTION': {
        'values': [-10]
    },
    'MOVED_UP': {
        'values': [-1]
    },
    'MOVED_DOWN': {
        'values': [-1]
    },
    'MOVED_LEFT': {
        'values': [-1]
    },
    'MOVED_RIGHT': {
        'values': [-1]
    },
    'WALKING_IN_CIRCLES': {
        'values': [-40]
    },
    'INVALID_BOMB_DROP': {
        'values': [-10]
    },
    'SURVIVED_ROUND': {
        'values': [150]
    },
    'KILLED_OPPONENT': {
        'values': [140]
    },
    'GOT_KILLED_BY_OPPONENT': {
        'values': [-150]
    },
    
    ## meta parameters
    'hidden_size1': {'values': [256]},
    'hidden_size2': {'values': [256]},
    'dropout': {'values': [0.15]}, 
    'BATCH_SIZE': {'values': [128]}, 
    'GAMMA': {'values': [0.7]},  # Discount
    
    ## training duration dependent
    # train from ground up
    'memory_history_size': {'values': [20000]},  # Different memory sizes
    'lr_patience': { 'values': [50] },
    'min_lr': {'values': [0.00005]},  # Minimum learning rate
    'lr': {'values': [0.0005]},  # Learning rate range
    'EPS_START': {'values': [0.3]},  # Exploration end
    'EPS_DECAY': {'values': [0.999]},  # Exploration end
    'EPS_END': {'values': [0.03]},  # Exploration end
    }
}
