import wandb
import os
from sweep_utils import execute_sweep_iteration  # Funktion wird wiederverwendet
from utils import copy_ground_model_if_exists
import multiprocessing

def load_sweep_config(sweep_config_path):
    config = {}
    with open(sweep_config_path, 'r') as f:
        exec(f.read(), config)
    return config['sweep_config']

def run_single_sweep(sweep_id, map_name, n_rounds, rename_model_bool):
    """Funktion, die einen einzelnen Sweep-Run ausführt."""
    wandb.agent(sweep_id, lambda: execute_sweep_iteration(map_name, n_rounds, rename_model_bool=rename_model_bool), count=1)

def run_sweep(sweep_config_path, rename_model_bool=True):
    sweep_config = load_sweep_config(sweep_config_path)
    map_name = sweep_config['parameters']['map']['value']
    n_rounds = sweep_config['parameters']['n_rounds']['value']
    run_quantity = sweep_config['parameters']['run_quantity']['value']
    
    print(f"Stage: Starte Sweep auf {map_name} für {n_rounds} runden")
    # Sweep-ID erstellen
    sweep_id = wandb.sweep(sweep_config)
    
    if parallelize_runs:
        num_parallel_runs = 8 # da ich thinkspad über 8 logische CPU-Kerne verfügt
        processes = []

        # Starte mehrere parallele Prozesse
        for _ in range(num_parallel_runs):
            p = multiprocessing.Process(target=run_single_sweep, args=(sweep_id, map_name, n_rounds, rename_model_bool))
            p.start()
            processes.append(p)

        # Warte, bis alle Prozesse beendet sind
        for p in processes:
            p.join()

    else:
        # Sequentieller Sweep (keine Parallelisierung)
        wandb.agent(sweep_id, lambda: execute_sweep_iteration(map_name, n_rounds, rename_model_bool=rename_model_bool), count=run_quantity)


parallelize_runs = False  # Setze hier auf False, wenn keine Parallelisierung gewünscht ist
train_with_ground_model = True

def main():
    
    if train_with_ground_model:
        copy_ground_model_if_exists()
    else:
        print("Don't want to use ground model...")
    
    print("Stage 1:")
    run_sweep('sweep_config_res_17-08.py', rename_model_bool=True)
    
    # print("Stage 2: ")
    # run_sweep('sweep_config_stage-2.py', rename_model_bool=False)

if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DEBUG"] = "1"
    
    main()
