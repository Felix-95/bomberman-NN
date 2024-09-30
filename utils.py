import os 
from datetime import datetime
import shutil

def delete_model_if_exists():
    # Pfad zum Modell und Standardname
    path = "agent_code\\ffm_agent"
    default_model_name = "my-saved-model.pt"
 
    if os.path.isfile(os.path.join(path, default_model_name)):
        os.remove(os.path.join(path, default_model_name))
        print("Previous model deleted.")

# def copy_ground_model_if_exists():
#     path = "agent_code\\ffm_agent"
#     default_model_name = "ground-model.pt"
#     current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     new_model_name = f"my-saved-model.pt"
    
#     if os.path.isfile(os.path.join(path, default_model_name)):
#         os.rename(os.path.join(path, default_model_name), os.path.join(path, new_model_name))
#         print(f"Ground model copied to {new_model_name}.")

def copy_ground_model_if_exists():
    agent_code_path = os.path.join("agent_code", "ffm_agent")
    proven_models_path = os.path.join(agent_code_path, "proven_models")
    proven_model_name = "ground-model.pt"
    new_model_name = "my-saved-model.pt"
    
    # Pfad zur Datei
    default_model_path = os.path.join(proven_models_path, proven_model_name)
    new_model_path = os.path.join(agent_code_path, new_model_name)

    # Prüfe, ob die Datei existiert
    if os.path.isfile(default_model_path):
        # Kopiere die Datei
        shutil.copy(default_model_path, new_model_path)
        print(f"Ground model copied to {new_model_name}.")
    else:
        print(f"Ground model {proven_model_name} does not exist.")

def rename_model():
    path = os.path.join("agent_code", "ffm_agent")
    default_model_name = "my-saved-model.pt"
    
    if os.path.isfile(os.path.join(path, default_model_name)):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_model_name = f"model-{current_time}.pt"
        os.rename(os.path.join(path, default_model_name), os.path.join(path, new_model_name))
        print(f"Model renamed to {new_model_name}.")