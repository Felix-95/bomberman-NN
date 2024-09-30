import matplotlib.pyplot as plt
import numpy as np
import os

def plot_action_distribution(action_counts):
    episodes = len(action_counts)
    actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT'] 
    
    # prozentualer Anteil jeder Aktion pro Episode
    action_percentages = np.array(action_counts) / (np.sum(action_counts, axis=1)[:, np.newaxis] + 0.001) * 100

    plt.figure(figsize=(12, 6))
    for i, action in enumerate(actions):
        plt.plot(range(1, episodes+1), action_percentages[:, i], label=action)

    plt.xlabel('Episode')
    plt.ylabel('Aktion Ausführung (%)')
    plt.title('Verteilung der Aktionen über Episoden')
    plt.legend()
    plt.grid(True)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Definiere den relativen Pfad zu dem Ordner, in dem die Dateien gespeichert werden sollen
    folder_path = os.path.join(current_dir, 'action_logs')
    os.makedirs(folder_path, exist_ok=True)

    # Speichere die Datei im angegebenen Ordner
    file_path = os.path.join(folder_path, f'action_distribution_episode.png')
    plt.savefig(file_path)

    plt.close()