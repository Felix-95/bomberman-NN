import wandb

import os
os.environ["WANDB_MODE"]="offline"

class WandBLogger:
    def __init__(self, project_name, config=None):
        """
        Initialisiert das wandb-Logging.

        :param project_name: Name des Projekts in wandb.
        :param config: Optionales Konfigurationsobjekt (z.B. Hyperparameter), das in wandb gespeichert werden soll.
        """
        self.project_name = project_name
        self.config = config
        self.run = None
        self.init_wandb()

    def init_wandb(self):
        """
        Initialisiert das wandb-Projekt.
        """
        if self.config:
            self.run = wandb.init(project=self.project_name, settings=wandb.Settings(console="wrap"), mode="offline", config=self.config)
        else:
            self.run = wandb.init(project=self.project_name, settings=wandb.Settings(console="wrap"), mode="offline")

    def log(self, metrics):
        """
        Protokolliert Metriken in wandb.

        :param metrics: Ein Dictionary, das die zu protokollierenden Metriken enthält. Beispiel: {"loss": 0.5, "accuracy": 0.9}
        """
        if self.run:
            wandb.log(metrics)
        else:
            raise RuntimeError("WandBLogger is not initialized. Call init_wandb() first.")

    def finish(self):
        """
        Finish current Wandb run
        """
        if self.run:
            self.run.finish()
