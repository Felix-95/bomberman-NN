# bomberman_rl

venv/Scripts/activate
python main.py play --agents rule_based_agent --train 0 --scenario coin-heaven

## Weights and Biases - Statistics
# 1. Install wandb via requierements.txt
# 2. Create an account on Weights and Biases
# 3. Copy your key
wandb login
# 4. Insert the key
# Done

## train our agent with no gui, 1000 episodes and with scenario 'coin-heaven'
python main.py play --n-rounds=1000  --no-gui --agents ffm_agent --train 1 --scenario coin-heaven

            
# felix training setup
"args": [
                "play",
                "--agents",
                "ffm_agent",
                "--scenario",
                "coin-heaven",
                "--model-file-name",
                "my-custom-model-15-23.pt",
                "--no-gui",
                "--n-rounds=300",
                "--train",
                "1",
                // "--seed",
                // "121"
            ]
            
# felix test setup
"args": [
                "play",
                "--agents",
                "ffm_agent",
                "--scenario",
                "coin-heaven",
                "--model-file-name",
                "my-custom-model-15-23.pt",
            ]            

## replay
python main.py replay "replays/Round 01 (2024-08-19 23-48-27).pt"

## plotting e.g. plot_loss_rewards - assuming your working directory is 'bomerbman_rl'
python ./agent_code/ffm_agent/plotting/plot_loss_rewards.py

## Training loot create
python main.py play --n-rounds=100 --no-gui --agents ffm_agent --train 1 --scenario loot-crate

## Training coin-heaven
python main.py play --n-rounds=100 --no-gui --agents ffm_agent --train 1 --scenario coin-heaven

## running
python main.py play  --agents ffm_agent  --scenario loot-crate