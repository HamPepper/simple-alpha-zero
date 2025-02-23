import sys
import os
import json
import pickle

import numpy as np

from datetime import datetime

from models.mlp import MLP
from models.minivgg import MiniVGG
from models.smallvgg import SmallVGG
from models.bigvgg import BigVGG
from models.resnet import ResNet
from models.senet import SENet

from games.tictactoe import TicTacToe
from games.leapfrog import ThreePlayerLeapFrog
from games.connect4 import Connect4
from games.guessit import TwoPlayerGuessIt

from neural_network import NeuralNetwork
from trainer import Trainer


# Load in a run configuration
with open(sys.argv[1], "r") as f:
    config = json.loads(f.read())

# Determine work mode
mode = "export" if len(sys.argv) == 2 else sys.argv[2]

# Instantiate
game = globals()[config["game"]]()
model_class = globals()[config["model"]]
sims = config["num_simulations"]
cuda = config["cuda"]
num_updates = config["num_updates"]

nn = NeuralNetwork(game=game, model_class=model_class, lr=config["lr"],
    weight_decay=config["weight_decay"], batch_size=config["batch_size"], cuda=cuda)
trainer = Trainer(game=game, nn=nn, num_simulations=sims,
num_games=config["num_games"], num_updates=num_updates,
buffer_size_limit=config["buffer_size_limit"], cpuct=config["cpuct"],
num_threads=config["num_threads"])

max_iterations = 20


# Import-training loop
if mode == "import":
    print("IMPORTING")
    with open("export/training_data.pkl", "rb") as file:
        data = pickle.load(file)

    print("TRAINING")
    examples = []

    for data_in_single_sim_batch in data:
        for game in data_in_single_sim_batch:
            wrapped = np.array(game, dtype=object)
            examples.append(wrapped)

        print("Training examples:", len(examples))

        mean_loss = None
        count = 0
        for _ in range(num_updates):
            nn.train(np.array(examples, dtype=object))
            new_loss = nn.latest_loss.item()
            if mean_loss is None:
                mean_loss = new_loss
            else:
                (mean_loss*count + new_loss)/(count+1)
            count += 1

        print("Average train error:", mean_loss)

elif mode == "export":
    print("EXPORTING")
    last_idx = 0
    data = []

    for _ in range(max_iterations):
        trainer.policy_iteration(verbose=True)

        # save games generated in this iteration
        data_in_single_sim_batch = trainer.training_data[last_idx:].tolist()
        data.append(data_in_single_sim_batch)
        last_idx = len(trainer.training_data)

    directory = "export"
    if not os.path.exists(directory):
        os.makedirs(directory)
    data_path = f"{directory}/training_data.pkl"

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    data_backup_path = f"{directory}/training_data-{timestamp}.pkl"

    with open(data_path, "wb") as file:
        pickle.dump(data, file)
    with open(data_backup_path, "wb") as file:
        pickle.dump(data_backup_path, file)

else:
    print("Invalid mode:", mode)
    sys.exit(1)