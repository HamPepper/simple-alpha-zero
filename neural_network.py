import torch
import numpy as np
import os

# Object that manages interfacing data with the underlying PyTorch model, as well as checkpointing models.
class NeuralNetwork():

    def __init__(self, game, model_class, lr=1e-3, weight_decay=1e-8, batch_size=64, cuda=False):
        self.game = game
        self.batch_size = batch_size
        input_shape = game.get_initial_state().shape
        p_shape = game.get_available_actions(game.get_initial_state()).shape
        self.model = model_class(input_shape, p_shape)
        self.cuda = cuda
        if self.cuda:
            self.model = self.model.to('cuda')
            self.model = torch.nn.DataParallel(self.model)
        if len(list(self.model.parameters())) > 0:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)


    # Incoming data is a numpy array containing (state, prob, outcome) tuples.
    def train(self, data):
        self.model.train()
        batch_size=self.batch_size
        idx = np.random.randint(len(data), size=batch_size)
        batch = data[idx]
        states = np.stack(batch[:,0])
        x = torch.from_numpy(states)
        p_pred, v_pred = self.model(x)
        v_pred = v_pred.view(-1)
        p_gt, v_gt = batch[:,1], torch.from_numpy(batch[:,2].astype(np.float32))
        if self.cuda:
            v_gt = v_gt.cuda()
        loss = self.loss(states, (p_pred, v_pred), (p_gt, v_gt))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.latest_loss = loss


    # Given a single state s, does inference to produce a distribution of valid moves P and a value V.
    def predict(self, s):
        self.model.eval()
        input_s = np.array([s])
        with torch.no_grad():
            input_s = torch.from_numpy(input_s)
            p_logits, v = self.model(input_s)
            p, v = self.get_valid_dist(s, p_logits[0]).cpu().numpy().squeeze(), v.cpu().numpy().reshape(-1)[0] # EXP because log softmax
        return p, v


    # MSE + Cross entropy
    def loss(self, states, prediction, target):
        batch_size = len(states)
        p_pred, v_pred = prediction
        p_gt, v_gt = target
        v_loss = ((v_pred - v_gt)**2).sum()
        p_loss = 0
        for i in range(batch_size):
            gt = torch.from_numpy(p_gt[i].astype(np.float32))
            if self.cuda:
                gt = gt.cuda()
            s = states[i]
            logits = p_pred[i]
            pred = self.get_valid_dist(s, logits, log_softmax=True)
            p_loss += -torch.sum(gt*pred)
        return p_loss + v_loss


    # Takes one state and logit set as input, produces a softmax/log_softmax over the valid actions.
    def get_valid_dist(self, s, logits, log_softmax=False):
        mask = torch.from_numpy(self.game.get_available_actions(s).astype(np.uint8)).bool()
        if self.cuda:
            mask = mask.cuda()
        selection = torch.masked_select(logits, mask)
        dist = torch.nn.functional.log_softmax(selection, dim=-1)
        if log_softmax:
            return dist
        return torch.exp(dist)


    # Saves the current network along with its current pool of training data and training error history.
    # Provide the name of the save file.
    def save(self, name, training_data, error_log):
        network_name = self.model.module.__class__.__name__ if self.cuda else self.model.__class__.__name__
        directory = "checkpoints/{}-{}".format(self.game.__class__.__name__, network_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        network_path = "{}/{}.ckpt".format(directory, name)
        data_path = "{}/training.data".format(directory)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'error_log': error_log,
            }, network_path)
        torch.save({
            'training_data': training_data,
            }, data_path)


    # Loads the network at the given name.
    # Optionally, also load and return the training data and training error history.
    def load(self, name, load_supplementary_data=False):
        network_name = self.model.module.__class__.__name__ if self.cuda else self.model.__class__.__name__
        directory = "checkpoints/{}-{}".format(self.game.__class__.__name__, network_name)
        network_path = "{}/{}.ckpt".format(directory, name)
        network_checkpoint = torch.load(network_path)
        self.model.load_state_dict(network_checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(network_checkpoint['optimizer_state_dict'])
        if load_supplementary_data:
            data_path = "{}/training.data".format(directory)
            data_checkpoint = torch.load(data_path)
            return data_checkpoint['training_data'], network_checkpoint['error_log']


    # Utility function for listing all available model checkpoints.
    def list_checkpoints(self):
        network_name = self.model.module.__class__.__name__ if self.cuda else self.model.__class__.__name__
        path = "checkpoints/{}-{}/".format(self.game.__class__.__name__, network_name)
        if  not os.path.isdir(path):
            return []
        return sorted([filename.split(".ckpt")[0] for filename in os.listdir(path) if filename.endswith(".ckpt")])

