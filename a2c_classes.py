import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T


class Actor(nn.Module):
    def __init__(self, n_inputs, n_outputs, LR):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(n_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)

        self.mu = nn.Linear(64, n_outputs)
        self.var = nn.Linear(64, n_outputs)

        self.optimizer = optim.RMSprop(self.parameters(), LR)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        mu = F.tanh(self.mu(x))
        var = F.softplus(self.var(x))

        return mu, var


class Critic(nn.Module):
    def __init__(self, n_inputs, n_outputs, LR):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(n_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.out = nn.Linear(64, n_outputs)

        self.optimizer = optim.RMSprop(self.parameters(), LR)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)
