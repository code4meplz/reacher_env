from a2c_classes import Actor, Critic
from collections import deque
import torch as T
import torch.nn.functional as F

from torch.distributions.normal import Normal


class Agent:
    def __init__(self, n_inputs, n_outputs, lr_rates, gamma):
        self.device = T.device('cuda:0')
        self.GAMMA = gamma
        self.ENTROPY_BETA = 0.001

        self.actions = None
        self.entropy = None
        self.log_prob = None

        self.memory = deque(maxlen=10)

        self.inputs = n_inputs
        self.out_actor, self.out_critic = n_outputs
        self.LR_actor, self.LR_critic = lr_rates

        self.actor = Actor(self.inputs, self.out_actor, self.LR_actor)
        self.critic = Critic(self.inputs, self.out_critic, self.LR_critic)

    def choose_action(self, state):
        state = T.Tensor(state)

        self.actor.eval()

        mus, sigmas = self.actor(state)

        normal = Normal(mus, sigmas)
        self.entropy = normal.entropy()
        self.actions = normal.sample()
        self.log_prob = normal.log_prob(self.actions)

        return self.actions.numpy()

    def learn_from_exp(self):

        self.actor.train()
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        states, rewards, new_states, dones, log_probs, entropy = zip(*self.memory)

        log_probs = T.cat(log_probs)
        entropy = T.cat(entropy)

        states = T.Tensor(states)
        new_states = T.Tensor(new_states)

        state_v = self.critic(states)
        new_state_v = self.critic(new_states)

        unrolled_r = self.unroll_rewards(rewards, new_state_v, dones)

        td_error = F.mse_loss(state_v, unrolled_r)
        td_error.backward()
        self.critic.optimizer.step()

        adv = unrolled_r - state_v.squeeze(-1)
        adv = adv.detach()

        #adv = (adv-adv.mean())/adv.std()

        loss = -(log_probs.sum(dim=1) * adv.detach()).mean() + self.ENTROPY_BETA * entropy.sum(dim=1).mean()

        loss.backward()
        self.actor.optimizer.step()

        self.memory.clear()

    def unroll_rewards(self, rewards, new_states_v, dones):

        unrolled_rewards = [new_states_v[-1] if dones[-1] == 0 else T.tensor([0.], requires_grad=True)]
        for r in reversed(rewards):
            unrolled_rewards.append(self.GAMMA * unrolled_rewards[-1] + r)
        return T.cat([x for x in reversed(unrolled_rewards[:-1])])
