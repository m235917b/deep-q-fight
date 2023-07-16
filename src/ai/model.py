"""
This file contains the model for the agents and their training and testing algorithm.
"""

import math
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim

from game.constants import VISUAL_RESOLUTION
from ai.game_adapter import N_NV_INPUTS, N_ACTIONS

torch.autograd.set_detect_anomaly(True)

# Uses GPU for the neural networks if available, otherwise CPU
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(_DEVICE)

# The datatype for the transitions in replay memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward',
                         'hidden', 'hs', 'next_hidden', 'next_hs'))

# the number of games the agents should be tested on, after each training game
_EVAL_ROUNDS = 5

# the length of the state sequence that the neural nets get as input sequence
_LEN_SEQUENCE = 10
# The max length of the replay memory. If this has been reached, the oldest sequence gets replaces be the newest.
_LEN_REPLAY_MEM = 1000
# only save the last _LEN_SEQUENCE frames every _FRAME_SKIPPING frame as a sequence in replay memory
_FRAME_SKIPPING = 5
# train the neural nets every _EPOCH_INTERVAL * _FRAME_SKIPPING on one batch from the replay memory
_EPOCH_INTERVAL = 5

# The size of the batch the agents train on the replay memory. Each batch contain _BATCH_SIZE sequences.
_BATCH_SIZE = 200
"""The gamma value for the DQN algorithm. This specifies how much an agent weights future rewards.
A higher value means, the agent tries to maximize rewards tha lie far in the future, while a 
smaller value, means the agent only tries to maximize rewards in the immediate future."""
_GAMMA = 0.9
# the starting probability with which the agents choose a random action during training
_EPS_START = 0.9
# the end probability with which the agents choose a random action during training after infinite frames
_EPS_END = 0.05
# the decay rate with which the probability to choose a random action during training converges to _EPS_END
_EPS_DECAY = 30  # 25
# the soft update parameter for updating the weights of the target-network
_TAU = .005  # 0.005
# the learning rate with which the weights of the policy-network get updated
_LR = 1e-3 / 2.  # 1e-3
# a clipping value to avoid exploding gradients in the neural networks
_CLIP = 300

# the input size of the neural network
_INPUT_SIZE = VISUAL_RESOLUTION * 2 + N_NV_INPUTS
# the size of the hidden layers of the RNN
_RNN_HIDDEN_SIZE = 200
# the number of layers in the RNN
_RNN_LAYERS = 5


class ReplayMemory(object):
    """
    The replay memory of the agents. Each agent has its own replay memory.
    The replay memory stores entire sequences for each transition and
    can return a batch of randomly selected transitions from its memory.
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class FrameBuffer(object):
    """
    The frame buffer is used to store the currently observed sequence of the agents. It always contains
    the last _LEN_SEQUENCE frames of the game. Each agent has its own frame buffer. With this class the
    sequences that get pushed into the replay memory are build. The frame buffer is implemented as a queue
    of frames that can return the entire queue.

    It also returns the hidden and cell states that the network had before processing the respective sequence.
    This is done, so the network can be initialized with these states before processing the current sequence.
    Thus, the network can use past states and model the entire game as one continual sequence, even though
    the sequences are truncated by _LAST_FRAMES.

    This works, because a RNN processes each sequence element individually and builds up a memory of the last
    frames from the sequence using the states as some kind of accumulated value for the last processed frames.
    In this way, if the last states before processing the sequence get carried over for the current sequence,
    the network generalizes this process to all sequences, or individual frames.
    """

    def __init__(self, capacity):
        self.__capacity = capacity
        self.__memory = deque([], maxlen=capacity)
        self.__hidden_memory = deque([], maxlen=capacity)
        self.__cell_memory = deque([], maxlen=capacity)

    def init(self, frame, hidden, cell):
        for i in range(self.__capacity):
            self.__memory.append(frame)
            self.__hidden_memory.append(hidden)
            self.__cell_memory.append(cell)

    def push(self, state, prev_hidden, prev_cell):
        """
        Pushes one single Frame in the queue.
        """

        self.__memory.append(state)
        self.__hidden_memory.append(prev_hidden)
        self.__cell_memory.append(prev_cell)

    def last_frames(self):
        """
        Returns the entire queue.
        """

        return (torch.stack(list(self.__memory), 0),
                self.__hidden_memory[0], self.__cell_memory[0])

    def __len__(self):
        return len(self.__memory)


def init_hidden(batch_size=None):
    """Initializes the hidden or cell states for the """

    if batch_size is None:
        return torch.zeros(_RNN_LAYERS, _RNN_HIDDEN_SIZE, device=_DEVICE)
    else:
        return torch.zeros(_RNN_LAYERS, batch_size, _RNN_HIDDEN_SIZE, device=_DEVICE)


class DQN(nn.Module):
    """The Deep-Q-Network is a neural network consisting of a multilayer LSTM network
    and an output layer that maps the hidden state of the last LSTM layer from the last time step
    to the output vector space. The output vector space is a vector representing the estimated
    probability of the agent for each possible action."""

    def __init__(self):
        super(DQN, self).__init__()

        # create the multilayer RNN
        self.__rnn = nn.LSTM(_INPUT_SIZE, _RNN_HIDDEN_SIZE, _RNN_LAYERS)

        # create the output layer to map the output of the RNN to the desired output space
        self.__output_layer = nn.Sequential(nn.Linear(_RNN_HIDDEN_SIZE, N_ACTIONS),
                                            nn.Softmax(dim=-1))

    def forward(self, x, hidden, cell):
        """
        Forward method forwards the given input and hidden and cell states of the agent
        and returns the output of the neural network.

        Parameters
        ----------
        x : torch.Tensor, shape(_INPUT_SIZE, _BATCH_SIZE, _LEN_SEQUENCE) for batched input,
        shape(_INPUT_SIZE, _LEN_SEQUENCE) for unbatched input
            the observed state sequence of the agent
        hidden : torch.Tensor, shape(_RNN_LAYERS, _BATCH_SIZE, _RNN_HIDDEN_SIZE) for batched input,
        shape(_RNN_LAYERS, _RNN_HIDDEN_SIZE) for unbatched input
            the initial hidden state for the sequence
        cell : torch.Tensor, shape(_RNN_LAYERS, _BATCH_SIZE, _RNN_HIDDEN_SIZE) for batched input,
        shape(_RNN_LAYERS, _RNN_HIDDEN_SIZE) for unbatched input
            the initial cell state for the sequence

        Returns
        -------
        out : torch.Tensor, shape(_BATCH_SIZE, N_ACTIONS) for batched input,
        shape(N_ACTIONS) for unbatched input
            the estimated probability of the agent for each possible action
        hidden : torch.Tensor, shape(_RNN_LAYERS, _BATCH_SIZE, _RNN_HIDDEN_SIZE) for batched input,
        shape(_RNN_LAYERS, _BATCH_SIZE, _RNN_HIDDEN_SIZE) for unbatched input
            the hidden state of the last time step state after processing the sequence
        cell : torch.Tensor, shape(_RNN_LAYERS, _BATCH_SIZE, _RNN_HIDDEN_SIZE) for batched input,
        shape(_RNN_LAYERS, _BATCH_SIZE, _RNN_HIDDEN_SIZE) for unbatched input
            the cell state of the last time step state after processing the sequence
        """

        # forward th input through the multilayer RNN
        x, (hidden, state) = self.__rnn(x, (hidden, cell))

        # forward the last hidden state of the last time step through the output layer
        out = hidden[-1]
        out = self.__output_layer(out)
        return out, hidden, state


class Trainer:
    """
    This class implements the training algorithm for the agents.
    """

    def __init__(self, env, path_eval):
        self.__agents = env.n_agents()
        self.__env = env
        self.__len_replay_buffer = _LEN_REPLAY_MEM
        self.__nets = []
        self.__steps_done = 0

        for net in range(self.__agents):
            policy = DQN().to(_DEVICE).eval()
            target = DQN().to(_DEVICE).eval()

            target.load_state_dict(policy.state_dict())

            self.__nets.append((policy, target,
                                ReplayMemory(self.__len_replay_buffer),
                                FrameBuffer(_LEN_SEQUENCE),
                                optim.AdamW(policy.parameters(), lr=_LR, amsgrad=True)))

        self.__eval_nets = [net.eval() for _, net in sorted(torch.load(path_eval).items())[:2]] + \
                           [net[1] for net in self.__nets[2:]]

        torch.set_grad_enabled(False)

    def __select_action(self, net, state, hidden, hs):
        """
        Selects a random action based on the current probability based on _EPS_DECAY,
        otherwise selects the action based on the output of the neural network net
        and the observed state (single frame, or sequence of frames).
        """

        sample = random.random()
        eps_threshold = _EPS_END + (_EPS_START - _EPS_END) * math.exp(-1. * self.__steps_done / _EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                action, hidden, hs = net(state, hidden, hs)
                action = action.max(-1)[1].view(1)
        else:
            _, hidden, hs = net(state, hidden, hs)
            action = torch.tensor([random.randint(0, self.__env.actions() - 1)],
                                  device=_DEVICE, dtype=torch.long)

        return action, hidden, hs

    def __optimize_models(self):
        """
        Optimizes the neural networks on one single batch with gradient descent, using _GAMMA
        as learning rate and SmoothL1Loss as loss function.

        It should be noted, that the networks can not learn to improve the error with regards to the
        last hidden and cell states before processing the entire sequence, because of truncated-
        back-propagation-through-time. But the network still gets initialized with them to accurately
        calculate the q-values based on the sequence. The networks can still learn to generalize over
        more frames than _LAST_FRAMES, because if they learn temporal dependencies between observed
        states within a sequence, they might be able to generalize this knowledge over longer distances
        during inference.
        """

        torch.set_grad_enabled(True)

        for net in self.__nets:
            if len(net[2]) < _BATCH_SIZE:
                return

            net[0].train()

            # samples the batch from replay memory

            transitions = net[2].sample(_BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            # transform the batch to torch tensors

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                          device=_DEVICE, dtype=torch.bool)
            non_final_next_states = torch.stack([s for s in batch.next_state if s is not None], 1)
            state_batch = torch.stack(batch.state, 1)
            action_batch = torch.stack(batch.action, 0)
            reward_batch = torch.stack(batch.reward, 0)
            hidden_batch = torch.stack(batch.hidden, 1).detach()
            cell_batch = torch.stack(batch.hs, 1).detach()

            # get the q-values for each action from the policy network for each batch element
            state_action_values = net[0](state_batch,
                                         hidden_batch,
                                         cell_batch)[0].gather(1, action_batch)

            next_state_values = torch.zeros(_BATCH_SIZE, device=_DEVICE)

            """Get the predicted q-value for the action with the maximal q-value
            from the target-network for each batch element."""
            with torch.no_grad():
                # transform the batch of next hidden and cell states to torch tensors

                next_hidden_batch = [hidden for hidden in batch.next_hidden if hidden is not None]
                next_hidden_batch = torch.stack(next_hidden_batch, 1)
                next_cell_batch = [cell for cell in batch.next_hs if cell is not None]
                next_cell_batch = torch.stack(next_cell_batch, 1)
                next_state_values[non_final_mask] = net[1](non_final_next_states,
                                                           next_hidden_batch,
                                                           next_cell_batch)[0].max(-1)[0]

            """Add the real reward for the current frame based on the chosen action to the estimated
            future reward of the target-network weighted by _GAMMA"""
            expected_state_action_values = (next_state_values.unsqueeze(1) * _GAMMA) + reward_batch

            # calculate the error of the real and predicted reward and the estimated reward for the current frame
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values)

            # perform gradient descent on the batch based on the calculated error
            net[4].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net[0].parameters(), _CLIP)
            net[4].step()

        torch.set_grad_enabled(False)
        for n in self.__nets:
            n[0].eval()

    def evaluate(self):
        """
        Evaluates the agents of team 1 in a test game against team 2 of a specified
        model that has already been trained.
        """

        states = [torch.tensor(inputs, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
                  for inputs in self.__env.reset(randomize=True)]

        hidden = [(init_hidden(), init_hidden())] * self.__agents

        done = False

        while not done and not self.__env.quit():
            outputs = [self.__eval_nets[net](states[net], hidden[net][0], hidden[net][1])
                       for net in range(self.__agents)]
            hidden = [(out[1], out[2]) for out in outputs]
            actions = [(out[0].max(-1)[-1].view(1)) for out in outputs]

            observations, _, done = self.__env.step(actions, record_stats=True)

            states = [torch.tensor(observation, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
                      for observation in observations]

    def train_episode(self):
        """
        Trains the agents for one episode (one training game) by playing the game based on actions
        chosen by __select_action() and training the networks using gradient descent on sampled
        batches from the replay memory.
        """

        # initialize the lastly observed game states for each agent
        last_states = [torch.tensor(inputs, dtype=torch.float32, device=_DEVICE)
                       for inputs in self.__env.reset(randomize=True)]

        # initialize the hidden and cell states for the agents
        for agent in range(self.__agents):
            self.__nets[agent][3].init(last_states[agent], init_hidden(), init_hidden())

        # get the initial sequence of the last frames (all initialized to 0) from the frame buffer
        last_frames = [nets[3].last_frames() for nets in self.__nets]

        done = False
        self.__steps_done = 0

        """Play as long as the game hasn't been won, or truncated by the maximum number of frames,
        or the user quits training with the ESC-key"""
        while not done and not self.__env.quit():
            """Get the selected action based on __select_action() and the hidden and cell states of
            the last element of the processed sequence of the last observed frames."""
            outputs = [self.__select_action(self.__nets[net][0], last_frames[net][0],
                                            last_frames[net][1], last_frames[net][2])
                       for net in range(self.__agents)]
            # get the hidden and cell states from the RNN after processing the sequence
            next_hidden = [(action[1], action[2]) for action in outputs]
            # get the selected action
            actions = [action[0] for action in outputs]

            rewards_acc = [0. for _ in range(self.__agents)]

            """Continue game for _FRAME_SKIPPING frames, continually choosing the selected action from
            actions each of these frames and adding all the rewards gained during these frames.
            The observed frames get pushed onto the frame buffer."""
            for _ in range(_FRAME_SKIPPING):
                # continue game for one frame and get the observation and reward for that frame for each agent
                observations, rewards, done = self.__env.step(actions)

                # add the rewards
                rewards_acc = [r1 + r2 for r1, r2 in zip(rewards_acc, rewards)]
                # convert the states to torch tensors for the neural networks
                next_states = [torch.tensor(observation, dtype=torch.float32, device=_DEVICE)
                               for observation in observations]

                # push the observed frame for each agent into its frame buffer
                for i in range(self.__agents):
                    self.__nets[i][3].push(next_states[i], next_hidden[i][0], next_hidden[i][1])

            # convert the rewards to torch tensors
            rewards_acc = [torch.tensor([reward], device=_DEVICE) for reward in rewards_acc]
            # sample the sequence of the last _LAST_FRAMES from the frame buffer
            next_frames = [net[3].last_frames() for net in self.__nets]

            self.__steps_done += 1

            """Push the observed sequence and its transition from the last observed sequence into
            the replay memory together with the observed reward and the hidden and cell states
            before processing the respective sequences for each agent."""
            for agent in range(self.__agents):
                if done:
                    self.__nets[agent][2].push(last_frames[agent][0], actions[agent],
                                               None, rewards_acc[agent],
                                               last_frames[agent][1], last_frames[agent][2],
                                               None, None)
                else:
                    self.__nets[agent][2].push(last_frames[agent][0], actions[agent],
                                               next_frames[agent][0], rewards_acc[agent],
                                               last_frames[agent][1], last_frames[agent][2],
                                               next_frames[agent][1], next_frames[agent][2])

            # set the last observed sequence to the currently observed sequence
            last_frames = next_frames

            # train the neural networks on a batch from the replay memory
            if self.__steps_done % _EPOCH_INTERVAL == 0:
                self.__optimize_models()

            # update weights in target-networks for each agent according to soft-update-rule
            for net in self.__nets:
                policy_net_state_dict = net[0].state_dict()
                target_net_state_dict = net[1].state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = (policy_net_state_dict[key] * _TAU +
                                                  target_net_state_dict[key] * (1 - _TAU))

                net[1].load_state_dict(target_net_state_dict)

        # evaluate the agents on test games after they trained on the current game
        for _ in range(_EVAL_ROUNDS):
            self.evaluate()

    def save_policy(self, path):
        torch.save({i: self.__nets[i][0] for i in range(self.__agents)}, path)

    def save_target(self, path):
        torch.save({i: self.__nets[i][1] for i in range(self.__agents)}, path)


class Tester:
    """
    This class is used to test the agents in test games.
    """

    def __init__(self, env, path, random_agents=False):
        self.__agents = env.n_agents()
        self.__env = env
        self.__random_team = random_agents

        self.__nets = [net.eval() for _, net in sorted(torch.load(path).items())]

        # uncomment one of the lines for testing an agent with a copy of itself
        # self.__nets[1] = self.__nets[0]
        # self.__nets[0] = self.__nets[1]

        torch.set_grad_enabled(False)

    def evaluate(self):
        states = [torch.tensor(inputs, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
                  for inputs in self.__env.reset(randomize=True)]

        hidden = [(init_hidden(), init_hidden())] * self.__agents

        done = False

        while not done and not self.__env.quit():
            outputs = [self.__nets[net](states[net], hidden[net][0], hidden[net][1])
                       for net in range(self.__agents)]
            hidden = [(out[1], out[2]) for out in outputs]
            actions = [(out[0].max(-1)[-1].view(1)) for out in outputs]

            if self.__random_team:
                actions[0] = random.randint(0, self.__env.actions() - 1)
                actions[1] = random.randint(0, self.__env.actions() - 1)
                actions[2] = random.randint(0, self.__env.actions() - 1)
                actions[3] = random.randint(0, self.__env.actions() - 1)

            observations, _, done = self.__env.step(actions, record_stats=True)

            states = [torch.tensor(observation, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
                      for observation in observations]
