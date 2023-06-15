import time
import random
import torch
from torch.multiprocessing import Queue, Process
from collections import namedtuple  

from src.Statistics import SummaryType
from src.GymEnv import make_env


class Agent(Process):
    """Agent process used to gather trajectories of experiences asynchronously"""

    def __init__(
        self,
        id_,
        prediction_queue,
        training_queue,
        states,
        exit_flag,
        statistics_queue,
        episode_counter,
        observation_shape,
        action_space,
        device,
        step_max,
        lstm_hidden_size=256,
    ):

        # Calling parent class constructor
        super(Agent, self).__init__()

        self.id = id_

        # We set the worker as a daemon child
        # When the main process stops, it also breaks the workers
        self.daemon = True

        self.device = device

        # Prediction queue to send the actions to
        self.prediction_queue = prediction_queue

        self.training_queue = training_queue
        self.stats_queue = statistics_queue
        self.action_queue = Queue(maxsize=1)

        self.states = states

        self.step_max = step_max

        self.memory = AgentMemory(
            num_steps=self.step_max,
            observation_shape=observation_shape,
            lstm_hidden_size=lstm_hidden_size,
            action_space=action_space,
        )
        self.memory.to(self.device)

        # Episodes
        self.episode_counter = episode_counter
        self.channels = observation_shape[0]
        self.heigh = observation_shape[1]
        self.width = observation_shape[2]

        # Set exit as global value between processes
        self.exit = exit_flag

    def run(self):
        """Starts the Agent process. It can be stopped thanks to the self.exit flag."""

        # Strating the process
        super(Agent, self).run()

        # Counter for the n-step return
        step = 0

        # Create a new environnement
        done = True

        # exit_flag is a shared torch.multiprocessing value 
        j = 0
        actions = [[15,0.6,0.3],[15,-0.3,-0.6]]
        while not self.exit.value:

            if done:
                # We start a new episode
                # Selecting a random state
                state = random.choice(self.states)
                self.env = make_env(
                    state=state, stacks=self.channels, size=(self.width, self.heigh)
                )

                obs = self.env.reset()
                done = False

                # Accumulated reward from the episode
                episode_reward = 0

                # Initialisation of LSTM memory
                # Shape should be num_layers, batch_size, hidden_size
                lstm_hxs = [torch.zeros((1, 1, 256)).to(self.device)] * 2

            # obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device)
            obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device)

            # Sending to predictor
            self.prediction_queue.put((self.id, obs_tensor, lstm_hxs))

            # Receiving the actions
            action, log_prob, lstm_hxs = self.action_queue.get()

            lstm_hxs = [item.to(self.device) for item in lstm_hxs]
            
            # obs, reward, done, info = self.env.step(action)
            action = torch.tensor(action.tolist()[0])
            obs, reward, done, info = self.env.step(action.tolist())
            
            self.env.render(mode="human")
            
            # Update the trajectory with the latest step
            self.memory.append_(
                observation=obs_tensor,
                action=action,
                reward=torch.tensor(reward),
                log_prob=log_prob,
                done=torch.tensor(done),
            )

            episode_reward += reward
            
            # We reset our environnement if the game is done
            if step == self.step_max:

                assert self.memory.step == self.step_max + 1, "Length issue"

                # Converting the data before sending
                self.training_queue.put(self.memory.enqueue())

                # Move the last experience
                self.memory.reset(initial_lstm_state=lstm_hxs)

                # Reinialize for next step
                step = 0

            # The step counter is placed here because of the first iteration
            # Coincides with the "length" of the trajectory buffer
            step += 1

            # Statistics about the episode
            if done:
                self.episode_counter.value += 1
                
                self.stats_queue.put(
                    (SummaryType.SCALAR, "episode/cumulated_reward", episode_reward)
                )
                self.stats_queue.put(
                    (
                        SummaryType.SCALAR,
                        "episode/nb_episodes",
                        self.episode_counter.value,
                    )
                )

                # Only statistic that is logged
                print(
                    f"Episode n° {self.episode_counter.value} finished \
                    \t State : {state} \
                    \t Cumulated reward {episode_reward} \
                    \t Action {action}"
                )

                

                
                # Reset the episode
                self.env.close()

        # The background process must be alive for the Trainer
        # Tensors are passed as reference in pytorch
        time.sleep(1)


Trajectory = namedtuple(
    "Trajectory",
    [
        "length",
        "observations",
        "actions",
        "rewards",
        "log_probs",
        "done",
        "lstm_initial_hidden",
        "lstm_initial_cell",
    ],
)


class AgentMemory(object):
    """Storage for the Agent experiences"""

    def __init__(self, num_steps, observation_shape, lstm_hidden_size, action_space):
        self.observations = torch.zeros(1 + num_steps, *observation_shape)
        self.lstm_initial_hidden = torch.zeros(1, 1, lstm_hidden_size)
        self.lstm_initial_cell = torch.zeros(1, 1, lstm_hidden_size)
        self.actions = torch.zeros(1 + num_steps, action_space)
        self.rewards = torch.zeros(1 + num_steps, 1)
        self.log_probs = torch.zeros(1 + num_steps, 1)
        self.done = torch.zeros(1 + num_steps, 1)
        self.step = 0

    def to(self, device):
        self.observations.to(device)
        self.lstm_initial_hidden.to(device)
        self.lstm_initial_cell.to(device)
        self.actions.to(device)
        self.rewards.to(device)
        self.log_probs.to(device)
        self.done.to(device)

    # Inplace operation
    def append_(self, observation, action, reward, log_prob, done):
        self.observations[self.step].copy_(observation)
        self.actions[self.step].copy_(action)
        self.rewards[self.step].copy_(reward)
        self.log_probs[self.step].copy_(log_prob)
        self.done[self.step].copy_(done)
        self.step += 1

    def reset(self, initial_lstm_state):
        # No need to zero the tensors
        self.observations[0].copy_(self.observations[-1])
        self.lstm_initial_hidden.copy_(initial_lstm_state[0])
        self.lstm_initial_cell.copy_(initial_lstm_state[1])
        self.actions[0].copy_(self.actions[-1])
        self.rewards[0].copy_(self.rewards[-1])
        self.log_probs[0].copy_(self.log_probs[-1])
        self.done[0].copy_(self.done[-1])
        # Length of the current trajectory
        self.step = 1

    def enqueue(self, device=torch.device("cuda")):
        # to() -> if already on correct device this is a no-op
        return Trajectory(
            length=self.step,
            # Sequence and bootstrapping
            observations=self.observations[: self.step].clone().to(device),
            actions=self.actions[: self.step].clone().to(device),
            done=self.done[: self.step].clone().to(device),
            # Full sequence
            rewards=self.rewards[: self.step - 1].clone().to(device),
            log_probs=self.log_probs[: self.step - 1].clone().to(device),
            # Initial hidden states
            lstm_initial_hidden=self.lstm_initial_hidden.clone().to(device),
            lstm_initial_cell=self.lstm_initial_cell.clone().to(device),
        )
