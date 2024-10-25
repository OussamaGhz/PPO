import pandas as pd
import numpy as np
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO


class SimpleBandwidthEnv(gym.Env):
    def __init__(self, data, total_bandwidth=10000, cir=1000):
        super(SimpleBandwidthEnv, self).__init__()
        self.data = data
        self.total_bandwidth = total_bandwidth
        self.cir = cir
        self.num_users = 10
        self.current_step = 0

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=cir, high=total_bandwidth, shape=(self.num_users,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.num_users, 2), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_state(), {}

    def _get_state(self):
        """
        Get the current state, with requested bandwidth and initial allocations.
        """
        unique_dates = self.data["Date"].unique()

        # Ensure `self.current_step` is within bounds
        if self.current_step >= len(unique_dates):
            self.current_step = 0  # Reset step if it exceeds available data

        current_time = unique_dates[self.current_step]
        current_data = self.data[self.data["Date"] == current_time].sort_values("DID")
        requested_bandwidths = current_data["BW_REQUESTED"].values
        initial_allocations = np.ones(self.num_users) * self.cir
        state = np.column_stack((requested_bandwidths, initial_allocations))
        return state

    def step(self, action):
        # Clip action to ensure it respects CIR and total bandwidth constraints
        allocated_bandwidth = np.clip(action, self.cir, self.total_bandwidth)
        requested_bandwidths = self._get_state()[:, 0]

        # Calculate rewards based on efficiency and penalty for over-allocation
        efficiency_reward = (
            np.sum(
                np.minimum(allocated_bandwidth, requested_bandwidths)
                / requested_bandwidths
            )
            / self.num_users
        )
        over_allocation_penalty = (
            max(0, np.sum(allocated_bandwidth) - self.total_bandwidth) * -0.1
        )
        reward = efficiency_reward + over_allocation_penalty

        # Advance to the next time step and check for end of data
        self.current_step += 1
        done = self.current_step >= len(self.data["Date"].unique())
        truncated = False

        return self._get_state(), reward, done, truncated, {}


def preprocess_data(file_path):
    columns = ["DID", "Date", "BW_REQUESTED"]
    data = pd.read_csv(file_path, sep=";", header=0)
    data["Date"] = pd.to_datetime(
        data["Date"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    return data


# Load the dataset
file_path = "train_data.csv"
data = preprocess_data(file_path)

# Initialize the environment with the dataset and CIR constraint
env = SimpleBandwidthEnv(data)

# Initialize and train the PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model
model.save("simple_ppo_bandwidth_model")

# Load and evaluate the trained model
model = PPO.load("simple_ppo_bandwidth_model")
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    print(f"Step Reward: {reward}")

print(f"Total Reward: {total_reward}")
