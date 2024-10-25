import pandas as pd
import numpy as np
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# ---------- Data Preprocessing ----------
def preprocess_data(file_path):
    """
    Preprocess the CSV file to extract relevant information.
    Assumes the dataset has columns for user ID, Date, and requested bandwidth.
    """
    columns = ['DID', 'Date', 'BW_REQUESTED']
    data = pd.read_csv(file_path, sep=';', names=columns, header=0)  # Skip the header row
    
    # Convert Date to datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M')
    
    return data

# Load and preprocess the data
file_path = 'train_data.csv'  # Your dataset file path
data = preprocess_data(file_path)

# ---------- Bandwidth Environment ----------
class BandwidthEnv(gym.Env):
    """
    Custom environment for bandwidth allocation using PPO.
    The agent adjusts MIR for each user based on their requests and system constraints.
    """
    def __init__(self, data, total_bandwidth=10000):
        super(BandwidthEnv, self).__init__()
        self.data = data
        self.total_bandwidth = total_bandwidth
        self.current_step = 0
        self.unique_dates = self.data['Date'].unique()

        # Action space: MIR adjustments for each of the 10 users (continuous)
        self.action_space = spaces.Box(low=1, high=total_bandwidth, shape=(10,), dtype=np.float32)

        # Observation space: Requested bandwidth, MIR, time of day, and a placeholder
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(10, 4), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_state().astype(np.float32), {}

    def _get_state(self):
        """
        Get the state for the current time step.
        Includes requested bandwidths and time of day for all users.
        """
        current_time = self.unique_dates[self.current_step]
        current_data = self.data[self.data['Date'] == current_time]
        requested_bandwidths = current_data['BW_REQUESTED'].values
        time_of_day = datetime.strptime(str(current_time), '%Y-%m-%d %H:%M:%S').hour
        
        # Assume initial MIRs are 1 Mbps for each user
        initial_MIRs = np.ones(10)
        
        # Placeholder for additional information
        placeholder = np.zeros(10)
        
        # Combine requested bandwidths, initial MIRs, time of day, and placeholder
        state = np.column_stack((requested_bandwidths, initial_MIRs, np.full(10, time_of_day), placeholder))
        return state

    def step(self, action):
        """
        Apply the action (MIR adjustments) and calculate the reward.
        """
        new_MIRs = action  # Actions correspond to the new MIR values for each user
        
        # Calculate total allocated bandwidth
        allocated_bandwidth = sum(new_MIRs)
        
        # Calculate the reward, penalties, and update the environment state
        reward, over_penalty, abuse_penalty = self.calculate_reward(new_MIRs, allocated_bandwidth)
        
        # Move to the next time step
        self.current_step += 1
        terminated = self.current_step >= len(self.unique_dates)  # End of the dataset
        truncated = False  # No truncation criteria in this example
        
        if terminated:
            return self._get_state().astype(np.float32), reward, terminated, truncated, {}
        
        return self._get_state().astype(np.float32), reward, terminated, truncated, {}

    def calculate_reward(self, MIRs, total_allocated_bandwidth):
        """
        Calculate the reward for the current step based on efficiency and penalties.
        """
        # Penalty for over-allocating bandwidth
        over_penalty = self.over_allocation_penalty(total_allocated_bandwidth)
        
        # Placeholder for abusive usage penalty (can be refined)
        abuse_penalty = 0
        
        # Efficiency calculation for each user
        efficiencies = [self.calculate_efficiency(MIR, req_bw) for MIR, req_bw in zip(MIRs, self._get_state()[:, 0])]
        total_efficiency = sum(efficiencies)
        
        # Reward = total efficiency - penalties
        reward = total_efficiency - over_penalty - abuse_penalty
        return reward, over_penalty, abuse_penalty

    def over_allocation_penalty(self, total_allocated_bandwidth, beta=3):
        """
        Calculate the penalty for over-allocating bandwidth beyond the system's capacity.
        """
        if total_allocated_bandwidth > self.total_bandwidth:
            penalty = beta * (total_allocated_bandwidth - self.total_bandwidth) / self.total_bandwidth
        else:
            penalty = 0
        return penalty

    def calculate_efficiency(self, MIR, user_request):
        """
        Calculate the efficiency (allocation ratio) for each user.
        """
        if user_request >= MIR:
            return MIR / user_request
        else:
            return 1.0  # Fully satisfied request

    def render(self, mode='human'):
        pass

# ---------- Model Training Using PPO ----------
# Create the environment
env = BandwidthEnv(data)

# Check if the environment adheres to Gym API standards
check_env(env)

# Create and configure the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the PPO model on the environment
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_bandwidth_model")

# ---------- Model Evaluation ----------
# Load the trained model for testing
model = PPO.load("ppo_bandwidth_model")

# Reset the environment for evaluation
obs, _ = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    print(f"Reward: {reward}")