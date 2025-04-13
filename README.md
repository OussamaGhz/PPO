# PPO Bandwidth Allocation Documentation

## Project Overview

This repository contains an implementation of a Proximal Policy Optimization (PPO) reinforcement learning model for dynamic bandwidth allocation in a network environment. The system optimizes bandwidth distribution among multiple users based on their requests, usage patterns, and time of day.

## Key Components

### 1. Environment Simulation

The project implements custom Gymnasium environments that simulate bandwidth allocation:

- **SimpleBandwidthEnv**: A more sophisticated environment with abusive usage detection
- **BandwidthEnv**: A simpler implementation focused on basic allocation

### 2. Data Processing

The system uses CSV data with the following structure:
- **DID**: User identifier
- **Date**: Timestamp of the bandwidth request
- **BW_REQUESTED**: Amount of bandwidth requested

### 3. Reinforcement Learning Model

The project uses Stable Baselines3's PPO implementation to learn optimal bandwidth allocation policies with:
- Continuous action space for bandwidth allocation
- Observation space including requested bandwidth, initial allocations, and time features

## Technical Details

### Environment Parameters

- **Total Bandwidth**: The total available bandwidth for allocation (default: 10,000 units)
- **CIR (Committed Information Rate)**: Minimum guaranteed bandwidth per user (default: 1,000 units)
- **MIR (Maximum Information Rate)**: Maximum bandwidth a user can receive

### Reward System

The reward function balances multiple factors:
1. **Efficiency Reward**: Maximizes bandwidth allocation efficiency across users
2. **Over-allocation Penalty**: Penalizes exceeding total available bandwidth
3. **Abusive Usage Penalty**: Penalizes users who repeatedly exceed their allocated bandwidth

### Training Process

1. Data is preprocessed from CSV into pandas DataFrame
2. The environment is initialized with the dataset
3. PPO model is trained on the environment
4. Trained model is saved and can be loaded for evaluation

## Usage

```python
# Load and preprocess data
file_path = "train_data.csv"
data = preprocess_data(file_path)

# Initialize environment
env = SimpleBandwidthEnv(data)

# Train model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save model
model.save("simple_ppo_bandwidth_model")

# Load and evaluate model
model = PPO.load("simple_ppo_bandwidth_model")
obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
```

## Dependencies

- pandas
- numpy
- gymnasium
- stable-baselines3
- datetime

## Project Structure

- **main.py**: Main implementation with SimpleBandwidthEnv
- **py.py**: Alternative implementation with BandwidthEnv
- **train_data.csv**: Dataset containing bandwidth request information

## Future Improvements

1. More sophisticated detection of abusive usage patterns
2. Additional features in the observation space (e.g., historical usage patterns)
3. Hyperparameter optimization for PPO
4. Integration with real-time network monitoring systems
