# Deep Reinforcement Learning for Robot Navigation

A comprehensive implementation of various Deep Reinforcement Learning algorithms for autonomous robot navigation using Isaac Sim simulation environment.

## Overview

This project contains implementations of multiple state-of-the-art reinforcement learning algorithms applied to robot navigation tasks. The project was developed as part of a Master's thesis research on comparing different RL approaches for autonomous navigation in simulated environments.

**IMPORTANT**: The Isaac Sim environment is a private submodule due to proprietary licensing restrictions. To access the complete simulation environment, please contact the author for repository access.

## Algorithms Implemented

- **Deep Q Network (DQN)** - Value-based RL with experience replay [1]
- **Deep Deterministic Policy Gradient (DDPG)** - Actor-critic method for continuous control [2]
- **Twin Delayed Deep Deterministic Policy Gradient (TD3)** - Improved DDPG with target policy smoothing [3]
- **TD3 Multi** - Multi Robot Implementation of Twin DDPG [4]
- **DQN Multi** - Multi Robot Implementation of DQN [1]

## Project Structure

```
├── src/
│   ├── algorithms/          # RL algorithm implementations
│   │   ├── dqn/            # Deep Q-Network implementations
│   │   ├── ddpg/           # DDPG implementations
│   │   ├── td3/            # TD3 implementations
│   │   └── gdae/           # GDAE implementations
│   ├── utils/              # Utility functions and helpers
│   └── memory/             # Experience replay and memory management
├── experiments/
│   ├── models/             # Trained model weights
│   ├── logs/               # Training logs and TensorBoard data
│   └── results/            # Hyperparameter Experimentation logs
├── docs/                   # Documentation and thesis materials
├── data/                   # ISAAC Sim Environment
├── assets/                 # Images, plots, and visualizations
│   ├── images/
│   └── plots/
└── tests/                  # Unit tests and validation scripts
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- Isaac Sim (for simulation environment)
- CUDA (recommended for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dqn_robot_navigation.git
cd dqn_robot_navigation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Set up Isaac Sim environment submodule** (Required):
```bash
# The Isaac Sim environment is a private submodule
# Contact the author for access to the proprietary simulation environment
git submodule update --init --recursive
```

4. Set up Isaac Sim environment according to NVIDIA's documentation

## Usage

### Training Models
Firstly Python Path must be set to use the python executable in ISAAC Sim configuration's worspace folder:

```bash
PYTHONPATH = "{Path/to_isaac_sim/python_executable_folder}:$PYTHONPATH"
```
Each algorithm can be trained independently:

```bash
# Train DQN
python src/algorithms/dqn/deepq_torch.py

# Train DDPG
python src/algorithms/ddpg/DDPG.py

# Train TD3
python src/algorithms/td3/td3.py
```

### Evaluation

```bash
# Evaluate trained models
python src/evaluate.py --algorithm ddpg --model_path experiments/models/ddpg_best.pth
```

## Research Context

This work contributes to the field of autonomous robotics by:

1. **Comparative Analysis**: Systematic comparison of multiple DRL algorithms
2. **Simulation Integration**: Integration with Isaac Sim physics engine
3. **Multi-Agent Support**: Extensions for multi-agent navigation scenarios
4. **Hyperparameter Optimization**: Comprehensive hyperparameter tuning results

## Key Features

- **GPU Acceleration**: CUDA support for faster training
- **TensorBoard Integration**: Real-time training visualization
- **Experience Replay**: Efficient memory management for off-policy learning
- **Hyperparameter Logging**: Systematic tracking of experiment parameters

##Bibliography
1. H. Anas, W. H. Ong, and O. A. Malik, “Comparison of deep q-learning, q-learning and
sarsa reinforced learning for robot local navigation,” in Robot Intelligence Technology and
Applications 6 (J. Kim, B. Englot, H.-W. Park, H.-L. Choi, H. Myung, J. Kim, and J.-H.
Kim, eds.), (Cham), pp. 443–454, Springer International Publishing, 2022.
2. L. Tai, G. Paolo, and M. Liu, “Virtual-to-real deep reinforcement learning: Continuous
control of mobile robots for mapless navigation,” 2017 IEEE/RSJ International Conference
on Intelligent Robots and Systems (IROS), pp. 31–36, 2017.
3. R. Cimurs, I. H. Suh, and J. H. Lee, “Goal-driven autonomous exploration through deep
reinforcement learning,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 730–737,
2022. 
4. H. Anas, O. W. Hong, and O. A. Malik, “Deep reinforcement learning-based mapless
crowd navigation with perceived risk of the moving crowd for mobile robots,” 2023.
---

*This project represents the culmination of Master's thesis research in Deep Reinforcement Learning for Robotics.*