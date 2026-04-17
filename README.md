# Cooperative Jamming for Physical Layer Security using Deep Reinforcement Learning

This repository contains a Python/Gymnasium simulation of Physical Layer Security (PLS) in a wireless network. It leverages a Deep Reinforcement Learning (DRL) agent to optimize transmit power for Cooperative Friendly Jamming (CFJ).

## 📄 About the Project
This project is based on the system model proposed in the 2024 paper: *"Cooperative Jamming for Physical Layer Security Enhancement Using Deep Reinforcement Learning"* (Hoseini et al.). 

While the baseline paper assumes the network has perfect knowledge of the eavesdropper's location, **this project introduces a novel contribution: Imperfect/Outdated Channel State Information (CSI).** The environment introduces realistic noise/lag into the eavesdropper's coordinates, forcing the Soft Actor-Critic (SAC) agent to learn a robust jamming power allocation strategy under uncertainty.

## ✨ Key Features
* **Custom Gymnasium Environment:** Simulates a Multi-Input Single-Output (MISO) wireless grid using Friis transmission equations.
* **Smart User Association:** Automatically maps legitimate users to the optimal Access Point (AP) before power allocation.
* **Continuous Power Control:** Uses the state-of-the-art Soft Actor-Critic (SAC) algorithm to assign continuous transmit power levels (0 to 1 Watt) to data-transmitting APs and jamming APs.
* **Imperfect CSI Simulation:** Modifies the agent's observation space to simulate real-world radar lag and measurement errors.

## 📂 Repository Structure
* `envs/cfj_env.py`: The custom Gymnasium environment (Physics, State, Action Space, Reward Function).
* `train.py`: The training script that initializes and trains the SAC agent using Stable-Baselines3.
* `test.py`: The evaluation script to test the trained model and generate performance graphs.
* `models/`: Directory where the trained `.zip` model files are saved.
* `results/`: Directory for output graphs (e.g., Secrecy Capacity vs. Number of APs).

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/pls-drl-cooperative-jamming.git](https://github.com/YOUR_USERNAME/pls-drl-cooperative-jamming.git)
   cd pls-drl-cooperative-jamming

   ### Create a virtual environment
   python -m venv pls_env
# Windows:
pls_env\Scripts\activate
# Mac/Linux:
source pls_env/bin/activate

pip install -r requirements.txt

pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org

python train.py

python test.py