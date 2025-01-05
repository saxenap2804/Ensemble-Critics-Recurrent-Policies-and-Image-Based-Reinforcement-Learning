# Ensemble Critics, Recurrent Policies, and Image-Based Reinforcement Learning: 

This project focuses on implementing and analyzing advanced reinforcement learning 
techniques across three distinct tasks: Specifically, it aims to: (1) Ensemble Critics for an accurate 
estimate of the value function, (2) Recurrent Policies in complex environments with limited 
observability, and (3) Image-Based Policy Learning for large-scale inputs.

Environment: This project is created with the inverted pendulum swingup problem as the base using Gymnasium. The system consists of a pendulum attached at one end to a fixed point, and the other end being free. The pendulum starts in a random position and the goal is to apply torque on the free end to swing it into an upright position, with its center of gravity right above the fixed point.

## Task 1: Ensemble of Critics

### Objective:
Consider C=5 instead of the single randomly chosen critic to increase the reliability of the value 
estimation. 
### Approach:
1. Architecture:
o Developed B=5 independent critic networks, each with separate parameters 𝜃𝑏.
o The final value function is the average of all critic’s outputs.
2. Training Process:
o The loss function for each critic was the Temporal Difference (TD) error. 
 Bootstrapping introduced diversity in critics, improving overall robustness.
3. Visualization:
o Performances of the ensemble critics were tested by learning curves.
o Noisy loss curves and 2D landscape plots of V(s) were created to visualize stability and 
robustness.


### Results:

1.Learning Curves-



2.Noisy Loss Curves-



3.2D Landscape Plots-



## Task 2: Recurrent Policy under Partial Observability


### Objective:
To train a recurrent policy that operates effectively in a partially observable environment, where 
the agent receives angular velocity as the only observation.
### Approach:
1. State Representation:
o Input was restricted to angular velocity, excluding angle information.
o This created a partially observable Markov Decision Process (POMDP).
2. Recurrent Policy Architecture:
o Introduced an LSTM (Long Short-Term Memory) layer in the policy network to model 
temporal dependencies.
3. Training Process:
o Optimized the policy using the REINFORCE algorithm
4. Visualization:
o Learning curves demonstrated the agent’s ability to adapt to limited observations.
o Noisy loss curves revealed training stability and convergence.


### Results:

Learning Curves-

Noisy Loss Curves-



## Task 3: Image-Based Policy Learning



### Objective:
Train a policy using image sequences as inputs instead of state vectors. Temporal information 
was incorporated using stacked images or ConvLSTM layers.
### Approach:
1. Input Representation:
o State represented as sequences of images.
o Explored two configurations: 
▪ Stacked Images: Concatenated multiple image frames into a single input 
vector.
▪ ConvLSTM: Used recurrent convolutional layers to learn temporal features.
2. Network Architecture:
o Convolutional Layers: Extracted spatial features from input images.
o Stacked Images: Flattened features of multiple frames were combined for policy 
training.
o ConvLSTM: Modeled temporal dependencies directly from image sequences.
3. Training Process:
o Used actor-critic methods with the following loss functions.

### Results:

Learning Curves-


Noisy Loss Curves-

