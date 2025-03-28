**DRL Robot navigation in IR-SIM**

Deep Reinforcement Learning algorithm implementation for simulated drone navigation in IR-SIM. Using 2D laser sensor data
and information about the goal point a robot learns to navigate to a specified point in the environment.


**Installation**

* Package versioning is managed with poetry \
`pip install poetry`
* Clone the repository \
`git clone`
* Navigate to the cloned location and install using poetry \
`poetry install https://github.com/williamcheong0616/DRL-drone-navigation-IR-SIM.git`

**Training the model**

* Run the training by executing the train.py file \
`poetry run python robot_nav/train.py`

* To open tensorbord, in a new terminal execute \
`tensorboard --logdir runs`



**Sources**

| Package/Model |                                           Description                                           |                    Model                           Source | 
|:--------------|:-----------------------------------------------------------------------------------------------:|----------------------------------------------------------:| 
| IR-SIM        |                                  Light-weight robot simulator                                   |                       https://github.com/hanruihua/ir-sim | 
| TD3           |                      Twin Delayed Deep Deterministic Policy Gradient model                      | https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2 | 
| SAC           |                                     Soft Actor-Critic model                                     |                https://github.com/denisyarats/pytorch_sac | 
| PPO           |                               Proximal Policy Optimization model                                |            https://github.com/nikhilbarhate99/PPO-PyTorch | 
| DDPG          |                            Deep Deterministic Policy Gradient model                             |                                          Updated from TD3 | 
| CNNTD3        |                          TD3 model with 1D CNN encoding of laser state                          |https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM|
| RCPG          | Recurrent Convolution Policy Gradient - adding recurrence layers (lstm/gru/rnn) to CNNTD3 model |https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM|


