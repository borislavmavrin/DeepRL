![Breakout](breakout.gif)
## Deep Reinforcement Learning algorithms
This repo contains my implementations of Deep Reinforcement Learning algorithms I used in research:
 - DQN (by Deepmind)
 - QR-DQN (by Deepmind)
 - DLTV (mine)
 - DQN-decor (mine)

I tried to make the code concise.

Other features:
 1. Experience is collected in a separate thread which gives a slight improvement in runtime.
 2. Training reward is logged using tensorboard
 3. Hyperparameters (with some exceptions - see the Configs) are taken from [Rainbow](https://arxiv.org/abs/1710.02298)
## How to run
1. setup virtualenv (or conda):
 - `virtualenv -p python3 p3`
 - `source p3/bin/activate`
 - `pip install -r requirements.txt`
2. train agent (see the empirical results in the papers):
 - [Human-level control through deep reinforcement learning (original DQN)](https://deepmind-data.storage.googleapis.com/assets/papers/DeepMindNature14236Paper.pdf)
    - `python dqn/agent.py`
 - [Distributional Reinforcement Learning With Quantile Regression](https://ojs.aaai.org/index.php/AAAI/article/download/11791/11650) 
    - `python qr-dqn/agent.py`
 - [Distributional Reinforcement Learning for Efficient Exploration](http://proceedings.mlr.press/v97/mavrin19a/mavrin19a.pdf)
    - `python dltv/agent.py`
 - [Deep reinforcement learning with decorrelation](https://arxiv.org/pdf/1903.07765.pdf)
   - `python dqn-decor/agent.py`
3. check training reward in tensorboard:
 - `tensorboard --logdir=/tmp/tf_logs`