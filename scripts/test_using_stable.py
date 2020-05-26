import gym
import numpy as np

from rl_with_teachers.envs import *

from stable_baselines.ddpg.policies import MlpPolicy
# from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

env = gym.make('PickPlaceRandomGoal-v0')

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = None #OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=400)
model.save("ddpg_pickplace")

obs = env.reset()
for i in range(50):
    import pdb; pdb.set_trace()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
