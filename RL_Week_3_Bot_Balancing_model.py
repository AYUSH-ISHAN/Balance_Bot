import numpy as np
import os
import gym
import pybullet as p
import tensorflow as tf
import stable_baselines
from stable_baselines import DQN
# from stable_baselines import TRPO
from stable_baselines import A2C
from stable_baselines import ACER
# from stable_baselines import PPO1
from stable_baselines import PPO2
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env

if __name__ == "__main__":

    env = gym.make("BotBALANCE202036-v0")  # change the name of environment being loaded as gym
    time_steps = int(1e6)  # int(1e4)   # int(1e5)  # 2000
    for i in range(4):
        if i == 0:
            print("DQN")
            checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='D://ROL_Summer_Camp',
                                                     name_prefix='Model_DQN')
            model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1,
                        tensorboard_log="D://ROL_Summer_Camp/model")
            model.learn(total_timesteps=time_steps, tb_log_name="main_model", reset_num_timesteps=False,
                        callback=checkpoint_callback)
            model.save("Model_DQN")

#         elif i == 1:
#           model = TRPO(MlpPolicy, env, verbose=1)
#           model.learn(total_timesteps=1000)
#           model.save("trpo_balance")

#           model = TRPO.load("trpo_balance")

#           obs = env.reset()
#           while True:
#               action, _states = model.predict(obs)
#               obs, rewards, dones, info = env.step(action)

        elif i == 1:
          print("A2C")
          model = A2C(MlpPolicy, env, verbose=1)
          model.learn(total_timesteps=1000)   # 25000
          model.save("a2c_balance")

          model = A2C.load("a2c_balance")

          obs = env.reset()
          while True:
              action, _states = model.predict(obs)
              obs, rewards, dones, info = env.step(action)

        elif i == 2:
          print("ACER")
          model = ACER(MlpPolicy, env, verbose=1)
          model.learn(total_timesteps=1000)
          model.save("acer_balance_bot")

          model = ACER.load("acer_balance_bot")

          obs = env.reset()
          while True:
              action, _states = model.predict(obs)
              obs, rewards, dones, info = env.step(action)

#         elif i == 4:
#           model = PPO1(MlpPolicy, env, verbose=1)
#           model.learn(total_timesteps=1000)
#           model.save("ppo1_balance_bot")

#           model = PPO1.load("ppo1_balance_bot")

#           obs = env.reset()
#           while True:
#               action, _states = model.predict(obs)
#               obs, rewards, dones, info = env.step(action)

        elif i == 2:
          print()
          model = PPO2(MlpPolicy, env, verbose=1)
          model.learn(total_timesteps=1000)
          model.save("ppo2_balance_bot")

          model = PPO2.load("ppo2_balance_bot")

          obs = env.reset()
          while True:
              action, _states = model.predict(obs)
              obs, rewards, dones, info = env.step(action)
              env.render()

## after looking the reward of these models I picked up the one with maximum reward and declared that as my final Model.


