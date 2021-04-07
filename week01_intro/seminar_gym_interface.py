#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
if 'google.colab' in sys.modules and not os.path.exists('.setup_complete'):
    get_ipython().system('wget -q https://raw.githubusercontent.com/yandexdataschool/Practical_RL/master/setup_colab.sh -O- | bash')

    get_ipython().system('touch .setup_complete')

# This code creates a virtual display to draw game images on.
# It will have no effect if your machine has a monitor.
if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY")) == 0:
    get_ipython().system('bash ../xvfb start')
    os.environ['DISPLAY'] = ':1'


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### OpenAI Gym
# 
# We're gonna spend several next weeks learning algorithms that solve decision processes. We are then in need of some interesting decision problems to test our algorithms.
# 
# That's where OpenAI Gym comes into play. It's a Python library that wraps many classical decision problems including robot control, videogames and board games.
# 
# So here's how it works:

# In[3]:


import gym

env = gym.make("MountainCar-v0")
env.reset()

plt.imshow(env.render('rgb_array'))
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)


# Note: if you're running this on your local machine, you'll see a window pop up with the image above. Don't close it, just alt-tab away.

# ### Gym interface
# 
# The three main methods of an environment are
# * `reset()`: reset environment to the initial state, _return first observation_
# * `render()`: show current environment state (a more colorful version :) )
# * `step(a)`: commit action `a` and return `(new_observation, reward, is_done, info)`
#  * `new_observation`: an observation right after committing the action `a`
#  * `reward`: a number representing your reward for committing action `a`
#  * `is_done`: True if the MDP has just finished, False if still in progress
#  * `info`: some auxiliary stuff about what just happened. For now, ignore it.

# In[4]:


obs0 = env.reset()
print("initial observation code:", obs0)

# Note: in MountainCar, observation is just two numbers: car position and velocity


# In[5]:


print("taking action 2 (right)")
new_obs, reward, is_done, _ = env.step(2)

print("new observation code:", new_obs)
print("reward:", reward)
print("is game over?:", is_done)

# Note: as you can see, the car has moved to the right slightly (around 0.0005)


# ### Play with it
# 
# Below is the code that drives the car to the right. However, if you simply use the default policy, the car will not reach the flag at the far right due to gravity.
# 
# __Your task__ is to fix it. Find a strategy that reaches the flag. 
# 
# You are not required to build any sophisticated algorithms for now, and you definitely don't need to know any reinforcement learning for this. Feel free to hard-code :)

# In[6]:


from IPython import display

# Create env manually to set time limit. Please don't change this.
TIME_LIMIT = 250
env = gym.wrappers.TimeLimit(
    gym.envs.classic_control.MountainCarEnv(),
    max_episode_steps=TIME_LIMIT + 1,
)
actions = {'left': 0, 'stop': 1, 'right': 2}


# In[7]:


def policy(obs, t):
    # Write the code for your policy here. You can use the observation
    # (a tuple of position and velocity), the current time step, or both,
    # if you want.
    position, velocity = obs
    
    VELOCITY_THRESHOLD = 1e-3
    if velocity >= VELOCITY_THRESHOLD or velocity < 0 and velocity > -VELOCITY_THRESHOLD:
        return actions['right']
    if velocity <= -VELOCITY_THRESHOLD or velocity > 0 and velocity < VELOCITY_THRESHOLD:
        return actions['left']
    
    return actions['right']


# In[8]:


plt.figure(figsize=(4, 3))
display.clear_output(wait=True)

obs = env.reset()
for t in range(TIME_LIMIT):
    plt.gca().clear()
    
    action = policy(obs, t)  # Call your policy
    obs, reward, done, _ = env.step(action)  # Pass the action chosen by the policy to the environment
    
    # We don't do anything with reward here because MountainCar is a very simple environment,
    # and reward is a constant -1. Therefore, your goal is to end the episode as quickly as possible.

    # Draw game image on display.
    plt.imshow(env.render('rgb_array'))
    plt.title(obs)
    
    display.display(plt.gcf())
    display.clear_output(wait=True)

    if done:
        print("Well done!")
        break
else:
    print("Time limit exceeded. Try again.")

display.clear_output(wait=True)


# In[10]:


assert obs[0] > 0.47
print("You solved it!")


# In[ ]:




