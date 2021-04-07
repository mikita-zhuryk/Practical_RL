#!/usr/bin/env python
# coding: utf-8

# ### Markov decision process
# 
# This week's methods are all built to solve __M__arkov __D__ecision __P__rocesses. In the broadest sense, an MDP is defined by how it changes states and how rewards are computed.
# 
# State transition is defined by $P(s' |s,a)$ - how likely are you to end at state $s'$ if you take action $a$ from state $s$. Now there's more than one way to define rewards, but we'll use $r(s,a,s')$ function for convenience.
# 
# _This notebook is inspired by the awesome_ [CS294](https://github.com/berkeleydeeprlcourse/homework/blob/36a0b58261acde756abd55306fbe63df226bf62b/hw2/HW2.ipynb) _by Berkeley_

# For starters, let's define a simple MDP from this picture:
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/a/ad/Markov_Decision_Process.svg" width="400px" alt="Diagram by Waldoalvarez via Wikimedia Commons, CC BY-SA 4.0"/>

# In[1]:


import sys, os
if 'google.colab' in sys.modules and not os.path.exists('.setup_complete'):
    get_ipython().system('wget -q https://raw.githubusercontent.com/yandexdataschool/Practical_RL/master/setup_colab.sh -O- | bash')
    get_ipython().system('wget -q https://raw.githubusercontent.com/yandexdataschool/Practical_RL/master/week02_value_based/mdp.py')
    get_ipython().system('touch .setup_complete')

# This code creates a virtual display to draw game images on.
# It will have no effect if your machine has a monitor.
if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY")) == 0:
    get_ipython().system('bash ../xvfb start')
    os.environ['DISPLAY'] = ':1'


# In[2]:


transition_probs = {
    's0': {
        'a0': {'s0': 0.5, 's2': 0.5},
        'a1': {'s2': 1}
    },
    's1': {
        'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
        'a1': {'s1': 0.95, 's2': 0.05}
    },
    's2': {
        'a0': {'s0': 0.4, 's2': 0.6},
        'a1': {'s0': 0.3, 's1': 0.3, 's2': 0.4}
    }
}
rewards = {
    's1': {'a0': {'s0': +5}},
    's2': {'a1': {'s0': -1}}
}

from mdp import MDP
mdp = MDP(transition_probs, rewards, initial_state='s0')


# We can now use MDP just as any other gym environment:

# In[3]:


print('initial state =', mdp.reset())
next_state, reward, done, info = mdp.step('a1')
print('next_state = %s, reward = %s, done = %s' % (next_state, reward, done))


# but it also has other methods that you'll need for Value Iteration

# In[4]:


print("mdp.get_all_states =", mdp.get_all_states())
print("mdp.get_possible_actions('s1') = ", mdp.get_possible_actions('s1'))
print("mdp.get_next_states('s1', 'a0') = ", mdp.get_next_states('s1', 'a0'))
print("mdp.get_reward('s1', 'a0', 's0') = ", mdp.get_reward('s1', 'a0', 's0'))
print("mdp.get_transition_prob('s1', 'a0', 's0') = ", mdp.get_transition_prob('s1', 'a0', 's0'))


# ### Optional: Visualizing MDPs
# 
# You can also visualize any MDP with the drawing fuction donated by [neer201](https://github.com/neer201).
# 
# You have to install graphviz for system and for python. 
# 
# 1. * For ubuntu just run: `sudo apt-get install graphviz` 
#    * For OSX: `brew install graphviz`
# 2. `pip install graphviz`
# 3. restart the notebook
# 
# __Note:__ Installing graphviz on some OS (esp. Windows) may be tricky. However, you can ignore this part alltogether and use the standart vizualization.

# In[5]:


from mdp import has_graphviz
from IPython.display import display
print("Graphviz available:", has_graphviz)


# In[6]:


if has_graphviz:
    from mdp import plot_graph, plot_graph_with_state_values, plot_graph_optimal_strategy_and_state_values
    display(plot_graph(mdp, graph_size="50,50"))


# ### Value Iteration
# 
# Now let's build something to solve this MDP. The simplest algorithm so far is __V__alue __I__teration
# 
# Here's the pseudo-code for VI:
# 
# ---
# 
# `1.` Initialize $V^{(0)}(s)=0$, for all $s$
# 
# `2.` For $i=0, 1, 2, \dots$
#  
# `3.` $ \quad V_{(i+1)}(s) = \max_a \sum_{s'} P(s' | s,a) \cdot [ r(s,a,s') + \gamma V_{i}(s')]$, for all $s$
# 
# ---

# First, let's write a function to compute the state-action value function $Q^{\pi}$, defined as follows
# 
# $$Q_i(s, a) = \sum_{s'} P(s' | s,a) \cdot [ r(s,a,s') + \gamma V_{i}(s')]$$
# 

# In[7]:


def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """

    v = 0
    for next_state in mdp.get_next_states(state, action):
        v += (mdp.get_transition_prob(state, action, next_state)
              * (mdp.get_reward(state, action, next_state) + gamma * state_values[next_state]))

    return v


# In[8]:


import numpy as np
test_Vs = {s: i for i, s in enumerate(sorted(mdp.get_all_states()))}
assert np.isclose(get_action_value(mdp, test_Vs, 's2', 'a1', 0.9), 0.69)
assert np.isclose(get_action_value(mdp, test_Vs, 's1', 'a0', 0.9), 3.95)


# Using $Q(s,a)$ we can now define the "next" V(s) for value iteration.
#  $$V_{(i+1)}(s) = \max_a \sum_{s'} P(s' | s,a) \cdot [ r(s,a,s') + \gamma V_{i}(s')] = \max_a Q_i(s,a)$$

# In[9]:


def get_new_state_value(mdp, state_values, state, gamma):
    """ Computes next V(s) as in formula above. Please do not change state_values in process. """
    if mdp.is_terminal(state):
        return 0

    v = -1e30
    action = None
    for possible_action in mdp.get_possible_actions(state):
        cur_v = get_action_value(mdp, state_values, state, possible_action, gamma)
        if cur_v > v:
            v = cur_v
            action = possible_action
    
    return v


# In[10]:


test_Vs_copy = dict(test_Vs)
assert np.isclose(get_new_state_value(mdp, test_Vs, 's0', 0.9), 1.8)
assert np.isclose(get_new_state_value(mdp, test_Vs, 's2', 0.9), 1.08)
assert np.isclose(get_new_state_value(mdp, {'s0': -1e10, 's1': 0, 's2': -2e10}, 's0', 0.9), -13500000000.0),     "Please ensure that you handle negative Q-values of arbitrary magnitude correctly"
assert test_Vs == test_Vs_copy, "Please do not change state_values in get_new_state_value"


# Finally, let's combine everything we wrote into a working value iteration algo.

# In[11]:


# parameters
gamma = 0.9            # discount for MDP
num_iter = 100         # maximum iterations, excluding initialization
# stop VI if new values are this close to old values (or closer)
min_difference = 0.001

# initialize V(s)
state_values = {s: 0 for s in mdp.get_all_states()}

if has_graphviz:
    display(plot_graph_with_state_values(mdp, state_values))

for i in range(num_iter):

    # Compute new state values using the functions you defined above.
    # It must be a dict {state : float V_new(state)}
    new_state_values = {s: get_new_state_value(mdp, state_values, s, gamma) for s in state_values.keys()}

    assert isinstance(new_state_values, dict)

    # Compute difference
    diff = max(abs(new_state_values[s] - state_values[s])
               for s in mdp.get_all_states())
    print("iter %4i   |   diff: %6.5f   |   " % (i, diff), end="")
    print('   '.join("V(%s) = %.3f" % (s, v) for s, v in state_values.items()))
    state_values = new_state_values

    if diff < min_difference:
        print("Terminated")
        break


# In[12]:


if has_graphviz:
    display(plot_graph_with_state_values(mdp, state_values))


# In[13]:


print("Final state values:", state_values)

assert abs(state_values['s0'] - 3.781) < 0.01
assert abs(state_values['s1'] - 7.294) < 0.01
assert abs(state_values['s2'] - 4.202) < 0.01


# Now let's use those $V^{*}(s)$ to find optimal actions in each state
# 
#  $$\pi^*(s) = argmax_a \sum_{s'} P(s' | s,a) \cdot [ r(s,a,s') + \gamma V_{i}(s')] = argmax_a Q_i(s,a)$$
#  
# The only difference vs V(s) is that here we take not max but argmax: find action such with maximum Q(s,a).

# In[14]:


def get_optimal_action(mdp, state_values, state, gamma=0.9):
    """ Finds optimal action using formula above. """
    if mdp.is_terminal(state):
        return None

    v = -1e30
    action = None
    for possible_action in mdp.get_possible_actions(state):
        cur_v = get_action_value(mdp, state_values, state, possible_action, gamma)
        if cur_v > v:
            v = cur_v
            action = possible_action

    return action


# In[15]:


assert get_optimal_action(mdp, state_values, 's0', gamma) == 'a1'
assert get_optimal_action(mdp, state_values, 's1', gamma) == 'a0'
assert get_optimal_action(mdp, state_values, 's2', gamma) == 'a1'

assert get_optimal_action(mdp, {'s0': -1e10, 's1': 0, 's2': -2e10}, 's0', 0.9) == 'a0',     "Please ensure that you handle negative Q-values of arbitrary magnitude correctly"
assert get_optimal_action(mdp, {'s0': -2e10, 's1': 0, 's2': -1e10}, 's0', 0.9) == 'a1',     "Please ensure that you handle negative Q-values of arbitrary magnitude correctly"


# In[16]:


if has_graphviz:
    display(plot_graph_optimal_strategy_and_state_values(mdp, state_values, get_action_value))


# In[17]:


# Measure agent's average reward

s = mdp.reset()
rewards = []
for _ in range(10000):
    s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
    rewards.append(r)

print("average reward: ", np.mean(rewards))

assert(0.40 < np.mean(rewards) < 0.55)


# ### Frozen lake

# In[18]:


from mdp import FrozenLakeEnv
mdp = FrozenLakeEnv(slip_chance=0)

mdp.render()


# In[19]:


def value_iteration(mdp, state_values=None, gamma=0.9, num_iter=1000, min_difference=1e-5, verbose=True):
    """ performs num_iter value iteration steps starting from state_values. Same as before but in a function """
    state_values = state_values or {s: 0 for s in mdp.get_all_states()}
    it = 0
    for i in range(num_iter):

        # Compute new state values using the functions you defined above. It must be a dict {state : new_V(state)}
        new_state_values = {s: get_new_state_value(mdp, state_values, s, gamma) for s in state_values.keys()}

        assert isinstance(new_state_values, dict)

        # Compute difference
        diff = max(abs(new_state_values[s] - state_values[s])
                   for s in mdp.get_all_states())

        if verbose:
            print("iter %4i   |   diff: %6.5f   |   V(start): %.3f " %
                  (i, diff, new_state_values[mdp._initial_state]))

        state_values = new_state_values
        it += 1
        if diff < min_difference:
            break

    return state_values, it


# In[20]:


state_values, _ = value_iteration(mdp)


# In[21]:


s = mdp.reset()
mdp.render()
for t in range(100):
    a = get_optimal_action(mdp, state_values, s, gamma)
    print(a, end='\n\n')
    s, r, done, _ = mdp.step(a)
    mdp.render()
    if done:
        break


# ### Let's visualize!
# 
# It's usually interesting to see what your algorithm actually learned under the hood. To do so, we'll plot state value functions and optimal actions at each VI step.

# In[22]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


def draw_policy(mdp, state_values):
    plt.figure(figsize=(3, 3))
    h, w = mdp.desc.shape
    states = sorted(mdp.get_all_states())
    V = np.array([state_values[s] for s in states])
    Pi = {s: get_optimal_action(mdp, state_values, s, gamma) for s in states}
    plt.imshow(V.reshape(w, h), cmap='gray', interpolation='none', clim=(0, 1))
    ax = plt.gca()
    ax.set_xticks(np.arange(h)-.5)
    ax.set_yticks(np.arange(w)-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:4, 0:4]
    a2uv = {'left': (-1, 0), 'down': (0, -1), 'right': (1, 0), 'up': (0, 1)}
    for y in range(h):
        for x in range(w):
            plt.text(x, y, str(mdp.desc[y, x].item()),
                     color='g', size=12,  verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
            a = Pi[y, x]
            if a is None:
                continue
            u, v = a2uv[a]
            plt.arrow(x, y, u*.3, -v*.3, color='m',
                      head_width=0.1, head_length=0.1)
    plt.grid(color='b', lw=2, ls='-')
    plt.show()


# In[23]:


state_values = {s: 0 for s in mdp.get_all_states()}

for i in range(10):
    print("after iteration %i" % i)
    state_values, _ = value_iteration(mdp, state_values, num_iter=1)
    draw_policy(mdp, state_values)
# please ignore iter 0 at each step


# In[24]:


from IPython.display import clear_output
from time import sleep
mdp = FrozenLakeEnv(map_name='8x8', slip_chance=0.1)
state_values = {s: 0 for s in mdp.get_all_states()}

for i in range(30):
    clear_output(True)
    print("after iteration %i" % i)
    state_values, _ = value_iteration(mdp, state_values, num_iter=1)
    draw_policy(mdp, state_values)
    sleep(0.5)
# please ignore iter 0 at each step


# Massive tests

# In[25]:


mdp = FrozenLakeEnv(slip_chance=0)
state_values, _ = value_iteration(mdp)

total_rewards = []
for game_i in range(1000):
    s = mdp.reset()
    rewards = []
    for t in range(100):
        s, r, done, _ = mdp.step(
            get_optimal_action(mdp, state_values, s, gamma))
        rewards.append(r)
        if done:
            break
    total_rewards.append(np.sum(rewards))

print("average reward: ", np.mean(total_rewards))
assert(1.0 <= np.mean(total_rewards) <= 1.0)
print("Well done!")


# In[26]:


# Measure agent's average reward
mdp = FrozenLakeEnv(slip_chance=0.1)
state_values, _ = value_iteration(mdp)

total_rewards = []
for game_i in range(1000):
    s = mdp.reset()
    rewards = []
    for t in range(100):
        s, r, done, _ = mdp.step(
            get_optimal_action(mdp, state_values, s, gamma))
        rewards.append(r)
        if done:
            break
    total_rewards.append(np.sum(rewards))

print("average reward: ", np.mean(total_rewards))
assert(0.8 <= np.mean(total_rewards) <= 0.95)
print("Well done!")


# In[27]:


# Measure agent's average reward
mdp = FrozenLakeEnv(slip_chance=0.25)
state_values, _ = value_iteration(mdp)

total_rewards = []
for game_i in range(1000):
    s = mdp.reset()
    rewards = []
    for t in range(100):
        s, r, done, _ = mdp.step(
            get_optimal_action(mdp, state_values, s, gamma))
        rewards.append(r)
        if done:
            break
    total_rewards.append(np.sum(rewards))

print("average reward: ", np.mean(total_rewards))
assert(0.6 <= np.mean(total_rewards) <= 0.7)
print("Well done!")


# In[28]:


# Measure agent's average reward
mdp = FrozenLakeEnv(slip_chance=0.2, map_name='8x8')
state_values, _ = value_iteration(mdp)

total_rewards = []
for game_i in range(1000):
    s = mdp.reset()
    rewards = []
    for t in range(100):
        s, r, done, _ = mdp.step(
            get_optimal_action(mdp, state_values, s, gamma))
        rewards.append(r)
        if done:
            break
    total_rewards.append(np.sum(rewards))

print("average reward: ", np.mean(total_rewards))
assert(0.6 <= np.mean(total_rewards) <= 0.8)
print("Well done!")


# # HW Part 1: Value iteration convergence
# 
# ### Find an MDP for which value iteration takes long to converge  (0.5 pts)
# 
# When we ran value iteration on the small frozen lake problem, the last iteration where an action changed was iteration 6--i.e., value iteration computed the optimal policy at iteration 6. Are there any guarantees regarding how many iterations it'll take value iteration to compute the optimal policy? There are no such guarantees without additional assumptions--we can construct the MDP in such a way that the greedy policy will change after arbitrarily many iterations.
# 
# Your task: define an MDP with at most 3 states and 2 actions, such that when you run value iteration, the optimal action changes at iteration >= 50. Use discount=0.95. (However, note that the discount doesn't matter here--you can construct an appropriate MDP with any discount.)
# 
# Note: value function must change at least once after iteration >=50, not necessarily change on every iteration till >=50.

# In[29]:


# transition_probs = {
#     's0': {'a0': {'s1': 0.95, 's2': 0.05}},
#     's1': {'a1': {'s0': 0.95, 's2': 0.05}}
# }
# rewards = {
#     's0': {'a0': {'s1': 1, 's2': 0.5}},
#     's1': {'a1': {'s0': 1, 's2': 0.5}}
# }

transition_probs = {
    's0': {
        'a0': {'s0': 0.999, 's1': 0.001}
    },
    's1': {
        'a0': {'s1': 1},
        'a1': {'s1': 0.98435, 's2': 0.01565}
    },
    's2': {
        'a0': {'s2': 1}
    }
}
rewards = {
    's0': {'a0': {'s0': -1, 's1': +1e3}},
    's1': {'a0': {'s1': +1e3}, 'a1': {'s1': -5e2, 's2': +1e4}},
    's2': {'a0': {'s2': +1e4}}
}

gamma = 0.95

from mdp import MDP
from numpy import random
mdp_bonus = MDP(transition_probs, rewards, initial_state='s0')
# Feel free to change the initial_state


# In[30]:


if has_graphviz:
    display(plot_graph(mdp_bonus, graph_size="50,50"))


# In[31]:


state_values = {s: 0 for s in mdp_bonus.get_all_states()}
policy = np.array([get_optimal_action(mdp_bonus, state_values, state, gamma)
                   for state in sorted(mdp_bonus.get_all_states())])

for i in range(100):
    print("after iteration %i" % i)
    state_values, _ = value_iteration(mdp_bonus, state_values, num_iter=1)

    print(policy)
    new_policy = np.array([get_optimal_action(mdp_bonus, state_values, state, gamma)
                           for state in sorted(mdp_bonus.get_all_states())])

    n_changes = (policy != new_policy).sum()
    print("N actions changed = %i \n" % n_changes)
    policy = new_policy

# please ignore iter 0 at each step


# ### Value iteration convervence proof (0.5 pts)
# **Note:** Assume that $\mathcal{S}, \mathcal{A}$ are finite.
# 
# Update of value function in value iteration can be rewritten in a form of Bellman operator:
# 
# $$(TV)(s) = \max_{a \in \mathcal{A}}\mathbb{E}\left[ r_{t+1} + \gamma V(s_{t+1}) | s_t = s, a_t = a\right]$$
# 
# Value iteration algorithm with Bellman operator:
# 
# ---
# &nbsp;&nbsp; Initialize $V_0$
# 
# &nbsp;&nbsp; **for** $k = 0,1,2,...$ **do**
# 
# &nbsp;&nbsp;&nbsp;&nbsp; $V_{k+1} \leftarrow TV_k$
# 
# &nbsp;&nbsp;**end for**
# 
# ---
# 
# In [lecture](https://docs.google.com/presentation/d/1lz2oIUTvd2MHWKEQSH8hquS66oe4MZ_eRvVViZs2uuE/edit#slide=id.g4fd6bae29e_2_4) we established contraction property of bellman operator:
# 
# $$
# ||TV - TU||_{\infty} \le \gamma ||V - U||_{\infty}
# $$
# 
# For all $V, U$
# 
# Using contraction property of Bellman operator, Banach fixed-point theorem and Bellman equations prove that value function converges to $V^*$ in value iterateion$
# 
# **Proof.**
# 
# Consider $||V_{k+1} - V^*||_{\infty}$. As $V^*$ is a fixed point of operator $T$, $TV^* = V^*$.
# $$\begin{align}
# ||V_{k+1} - V^*|| = ||TV_k - TV^*|| \le&\ \gamma ||V_k - V^*|| = \gamma^{k+1}||V_0-V^*|| \implies \\
# 0 \le \lim\limits_{k \to +\infty} ||V_{k+1} - V^*|| \le&\ \lim\limits_{k \to +\infty} \gamma^{k+1}||V_0-V^*|| = [0 < \gamma < 1] = 0 \implies \\
# \lim\limits_{k \to +\infty} ||V_{k+1} - V^*|| =&\ 0
# \end{align}$$
# 
# **Q.E.D.**

# ### Bonus. Asynchronious value iteration (2 pts)
# 
# Consider the following algorithm:
# 
# ---
# 
# Initialize $V_0$
# 
# **for** $k = 0,1,2,...$ **do**
# 
# &nbsp;&nbsp;&nbsp;&nbsp; Select some state $s_k \in \mathcal{S}$    
# 
# &nbsp;&nbsp;&nbsp;&nbsp; $V(s_k) := (TV)(s_k)$
# 
# **end for**
# 
# ---
# 
# 
# Note that unlike common value iteration, here we update only a single state at a time.
# 
# **Homework.** Prove the following proposition:
# 
# If for all $s \in \mathcal{S}$, $s$ appears in the sequence $(s_0, s_1, ...)$ infinitely often, then $V$ converges to $V*$
# 
# **Proof.**
# 
# Consider vector $\Delta^{t+1} = |V_{t+1} - V^*| = [|V_{t+1}(s_i) - V^*(s_i)|]_{i = \overline{1,|\mathcal{S}|}}$.
# 
# Let's split sequence $S$ of $s_i$-s into all subsequences with same index $S_i$. It's guaranteed that all of them are infinite, that is $|S_i| \to +\infty$. Then, we take a look at $||V_{t+1}-V^*||_{\infty}$. It's obvious that $||V_{t+1}-V^*||_{\infty} \le ||V_{t+1}-V^*||_1 = \sum\limits_{i=1}^{|\mathcal{S}|} \Delta_i^{t+1}$.
# 
# Taking a limit on both sides:
# $$0 \le \lim\limits_{t \to +\infty} ||V_{t+1}-V^*||_{\infty} \le \lim\limits_{t \to +\infty}\sum\limits_{i=1}^{|\mathcal{S}|} \Delta_i^{t+1} = \sum\limits_{i=1}^{|\mathcal{S}|}\lim\limits_{t \to +\infty} \Delta_i^{t+1}$$
# 
# Last equality is valid for finite $|\mathcal{S}|$ iff $\lim\limits_{t \to +\infty} \Delta_i^{t+1}$ exists for all $i$. But
# $$\lim\limits_{t \to +\infty} \Delta_i^{t+1} = \lim\limits_{t \to +\infty} |V_{t+1}(s_i) - V^*(s_i)|$$
# 
# Introducing $S_i(t) = \sum\limits_{j=1}^{t} \mathcal{I}(S[j] = s_i)$ where $\mathcal{I}$ is the indicator function. Then, in terms of $S_i(t)$ $s_k$ appearing infinitely many times in $S$ means that $S_i(t) \to +\infty$ as $t \to +\infty$. And limit above can be expressed in terms of $S_i(t)$ as well!
# 
# $$\lim\limits_{t \to +\infty} |V_{t+1}(s_i) - V^*(s_i)| \le \lim\limits_{t \to +\infty} \gamma^{S_i(t)} |V_0(s_i) - V^*(s_i)| = 0$$
# 
# Substituting this gives us:
# $$0 \le \lim\limits_{t \to +\infty} ||V_{t+1}-V^*||_{\infty} \le \sum\limits_{i=1}^{|\mathcal{S}|}\lim\limits_{t \to +\infty} \gamma^{S_i(t)} |V_0(s_i) - V^*(s_i)| = [0 < \gamma < 1] = 0$$
# 
# Then $\lim\limits_{t \to +\infty} ||V_{t+1}-V^*||_{\infty} = 0$. **Q.E.D.**

# # HW Part 2: Policy iteration
# 
# ## Policy iteration implementateion (2 pts)
# 
# Let's implement exact policy iteration (PI), which has the following pseudocode:
# 
# ---
# Initialize $\pi_0$   `// random or fixed action`
# 
# For $n=0, 1, 2, \dots$
# - Compute the state-value function $V^{\pi_{n}}$
# - Using $V^{\pi_{n}}$, compute the state-action-value function $Q^{\pi_{n}}$
# - Compute new policy $\pi_{n+1}(s) = \operatorname*{argmax}_a Q^{\pi_{n}}(s,a)$
# ---
# 
# Unlike VI, policy iteration has to maintain a policy - chosen actions from all states - and estimate $V^{\pi_{n}}$ based on this policy. It only changes policy once values converged.
# 
# 
# Below are a few helpers that you may or may not use in your implementation.

# In[32]:


transition_probs = {
    's0': {
        'a0': {'s0': 0.5, 's2': 0.5},
        'a1': {'s2': 1}
    },
    's1': {
        'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
        'a1': {'s1': 0.95, 's2': 0.05}
    },
    's2': {
        'a0': {'s0': 0.4, 's1': 0.6},
        'a1': {'s0': 0.3, 's1': 0.3, 's2': 0.4}
    }
}
rewards = {
    's1': {'a0': {'s0': +5}},
    's2': {'a1': {'s0': -1}}
}

from mdp import MDP
mdp = MDP(transition_probs, rewards, initial_state='s0')


# Let's write a function called `compute_vpi` that computes the state-value function $V^{\pi}$ for an arbitrary policy $\pi$.
# 
# Unlike VI, this time you must find the exact solution, not just a single iteration.
# 
# Recall that $V^{\pi}$ satisfies the following linear equation:
# $$V^{\pi}(s) = \sum_{s'} P(s,\pi(s),s')[ R(s,\pi(s),s') + \gamma V^{\pi}(s')]$$
# 
# You'll have to solve a linear system in your code. (Find an exact solution, e.g., with `np.linalg.solve`.)

# In[33]:


def compute_vpi(mdp, policy, gamma):
    """
    Computes V^pi(s) FOR ALL STATES under given policy.
    :param policy: a dict of currently chosen actions {s : a}
    :returns: a dict {state : V^pi(state) for all states}
    """
    states = mdp.get_all_states()
    state_to_idx = dict(zip(states, np.arange(len(states))))
    system = np.diag(np.ones(len(states)))
    rhs = np.zeros(len(states))
    for (i, state) in enumerate(states):
        policy_action = policy[state]
        if policy_action:  # if state has no actions, policy_action will be an empty string
            for next_state in mdp.get_next_states(state, policy_action):
                p = mdp.get_transition_prob(state, policy_action, next_state)
                r = mdp.get_reward(state, policy_action, next_state)
                rhs[i] += p * r
                system[i][state_to_idx[next_state]] -= gamma * p
            
    vpi = np.linalg.solve(system, rhs).tolist()
    return dict(zip(states, vpi))


# In[34]:


test_policy = {s: np.random.choice(
    mdp.get_possible_actions(s)) for s in mdp.get_all_states()}
new_vpi = compute_vpi(mdp, test_policy, gamma)

print(new_vpi)

assert type(
    new_vpi) is dict, "compute_vpi must return a dict {state : V^pi(state) for all states}"


# Once we've got new state values, it's time to update our policy.

# In[35]:


def compute_new_policy(mdp, vpi, gamma):
    """
    Computes new policy as argmax of state values
    :param vpi: a dict {state : V^pi(state) for all states}
    :returns: a dict {state : optimal action for all states}
    """
    states = mdp.get_all_states()
    q_pi = lambda state, action: np.sum([mdp.get_transition_prob(state, action, next_state)
                                         * (mdp.get_reward(state, action, next_state) + gamma * vpi[next_state])
                                         for next_state in mdp.get_next_states(state, action)])
    policy = {}
    for state in states:
        actions = mdp.get_possible_actions(state)
        if actions:
            policy[state] = actions[np.argmax([q_pi(state, action) for action in actions])]
        else:
            policy[state] = ''
            
    return policy


# In[36]:


new_policy = compute_new_policy(mdp, new_vpi, gamma)

print(new_policy)

assert type(
    new_policy) is dict, "compute_new_policy must return a dict {state : optimal action for all states}"


# __Main loop__

# In[37]:


def policy_iteration(mdp, policy=None, gamma=0.9, num_iter=1000, min_difference=1e-5, iter_callback=None):
    """ 
    Run the policy iteration loop for num_iter iterations or till difference between V(s) is below min_difference.
    If policy is not given, initialize it at random.
    """
    def get_random_policy():
        policy = {}
        states = mdp.get_all_states()
        for state in states:
            actions = mdp.get_possible_actions(state)
            if actions:
                policy[state] = actions[np.random.randint(0, len(actions))]
            else:
                policy[state] = ''
        return policy
    
    def diff(vpi, new_vpi):
        vpi_np = np.array(list(vpi.values()))
        new_vpi_np = np.array(list(new_vpi.values()))
        return np.linalg.norm(vpi_np - new_vpi_np, ord=np.inf)
    
    if policy is None:
        policy = get_random_policy()
    state_values = compute_vpi(mdp, policy, gamma)
    it = 0
    for i in range(num_iter):
        new_policy = compute_new_policy(mdp, state_values, gamma)
        new_state_values = compute_vpi(mdp, new_policy, gamma)
        d = diff(state_values, new_state_values)
        policy_is_optimal = d < min_difference
        state_values = new_state_values
        policy = new_policy
        it += 1
        if iter_callback:
            iter_callback(it=it, diff=d, policy=policy, vpi=state_values)
        if policy_is_optimal:
            break

    return state_values, policy, it


# __Your PI Results__

# In[38]:


gamma = 0.95


# In[39]:


def play_vi_game(mdp, gamma=0.9, verbose=True):
    if verbose:
        print('Value iteration log')
    state_values, total_iters = value_iteration(mdp, gamma=gamma, verbose=verbose)

    total_rewards = []
    for game_i in range(1000):
        s = mdp.reset()
        rewards = []
        for t in range(100):
            s, r, done, _ = mdp.step(
                get_optimal_action(mdp, state_values, s, gamma))
            rewards.append(r)
            if done:
                break
        total_rewards.append(np.sum(rewards))
    return np.mean(total_rewards), total_iters

def play_pi_game(mdp, gamma=0.9, verbose=True):
    pi_callback = lambda it, diff, policy, vpi: 1
    if verbose:
        print('Policy iteration log')
        pi_callback = lambda it, diff, policy, vpi:            print(f'iter {it:4d}   |   diff: {diff:.5f}   |   V(start): {vpi[(0,0)]:.3f}')
    vpi, policy, total_iters = policy_iteration(mdp, gamma=gamma, iter_callback=pi_callback)
    
    total_rewards = []
    for game_i in range(1000):||
        s = mdp.reset()
        rewards = []
        for t in range(100):
            s, r, done, _ = mdp.step(policy[s])
            rewards.append(r)
            if done:
                break
        total_rewards.append(np.sum(rewards))
    return np.mean(total_rewards), total_iters


# In[40]:


from tqdm import tqdm

def print_mdp_comparison(mdp, mdp_name, gamma=0.9, verbose=False):
    print('###############################################')
    print(f'Playing MDP {mdp_name}')
    print('###############################################')
    
    vi_history = []
    vi_iters = []
    pi_history = []
    pi_iters = []
    
    for i in tqdm(range(25)):
        vi_r, vi_it = play_vi_game(mdp, gamma, verbose)
        pi_r, pi_it = play_pi_game(mdp, gamma, verbose)
        vi_history.append(vi_r)
        vi_iters.append(vi_it)
        pi_history.append(pi_r)
        pi_iters.append(pi_it)

    print('Average VI reward stats')
    print(f'Mean: {np.mean(vi_history)}, std: {np.std(vi_history)}')
    print(f'Mean iter: {np.mean(vi_iters)}')
    print('Average PI reward stats')
    print(f'Mean: {np.mean(pi_history)}, std: {np.std(pi_history)}')
    print(f'Mean iter: {np.mean(pi_iters)}')
    
MDPs = [mdp_bonus,
        FrozenLakeEnv(slip_chance=0.25),
        FrozenLakeEnv(slip_chance=0.2, map_name='8x8')]
MDP_names = ['Bonus 1',
             'Small lake',
             'Big lake'
             ]

for (mdp, mdp_name) in zip(MDPs, MDP_names):
    print_mdp_comparison(mdp, mdp_name, gamma)


# **Tricky MDP:**
# 
# For tricky MDP policy iteration achieves better values in only a few iterations. Value iteration has difficulties converging because the probability of getting into very profitable state is rather low and discounting takes a lot of time to have an effect.
# 
# **Frozen lake envs:**
# 
# Stats above suggest that policy iteration converges to something slightly worse than optimal value iteration policy. Even though we calculate average reward estimate based on 25 runs of both algorithms, it still may differ quite significantly between runs. If we actually run some statistical test, I guess it'll tell us that they actually have the same average reward. This might be the case because environments are quite straight-forward and both algorithms converge to globally optimal policy. Apart from that, PI seems to be more stable than VI overall (its std is lower than that of VI).

# ## Policy iteration convergence (3 pts)
# 
# **Note:** Assume that $\mathcal{S}, \mathcal{A}$ are finite.
# 
# We can define another Bellman operator:
# 
# $$(T_{\pi}V)(s) = \mathbb{E}_{r, s'|s, a = \pi(s)}\left[r + \gamma V(s')\right]$$
# 
# And rewrite policy iteration algorithm in operator form:
# 
# 
# ---
# 
# Initialize $\pi_0$
# 
# **for** $k = 0,1,2,...$ **do**
# 
# &nbsp;&nbsp;&nbsp;&nbsp; Solve $V_k = T_{\pi_k}V_k$   
# 
# &nbsp;&nbsp;&nbsp;&nbsp; Select $\pi_{k+1}$ s.t. $T_{\pi_{k+1}}V_k = TV_k$ 
# 
# **end for**
# 
# ---
# 
# To prove convergence of the algorithm we need to prove two properties: contraction an monotonicity.
# 
# #### Monotonicity (0.5 pts)
# 
# For all $V, U$ if $V(s) \le U(s)$   $\forall s \in \mathcal{S}$ then $(T_\pi V)(s) \le (T_\pi U)(s)$   $\forall s \in  \mathcal{S}$
# 
# **Proof.**
# 
# $$\begin{align}
# V(s) \le U(s) \implies V(s) - U(s) \le&\ 0 \ \forall s \in \mathcal{S} \\
# T_\pi V(s) - T_\pi U(s) =&\ \mathbb{E}_{r, s'|s, a = \pi(s)}\left[r + \gamma V(s')\right] - \mathbb{E}_{r, s'|s, a = \pi(s)}\left[r + \gamma U(s')\right] \\
# =&\ \mathbb{E}_{r, s'|s, a = \pi(s)}\left[\gamma(V(s') - U(s'))\right] \\
# =&\ \sum\limits_{s'} \underbrace{P(r, s' | s, a = \pi(s))\gamma}_{\ge 0}\underbrace{(V(s') - U(s'))}_{\le 0} \le 0
# \end{align}
# $$
# 
# **Q.E.D.**
# 
# #### Contraction (1 pts)
# 
# $$
# ||T_\pi V - T_\pi U||_{\infty} \le \gamma ||V - U||_{\infty}
# $$
# 
# for all $V, U$.
# 
# **Proof.**
# 
# $$\begin{align}
# ||T_\pi V - T_\pi U||_{\infty} =&\ \max\limits_s \left|\sum\limits_{s'} P(r, s' | s, a = \pi(s))\gamma(V(s') - U(s'))\right| \le \\
# \le&\ \max\limits_s \sum\limits_{s'} \gamma\left|P(r, s' | s, a = \pi(s))(V(s') - U(s'))\right| \le \\
# \le&\ \max\limits_s \sum\limits_{s'} \gamma\max\limits_{s'}|V(s') - U(s')|P(r, s' | s, a = \pi(s)) = \\
# =&\ \gamma||V - U||_{\infty}\max\limits_s \underbrace{\sum\limits_{s'} P(r, s' | s, a = \pi(s))}_{=1} = \\
# =&\ \gamma||V - U||_{\infty}
# \end{align}$$
# 
# **Q.E.D.**
# 
# #### Convergence (1.5 pts)
# 
# Prove that there exists iteration $k_0$ such that $\pi_k = \pi^*$ for all $k \ge k_0$
# 
# **Proof.**
# 
# It's easy to see that $V_k \le V_{k+1} \le V^*$. Using monotonicity, we conclude that $T_{\pi_{k+1}} V_k \le T_{\pi_{k+1}}V_{k+1} = V_{k+1}$. Now, consider
# 
# $$|V_{k+1}-V^*| = V^* - V_{k+1} \le [-V_{k+1} \le -T_{\pi_{k+1}}V_k] \le V^* - T_{\pi_{k+1}}V_k = T_{\pi_{k+1}}V^* - T_{\pi_{k+1}}V_k = |T_{\pi_{k+1}}V^* - T_{\pi_{k+1}}V_k|$$
# 
# Then
# $$||V_{k+1}-V^*||_{\infty} \le ||T_{\pi_{k+1}}V^* - T_{\pi_{k+1}}V_k||_{\infty} \le \gamma||V_k-V^*||_{\infty} \le \gamma^k||V_0-V^*||_{\infty}$$
# 
# Taking limit on both sides, we prove that $V_{k+1}$ converges to $V^*$ as $k$ approaches infinity. Using $\epsilon-\delta$ formalism, this means that
# $$\forall \epsilon > 0 \ \exists k_0(\epsilon): \forall k \ge k_0(\epsilon) \ ||V_k-V^*||_{\infty} \le \epsilon$$
# 
# Because $\mathcal{S}$, $\mathcal{A}$ are finite, there exists a finite sequence $E = ||V_{\pi_i} - V^*||_{\infty},  i=\overline{1,|\mathcal{S}||\mathcal{A}|}$. Denote $\epsilon_0 = \min\{E_i | E_i \ne 0\}$ (this minimum always exists because sequence is finite) and take $\epsilon = \frac{\epsilon_0}{2}$. The only possible value for $||V_k-V^*||_{\infty}$ is then $0$ (optimal policy always exists and $||V_{\pi^*} - V^*||_{\infty} = 0$ because $V_{\pi^*} = V^*$). And due to definition of the limit, there has to exist some $k_0$ such that for all $k \ge k_0$ $||V_k-V^*||_{\infty} = 0$. Then $\pi_k = \pi^*$. **Q.E.D.**

# In[ ]:




