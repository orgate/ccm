
# coding: utf-8

# # Homework - Reinforcement Learning - Part A (40/100 points)

# by *Todd Gureckis* and *Brenden Lake*  
# Computational Cognitive Modeling  
# NYU class webpage: https://brendenlake.github.io/CCM-site/  
# email to course instructors: instructors-ccm-spring2019@nyuccl.org

# <div class="alert alert-danger" role="alert">
#   This homework is due before midnight on March 27, 2019. 
# </div>

# ---

# ## Reinforcement Learning
# 
# Reinforcement learning (RL) is a topic in machine learning and psychology/neuroscience which considers how an emboddied agent should learn to make decisions in an environment in order to maximize reward.  You could definitely do worse things in life than to read the classic text on RL by Sutton and Barto:
# 
# * Sutton, R.S. and Barto, A.G. (2017) Reinforcement Learning: An Introduction.  MIT Press, Cambridge, MA. [<a href="http://incompleteideas.net/book/the-book-2nd.html">available online for free!</a>]
# 
# 
# The standard definition of the RL problem can be summarized with this figure:
# 
# <img src='images/rlsutton.png' width='300'>
# 
# The agent at each time point chooses an action which influences the state of the world according to the rules of the environment (e.g., spatial layout of a building or the very nature of physics).  This results in a new state ($s_{t+1}$) and possibly a reward ($r_{t+1}$).  The agent then receives the new state and the reward signal and updates in order to choose the next action.  The goal of the agent is to maximize the reward received over the long run.  In effect this approach treats life like an optimal control problem (one where the goal is to determine the best actions to take for each possible state).
# 
# The simplicity and power of this framework has made it very influential in recent years in psychology and computer science.  Recently more advanced techniques for solving RL problems have been scaled to show impressive performance on complex, real-world tasks.  For example, so called "deep RL" system which combine elements of deep convolutional nets and reinforcement learning algorithms can learn to play classic video games at near-human performance, simply by aiming to earn points in the game:
# 
# <img src='images/deepql.png' width='350'>
# 
# * Mnih, V. et al. (2015) Human-level control through deep reinforcement learning.  *Nature*, 518, 529. [<a href="https://www.nature.com/articles/nature14236">pdf</a>]
# 
# 
# In this homework we will explore some of the underlying principles which support these advances.
# 
# The homework is divided into two parts:
# 1. The first part (this notebook) explores different solution methods to the problem of behaving optimally in a *known* environment.  
# 2. The [second part](Homework-RL-B.ipynb) explores some basic issues in learning to choose effectively in an *unknown* enviroment.  
# 
# **References:**
# * Sutton, R.S. and Barto, A.G. (2017) Reinforcement Learning: An Introduction.  MIT Press, Cambridge, MA.
# * Gureckis, T.M. and Love, B.C. (2015) Reinforcement learning: A computational perspective. Oxford Handbook of Computational and Mathematical Psychology, Edited by Busemeyer, J.R., Townsend, J., Zheng, W., and Eidels, A., Oxford University Press, New York, NY.
# * Daw, N.S. (2013) “Advanced Reinforcement Learning” Chapter in Neuroeconomics: Decision making and the brain, 2nd edition
# * Niv, Y. and Schoenbaum, G. (2008) “Dialogues on prediction errors” *Trends in Cognitive Science*, 12(7), 265-72.
# * Nathaniel D. Daw, John P. O’Doherty, Peter Dayan, Ben Seymour & Raymond J. Dolan (2006). Cortical substrates for exploratory decisions in humans. *Nature*, 441, 876-879.

# # Learning and deciding in a known world

# Reinforcement learning is a collection of methods and techniques for learning to make good or optimal sequential decisions.  As described in the lecture, the basic definition of the RL problem (see above) is quite general and therefore there is more than one way to solve an RL problem (and even multiple ways to define what the RL problem is).
# 
# In this homework we are going to take one simple RL problem: navigation in a grid-world maze, and explore two different ways of solving this decision problem.
# 
# - The first method is going to be policy-iteration or dynamic programming.  
# - The second method is going to be monte-carlo simulation.
# 
# By seeing the same problem solved multiple ways, it helps to reinforce the differences between these different approaches and the features of the algorithms that are interesting from the perspective of human decision making.

# ## The problem defintion
# 
# The problem we will consider is a grid world task.  The grid is a collection of rooms.  Within each room there are four possible actions (move up, down, left, right).  There are also walls in the maze that the agent cannot move through (indicated in grey below).  There are two special states, $S$ which is the start state, and $G$ which is the goal state.  The agent will start at location $S$ and aims to arrive at $G$.  When the agents moves into the $G$ state they earn a reward of +10.  If they walk off the edge of the maze, they receive a -1 reward and are returned to the $S$ state.  $G$ is an absorbing state in the sense that you can think of the agent as never leaving that state once they arrive there.
# 
# The specific gridworld we will consider looks like this:
# 
# <img src='images/gridworld.png' width='500'>
# 
# The goal of the agent to determine the optimal sequential decision making policy to arrive at state $G$.
# 
# To help you with this task we provide a simple `GridWorld()` class that makes it easy to specify parts of the gridworld environment and provides access to some of the variables you will need in constructing your solutions to the homework.  In order to run the gridworld task you need to first execute the following cell:

# <div class="alert alert-warning" role="alert">
#   <strong>Warning!</strong> Before running other cells in this notebook you must first successfully execute the following cell which includes some libraries.
# </div>

# In[1]:


# import the gridworld library
import numpy as np
import random
import math
import statistics
from copy import deepcopy
from IPython.display import display, Markdown, Latex, HTML
from gridworld import GridWorld, random_policy

# The following cell sets up the grid world defined above including the spatial layout and then a python dictionary called `rewards` that determines which transitions between states result in a reward of a given magnitude.

# In[2]:


gridworld = [
       [ 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'x', 'g'],
       [ 'o', 'o', 'x', 'o', 'o', 'o', 'o', 'x', 'o'],
       [ 'o', 'o', 'x', 'o', 'o', 'o', 'o', 'x', 'o'],
       [ 'o', 'o', 'x', 'o', 'o', 'o', 'o', 'o', 'o'],
       [ 'o', 'o', 'o', 'o', 'o', 'x', 'o', 'o', 'o'],
       [ 's', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    ] # the problem described above, 'x' is a wall, 's' is start, 'g' is goal, and 'o' is a normal room

mygrid = GridWorld(gridworld)
#mygrid.raw_print()  # print out the grid world
#mygrid.index_print() # print out the indicies of each state
#mygrid.coord_print() # print out the coordinates of each state (helpful in your code)

# define the rewards as a hash table
rewards={}

# mygrid.transitions contains all the pairwise state-state transitions allowed in the grid
# for each state transition intialize the reward to zero
for start_state in mygrid.transitions:
    for action in mygrid.transitions[start_state].keys():
        next_state = mygrid.transitions[start_state][action]
        rewards[str([start_state, action, next_state])] = 0.0

# now set the reward for moving up into state 8 (the goal state) to +10
rewards[str([17, 'up', 8])] = 10

# now set the penalty for walking off the edge of the grid and returning to state 45 (the start state)
for i in [0,1,2,3,4,5,6,7]:
    rewards[str([i, 'up', 45])] = -1
for i in [0,9,18,27,36,45]:
    rewards[str([i, 'left', 45])] = -1
for i in [45,46,47,48,49,50,51,52,53]:
    rewards[str([i, 'down', 45])] = -1
for i in [8,17,26,35,44,53]:
    rewards[str([i, 'right', 45])] = -1


# <div class="alert alert-info" role="alert">
# 
# Notice that the above printouts show the grid but also an array of the indexes and coordinated of each location on the grid.  You will need these to help you analyze your solution to the homework so it can be frequently helpful to refer back to these outputs.
# </div>

# In order to solve this problem using dynamic programming the agent needs to maintain two key representations.  One is the value of each state under the current policy, $V^\pi$, and the other is the policy $\pi(s,a)$.  The following cell initializes a new value table and a new random policy and uses functions provided in `GridWorld` to print these out in the notebook in a friendly way.

# In[3]:


value_table=np.zeros((mygrid.nrows,mygrid.ncols))
display(Markdown("**Current value table for each state**"))
#mygrid.pretty_print_table(value_table)

policy_table = [[random_policy() for i in range(mygrid.ncols)] for j in range(mygrid.nrows)]
display(Markdown("**Current (randomized) policy**"))
#mygrid.pretty_print_policy_table(policy_table)


# <div class="alert alert-info" role="alert">
# 
# Note how the current policy is random with the different arrows within each state pointing in different, sometimes opposing directions.  Your goal is to solve for the best way to orient those arrows.</div>

# ## Dynamic Programming via Policy Iteration

# <div class="alert alert-success" role="alert">
# <h3> Problem 1 (15 points) </h3><br>
#   Remember that the Bellman equation that recursively relates the value of any state to any other state is like this:
# 
# $\large V^{\pi}(s) = \sum_{a} \pi(s,a) \sum_{s'} \mathcal{P}^{a}_{ss'} [ \mathcal{R}^{a}_{ss'}+\gamma V^{\pi}(s')]$
# 
# Your job in this first exercise is to set up a dynamic programming solution to the provided gridworld problem.  Your should implement two steps.  The first is policy evaluation which means given a policy (in `policy_table`) update the `value_table` to be consistent with that policy.  Your algorithm should do this by visiting each state in a random order and updating it's value in the `value_table` (this is known as asychronous update since you are changing the values in-place).  
# 
# The next step is policy improvement where you change the policy to maximize expected long-run reward in each state by adjusting which actions you should take (this means changing the values in `policy_table`).  We will only consider deterministic policies in this case.  Thus your algorithm should always choose one particular action to take in each state even if two actions are similarly valued.
# 
# The algorithm you write should iterate sequentially between policy evaluate and (greedy) policy improvement for at least 2000 iterations.
# 
# <img src="images/sutton-iteration.png">
# 
# (Figure from <a href="http://www.incompleteideas.net/book/the-book-2nd.html">Sutton and Barto</a> text)
# 
# To gain some intuition about how preferences for the future impact the resulting policies, run your algorithm twice, once with $\gamma$ set to zero (as in lecture) and another with $\gamma$ set to 0.9 and output the resulting policy and value table using `mygrid.pretty_print_policy_table()` and `mygrid.pretty_print_table()`.
# </div>
# 
# 
# 
# 

# <div class="alert alert-info" role="alert">
#   <strong>Info</strong> <br> 
#   The `GridWorld` class provides some helpful functions that you will need in your solution.  The following code describes these features
# </div>

# <div class="alert alert-info" role="alert">
# 
# Only some states are "valid" i.e., are not walls.  `mygrid.valid_states` is a python dictionary containing those states.  The keys of this dictionary are id numbers for each state (see the output of `mygrid.index_print()`) and the values are coordinates (see the output of `mygrid.coord_print()`).  Your algorithm will want to iterate over this list to update the value of each "valid" state.
# </div>

# In[4]:


mygrid.valid_states  # output the indexes and coordinates of the valid states


# <div class="alert alert-info" role="alert">
# As the previous command makes clear, there are two ways of referencing a state: by its id number or by its coordinates.  Two functions let you swap between those:
# 
# - `mygrid.index_to_coord(index)` converts a index (e.g., 1-100) to a coordinate (i,j)
# - `mygrid.coord_to_index(coord)` takes a tuples representing the coordinate (i,j) and return the index (e.g., 1-100)
# 
# Both the value table (`value_table`) and policy table (`policy_table`) are indexed using coordinates.
# </div>

# <div class="alert alert-info" role="alert">
#     
# A key variable for your algorithm is $\mathcal{P}^{a}_{ss'}$ which is the probability of reaching state $s'$ when in state $s$ and taking action $a$.  We assume that the world is deterministic here so these probabilities are always 1.0.  However, some states do not lead to immediately adjacent cells but instead return to the start state (e.g., walking off the edge of the grid).  `mygrid.transitions` contains a nested hash table that contains this information for your gridworld.  For example consider state 2:
# </div>
# 

# In[5]:


state = 3
mygrid.transitions[state]


# <div class="alert alert-info" role="alert">
# 
# The output of the above command is a python dictionary showing what next state you will arrive at if you chose the given actions.  Thus `mygrid.transitions[2]['down']` would return state id 2 because you will hit the wall and thus not change state.  Whereas `mygrid.transitions[2]['left']` will move to state 1.  The `mygrid.transitions` dictionary thus provides all the information necessary to represent $P^a_{ss'}$.  The world is deterministic so taking an action in a given state will always move the agent to the next corresponding state with probability 1.
# </div>

# <div class="alert alert-info" role="alert">
#     
# The next variable you will need is the reward function.  Rewards are delivered anytime the agent makes a transition from one state to another using a particular action.  Thus this variable is written $\mathcal{R}^{a}_{ss'}$ in the equation above.  You can access this programmatically using the python dictionary 
# `rewards` which we ourselves defined above.  The `rewards` dictionary defines the reward for taking a particular action in a particular state and then arriving at a new state $s'$.  To look up the reward for a particular $<s,a,s'>$ triplet you create a list with these variables in index format, convert it to a string, and look it up in the dictionary.  For example the reward for being in state 17, choosing up, and then arriving in state 8 is:
# </div>
# 
# 

# In[6]:


state = 17
next_state = 8
action = "up"
rewards[str([state, action, next_state])]


# <div class="alert alert-info" role="alert">
# This should be the required ingredients to solve both the policy evaluation and policy improvement functions that you will need to write.  If you need further information you can read the `GridWorld` class directly in <a href="gridworld.py">gridworld.py</a>. 
# </div>
# 
# 

# ### Your solution:

# Implement the two major steps of your algorithm as the following two functions.  Then write code that iterates between them for the specified number of steps and inspect the final solution.  **Some scaffolding code has been provided for you so all you have to implement is the sections noted in the comments**

# In[7]:


def policy_evaluate(mygrid, value_table, policy_table, GAMMA):
    valid_states = list(mygrid.valid_states.keys())
    random.shuffle(valid_states)

    for state in valid_states:
        sx,sy = mygrid.index_to_coord(state)
        new_value = 0.0
        for action in mygrid.transitions[state].keys():
            #pass
            next_state = mygrid.transitions[state][action]  # Added by Alfred
            nsx,nsy = mygrid.index_to_coord(next_state)  # Added by Alfred
            new_value = new_value + policy_table[sx][sy][action]*(rewards[str([state, action, next_state])] + GAMMA*value_table[nsx][nsy])  # Added by Alfred
            # PART 1: HOMEWORK: compute what the new value of the give state should be
            # here!!!  This is your homework problem***************************************
            # delete the pass
        value_table[sx][sy] = new_value

# this is a helper function that will take a set of q-values and convert them into a greedy decision strategy        
def be_greedy(q_values):
    if len(q_values)==0:
        return {}
    
    keys = list(q_values.keys())
    vals = [q_values[i] for i in keys]    
    maxqs = [i for i,x in enumerate(vals) if x==max(vals)]
    if len(maxqs)>1:
        pos = random.choice(maxqs)
    else:
        pos = maxqs[0]
    policy = deepcopy(q_values)
    for i in policy.keys():
        policy[i]=0.0
    policy[keys[pos]]=1.0
    return policy

def policy_improve(mygrid, value_table, policy_table, GAMMA):
    # for each state
    valid_states = list(mygrid.valid_states.keys())

    for state in valid_states:
        # compute the Q-values for each action
        q_values = {}
        for action in mygrid.transitions[state].keys():
            # update the q-values here for each action here
            # and store them in a variable called qval
            # PROBLEM 3: HOMEWORK: Update the returns here using the first-visit algorithm
            #*********************
            next_state = mygrid.transitions[state][action]  # Added by Alfred
            nsx,nsy = mygrid.index_to_coord(next_state)  # Added by Alfred
            qval = rewards[str([state, action, next_state])] + GAMMA*value_table[nsx][nsy] # Added by Alfred
            q_values[action]=qval
        newpol = be_greedy(q_values) # take this dictionary and convert into a greedy policy
        # then update the policy table printing to allow more complex policies
        sx,sy = mygrid.index_to_coord(state)
        for action in mygrid.transitions[state].keys():
            policy_table[sx][sy][action] = newpol[action]


# The following code actually runs the policy iteration algorithm cycles.  You should play with the parameters of this simulation until you are sure that your algorithm has converged and that you understand how the various parameters influence the obtained solutions.

# In[8]:

'''
mygrid.pretty_print_table(value_table)
mygrid.pretty_print_policy_table(policy_table)

GAMMA=0.9  # run your algorithm from 
           # above with different settings of GAMMA 
           # (Specifically 0 and 0.9 to see how the resulting value function and policy changein)
for i in range(2000):
    policy_evaluate(mygrid, value_table, policy_table, GAMMA)
    policy_improve(mygrid, value_table, policy_table, GAMMA)

mygrid.pretty_print_table(value_table)
mygrid.pretty_print_policy_table(policy_table)
'''

# <div class="alert alert-info" role="alert">
# Your final policy should look something like this for $\gamma=0.0$:
# 
# <img src="images/gammazerodpsolution.png" width="250">
# 
# and like this for $\gamma=0.9$
# 
# <img src="images/gamma09dpsolution.png" width="500">
# 
# Although note that your solution may not be identical because we are doing greedy action selection and randomly choosing one perferred action in the case that there are ties (partly because it is harder to display stochastic policies as a grid).
# </div>

# ## First Visit Monte-Carlo

# In the previous exercise you solved the sequential decision making problem using policy iteration.  However, you relied heavily on the information provided by the `GridWorld()` class, especially $\mathcal{P}^{a}_{ss'}$ (`mygrid.transitions`) and $\mathcal{R}^{a}_{ss'}$ (`rewards`).  These values are not commonly known when an agent faces an environment.  In this step of the homework you will solve the same grid world problem this time using Monte-Carlo.  
#     
# [Monte Carlo methods](https://en.wikipedia.org/wiki/Monte_Carlo_method) are ones where stochastic samples are drawn from a problem space and aggregated to estimate quantities of interest.  In this case we want to average the expected rewards available from a state going forward.  Thus, we will use Monte Carlo methods to estimate the value of particular actions or states.
# 
# The specific Monte-Carlo algorithm you should use is known as First-Visit Monte Carlo (described in lecture).  According to this algorithm, each time you first visit a state (or state-action pair) you record the rewards received until the end of an episode.  You do this many times and then average together the rewards received to estimate the value of the state or action.
# 
# Then, as you did in problem 1, you adjust your policy to become greedy with respect to the values you have estimated.
# 
# There are two significant conceptual changes in applying Monte-Carlo to the gridworld problem.  First is that rather than estimate the value of each state $V^{\pi}(s)$ under the current policy $\pi$, it makes more sense to estimate the value of each state-action pair, $Q^{\pi}(s,a)$, directly.  The reason is that in your previous solution, in order to determine the optimal policy, you likely had to know $\mathcal{P}^{a}_{ss'}$ to determine which action to perform and which state it would lead to.  Since we are trying to avoid accessing any explicit knowledge about the probabilities and rewards we cannot use this variable in our solution.  Thus, average the returns following the first visit to a particular action.
# 
# The second is what policy we should use for running our Monte Carlo updates.  If we randomly initialize the policy as we did above and then run it forward it is very easy for the runs to get caught in cycles and loops that never visit many of the states or ever encounters any rewards.  Thus, we will want to include some randomness in our simulations so that they have a non-zero probability of choosing different actions.  We will consider this issue in more detail in Part B of the homework.  For now use the $\epsilon$-greedy algorithm which choose the currently "best" action with probability $1-\epsilon$ and otherwise chooses randomly.  

# <div class="alert alert-success" role="alert">
# <h3> Problem 2 (15 points) </h3><br>
#   
#     
# In this exercise you should solve the problem introduced at the start of this notebook using Monte Carlo methods.  The pseudo code for your algorithm is described here:
# 
# ```
# Initialize, for all $s \in S$, $a \in A(s)$:
#     $Q(s,a)$ <- arbitrary
#     $\pi(s)$ <- arbitrary
#     $Returns(s,a)$ <- empty list
# 
# Repeat many times:
#     a) Generate an episode using $\pi$ with $\epislon$ probability of choosing an action at random
#     b) For each pair $s,a$ appearing in the episode
#         R <- return following the first occurence of $s,a$
#         Append R to $Returns(s,a)$
#         $Q(s,a)$ <- discounted_average(Returns(s,a))
#     c) For eash $s$ in episode:
#         $\pi(s)$ <- arg max_a Q(s,a)
# ```
# 
# When you compute the average returns you should weight them by them by $\gamma$ so that they reflect the discount rates described above.  Run your algorithm for both $\gamma=0.0$ and $\gamma=0.9$ and compare the resulting `policy_table` to the one you obtained in Problem 1.  They should work out to the same optimal policies, obtained using a quite different method, and one that in particular doesn't need an explicit model of the environment.
# </div>

# <div class="alert alert-info" role="alert">
#     
# There are a couple of hints that you will need to implement your solution which are provided by the `GridWorld` class.  The first is that you will still need to use the `rewards` dictionary from your solution to Problem 1 to compute when the rewards are delievered.  However instead of consulting this function arbitrarily you are using it just to sample the rewards when the correct event happens in your Monte Carlo simulation.
# 
# Second, you will need to find out what state you are in after taking an action in a given state.  The one-step transition dynamics of the gridworld can be simulated from the GridWorld class.  For example, to determine the state you would be in if you were in state 45 (the start state) and chose the action "up", "down", "left", or "right" is given by:
# </div>

# In[9]:


[mygrid.up(45), mygrid.down(45), mygrid.left(45), mygrid.right(45)]


# <div class="alert alert-info" role="alert">
# 
# Note that in this example, down and left walk off the edge of the environment and thus return the agent to the start state.
# 
# </div>

# <div class="alert alert-info" role="alert">
# 
# The following two functions implement the epsilon-greedy Monte Carlo sample from your gridworld task using a recursive function.  Although this is provided to you for free, you should try to understand the logic of these functions.
# </div>

# In[209]:


def epsilon_greedy(q_values, EPISILON):
    if random.random() < EPISILON:
        return random.choice(list(q_values.keys()))
    else:
        if q_values['up']==1.0:
            return 'up'
        elif q_values['right']==1.0:
            return 'right'
        elif q_values['down']==1.0:
            return 'down'
        elif q_values['left']==1.0:
            return 'left'

#recursively sample state-action transitions using epsilon greedy algorithm with a maximum recursion depth of 100.
def mc_episode(current_state, EPSILON, goal_state, policy_table, depth=0, max_depth=100):
    if current_state!=goal_state and depth<max_depth:
        sx, sy = mygrid.index_to_coord(current_state)
        action = epsilon_greedy(policy_table[sx][sy],EPSILON)
        if action == 'up':
            new_state = mygrid.up(current_state)
        elif action == 'right':
            new_state = mygrid.right(current_state)
        elif action == 'down':
            new_state = mygrid.down(current_state)
        elif action == 'left':
            new_state = mygrid.left(current_state)
        r = rewards[str([current_state,action,new_state])]
        return [[r, current_state, action]] + mc_episode(new_state, EPSILON, goal_state, policy_table, depth+1)
    else:
        return []


# <div class="alert alert-info" role="alert">
# 
#  Finally, each of your episodes should start from the start state (45) and terminate when the goal state is reached (state 8).  These have been hard coded for you below along with some initial data structures for managing the q-values, policy, and returns.
# </div>

# In[202]:


starting_state = 45
goal_state = 8 # terminate the MC roll out when you get to this state
GAMMA=0.9
EPSILON = 0.2


# set up initial data strucutres that might be useful for you
# q(s,a) - the q-values for each action in each state
def zero_q_values():
    qvals = {"up": 0.0, "right": 0.0, "down": 0.0, "left": 0.0}
    return qvals
q_value_table = [[zero_q_values() for i in range(mygrid.ncols)] for j in range(mygrid.nrows)]

# pi - the policy table
policy_table = [[random_policy() for i in range(mygrid.ncols)] for j in range(mygrid.nrows)]
display(Markdown("**Initial (randomized) policy**"))
#mygrid.pretty_print_policy_table(policy_table)

# dictionary for returns, can be filled in as more info is encountered
returns = {}


for i in range(0):  # you probably need to take many, many steps here and it make take some time to run
    episode = mc_episode(start_state, EPSILON, goal_state, policy_table)
    
    visited = {}
    for idx in range(len(episode)):
        item = episode[idx]
        qkey = str((item[1],item[2]))
        if qkey not in visited:
          qkey = 1 # Added by Alfred
            # first visit
            #PROBLEM 2 - update the returns dictionary to include the discounted average returns

    # update q-value-table
    for ret in returns.keys():
        s,a = eval(ret)
        sx, sy = mygrid.index_to_coord(s)
        q_value_table[sx][sy][a] = statistics.mean(returns[ret])

    # improve policy
    for sx in range(len(q_value_table)):
        for sy in range(len(row)):
            policy_table[sx][sy] = be_greedy(q_value_table[sx][sy])
        
display(Markdown("**Improved policy**"))
#mygrid.pretty_print_policy_table(policy_table)


# <div class="alert alert-success" role="alert">
# <h3> Problem 3 (10 points) </h3><br>
# 
# The two solution methods we just considered have different strengths and weaknesses.  Describe in your own words the things that make these solutions methods better or worse.  Your response should be 2-3 sentences and address both the computational efficiency of the algoriths, the amount of assumed knowledge of the environment, and the relationship between these methods to how humans might solve similar sequential decision making problems.
# <div>

# # Turning in homework
# 
# When you are finished with this notebook. Save your work in order to turn it in.  To do this select *File*->*Download As...*->*HTML*.
# 
# <img src="images/save-pdf.png" width="300">
# 
# You can turn in your assignments using NYU Classes webpage for the course (available on https://home.nyu.edu). **Make sure you complete all parts (A and B) of this homework.**

# ## Next steps...
# 
# So, far, so good...  Now move on to [Homework Part B](Homework-RL-B.ipynb) 
