import numpy as np
import matplotlib.pyplot as plt

""" This is a soultion to the MDP gamblers problem with value iteration"""

# setup of all function
states = np.linspace(1, 99, 99)
values = { i: 0 for i in states} # for each state give it a value of intially 0
values[0] = 0
values[100] = 1
optimal_policy = {i: 0 for i in states}


p_head = 0.25
p_tails = 1  - p_head
grad = 1000 # checking dp difference
theta = 0.00000001 # giving covergernce limit
step = 0
max_steps = 100
while max_steps > step:
  grad = 0

  values_temp = values
  for s in states:

    v = values[s]
    values[s] = 0


    # itearte through actions
    # iteration through rewards and possible states to find the transition probailbity
    # compute final reward

    actions = np.arange(0, min(s, 100 - s) + 1)
    for a in actions:
      val = 0
      s_primes = np.array([(s + a, p_head), (s - a, p_tails)])
      for sp, prob in s_primes:
        reward = 1 if sp == 100 else 0

        val += 1 * prob * ( 1 * values_temp[sp])

      if val > values[s]:
        values[s] = val
        optimal_policy[s] = a

    grad = max(grad, abs(v - values[s]))
  step += 1


fig, axs = plt.subplots(2)

axs[0].plot(states, np.array([values[i] for i in states]))

axs[1].bar(states, np.array([optimal_policy[i] for i in states]))

plt.show()