import numpy as np
import math


# create intial variations
max_cars = 5
lots = 2
states = np.array([max_cars for i in range(lots)]) # intial setup


#  define the following as such
# the first index respresnts state 1, second reprsents state 2
# from -min(5,k) is how many you can take from k and thus index min(5,k) is 0
# from 1 - min(5.i) is how many you can add
# for every state pair (k,i) you save a tuple (actions, 0-th index)
policy = {
  (k,i): (np.zeros((min(5,k) + min(5,i) + 1)), min(5,k)) for k in range(max_cars + 1) for i in range(max_cars + 1)
} # define the number of cars for each possible state

def print_policy():
  for i in policy.keys():
    arr, other = policy[i]
    print("action", np.argmax(arr) - other ,": for the policy of  ", i)


def get_action_number(policy_arr):
  arr, zero_index = policy_arr
  return np.argmax(arr) - zero_index



def get_possion_probability(costumers):
  ans = np.zeros((2,), dtype=float)
  return_expected = np.array([3. , 2. ])
  takeout_expected = np.array([3., 4.])

  for i in range(2):
    current_lot = costumers[i]
    current_expected = return_expected[i] if current_lot < 0 else takeout_expected[i]
    current_lot_prob = ((current_expected)**abs(current_lot) / math.factorial(abs(current_lot)) )* math.exp(-current_expected)
    ans[i] = current_lot_prob

  return ans

time_steps = 10
grad = 1000
gamma = 0.9
theta = 1
policy_stable = False

values = np.zeros((max_cars + 1, max_cars + 1))


#TODO function is wrong and needs to be fixed
def compute_values(state, action_number):

  first, second = state

  value = 0
  for i in range(max_cars + 1):
    for j in range(max_cars + 1):

      reward = 10 * (max(0, -i + (first + action_number) + max(0, -j + (second - action_number)))) - 2 * abs(action_number)

      possion_probabilites = get_possion_probability(np.array([
        -i + (first + action_number), -j + (second - action_number)
      ]))

      transition_probability = 1

      for p in possion_probabilites:
        transition_probability *= p
      value += transition_probability * (reward + gamma * values[i, j])

  return value

while policy_stable != True:

  policy_stable = True

  # policy iteration

  while grad > theta:

    grad = 0

    for i in range(max_cars + 1):
      for k in range(max_cars + 1):

        state = (i,k)
        v = values[i,k]
        current_action = get_action_number(policy[state])

        values[i,k] = compute_values(state, current_action)

        grad = max(grad, abs(v - values[i,k]))
    print(grad)
  grad = 1000 # reset grad

  # policy evaulation

  for i in range(max_cars + 1):
    for k in range(max_cars + 1):

      state = (i,k)
      current_policy = policy[state]
      arr, index = current_policy
      q_star = np.zeros((arr.shape[0],))

      for a in range(arr.shape[0]):
        current_action_policy = np.zeros((arr.shape[0], ))
        current_action_policy[a] = 1
        current_action = get_action_number((current_action_policy, index))

        q_star[a] = compute_values(state, action_number=current_action)

      updated_arr = 0 * arr # reset arr
      updated_arr[np.argmax(q_star)] = 1

      if np.argmax(updated_arr) != np.argmax(arr):
        policy_stable = False

      policy[state] = (updated_arr, index)
      print(updated_arr)


# create a graph of policy for all states coloring in the action
# with the highest probability
print_policy()
