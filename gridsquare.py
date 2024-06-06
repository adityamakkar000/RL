import numpy as np

# this is a soultion to Example 4.1 in sutto barton


# setup

""" We begin by creating states, values and policy array"""

num_squares = 4

# states is a 4 x 4 grid
states = np.zeros((num_squares, num_squares))
values = np.copy(states)

policy = np.zeros((num_squares, num_squares), dtype=np.ndarray)

# create equa distrubted policy
for i in range(num_squares):
  for k in range(num_squares):
    policy[i][k] = np.array([1/4, 1/4, 1/4, 1/4 ])


theta = 0.01
policy_stable = False
gamma = 1

def get_s_prime(action, state):

  # 0 -> up
  # 1 -> down
  # 2 -> right
  # 3 -> left

  first, second = state # unpack values
  if (first == 0 and second == 0) or (first == num_squares -1 and second == num_squares - 1):
    return state

  if(action == 0):
    return (max(first - 1, 0), second)

  if (action == 1) :
    return (min(first + 1, num_squares - 1), second)

  if (action == 2):
    return (first, min(second + 1, num_squares - 1))

  if (action == 3):
    return (first, max(0, second - 1))


def get_reward(state):
  first, second = state
  if ((first == 0 and second == 0) or (first == num_squares - 1 and second == num_squares - 1)):
    return 0

  return -1

# while policy_stable == False:
#  grad = 1000


steps = 0
max_steps = 1000
while policy_stable == False:

  grad = 1000
  while grad > theta:
    grad = 0
    values_temp = np.copy(values) # temp thing from old one
    for i in range(num_squares):
      for k in range(num_squares):

        state = (i,k)

        v = values[i,k]
        values[i,k] = 0

        for _, a in enumerate(policy[i,k]):
          s_prime = get_s_prime(_, state)
          reward = get_reward(state)

          values[i,k] += a * 1 * (reward + gamma * values_temp[s_prime[0], s_prime[1]])

        grad = max(grad, abs(v - values[i,k]))
    steps +=  1

    policy_stable = True
    # for i in range(num_squares):
    #   for k in range(num_squares):

    #     q_star = np.zeros((4,))
    #     state = (i,k)

    #     for _, a in enumerate(policy[i,k]):
    #       value = 0
    #       s_prime = get_s_prime(_, state)
    #       reward = get_reward(state)

    #       value = 1 * 1 * (reward + gamma * values[s_prime[0], s_prime[1]])

    #       q_star[_] = value

    #     policy_new = np.array([1 if i == np.argmax(q_star) else 0 for i in range(4)])
    #     if not np.array_equal(policy_new, policy[i,k]):
    #       policy[i,k] = policy_new
    #       policy_stable = False


print(values)