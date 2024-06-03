import numpy as np



# create the value_0 for the grid

grid_squares = 4
total_grid_squares = grid_squares * grid_squares
values = np.zeros((grid_squares, grid_squares))

# first training loop


states = np.arange(0, total_grid_squares).reshape(grid_squares, grid_squares)

policy = {i:[0.25, 0.25,0.25,0.25] for i in range(total_grid_squares)}

ts = np.ones((grid_squares, grid_squares))
ts[0,0] = 0
ts[grid_squares - 1, grid_squares -1 ] = 0

rewards = -1 * np.ones((grid_squares, grid_squares))
rewards[0,0] = 0
rewards[grid_squares - 1, grid_squares -1 ] = 0

grad = 10000
theta = 0.01
gamma = 1
actions = ['u', 'd', 'l', 'r']

def do_action(direction, state):

  first_index, second_index = state
  if (direction == 'u'):
    return (first_index - 1, second_index)
  elif(direction == 'd'):
    return  (first_index + 1, second_index)
  elif(direction == 'r'):
    return (first_index, second_index + 1)
  elif(direction == 'l'):
    return (first_index, second_index - 1)


def compute_values(state):

  first, second = state
  value = 0
  # sum over actions
  for _,a in enumerate(actions):
    # sum over states
    i, k = do_action(a, state)
    if(i < 0 or i >= grid_squares or k < 0 or k >= grid_squares):
      i, k = state
    if((first == 0 and second == 0) or (first == grid_squares - 1 and second == grid_squares - 1)):
      i,k = first, second
    value += policy[4*first + second][_] * ts[i, k] * (rewards[i,k] +  gamma * values[i,k])

  return value

def compute_q_star(state):

  first, second = state
  q_star = []
  # sum over actions
  for _,a in enumerate(actions):
    # sum over states
    i, k = do_action(a, state)
    value = 0

    if(i < 0 or i >= grid_squares or k < 0 or k >= grid_squares):
      i, k = state

    if((first == 0 and second == 0) or (first == grid_squares - 1 and second == grid_squares - 1)):
      i,k = first, second

    value += (ts[i, k] * (rewards[i,k] +  gamma * values[i,k]))
    q_star.append(value)

  return np.array(q_star)



print(values.shape)


policy_stable = False

while policy_stable == False:

  policy_stable = True

  while grad > theta :

    grad = 0

    for i in range(grid_squares):
      for k in range(grid_squares):
        state = (i,k)
        v = values[i,k]
        values[i,k] = compute_values(state)

        grad = max(grad, abs(v - values[i,k]))


  for i in range(grid_squares):
    for k in range(grid_squares):

      current_policy = policy[4*i + k]
      arr = compute_q_star((i,k))
      q_star = np.argmax(arr)

      policy[4*i + k ] = [1 if i == q_star else 0 for i in range(4)]

      if(policy[4*i + k] != current_policy):
        policy_stable = False

print(values)
coversion = {
1000: 'u',
100: 'd',
10: 'l',
1: 'r'
}
print(np.array([coversion[sum([10**(3-k) * policy[i][k] for k in range(4)])] for i in range(total_grid_squares)]).reshape(grid_squares, grid_squares))
