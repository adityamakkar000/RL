import numpy as np



# create the value_0 for the grid

grid_squares = 10
total_grid_squares = grid_squares * grid_squares
values = np.zeros((grid_squares, grid_squares))

# first training loop

policy = lambda a,s: 0.25 # equi distrubtion function

states = np.arange(0, total_grid_squares).reshape(grid_squares, grid_squares)

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
  for a in actions:
    # sum over states
    i, k = do_action(a, state)
    if(i < 0 or i >= grid_squares or k < 0 or k >= grid_squares):
      i, k = state
    value += 0.25 * ts[first, second] * (rewards[i,k] +  gamma * values[i,k])

  return value


while grad > theta:

  grad = 0

  for i in range(grid_squares):
    for k in range(grid_squares):
      state = (i,k)
      v = values[i,k]
      values[i,k] = compute_values(state)

      grad = max(grad, abs(v - values[i,k]))
  print(grad)

print(values)
