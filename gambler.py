import numpy as np





grad = 1000
theta = 0.001
gamma = 1
p_head  = 0.4


def compute_values(grad, theta, p_head, gamma):

  values = np.zeros((101,))
  values[0] = 0
  values[1] = 1




  return (values, optimal_values)



fig, axs = plt.subplots(2)

axs[0].plot(states[1:-1], values[1:-1])
axs[1].bar(states[1:-1], optimal_policy[1:-1])

plt.show()
