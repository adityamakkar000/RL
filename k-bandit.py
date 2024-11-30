import matplotlib.pyplot as plt
import numpy as np
import random as r


def get_array(k):
    return np.random.random((k,))


k = 100
lower = 0
upper = 5

eps_number = 10
eps = np.linspace(0, 1, eps_number)

time_steps = 1000000
reward = np.zeros((eps_number + 1, time_steps))
reward[0] = np.arange(1, time_steps + 1)

q_star = np.zeros((time_steps, k))

q_star[0] = get_array(k)
Q_star = np.ones((k,)) * 5

for i in range(1, time_steps):
    q_star[i] = q_star[i - 1] + np.random.normal(loc=0.0, scale=1, size=(1, k))
print(q_star)

for i in range(time_steps):
    reward[0, i] = np.random.normal(
        loc=q_star[i, np.argmax(q_star[i])], scale=1.0, size=(1,)
    )[0]
    if i > 0:
        reward[0, i] += reward[0, i - 1]
        reward[0, i] = reward[0, i]
for e in range(eps_number):
    x = np.array([eps[e], 1 - eps[e]])
    actions = np.zeros((k,))

    Q_star_current = np.copy(Q_star)

    Q_star_estimate = np.zeros((k,))
    Q_star_divsor = np.zeros((k,))

    samples = np.random.choice(x, time_steps, p=x)
    for i in range(time_steps):
        number = (
            np.random.randint(0, len(Q_star_current))
            if samples[i] == eps[e]
            else np.argmax(Q_star_current)
        )
        current_sample = np.random.normal(loc=q_star[i, number], scale=1.0, size=(1,))[
            0
        ]
        reward[e + 1, i] = current_sample

        if i > 0:
            reward[e + 1, i] += reward[e + 1, i - 1]
            reward[e + 1, i] = reward[e + 1, i]
        Q_star_estimate[number] += current_sample
        Q_star_divsor[number] += 1
        Q_star_current[number] = Q_star_estimate[number] / 10
intial = 'base'
for e in range(eps_number + 1):
    plt.plot(reward[e], label=f'epsilon = {eps[e-1] if e > 0 else intial}')
plt.xlabel('Time step')
plt.ylabel('Reward')
plt.legend()
plt.show()
