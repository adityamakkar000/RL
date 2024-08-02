import numpy as np
import math

total_cars = 20
states = [(i, k) for i in range(0, total_cars + 1) for k in range(0, total_cars + 1)]

values = np.array(
    [[5 for i in range(0, total_cars + 1)] for k in range(0, total_cars + 1)]
)

policy = {
    s: (np.zeros((1 + min(s[0], 5) + min(s[1], 5),)), min(s[0], 5)) for s in states
}

gradStable = False
theta = 0.1
gamma = 0.9


def get_prob(lam, n):
    first_term = lam**n
    fac = 1 / math.factorial(n)
    constant = math.exp(-lam)
    return first_term * fac * constant


while gradStable == False:
    grad = 100
    # policy evaulation
    while grad > theta:
        grad = 0
        values_temp = np.copy(values)
        for x in range(0, total_cars + 1):
            for y in range(0, total_cars + 1):
                state = (x, y)

                values[x, y] = 0

                action = policy[state]  # have my action tuple
                cars_to_transfer = np.argmax(action[0]) - action[1]

                x += cars_to_transfer
                y -= cars_to_transfer

                for cars_x_take in range(0, x):
                    for cars_x_return in range(0, total_cars - x):
                        for cars_y_take in range(0, y):
                            for cars_y_return in range(0, total_cars - y):
                                reward = (
                                    10 * (cars_x_take + cars_y_take)
                                    - 2 * cars_to_transfer
                                )
                                probability = (
                                    get_prob(3, cars_x_take)
                                    * get_prob(3, cars_x_return)
                                    * get_prob(4, cars_y_take)
                                    * get_prob(2, cars_y_return)
                                )
                                print(probability)
                                values[state[0], state[1]] += probability * (
                                    reward
                                    + gamma
                                    * values_temp[
                                        state[0]
                                        + cars_to_transfer
                                        - cars_x_take
                                        + cars_x_return,
                                        state[1]
                                        - cars_to_transfer
                                        - cars_y_take
                                        + cars_y_return
                                    ]
                                )
            grad = max(
                grad, abs(values_temp[state[0], state[1]] - values[state[0], state[1]])
            )
        # do somethig
    print(grad)
    # policy improvment
    gradStable = True
    for x in range(0, total_cars + 1):
        for y in range(0, total_cars + 1):
            state = (x, y)

            value = 0
            action = policy[state]
            q_star = np.zeros((len(action[0])))

            for i in range(len(action[0])):
                cars_to_transfer = i - action[1]

                x += cars_to_transfer
                y -= cars_to_transfer

                for cars_x_take in range(0, x):
                    for cars_x_return in range(0, total_cars - x):
                        for cars_y_take in range(0, y):
                            for cars_y_return in range(0, total_cars - y):
                                reward = (
                                    10 * (cars_x_take + cars_y_take)
                                    - 2 * cars_to_transfer
                                )
                                probability = (
                                    get_prob(3, cars_x_take)
                                    * get_prob(3, cars_x_return)
                                    * get_prob(4, cars_y_take)
                                    * get_prob(2, cars_y_return)
                                )
                                value += probability * (
                                    reward
                                    + gamma
                                    * values[
                                        state[0]
                                        + cars_to_transfer
                                        - cars_x_take
                                        + cars_x_return,
                                        state[1]
                                        - cars_to_transfer
                                        - cars_y_take
                                        + cars_y_return
                                    ]
                                )
                x -= cars_to_transfer
                y += cars_to_transfer
                q_star[i] = value
            if np.argmax(q_star) != np.argmax(action[0]):
                gradStable = False
                policy[state] = (0 * policy[state][0], policy[state][1])  # reset policy
                policy[state][0][np.argmax(q_star)] = 1
