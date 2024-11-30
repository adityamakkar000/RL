""" OFF policy MC control"""

import numpy as np


grid_x = 17
grid_y = 32

intial_definitions = [
    [3, 16],
    [2, 16],
    [2, 16],
    [1, 16],
    [0, 16],
    [0, 16],
    [
        0,
        9,
    ],
    [
        0,
        8,
    ],
    [
        0,
        8,
    ],
    [
        0,
        8,
    ],
    [
        0,
        8,
    ],
    [
        0,
        8,
    ],
    [
        0,
        8,
    ],
    [
        0,
        8,
    ],
    [
        1,
        8,
    ],
    [
        1,
        8,
    ],
    [
        1,
        8,
    ],
    [
        1,
        8,
    ],
    [
        1,
        8,
    ],
    [
        1,
        8,
    ],
    [
        1,
        8,
    ],
    [
        1,
        8,
    ],
    [
        2,
        8,
    ],
    [
        2,
        8,
    ],
    [
        2,
        8,
    ],
    [
        2,
        8,
    ],
    [
        2,
        8,
    ],
    [
        2,
        8,
    ],
    [
        2,
        8,
    ],
    [
        3,
        8,
    ],
    [3, 8],
    [3, 8],
]

finish_row = np.array([i for i in range(6)])


class RaceTrack:

    def __init__(self, grid_x, grid_y, intial_definitions, finish_row):
        grid_boundaries = -1 * np.ones((grid_y, grid_x))

        for row in range(grid_y):
            if row == grid_y - 1:
                grid_boundaries[
                    row, intial_definitions[row][0] : intial_definitions[row][1] + 1
                ] = 1
            else:
                grid_boundaries[
                    row, intial_definitions[row][0] : intial_definitions[row][1] + 1
                ] = 0

        finish_x = np.array([-1 for i in range(len(finish_row))])

        grid_boundaries[finish_row, finish_x] = 2

        self.finish = [(i, grid_x - 1) for i in range(len(finish_row))]
        self.grid_boundaries = grid_boundaries
        self.start = np.array(
            [
                (grid_y - 1, k)
                for k in range(
                    intial_definitions[grid_y - 1][0],
                    intial_definitions[grid_y - 1][1] + 1,
                )
            ]
        )
        self.grid_y = grid_y
        self.grid_x = grid_x

    def __repr__(self):
        return str(self.grid_boundaries)


class Racer:

    def __init__(self, racetrack):

        def generate_dict(obj, racetrack):
            return {
                (i, k, v_1, v_2): obj
                for i in range(grid_y)
                for k in range(grid_x)
                for v_1 in range(0, 6)
                for v_2 in range(0, 6)
                if racetrack.grid_boundaries[i, k] != -1
            }

        equal_policy = np.array([1 / 9 for _ in range(9)])
        determenistic_policy = np.eye(9)[0]

        self.racetrack = racetrack
        self.actions = np.array([(i, k) for i in range(-1, 2) for k in range(-1, 2)])
        self.values = generate_dict(np.random.rand(1), racetrack)
        self.Q = generate_dict(np.random.rand(9), racetrack)
        self.target_policy = generate_dict(determenistic_policy, racetrack)
        self.behaviour_policy = generate_dict(equal_policy, racetrack)

    def simulate(self):

        def get_action(actions, probs):
            rng = np.random.default_rng()
            choice = rng.choice(np.arange(actions.shape[0]), 1, p=probs)

            action = actions[choice][0]
            return choice, action

        states, actions, rewards = [], [], []

        index, intial_state = get_action(
            self.racetrack.start,
            np.array(
                [
                    1 / self.racetrack.start.shape[0]
                    for i in range(self.racetrack.start.shape[0])
                ]
            ),
        )
        state = (*intial_state, 0, 0)

        states.append(state)

        while self.racetrack.grid_boundaries[state[0], state[1]] != 2:

            current_cell_policy = self.behaviour_policy[state]
            index, action = get_action(self.actions, current_cell_policy)

            old_x, old_y, old_v1, old_v2 = state

            v1 = max(0, min(5, old_v1 + action[0]))
            v2 = max(0, min(5, old_v2 + action[1]))



            x = old_x - v1
            y = old_y + v2

            actions.append(index)
            rewards.append(-1)
            if (
                x >= self.racetrack.grid_y
                or x < 0
                or y >= self.racetrack.grid_x
                or y < 0
            ):
                if ((y - old_y) / (x - old_x)) in [
                    (i - old_y) / (k - old_x) for i, k in self.racetrack.finish
                ]:
                    return states, actions, rewards

                index, intial_state = get_action(
                    self.racetrack.start,
                    np.array(
                        [
                            1 / self.racetrack.start.shape[0]
                            for i in range(self.racetrack.start.shape[0])
                        ]
                    ),
                )
                state = (*intial_state, 0, 0)

            elif self.racetrack.grid_boundaries[state[0], state[1]] == -1:
                state = intial_state = get_action(
                    self.racetrack.start,
                    np.array(
                        [
                            1 / self.racetrack.start.shape[0]
                            for i in range(self.racetrack.start.shape[0])
                        ]
                    ),
                )
                state = (*intial_state, 0, 0)

            elif (x, y) in self.racetrack.finish:
                return states, actions, rewards

            else:
                state = (x, y, v1, v2)

            states.append(state)

        return states, actions, rewards


rt1 = RaceTrack(grid_x, grid_y, intial_definitions, finish_row)

racer = Racer(rt1)

print(racer.simulate())
