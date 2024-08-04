import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Monte Carlo ES (Exploring Starts), for estimating optimal policy on blackjack


class blackjack:
    def __init__(self) -> None:
        self.card_probs = {
            2: 4 / 52,
            3: 4 / 52,
            4: 4 / 52,
            5: 4 / 52,
            6: 4 / 52,
            7: 4 / 52,
            8: 4 / 52,
            9: 4 / 52,
            10: 16 / 52,
            'A': 4 / 52
        }

        self.cards_sum = {str(i): i for i in range(2, 11)}

        self.cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 'A']

    def sum(self, cards):
        usable_ace = False
        ace_count = 0
        current_sum = 0
        for card in cards:
            if card == 'A':
                usable_ace = True
                ace_count += 1
            else:
                current_sum += self.cards_sum[card]
        if usable_ace:
            current_sum += 11
            current_sum += ace_count - 1
            if current_sum > 21:
                current_sum -= 10
                usable_ace = False
        return current_sum, usable_ace

    def draw(self, number):
        return np.random.choice(
            self.cards, number, [self.card_probs[card] for card in self.cards]
        ).tolist()

    def players_turn(self, cards, dealers_showing, policy):
        episode = []
        actions = []
        psum, usableAce = self.sum(cards)
        episode.append((psum, usableAce, dealers_showing))
        while psum <= 21:
            action = np.argmax(
                policy[psum - 12, 1 if usableAce else 0, dealers_showing - 2]
            )
            if action == 0:
                actions.append(action)
                break
            else:
                actions.append(action)
                cards.append(self.draw(1)[0])
                psum, usableAce = self.sum(cards)
                if psum > 21:
                    break
                episode.append((psum, usableAce, dealers_showing))
        return episode, actions, psum

    def dealers_turn(self, cards):
        psum, usableAce = self.sum(cards)
        while psum < 17:
            cards.append(self.draw(1)[0])
            psum, usableAce = self.sum(cards)
        return psum

    def simulate_game(self, policy):
        players_cards = self.draw(2)
        dealers_cards = self.draw(2)

        dealers_showing, dealerace = self.sum([dealers_cards[0]])

        episode, actions, players_sum = self.players_turn(
            players_cards, dealers_showing, policy
        )
        dealers_sum = self.dealers_turn(dealers_cards)

        if players_sum > 21 or dealers_sum > players_sum:
            reward = -1
        elif dealers_sum > 21 or players_sum > dealers_sum:
            reward = 1
        elif dealers_sum == players_sum:
            reward = 0
        game_info = {
            'episode': episode,
            'actions': actions,
            'dealers_sum': dealers_sum,
            'reward': reward
        }

        return game_info

    def __call__(self, policy):
        return self.simulate_game(policy)


def plot_value_function(value_function, title):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(12, 22)
    y = np.arange(2, 12)
    x, y = np.meshgrid(x, y)
    z = value_function.T

    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Sum')
    ax.set_zlabel('Value')
    ax.set_title(title)

    plt.show()


def average(arr):
    return np.mean(arr)


value = np.zeros((10, 2, 10))

Q = np.zeros((10, 2, 10, 2))

returns = [
    [[[[] for p in range(2)] for j in range(10)] for k in range(2)] for i in range(10)
]

policy = np.ones((10, 2, 10, 2)) * 0.5

game = blackjack()
gamma = 1

games = 0
total_games = 10000000

while games < total_games:
    info = game(policy)
    episode = info['episode']
    actions = info['actions']
    g = info['reward']

    for t in range(len(episode) - 1, -1, -1):
        g = gamma * g
        current_episode = episode[t]
        current_action = actions[t]
        # if current_episode in episode[: t - 1] and current_action in episode[: t - 1]:
        #     print('hi')
        if current_episode[0] >= 12 and current_episode[0] <= 21:
            idx0, idx1, idx2 = (
                current_episode[0] - 12,
                1 if current_episode[1] == True else 0,
                current_episode[2] - 2
            )

            returns[idx0][idx1][idx2][current_action].append(g)
            Q[idx0, idx1, idx2, current_action] = average(
                returns[idx0][idx1][idx2][current_action]
            )
            index = np.argmax(Q[idx0, idx1, idx2])
            policy[idx0, idx1, idx2] = np.eye(2)[index]
            value[idx0, idx1, idx2] = np.max(Q[idx0, idx1, idx2])
    games += 1
    if games % 10000 == 0:
        print('completed ', games, ' games')
usable_ace_values = value[:, 1, :]
no_ace_values = value[:, 0, :]

plot_value_function(usable_ace_values, 'usableAce')
plot_value_function(no_ace_values, 'nousableAce')
