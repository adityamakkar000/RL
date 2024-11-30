import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import argparse
from tqdm import tqdm
# Monte Carlo ES (Exploring Starts), for estimating optimal policy on blackjack


parser = argparse.ArgumentParser(description="Monte Carlo Exploring Starts")
parser.add_argument("-games", type=int, default=10000000)
parser.add_argument("-gamma", type=float, default=1)

args = parser.parse_args()


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
            "A": 4 / 52,
        }

        self.cards_sum = {str(i): i for i in range(2, 11)}

        self.cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, "A"]

    def sum(self, cards):
        usable_ace = False
        ace_count = 0
        current_sum = 0
        for card in cards:
            if card == "A":
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

    def players_turn(self, cards, dealers_showing, policy, intial_action):
        episode = []
        actions = []
        psum, usableAce = self.sum(cards)
        episode.append((psum, usableAce, dealers_showing))
        actions.append(intial_action)
        if intial_action == 0:
            return episode, actions, psum

        if intial_action == 1:
            cards.append(self.draw(1)[0])
            psum, usableAce = self.sum(cards)
            if psum > 21:
                return episode, actions, psum

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

    def simulate_game(self, policy, intial_action):
        players_cards = self.draw(2)
        dealers_cards = self.draw(2)

        dealers_showing, dealerace = self.sum([dealers_cards[0]])

        episode, actions, players_sum = self.players_turn(
            players_cards, dealers_showing, policy, intial_action
        )
        dealers_sum = self.dealers_turn(dealers_cards)

        if players_sum > 21 or dealers_sum > players_sum:
            reward = -1
        elif dealers_sum > 21 or players_sum > dealers_sum:
            reward = 1
        elif dealers_sum == players_sum:
            reward = 0
        game_info = {
            "episode": episode,
            "actions": actions,
            "dealers_sum": dealers_sum,
            "reward": reward,
        }

        return game_info

    def __call__(self, policy, intial_action):
        return self.simulate_game(policy, intial_action)


def plot_value_function(value_function, title):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    x = np.arange(12, 22)
    y = np.arange(2, 12)
    x, y = np.meshgrid(x, y)
    z = value_function.T

    ax.plot_surface(x, y, z, cmap="viridis")
    ax.set_xlabel("Player Sum")
    ax.set_ylabel("Dealer Sum")
    ax.set_zlabel("Value")
    ax.set_title(title)

    plt.show()





value = np.zeros((10, 2, 10))
Q = np.zeros((10, 2, 10, 2))
returns = np.zeros((10, 2, 10, 2))
count = np.zeros((10, 2, 10, 2))
policy = np.ones((10, 2, 10, 2)) * 0.5

game = blackjack()
gamma = args.gamma

total_games = args.games

start = time.time()

with tqdm(total=total_games, desc="MCES Progress") as pbar:
    for games in range(total_games):
        info = game(policy, np.random.choice([0, 1], 1, [0.5, 0.5])[0])
        episode = info["episode"]
        actions = info["actions"]
        g = info["reward"]

        for t in range(len(episode) - 1, -1, -1):
            g = gamma * g
            current_episode = episode[t]
            current_action = actions[t]


            if current_episode not in episode[:t] and current_action not in actions[:t]:

                if current_episode[0] >= 12 and current_episode[0] <= 21:
                    idx0, idx1, idx2 = (
                        current_episode[0] - 12,
                        1 if current_episode[1] == True else 0,
                        current_episode[2] - 2,
                    )

                    count[idx0][idx1][idx2][current_action] += 1
                    returns[idx0][idx1][idx2][current_action] = returns[idx0][idx1][idx2][
                        current_action
                    ] + (1 / count[idx0][idx1][idx2][current_action]) * (
                        g - returns[idx0][idx1][idx2][current_action]
                    )

                    Q[idx0, idx1, idx2, current_action] = returns[idx0][idx1][idx2][
                        current_action
                    ]

                    index = np.argmax(Q[idx0, idx1, idx2])
                    policy[idx0, idx1, idx2] = np.eye(2)[index] + [-0.2, 0.2]
                    value[idx0, idx1, idx2] = np.max(Q[idx0, idx1, idx2])

        pbar.update(1)

        if (games + 1) % 10000 == 0:
            end = time.time() - start
            pbar.set_postfix({'Time': f"{end:.2f}s"})
            start = time.time()

usable_ace_values = value[:, 1, :]
no_ace_values = value[:, 0, :]

plot_value_function(usable_ace_values, "usableAce")
plot_value_function(no_ace_values, "nousableAce")

def plot_policy(policy):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    usable_ace_policy = policy[:, 1, :, 0]
    im1 = ax1.imshow(usable_ace_policy, cmap='coolwarm', aspect='auto', extent=[1.5, 10.5, 21.5, 11.5])
    ax1.set_title("π* (Usable ace)")
    ax1.set_xlabel("Dealer showing")
    ax1.set_ylabel("Player sum")
    ax1.set_xticks(range(2, 11))
    ax1.set_yticks(range(12, 22))
    ax1.text(0.05, 0.95, "STICK", transform=ax1.transAxes, fontsize=14, verticalalignment='top')
    ax1.text(0.05, 0.05, "HIT", transform=ax1.transAxes, fontsize=14, verticalalignment='bottom')
    plt.colorbar(im1, ax=ax1, label="Probability of STICK")

    no_usable_ace_policy = policy[:, 0, :, 0]
    im2 = ax2.imshow(no_usable_ace_policy, cmap='coolwarm', aspect='auto', extent=[1.5, 10.5, 21.5, 11.5])
    ax2.set_title("π* (No usable ace)")
    ax2.set_xlabel("Dealer showing")
    ax2.set_ylabel("Player sum")
    ax2.set_xticks(range(2, 11))
    ax2.set_yticks(range(12, 22))
    ax2.text(0.05, 0.95, "STICK", transform=ax2.transAxes, fontsize=14, verticalalignment='top')
    ax2.text(0.05, 0.05, "HIT", transform=ax2.transAxes, fontsize=14, verticalalignment='bottom')
    plt.colorbar(im2, ax=ax2, label="Probability of STICK")

    plt.tight_layout()
    plt.show()



plot_policy(policy)