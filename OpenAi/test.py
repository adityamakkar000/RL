import gymnasium as gym
import torch
from algo.gradMC import QPolicy
import time
env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset(seed=42)

policy = QPolicy(8, 4)
epsilon = 0.3
states = [observation]
actions = []
counts = []
rewards = []
gamma = 1
alpha = 0.01
epochs = 100

for _ in range(epochs):
    losses = []
    start = time.time()
    for l in range(100):
        q = policy(torch.from_numpy(observation))
        action = torch.randint(0,4, (1,)).item()

        if torch.randn(1,)[0] > epsilon:
            action = torch.argmax(q).item()

        actions.append(q[action])
        counts.append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        states.append(observation)

        if terminated or truncated:
            G_t = 0
            loss_total_ep = 0
            c = torch.zeros((len(states)-1, 4))
            for t in range(len(states) - 2, -1, -1):
                c[t,counts[t]] += 1
                policy.zero_grad()
                G_t = (rewards[t] + gamma * G_t)/(c[t,counts[t]])
                loss = 0.5 * (G_t - actions[t])**2
                loss_total_ep += loss.item()
                loss.backward()

                for p in policy.parameters():
                    p.data -= alpha * p.grad

            losses.append(loss_total_ep)
            states  = []
            counts = []
            actions = []
            rewards = []

            observation, info = env.reset()
            states.append(observation)
    end = time.time()
    total_time = end - start
    loss_total = sum(losses)/len(losses)
    print(f"epoch: {_} | loss: {loss_total:.4f} | total_games: {len(losses)} | time: {total_time:.2f} s")


env.close()
