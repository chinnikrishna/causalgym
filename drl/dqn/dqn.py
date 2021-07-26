import gym
import causalgym
import random

if __name__ == '__main__':
    env = gym.make('causalgym-task1-v0')
    num_episodes = 1
    done = False
    obs_img, done, reward = env.reset()
    cnt = 0
    while not done:
        action = random.choice(range(0, 5))
        print(action)
        obs_img, done, reward = env.step(action)
        cnt += 1
        if cnt > 200:
            break
        env.render()