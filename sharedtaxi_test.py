from sharedtaxi import SharedTaxi
import numpy as np
from numpy import random

if __name__ == '__main__':

    env = SharedTaxi(2, 2, 6, 2)

    #Random agent - uncomment to test
    """
    initial_state = env.s = 701
    done = False

    while not done:
        action = env.action_space.sample()
        last_state = env.s
        state, reward, done, info = env.step(action)
        print('state, action, next_state, reward - ', last_state, action, state, reward)

    if done:
        print('Done for state - ', initial_state)
    for s in range(1600):
        initial_state = env.reset()
        done = False
        epochs = 0

        while not done:
            action = env.action_space.sample()
            last_state = env.s
            state, reward, done, info = env.step(action)

            epochs = epochs + 1
            if epochs % 100000 == 0:
                print('States inital->last->current: ', initial_state, last_state, state)
                break

        if done:
            print('Done for state - ', initial_state)

    epochs = 0
    penalties, reward = 0, 0

    done = False

    while not done:
        action = env.action_space.sample()
        old_state = env.s
        state, reward, done, info = env.step(action)

        if reward < 0:
            penalties += 1
        
        # Put each rendered frame into dict for animation
        epochs += 1
        
    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))

    """

    #Q-learning based agent
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    for i in range(1, 1000000):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action) 
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward < 0:
                penalties += 1

            state = next_state
            epochs += 1
            
        if i % 100000 == 0:
            print(f"Episode: {i}")

    print("Training finished.\n")

    episodes = 10000
    total_epochs, total_penalties = 0, 0

    for episode in range(episodes):
        initial_state = state = env.reset()
        #initial_state = state = 946
        epochs, penalties, reward = 0, 0, 0

        done = False

        while not done:
            action = np.argmax(q_table[state])
            last_state = env.s
            state, reward, done, info = env.step(action)

            if reward < 0: penalties += 1

            epochs += 1
            if epochs % 100000 == 0:
                print(f"Epochs : {epochs}", "states initial->last->current", initial_state, last_state, state)

        if episode % 1000 == 0:
            print(f"Episode: {episode}")

        total_penalties += penalties
        total_epochs += epochs

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")
