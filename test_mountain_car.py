import gym
from keras.models import load_model
import numpy as np
model = load_model("car_model.h5")

env = gym.make("MountainCar-v0")

for i in range(100):
    state = env.reset()
    count = 0
    while True:
        env.render()
        count += 1
        action = np.argmax(model.predict(state.reshape(-1, 2)))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            print("游戏的次数:", count)
            break
