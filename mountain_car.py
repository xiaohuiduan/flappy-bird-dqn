import keras
import random
from collections import deque

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential


class Agent():
    def __init__(self, action_set, observation_space):
        """
        初始化
        :param action_set: 动作集合
        :param observation_space: 环境属性，我们需要使用它得到state的shape
        """
        # 奖励衰减
        self.gamma = 1.0
        # 从经验池中取出数据的数量
        self.batch_size = 50
        # 经验池
        self.memory = deque(maxlen=2000000)
        # 贪婪
        self.greedy = 1.0
        # 动作集合
        self.action_set = action_set
        # 环境的属性
        self.observation_space = observation_space
        # 神经网路模型
        self.model = self.init_netWork()

    def init_netWork(self):
        """
        构建模型
        :return: 模型
        """
        model = Sequential()
        # self.observation_space.shape[0]，state的变量的数量
        model.add(Dense(64 * 4, activation="tanh", input_dim=self.observation_space.shape[0]))
        model.add(Dense(64 * 4, activation="tanh"))
        # self.action_set.n 动作的数量
        model.add(Dense(self.action_set.n, activation="linear"))
        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=keras.optimizers.RMSprop(lr=0.001))
        return model

    def train_model(self):
        # 从经验池中随机选择部分数据
        train_sample = random.sample(self.memory, k=self.batch_size)

        train_states = []
        next_states = []
        for sample in train_sample:
            cur_state, action, r, next_state, done = sample
            next_states.append(next_state)
            train_states.append(cur_state)

        # 转成np数组
        next_states = np.array(next_states)
        train_states = np.array(train_states)
        # 得到next_state的q值
        next_states_q = self.model.predict(next_states)

        # 得到state的预测值
        state_q = self.model.predict_on_batch(train_states)

        # 计算Q现实
        for index, sample in enumerate(train_sample):
            cur_state, action, r, next_state, done = sample
            if not done:
                state_q[index][action] = r + self.gamma * np.max(next_states_q[index])
            else:
                state_q[index][action] = r

        self.model.train_on_batch(train_states, state_q)

    def add_memory(self, sample):
        self.memory.append(sample)

    def update_greedy(self):
        # 小于最小探索率的时候就不进行更新了。
        if self.greedy > 0.01:
            self.greedy *= 0.995

    def act(self, env, action):
        """
        执行动作
        :param env: 执行环境
        :param action: 执行的动作
        :return: ext_state, reward, done
        """
        next_state, reward, done, _ = env.step(action)

        if done:
            if reward < 0:
                reward = -100
            else:
                reward = 10
        else:
            if next_state[0] >= 0.4:
                reward += 1

        return next_state, reward, done

    def get_best_action(self, state):
        if random.random() < self.greedy:
            return self.action_set.sample()
        else:
            return np.argmax(self.model.predict(state.reshape(-1, 2)))


if __name__ == "__main__":
    # 训练次数
    episodes = 10000
    # 实例化游戏环境
    env = gym.make("MountainCar-v0")
    # 实例化Agent
    agent = Agent(env.action_space, env.observation_space)
    # 游戏中动作执行的次数（最大为200）
    counts = deque(maxlen=10)

    for episode in range(episodes):
        count = 0
        # 重置游戏
        state = env.reset()

        # 刚开始不立即更新探索率
        if episode >= 5:
            agent.update_greedy()

        while True:
            count += 1
            # 获得最佳动作
            action = agent.get_best_action(state)
            next_state, reward, done = agent.act(env, action)
            agent.add_memory((state, action, reward, next_state, done))
            # 刚开始不立即训练模型，先填充经验池
            if episode >= 5:
                agent.train_model()
            state = next_state
            if done:
                # 将执行的次数添加到counts中
                counts.append(count)
                print("在{}轮中，agent执行了{}次".format(episode + 1, count))

                # 如果近10次，动作执行的平均次数少于160，则保存模型并退出
                if len(counts) == 10 and np.mean(counts) < 160:
                    agent.model.save("car_model.h5")
                    exit(0)
                break
