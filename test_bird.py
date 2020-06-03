import numpy as np
from keras.models import load_model
from ple import PLE
from ple.games import FlappyBird
import time


class Agent():
    def __init__(self, action_set):
        self.model = self.init_netWork()
        self.action_set = action_set

    def get_state(self, state):
        """
        提取游戏state中我们需要的数据
        :param state: 游戏state
        :return: 返回提取好的数据
        """
        return_state = np.zeros((3,))
        dist_to_pipe_horz = state["next_pipe_dist_to_player"]
        dist_to_pipe_bottom = state["player_y"] - state["next_pipe_top_y"]
        velocity = state['player_vel']
        return_state[0] = dist_to_pipe_horz
        return_state[1] = dist_to_pipe_bottom
        return_state[2] = velocity
        return return_state

    def init_netWork(self):
        """
        构建模型
        :return:
        """
        return load_model("./tmp_model1500.h5")

    def get_best_action(self, state):
        return np.argmax(self.model.predict(state.reshape(-1, 3)))

    def act(self, p, action):
        """
        执行动作
        :param p: 通过p来向游戏发出动作命令
        :param action: 动作
        :return: 奖励
        """
        r = p.act(self.action_set[action])
        return r


if __name__ == "__main__":
    # 训练次数
    episodes = 20000
    # 实例化游戏对象
    game = FlappyBird()
    # 类似游戏的一个接口，可以为我们提供一些功能
    p = PLE(game, fps=30, display_screen=True)
    # 初始化
    p.init()
    # 实例化Agent，将动作集传进去
    agent = Agent(p.getActionSet())

    for episode in range(episodes):
        # 重置游戏
        p.reset_game()
        # 获得状态
        state = agent.get_state(game.getGameState())

        while True:
            # 获得最佳动作
            action = agent.get_best_action(state)
            # 然后执行动作获得奖励
            reward = agent.act(p, action)
            # 获得执行动作之后的状态
            next_state = agent.get_state(game.getGameState())
            state = next_state
            if p.game_over():
                print("当前分数为{}".format(p.score()))
                break
            # 让小鸟慢一点
            time.sleep(0.02)
