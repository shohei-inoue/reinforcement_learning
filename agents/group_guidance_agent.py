from environments.group_guidance_environment import GroupGuidanceEnv
from agents.red import Red
import pandas as pd
import numpy as np
import math

class GroupGuidanceAgent():
    """
    群誘導のエージェント
    """
    RED_NUM = 5
    RED_STEPS = 50
    SAVE_GIF = True

    def __init__(self, env: GroupGuidanceEnv, init_position: np.array):
        """
        GroupGuidanceAgentのコンストラクタ
        """
        self.env: GroupGuidanceEnv = env
        self.init_position = init_position


    def policy(self):
        """
        戦略
        TODO (仮置きとしてランダムな行動選択)
        """
        return self.env.action_space.sample()


    def run(self, episodes=10, agent_steps=50):
        """
        学習実行
        """
        for episode in range(episodes):
             # REDの初期化
            self.reds: list[Red] = []
            self.reds_data: list[pd.DataFrame] = []

            for i in range(self.RED_NUM):
                x_coordinate = self.init_position[1] + 5.0 * math.cos((2 * math.pi * i / (self.RED_NUM)))
                y_coordinate = self.init_position[0] + 5.0 * math.sin((2 * math.pi * i / (self.RED_NUM)))
                self.reds.append(
                    Red(
                        id=f'red_{i}', 
                        env=self.env, 
                        agent_position=self.init_position,
                        x=x_coordinate,
                        y=y_coordinate
                        ))
                self.reds_data.append(self.reds[i].data)

            print(f"Episode {episode+1} / {episodes}")
            state = self.env.reset(self.reds_data)

            # 1episodeごとの行動
            for step in range(agent_steps):

                # REDの行動
                for _ in range(self.RED_STEPS):
                    for i in range(len(self.reds)):
                        self.reds[i].step_motion()

                        # TODO REDのプロットを考える
                        self.reds_data[i] = self.reds[i].data
                    self.env.render(self.reds_data, self.SAVE_GIF)

                # エージェントの学習, 行動
                action = self.policy()
                state, reward, done, _ = self.env.step(action)
                
                # エージェントの状態の変化をredに反映
                for i in range(len(self.reds)):
                    self.reds[i].change_agent_state(self.env, state)

                print(f"Step {step+1}: Action={action}, State={state}, Reward={reward}")

                # self.env.render()

                if done:
                    print("Episode finished")
                    break
            
            if self.SAVE_GIF:
                self.env.save_gif(episode + 1)


    def log(self):
        pass


    def init_log(self):
        pass