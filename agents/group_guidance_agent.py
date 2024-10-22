from environments.group_guidance_environment import GroupGuidanceEnv
import red
import numpy as np
import math

class GroupGuidanceAgent():
    """
    群誘導のエージェント
    """
    RED_NUM = 10
    RED_STEPS = 100

    def __init__(self, env: GroupGuidanceEnv, init_position: np.array):
        """
        GroupGuidanceAgentのコンストラクタ
        """
        self.env: GroupGuidanceEnv = env
        self.reds: list[red.Red] = []
        for i in range(self.RED_NUM):
            self.reds.append(
                red.Red(
                    id=f'red_{i}', 
                    env=self.env, 
                    agent_position=init_position,
                    x=init_position[1] + 5.0 * math.cos((2 * math.pi / (self.RED_NUM - 1)) * (i - 1)),
                    y=init_position[0] + 5.0 * math.sin((2 * math.pi / (self.RED_NUM - 1)) * (i - 1)),
                    ))

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
            print(f"Episode {episode+1} / {episodes}")
            state = self.env.reset()

            # 1episodeごとの行動
            for step in range(agent_steps):

                # REDの行動
                for _ in range(self.RED_STEPS):
                    for i in range(len(self.reds)):
                        self.reds[i].step_motion()

                # エージェントの学習, 行動
                action = self.policy()
                state, reward, done, _ = self.env.step(action)
                print(f"Step {step+1}: Action={action}, State={state}, Reward={reward}")

                # TODO REDのプロットを考える
                self.env.render()

                if done:
                    print("Episode finished")
                    break


    def log(self):
        pass


    def init_log(self):
        pass