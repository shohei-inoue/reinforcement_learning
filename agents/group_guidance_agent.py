from environments.group_guidance_environment import GroupGuidanceEnv

class GroupGuidanceAgent():
    """
    群誘導のエージェント
    """
    def __init__(self, env: GroupGuidanceEnv):
        """
        GroupGuidanceAgentのコンストラクタ
        """
        self.env = env

    def policy(self):
        """
        戦略
        TODO (仮置きとしてランダムな行動選択)
        """
        return self.env.action_space.sample()


    def run(self, episodes=10, steps=50):
        """
        学習実行
        """
        for episode in range(episodes):
            print(f"Episode {episode+1} / {episodes}")
            state = self.env.reset()

            # 1episodeごとの行動
            for step in range(steps):
                action = self.policy()
                state, reward, done, _ = self.env.step(action)
                print(f"Step {step+1}: Action={action}, State={state}, Reward={reward}")

                self.env.render()

                if done:
                    print("Episode finished")
                    break


    def log(self):
        pass


    def init_log(self):
        pass