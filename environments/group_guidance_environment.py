import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class GroupGuidanceEnv(gym.Env):
    """
    群誘導の学習環境
    """
    # レンダリングモードの指定
    metadata = {'render.modes': ['human', 'rgb_array']}
    # 描画サイズ 
    ENV_HEIGHT = 60
    ENV_WIDTH = 150

    def __init__(self):
        """
        GroupGuidanceEnvのコンストラクタ
        行動空間：θ(0~360)
        状態空間:(x(無限), y(無限))
        報酬範囲：(0~1)(仮置き)
        """
        super(GroupGuidanceEnv, self).__init__()
        self.map = np.zeros((self.ENV_HEIGHT, self.ENV_WIDTH)) # マップ配列生成
        self.generate_obstacles() # 障害物の生成
        self.action_space = gym.spaces.Box(
            low=np.array([0.0]),
            high=np.array([360.0]),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low = np.array([-np.inf, -np.inf]),
            high = np.array([np.inf, np.inf]),
            dtype=np.float32
        )
        self.reward_range = (0, 1) # TODO 報酬幅仮置き
        # TODO 1.外部入力から決定に変更するかも?
        self.agent_position = np.array([0.0, 0.0]) # 初期位置
        self.r = 10 # outer_boundary(frontier_based)
        self.agent_trajectory = [self.agent_position.copy()] # agentの軌跡を保存(プロット用)
    

    def reset(self):
        """
        初期化関数
        """
        # TODO 1.外部入力から決定に変更するかも?
        self.agent_position = np.array([0.0, 0.0])
        return self.agent_position
    

    def step(self, action):
        """
        現在の状態から行動を経て次の状態に遷移
        """
        # 角度を取得
        theta = action[0]

        dx = self.r * np.cos(np.radians(theta))
        dy = self.r + np.sin(np.radians(theta))

         # 現在の位置の更新
        self.agent_position += np.array([dx, dy])

        # 軌跡用にagentの位置を追加
        self.agent_trajectory.append(self.agent_position.copy())

        # 観測を新しい位置に
        observation = self.agent_position

        # 報酬を計算
        # TODO 2.報酬の計算を変更(現在仮置き)
        reward = -np.linalg.norm(self.agent_position)

        # 終了条件
        # TODO 3. 終了条件を決定する
        done = False

        return observation, reward, done, {}
    

    def render(self, mode='rgb_array'):
        """
        レンダリング処理
        """
         # 図のサイズを指定
        plt.figure(figsize=(10, 4))
         # 環境のマップを描画
        plt.imshow(
            self.map,
            cmap='gray_r', 
            origin='upper',
            extent=[0, self.ENV_WIDTH, 0, self.ENV_HEIGHT]
            )
        # 探査中心位置の描画
        plt.scatter(
            self.agent_position[0],
            self.agent_position[1],
            color='red',
            s=100,
            label='Explore Center'
            )
        trajectory = np.array(self.agent_trajectory)
        plt.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color='blue',
            linewidth=1,
            label='Explore Center Trajectory',
            )
         # 探査領域を表示
        explore_area = Circle(
            self.agent_position,
            self.r, 
            color='black',
            fill=False,
            linewidth=1,
            label='Explore Area'
            )
        plt.gca().add_patch(explore_area)
        
        # 描画設定
        plt.xlim(0, self.ENV_WIDTH)
        plt.ylim(0, self.ENV_HEIGHT)
        plt.title('Explore Environment')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(False)
        plt.legend()

        plt.show()
        # print("Agent position: {}".format(self.agent_position))

    
    def generate_obstacles(self):
        """
        障害物の生成
        """
        # 壁の生成
        self.map[0, :] = 1 # 上辺
        self.map[self.ENV_HEIGHT - 1, :] = 1 # 下辺
        self.map[:, 0] = 1 # 左辺
        self.map[:, self.ENV_WIDTH - 1] = 1 # 右辺

        # TODO 他の障害物の生成








