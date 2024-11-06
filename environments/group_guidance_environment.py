import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio
import os
from datetime import datetime


class GroupGuidanceEnv(gym.Env):
    """
    群誘導の学習環境
    """
    # レンダリングモードの指定
    metadata = {'render.modes': ['human', 'rgb_array']}
    # 描画サイズ 
    ENV_HEIGHT = 60
    ENV_WIDTH = 150
    # 探査パラメータ
    OUTER_BOUNDARY = 10.0
    INNER_BOUNDARY = 0.0
    MEAN = 0.0
    VARIANCE = 10.0

    def __init__(self, init_position: np.array):
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
        self.init_position = init_position  # 初期位置
        self.agent_position = self.init_position
        self.agent_trajectory = [self.agent_position.copy()] # agentの軌跡を保存(プロット用)
        self.frames = [] # シミュレーション保存用
    

    def reset(self, add_data: pd.DataFrame | None):
        """
        初期化関数
        """
        self.agent_position = self.init_position
        # 描画を初期化
        self.agent_trajectory = [self.agent_position.copy()]
        if add_data is not None:
            self.render(add_data)
        else:
            self.render()

        return self.agent_position
    

    def step(self, action):
        """
        現在の状態から行動を経て次の状態に遷移
        """
        # 角度を取得
        theta = action[0]

        # 現在の位置の更新
        dx = self.OUTER_BOUNDARY * np.cos(np.radians(theta))
        dy = self.OUTER_BOUNDARY * np.sin(np.radians(theta))
        self.agent_position = self.next_position(dy, dx)

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
    

    def next_position(self, dy, dx):
        """
        障害物判定
        """
        # SAMPLING_NUM = 1000 # 軌跡線分のサンプリング数
        SAMPLING_NUM = max(150, int(np.ceil(np.linalg.norm([dy, dx]) * 10)))
        SAFE_DISTANCE = 1.0 # マップの安全距離

        for i in range(1, SAMPLING_NUM + 1):
            intermediate_position = np.array([
                self.agent_position[0] + (dy * i / SAMPLING_NUM),
                self.agent_position[1] + (dx * i / SAMPLING_NUM)
            ])

            # マップ内か判断
            if (0 < intermediate_position[0] < self.ENV_HEIGHT) and (0 < intermediate_position[1] < self.ENV_WIDTH):
                # サンプリング点が障害物でないか判断
                map_y = int(intermediate_position[0])
                map_x = int(intermediate_position[1])

                if self.map[map_y, map_x] == 1:
                    print(f"Obstacle collided at : {intermediate_position}")

                    # 障害物に衝突する事前位置を計算
                    collision_position = intermediate_position
                    direction_vector = collision_position - self.agent_position
                    norm_direction_vector = np.linalg.norm(direction_vector)

                    # if norm_direction_vector > SAFE_DISTANCE:
                    #     stop_position = self.agent_position + (direction_vector / norm_direction_vector) * (norm_direction_vector - SAFE_DISTANCE)
                    #     return stop_position
                    # else:
                    #     # 移動距離が安全距離より短い場合はそのまま停止
                    #     return self.agent_position
                    stop_position = self.agent_position + (direction_vector / norm_direction_vector) * (norm_direction_vector - SAFE_DISTANCE)
                    return stop_position

            else:
                continue
        
        return self.agent_position + np.array([dy, dx])


    

    def render(self, add_data: list[pd.DataFrame] | None = None, save_frames = False, mode = 'rgb_array'):
        """
        レンダリング処理
        """
        plt.gcf().clf()
         # 図のサイズを指定
        # plt.figure(figsize=(10, 4))
         # 環境のマップを描画
        plt.imshow(
            self.map,
            cmap='gray_r', 
            origin='lower',
            extent=[0, self.ENV_WIDTH, 0, self.ENV_HEIGHT]
            )
        # 探査中心位置の描画
        plt.scatter(
            x=self.agent_position[1],
            y=self.agent_position[0],
            color='blue',
            s=100,
            label='Explore Center'
            )
        trajectory = np.array(self.agent_trajectory)
        plt.plot(
            trajectory[:, 1],
            trajectory[:, 0],
            color='blue',
            linewidth=1,
            label='Explore Center Trajectory',
            )
         # 探査領域を表示
        explore_area = Circle(
            (self.agent_position[1], self.agent_position[0]),
            self.OUTER_BOUNDARY, 
            color='black',
            fill=False,
            linewidth=1,
            label='Wall or Obstacle'
            )
        plt.gca().add_patch(explore_area)

        # REDの描画
        if add_data is not None:
            for data in add_data:
                plt.scatter(
                    x = data['x'].iloc[-1],
                    y = data['y'].iloc[-1],
                    color='red',
                    s=10,
                    # label='sub robots'
                )
                plt.plot(
                    data['x'],
                    data['y'],
                    color='gray',
                    linewidth='0.5',
                    alpha=0.5
                    # label='sub robots trajectory'
                )
                # plt.scatter(
                #     x = data['x'],
                #     y = data['y'],
                #     color='gray',
                #     s=2,
                #     alpha=0.3
                #     # label='sub robots'
                # )
        
        # 描画設定
        plt.xlim(0, self.ENV_WIDTH)
        plt.ylim(self.ENV_HEIGHT, 0)
        plt.title('Explore Environment')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(False)
        if add_data is None:
            plt.legend()

        plt.draw()
        plt.pause(0.001)

        if save_frames:
            filename = f"frame_{len(self.frames)}.png"
            plt.savefig(filename)
            self.frames.append(filename)
            print(f"frame named: {filename}")
        # print("Agent position: {}".format(self.agent_position))


    def save_gif(self, episode, gif_name=None):
        """
        保存したフレームをGIFに変換
        """
        now = datetime.now()
        date_time_str = now.strftime("%Y%m%d_%H%M%S")
        if gif_name is None:
            gif_name = f"{date_time_str}_episode{episode}.gif"

        images = [imageio.imread(frame) for frame in self.frames]
        imageio.mimsave(f"gif/{gif_name}", images, duration=0.1)

        # 一時ファイルの削除
        for frame in self.frames:
            os.remove(frame)
        self.frames = []

    
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
        self.map[20, :20] = 1
        self.map[30:, 40] = 1
        self.map[:40, 70] = 1
        self.map[20:40, 100] = 1
    

    def _close(self):
        pass


    def _seed(self, seed=None):
        pass








