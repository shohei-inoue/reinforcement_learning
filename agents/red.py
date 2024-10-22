from environments.group_guidance_environment import GroupGuidanceEnv

import pandas  as pd
import numpy as np
import random
import math

class Red():
    """
    REDの確率密度制御を模倣したクラス
    x                   : x座標
    y                   : y座標
    point               : 座標
    amount_of_movement  : 移動量
    direction_angle     : ロボット正面からの移動角度
    """
    MIN_MOVEMENT = 2.0
    MAX_MOVEMENT = 3.0
    MAX_BOIDS_MOVEMENT = 3.0
    MIN_BOIDS_MOVEMENT = 2.0

    def __init__(
            self,
            id: str,
            env: GroupGuidanceEnv,
            agent_position,
            x: float = 0.0,
            y: float = 0.0, 
            step: int = 0, 
            amount_of_movement: float = 0.0, 
            direction_angle: float = 0.0, 
            collision_flag: bool = False, 
            boids_flag: int = 0, 
            estimated_probability: float = 0.0, 
            data: pd.DataFrame = pd.DataFrame(),):
        """
        REDのコンストラクタ
        """
        self.id = id
        self.env = env
        self.agent_position = agent_position
        self.step = step
        self.x = x
        self.y = y
        self.point = np.array([self.y, self.x])
        self.amount_of_movement = amount_of_movement
        self.direction_angle = direction_angle
        self.distance = np.linalg.norm(self.point - agent_position)
        self.azimuth = self.azimuth_adjustment(agent_position)
        self.collision_flag = collision_flag
        self.boids_flag = boids_flag
        self.estimated_probability = estimated_probability
        self.data = data
    

    def get_arguments(self):
        """
        データをデータフレーム形式にする
        """
        return pd.DataFrame({'id': [self.id],
                             'step': [self.step], 
                             'x': [self.x], 
                             'y': [self.y], 
                             'point': [self.point], 
                             'amount_of_movement': [self.amount_of_movement], 
                             'direction_angle': [self.direction_angle], 
                             'collision_flag': [self.collision_flag],
                             'boids_flag': [self.boids_flag],
                             'estimated_probability': [self.estimated_probability],
                             'distance': [None], 
                             'azimuth': [None],
                             })
    

    def azimuth_adjustment(self):
        """
        探査中心の方向角を計算
        """
        azimuth = 0.0
        if self.x != self.agent_position[1]:
            vec_d = np.array(self.point - self.agent_position)
            vec_x = np.array([0, self.x - self.agent_position[1]]) # TODO

            azimuth = np.rad2deg(math.acos(vec_d @ vec_x / (np.linalg.norm(vec_d) * np.linalg.norm(vec_x))))
        
        if self.x - self.agent_position[1] < 0:
            if self.y - self.agent_position[0] >= 0:
                azimuth = np.rad2deg(math.pi) - azimuth
            else:
                azimuth += np.rad2deg(math.pi)
        
        else:
            if self.y - self.agent_position[0] < 0:
                azimuth = np.rad2deg(2.0 * math.pi) - azimuth
        
        return azimuth


    def avoidance_behavior(self):
        """
        障害物回避行動
        """
        self.direction_angle = (self.direction_angle + random.uniform(90.0, 270.0)) % np.rad2deg(math.pi * 2.0)
        amount_of_movement = random.uniform(self.MIN_MOVEMENT, self.MAX_MOVEMENT)
        dx = amount_of_movement * math.cos(math.radians(self.direction_angle))
        dy = amount_of_movement * math.sin(math.radians(self.direction_angle))
        prediction_point = np.array([self.y + dy, self.x + dx])
        return prediction_point
    

    def forward_behavior(self, dy, dx):
        """
        直進行動処理
        """
        SAMPLING_NUM = max(150, int(np.ceil(np.linalg.norm([dy, dx]) * 10)))
        SAFE_DISTANCE = 1.0 # マップの安全距離

        for i in range(1, SAMPLING_NUM + 1):
            intermediate_position = np.array([
                self.agent_position[0] * (dy * i / SAMPLING_NUM),
                self.agent_position[1] * (dx * i / SAMPLING_NUM)
            ])

            if (0 < intermediate_position[0] < self.env.ENV_HEIGHT) and (0 < intermediate_position[1] < self.env.ENV_WIDTH):
                map_y = int(intermediate_position[0])
                map_x = int(intermediate_position[1])

                if self.env.map[map_y, map_x] == 1:
                    print(f"Obstacle collided at : {intermediate_position}")

                    # 障害物に衝突する事前位置を計算
                    collision_position = intermediate_position
                    direction_vector = collision_position - self.agent_position
                    norm_direction_vector = np.linalg.norm(direction_vector)

                    stop_position = self.agent_position + (direction_vector / norm_direction_vector) * (norm_direction_vector - SAFE_DISTANCE)

                    self.collision_flag = True

                    return stop_position
            else:
                continue
        
        self.collision_flag = False

        return self.agent_position + np.array([dy, dx])
    

    def boids_judgement(self):
        """
        boids行動を行うか判断する
        """
        if self.distance > self.env.OUTER_BOUNDARY:
            self.boids_flag = 1
        elif self.distance < self.env.INNER_BOUNDARY:
            self.boids_flag = 2
        else:
            self.boids_flag = 0
    

    def boids_behavior(self):
        """
        boids行動
        """
        self.direction_angle = self.azimuth
        if self.boids_flag == 1:
            if self.y - self.agent_position[0]:
                self.direction_angle += np.rad2deg(math.pi)
            else:
                self.direction_angle -= np.rad2deg(math.pi)
        
        amount_of_movement = random.uniform(self.MIN_BOIDS_MOVEMENT, self.MAX_BOIDS_MOVEMENT)
        dx = amount_of_movement * math.cos(math.radians(self.direction_angle))
        dy = amount_of_movement * math.sin(math.radians(self.direction_angle))
        prediction_point = np.array([self.y + dy, self.x + dx])
        return prediction_point
    

    def rejection_decision(self):
        """
        メトロポリス法による棄却決定
        """
        def distribution(distance, mean, variance):
            """
            正規分布
            """
            return 1 / math.sqrt(2 * math.pi) * math.exp(-(distance - mean) ** 2 / (2 * variance ** 2))
        

        while True:
            direction_angle = np.rad2deg(random.uniform(0.0, 2.0 * math.pi))
            amount_of_movement = random.uniform(self.MIN_MOVEMENT, self.MAX_MOVEMENT)
            dx = amount_of_movement * math.cos(math.radians(direction_angle))
            dy = amount_of_movement * math.sin(math.radians(direction_angle))
            prediction_point = np.array([self.y + dy, self.x + dx])
            distance = np.linalg.norm(prediction_point - self.agent_position)
            estimated_probability = distribution(distance, self.env.MEAN, self.env.VARIANCE)
            if self.estimated_probability == 0.0:
                self.estimated_probability = estimated_probability
                self.direction_angle = direction_angle
                return prediction_point
            else:
                continue

    
    def step_motion(self):
        """
        行動制御
        """
        if self.collision_flag:
            prediction_point = self.avoidance_behavior()
        else:
            self.boids_judgement()
            if self.boids_flag:
                prediction_point = self.boids_behavior()
            else:
                prediction_point = self.rejection_decision()
        
        self.point = self.forward_behavior(
            prediction_point[0] - self.point[0],
            prediction_point[1] - self.point[1]
            )
        
        self.y = self.point[0]
        self.x = self.point[1]
        self.distance = np.linalg.norm(self.point - self.agent_position)
        self.azimuth = self.azimuth_adjustment()
        self.step += 1

        self.data = pd.concat([self.data, self.get_arguments()])
