# 確率密度制御を用いた小型ロボット群の探査中心移動先決定のシミュレーション.
![20241106_211532_episode1](https://github.com/user-attachments/assets/9ed1efb1-7e01-4528-a17b-ad5fbddfa2a6)

## 使用環境
```
・python = 3.12.1
・numpy = 1.26.3
・gym = 0.26.2
・matplotlib = 3.8.3
・imageio = 2.36.0
```

### 環境構築
```
$ pip3 install numpy
$ pip3 install gym
$ pip3 install matplotlib
$ pip3 install imageio
```

## ディレクトリ構成
```
reinforcement_learning
    |
    | - environments
    |       | 
    |       | - group_guidance_environment.py
    |
    | - agents
    |       |
    |       | - group_guidance_agent.py
    |       | - red.py
    |
    | - gif
    |
    | - gitignore.txt
    | - README.md
```

## TODO
- azimuthの修正
- 終了条件の決定
- 報酬設計