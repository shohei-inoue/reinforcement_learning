from environments.group_guidance_environment import GroupGuidanceEnv
from agents.group_guidance_agent import GroupGuidanceAgent
import numpy as np

def main():
    init_position = np.array([10.0, 10.0])
    env = GroupGuidanceEnv(init_position)
    agent = GroupGuidanceAgent(env, init_position)
    agent.run(episodes=3, agent_steps=20)


if __name__ == '__main__':
    main()