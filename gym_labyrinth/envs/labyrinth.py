from typing import Optional
import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
from .generator import MazeGenerator

class LabyrinthEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, size: int = 10, seed: Optional[int] = None):
        self.seed(seed)
        self.size = size
        self._current_location = None
        self._old_location = None
        self.start_location = np.array([1, 1], dtype=np.int64)
        self.target_location = np.array([size - 2, size - 2], dtype=np.int64)

        self.maze = MazeGenerator(size=self.size).random_maze()

        self.observation_space = gym.spaces.Box(
            low=0, high=max(self.size - 1, 1), shape=(6,), dtype=np.int64
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

        self._figure = None
        self._ax = None
        self._agent_trail = []

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._agent_location = self.start_location.copy()
        self._agent_trail = []

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_info(self):
        return {}

    def _get_obs(self):
        current_pos = self._agent_location

        left = self.maze[current_pos[0] - 1, current_pos[1]]
        right = self.maze[current_pos[0] + 1, current_pos[1]]
        up = self.maze[current_pos[0], current_pos[1] - 1]
        down = self.maze[current_pos[0], current_pos[1] + 1]

        observation = np.array([
            self._agent_location[0],
            self._agent_location[1],
            left,
            right,
            up,
            down
        ], dtype=np.int64)
        return observation

    def step(self, action):
        action = int(action)
        direction = self._action_to_direction[action]

        self._old_location = self._agent_location.copy()

        new_location = self._agent_location + direction

        if (0 <= new_location[0] < self.maze.shape[0]) and (0 <= new_location[1] < self.maze.shape[1]):
            if self.maze[new_location[0], new_location[1]] == 0:
                self._agent_location = new_location
            else:
                pass
        else:
            pass

        self._agent_trail.append(self._old_location.copy())

        terminated = np.array_equal(self._agent_location, self.target_location)
        reward = self.calculate_reward()
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def calculate_reward(self):
        if np.array_equal(self._agent_location, self.target_location):
            reward = +1
        elif np.array_equal(self._agent_location, self._old_location):
            reward = -1
        else:
            reward = -0.01

        return reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def render(self, mode='human'):
        if self._figure is None:
            self._figure, self._ax = plt.subplots()
            plt.ion()
            plt.show()

        maze_rgb = np.zeros(self.maze.shape + (3,), dtype=float)

        maze_rgb[self.maze == 1] = [0, 0, 0]  # Black
        maze_rgb[self.maze == 0] = [1, 1, 1]  # White
       
        for pos in self._agent_trail:
            maze_rgb[pos[0], pos[1]] = [0.5, 0.5, 0.5]  # Grey

        start = self.start_location
        maze_rgb[start[0], start[1]] = [0, 1, 0]  # Green

        goal = self.target_location
        maze_rgb[goal[0], goal[1]] = [1, 0, 0]  # Red

        agent_pos = self._agent_location
        maze_rgb[agent_pos[0], agent_pos[1]] = [1.0, 0.5, 0.0]  # Orange

        self._ax.clear()
        self._ax.imshow(maze_rgb, origin='lower')

        self._ax.set_xticks([])
        self._ax.set_yticks([])

        plt.draw()
        plt.pause(0.001)

    def close(self):
        if self._figure:
            plt.close(self._figure)
            self._figure = None
            self._ax = None
