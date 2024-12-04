from typing import Optional
import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
from generator import MazeGenerator
import networkx as nx

class LabyrinthEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, size: int = 10, seed: Optional[int] = None):
        self.seed(seed)
        self.size = size
        self._current_location = None
        self._old_location = None
        self.start_location = np.array([1, 1], dtype=np.int64)
        self.target_location = np.array([size - 2, size - 2], dtype=np.int64)

        self.generator = MazeGenerator(size=self.size)
        self.maze = self.generator.random_maze()
        
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, -self.size, -self.size]),
            high=np.array([1, 1, 1, 1, self.size, self.size]),
            dtype=np.int64
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

        self.max_steps = size * 10  
        self.current_step = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # self.maze = self.generator.random_maze()

        self._agent_location = self.start_location.copy()
        self._agent_trail = []
        self.current_step = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_info(self):
        reached_goal = np.array_equal(self._agent_location, self.target_location)

        info = {
            'reached_goal': reached_goal,
        }
        return info

    def _get_obs(self):
        relative_goal = self.target_location - self._agent_location
        current_pos = self._agent_location

        left = self.maze[current_pos[0] - 1, current_pos[1]]
        right = self.maze[current_pos[0] + 1, current_pos[1]]
        up = self.maze[current_pos[0], current_pos[1] - 1]
        down = self.maze[current_pos[0], current_pos[1] + 1]

        observation = np.array([
            left,
            right,
            up,
            down,
            *relative_goal
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

        terminated = np.array_equal(self._agent_location, self.target_location)

        self._agent_trail.append(self._old_location.copy())
        self.current_step += 1

        if self.current_step >= self.max_steps:
            terminated = True

        reward = self.calculate_reward()
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def calculate_reward(self):
        if np.array_equal(self._agent_location, self.target_location):
            return 50
        elif np.array_equal(self._agent_location, self._old_location):
            return - 1
        else:
            path_len = self.shortest_path_length(self.maze, self._agent_location, self.target_location)
            return 1 / (path_len + 1) - 0.01  # Encourage shorter path and penalize steps

    def shortest_path_length(self, maze, start, goal):
        graph = nx.grid_graph(dim=[maze.shape[0], maze.shape[1]])
        for x in range(maze.shape[0]):
            for y in range(maze.shape[1]):
                if maze[x, y] == 1:
                    graph.remove_node((x, y))
        try:
            path_length = nx.shortest_path_length(graph, source=tuple(start), target=tuple(goal))
        except nx.NetworkXNoPath:
            print('No path found')
            path_length = float('inf')
        return path_length

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.seed(seed) 

        return [seed]

    def render(self, mode='console'):
        if mode == 'human':
            if self._figure is None:
                self._figure, self._ax = plt.subplots()
                plt.ion()
                plt.show()

            maze_rgb = np.zeros(self.maze.shape + (3,), dtype=float)

            maze_rgb[self.maze == 1] = [0, 0, 0]  # Black for walls
            maze_rgb[self.maze == 0] = [1, 1, 1]  # White for free spaces

            for pos in self._agent_trail:
                maze_rgb[pos[0], pos[1]] = [0.5, 0.5, 0.5]  # Grey for trail

            start = self.start_location
            maze_rgb[start[0], start[1]] = [0, 1, 0]  # Green for start

            goal = self.target_location
            maze_rgb[goal[0], goal[1]] = [1, 0, 0]  # Red for goal

            agent_pos = self._agent_location
            maze_rgb[agent_pos[0], agent_pos[1]] = [1.0, 0.5, 0.0]  # Orange for agent

            self._ax.clear()
            self._ax.imshow(maze_rgb, origin='lower')

            self._ax.set_xticks([])
            self._ax.set_yticks([])

            plt.draw()
            plt.pause(0.001)

        elif mode == 'console':
            maze_copy = self.maze.copy()

            maze_copy[tuple(self._agent_location)] = 2
            maze_copy[tuple(self.target_location)] = 3

            for row in maze_copy:
                line = ''.join(
                    '#' if cell == 1 else
                    'A' if cell == 2 else
                    'G' if cell == 3 else
                    ' '
                    for cell in row
                )
                print(line)
            print("\n")

    def close(self):
        if self._figure:
            plt.close(self._figure)
            self._figure = None
            self._ax = None
