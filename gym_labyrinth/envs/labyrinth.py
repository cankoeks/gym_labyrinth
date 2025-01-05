from typing import Optional
import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
from .generator import MazeGenerator
import networkx as nx
import random
import time

class LabyrinthEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'console'],
        'render_fps': 30
    }

    def __init__(self, size: int = 10, seed: Optional[int] = 0, maze_type: str = 'random', sparse_rewards: bool = False):
        self.action_space = gym.spaces.Discrete(4)
        self.seed(seed)
        self.size = size
        self._current_location = None
        self._old_location = None
        self.start_location = np.array([1, 1], dtype=np.int64)
        self.target_location = np.array([size - 2, size - 2], dtype=np.int64)
        self.sparse_rewards = sparse_rewards

        self.generator = MazeGenerator(size=self.size)
        if maze_type == 'random':
            self.maze = self.generator.random_maze()
        elif maze_type == 'empty':
            self.maze = self.generator.empty_maze()

        self.path_lengths = self.compute_shortest_path_lengths(self.maze, self.target_location)
        self.shortest_path = self.compute_shortest_path(self.start_location, self.target_location)

        self.maze[self.target_location[0], self.target_location[1]] = 3

        if not self.sparse_rewards:
            for pos in self.shortest_path[:-1]:
                if tuple(pos) != tuple(self.start_location):
                    self.maze[pos[0], pos[1]] = 2
            
        self.initial_maze = self.maze.copy()

        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, -self.size, -self.size]),
            high=np.array([3, 3, 3, 3, self.size, self.size]),
            dtype=np.int64
        )

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

        self.sparse_rewards = sparse_rewards

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.maze = self.initial_maze.copy()

        self.collected_rewards = set()
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
            'agent_location': self._agent_location,
        }
        return info

    def _get_obs(self):
        current_pos = self._agent_location

        def cell_state(x, y):
            if 0 <= x < self.size and 0 <= y < self.size:
                return self.maze[x, y]
            else:
                return 1

        left = cell_state(current_pos[0] - 1, current_pos[1])
        right = cell_state(current_pos[0] + 1, current_pos[1])
        up = cell_state(current_pos[0], current_pos[1] - 1)
        down = cell_state(current_pos[0], current_pos[1] + 1)

        observation = np.array([
            left,
            right,
            up,
            down,
            current_pos[0],
            current_pos[1]
        ], dtype=np.int64)
        return observation

    def step(self, action):
        action = int(action)
        direction = self._action_to_direction[action]

        self._old_location = self._agent_location.copy()
        new_location = self._agent_location + direction

        if (0 <= new_location[0] < self.maze.shape[0]) and (0 <= new_location[1] < self.maze.shape[1]):
            if self.maze[new_location[0], new_location[1]] in (0, 2, 3):
                self._agent_location = new_location

        terminated = np.array_equal(self._agent_location, self.target_location)
        self._agent_trail.append(self._old_location.copy())
        self.current_step += 1
        if self.current_step >= self.max_steps:
            terminated = True

        reward = self.calculate_reward()

        agent_pos = tuple(self._agent_location)
        if self.maze[agent_pos[0], agent_pos[1]] == 2 and agent_pos in self.collected_rewards:
            self.maze[agent_pos[0], agent_pos[1]] = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def calculate_reward(self):
        agent_pos = tuple(self._agent_location)

        if np.array_equal(self._agent_location, self.target_location):
            return 100

        if not self.sparse_rewards:
            if self.maze[agent_pos[0], agent_pos[1]] == 2 and agent_pos not in self.collected_rewards:
                self.collected_rewards.add(agent_pos)
                return 10
        
        return -1

    def compute_shortest_path(self, start, goal):
        graph = nx.grid_graph(dim=[self.maze.shape[0], self.maze.shape[1]])
        for x in range(self.maze.shape[0]):
            for y in range(self.maze.shape[1]):
                if self.maze[x, y] == 1:
                    graph.remove_node((x, y))
        try:
            path = nx.shortest_path(graph, source=tuple(start), target=tuple(goal))
        except nx.NetworkXNoPath:
            path = []
        return path

    def compute_shortest_path_lengths(self, maze, goal):
        graph = nx.grid_graph(dim=[maze.shape[0], maze.shape[1]])
        for x in range(maze.shape[0]):
            for y in range(maze.shape[1]):
                if maze[x, y] == 1:
                    graph.remove_node((x, y))
        path_lengths = nx.single_source_shortest_path_length(graph, source=tuple(goal))
        return path_lengths

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)
        return [seed]

    def render(self, mode='human'):
        if mode == 'human':
            if self._figure is None:
                self._figure, self._ax = plt.subplots()
                plt.ion()
                plt.show()

            maze_rgb = np.zeros(self.maze.shape + (3,), dtype=float)

            # Color scheme:
            # 1: wall (black)
            # 0: free (white)
            # 2: shortest path (yellow)
            # 3: goal (red)
            maze_rgb[self.maze == 1] = [0, 0, 0]
            maze_rgb[self.maze == 0] = [1, 1, 1]
            maze_rgb[self.maze == 2] = [0.8, 0.8, 0.2]
            maze_rgb[self.maze == 3] = [1, 0, 0]

            # Mark the agent's trail in grey
            for pos in self._agent_trail:
                maze_rgb[pos[0], pos[1]] = [0.5, 0.5, 0.5]

            # Mark the start in green
            start = self.start_location
            maze_rgb[start[0], start[1]] = [0, 1, 0]

            # Mark the agent in orange
            agent_pos = self._agent_location
            maze_rgb[agent_pos[0], agent_pos[1]] = [1.0, 0.5, 0.0]

            self._ax.clear()
            self._ax.imshow(maze_rgb, origin='lower')
            self._ax.set_xticks([])
            self._ax.set_yticks([])

            plt.draw()
            plt.pause(0.001)

        elif mode == 'console':
            maze_copy = self.maze.copy()
            maze_copy[self._agent_location[0], self._agent_location[1]] = 4

            for row in maze_copy:
                line = ''.join(
                    '#' if cell == 1 else
                    '.' if cell == 2 else
                    'G' if cell == 3 else
                    'A' if cell == 4 else
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

def main():
    env = LabyrinthEnv(size=8, sparse_rewards=True)

    while True:
        obs, info = env.reset()
        done = False
        
        while not done:
            env.render('human')

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            print(f"Reward: {reward:.1f}")
            time.sleep(0.1)

if __name__ == "__main__":
    main()
