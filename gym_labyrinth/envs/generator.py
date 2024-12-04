import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

class MazeGenerator:
    def __init__(self, size=10, start=np.array([1, 1]), goal= None ) -> None:
        self.start = start
        self.width = size
        self.height = size
        self.goal = np.array([size - 2, size - 2]) if goal is None else goal

    def random_maze(self, complexity=0.1, density=0.1):
        while True:
            width = (self.width // 2) * 2 + 1
            height = (self.height // 2) * 2 + 1
            maze = np.ones((height, width), dtype=int)

            stack = [self.start]
            maze[self.start[0], self.start[1]] = 0
            directions = np.array([(0, 2), (0, -2), (2, 0), (-2, 0)])

            while stack:
                x, y = stack.pop()  
                np.random.shuffle(directions)
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 1 <= nx < height - 1 and 1 <= ny < width - 1 and maze[nx, ny] == 1:
                        maze[nx, ny] = 0
                        maze[x + dx // 2, y + dy // 2] = 0
                        stack.append(np.array([nx, ny]))

            maze[height - 2, width - 2] = 0 
            num_walls = int(complexity * (height + width))
            num_passages = int(density * (height * width // 2))

            for _ in range(num_walls):
                x, y = np.random.randint(1, height - 2), np.random.randint(1, width - 2)
                if maze[x, y] == 0:
                    maze[x, y] = 1

            for _ in range(num_passages):
                x, y = np.random.randint(1, height - 2), np.random.randint(1, width - 2)
                if maze[x, y] == 1:
                    maze[x, y] = 0

            if self.validate_maze(maze):
                return maze
         
    def validate_maze(self, maze):
        height, width = maze.shape
        start = self.start
        goal = self.goal

        if maze[start[0], start[1]] == 1 or maze[goal[0], goal[1]] == 1:
            return False

        stack = [start]
        visited = set()

        while stack:
            x, y = stack.pop()

            if np.array_equal((x, y), goal):
                return True

            if (x, y) in visited:
                continue
            visited.add((x, y))

            for dx, dy in np.array([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and maze[nx, ny] == 0 and (nx, ny) not in visited:
                    stack.append(np.array([nx, ny]))

        return False

    def empty_maze(self):
        width = (self.width // 2) * 2 + 1
        height = (self.height // 2) * 2 + 1
        maze = np.zeros((height, width), dtype=int)

        maze[0, :] = 1
        maze[-1, :] = 1
        maze[:, 0] = 1
        maze[:, -1] = 1

        maze[self.start[0], self.start[1]] = 0
        maze[self.goal[0], self.goal[1]] = 2

        return maze

def visualize_maze(maze):
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap="binary", origin="upper")
    plt.xticks([])
    plt.yticks([])
    plt.title("Maze Visualization", fontsize=16)
    plt.show()

if __name__ == "__main__":
    maze = MazeGenerator(size=10).empty_maze()
    print(maze)