from gymnasium.envs.registration import register

register(
    id="Labyrinth-v0",
    entry_point="gym_labyrinth.envs:LabyrinthEnv",
    kwargs={"size": 10, "seed": 1, "maze_type": "random"},
)
