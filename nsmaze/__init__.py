"""ЛР3: поиск новизны (novelty search) + NEAT для робота в лабиринте."""
from .maze import Maze, generate_grid_maze, open_maze
from .simulator import RobotParams, simulate, behaviors
from .novelty import NoveltyArchive
from .evolution import Params, RunResult, run, run_many, METHODS
