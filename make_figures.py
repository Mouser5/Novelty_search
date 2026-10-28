import argparse
from collections import defaultdict
from pathlib import Path

from nsmaze import plots
from nsmaze.cli_utils import write_csv
from nsmaze.evolution import RunResult
from nsmaze.maze import Maze


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="папка с результатами run_experiment.py")
    a = ap.parse_args()
    root = Path(a.folder)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(lambda: defaultdict(list))
    for f in sorted(root.glob("*/*/*.pkl")):
        r = RunResult.load(f)
        groups[r.maze][r.params["method"]].append(r)
    if not groups:
        raise SystemExit(f"в {root} не найдено файлов */*/seed*.pkl")
    rows = []
    for maze_name, by_method in groups.items():
        maze = Maze.load(maze_name)
        for method, res in by_method.items():
            res.sort(key=lambda r: r.seed)
            rows += [r.summary() for r in res]
            for r in res:
                plots.figure_run(r, maze, fig_dir / f"run_{maze_name}_{method}_seed{r.seed}.png")
        plots.figure_heatmaps_compare(dict(by_method), maze, fig_dir / f"heatmaps_{maze_name}.png")
    plots.figure_methods_bars(rows, fig_dir / "methods.png")
    write_csv(rows, root / "summary.csv")
    print(f"Рисунки: {fig_dir}")


if __name__ == "__main__":
    main()
