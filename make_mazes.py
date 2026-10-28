import argparse
from pathlib import Path

from nsmaze.maze import MAZES_DIR, Maze, generate_grid_maze, open_maze
from nsmaze.cli_utils import write_csv
from nsmaze.plots import figure_mazes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5, 6], help="размеры сетки N×N")
    ap.add_argument("--cell", type=float, default=40.0, help="ширина коридора")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--braid", type=float, default=0.0, help="доля случайно удаляемых стен (петли)")
    ap.add_argument("--exit-mode", choices=["deceptive", "far"], default="deceptive")
    ap.add_argument("--out", default=str(MAZES_DIR))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mazes = [open_maze()]
    mazes[0].save(out / "open.txt")
    for n in a.sizes:
        m = generate_grid_maze(n, n, cell=a.cell, seed=a.seed, braid=a.braid, exit_mode=a.exit_mode,
                               name=f"grid{n}x{n}")
        m.save(out / f"{m.name}.txt")
        mazes.append(m)
    for name in ("medium", "hard"):
        mazes.append(Maze.load(MAZES_DIR / f"{name}.txt"))

    rows = []
    for m in mazes:
        d = m.difficulty()
        rows.append({"maze": m.name, **d})
        print(f"{m.name:10s} стен {d['walls']:3d}  путь {d['path']:7.1f}  по прямой {d['euclid']:6.1f}  "
              f"обманчивость {d['deception']}")
    write_csv(rows, out / "difficulty.csv")
    figure_mazes(mazes, out / "mazes.png")
    print(f"Лабиринты и галерея сохранены в {out}")


if __name__ == "__main__":
    main()
