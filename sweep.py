"""
    Характеристические карты: перебор одного или двух параметров модели.

Примеры:
    # влияние числа соседей k на сложном лабиринте
    python sweep.py --maze hard --param k --values 1 3 7 15 30 --runs 10

    # порог архива ρmin (без динамической подстройки, чтобы порог не менялся)
    python sweep.py --maze hard --param threshold --values 1 3 6 12 24 --dynamic-threshold false --runs 10

    # две оси: размер популяции × вероятность добавления нейрона
    python sweep.py --maze medium --param pop_size --values 50 100 250 \
                    --param2 node_add_prob --values2 0.03 0.1 0.3 --runs 5

    # описание поведения: точка остановки или траектория
    python sweep.py --maze hard --param behavior --values endpoint trajectory --threshold 20 --runs 10

Результаты: <out>/sweep.csv и рисунок <out>/sweep_<param>.png.
"""
import argparse
import dataclasses
import itertools
from pathlib import Path

from nsmaze.cli_utils import add_params_args, cast_value, params_from_args, write_csv
from nsmaze.evolution import METHODS, Params, run_many
from nsmaze.maze import Maze
from nsmaze import plots


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maze", default="hard")
    ap.add_argument("--method", default="novelty", choices=METHODS)
    ap.add_argument("--param", required=True, help="имя параметра (как в nsmaze.evolution.Params)")
    ap.add_argument("--values", nargs="+", required=True)
    ap.add_argument("--param2", default=None)
    ap.add_argument("--values2", nargs="+", default=None)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    add_params_args(ap)
    a = ap.parse_args()

    names = {f.name for f in dataclasses.fields(Params)}
    for p in (a.param, a.param2):
        if p is not None and p not in names:
            ap.error(f"неизвестный параметр {p}")
    maze = Maze.load(a.maze)
    out = Path(a.out or f"results/sweep_{maze.name}_{a.param}" + (f"_{a.param2}" if a.param2 else ""))
    out.mkdir(parents=True, exist_ok=True)

    v1 = [cast_value(a.param, v) for v in a.values]
    v2 = [cast_value(a.param2, v) for v in a.values2] if a.param2 else [None]
    seeds = list(range(a.seed0, a.seed0 + a.runs))
    rows = []
    for x, y in itertools.product(v1, v2):
        p = params_from_args(a, a.method)
        setattr(p, a.param, x)
        tag = f"{a.param}={x}"
        if a.param2:
            setattr(p, a.param2, y)
            tag += f"_{a.param2}={y}"
        print(f"\n=== {tag} ===", flush=True)
        res = run_many(a.maze, p, seeds, workers=a.workers, out_dir=out / "runs", tag=tag + "_")
        rows += [r.summary() for r in res]
        write_csv(rows, out / "sweep.csv")

    if a.param2:
        plots.figure_sweep_2d(rows, a.param, a.param2, out / f"sweep_{a.param}_{a.param2}.png", maze.name)
    else:
        plots.figure_sweep_1d(rows, a.param, out / f"sweep_{a.param}.png", maze.name)
    print(f"\nТаблица: {out / 'sweep.csv'}\nРисунок: {out}")


if __name__ == "__main__":
    main()
