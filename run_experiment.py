"""Серия запусков: один или несколько методов на одном или нескольких лабиринтах.

Примеры:
    # быстрая проверка работоспособности (≈1–2 мин)
    python run_experiment.py --maze medium --method novelty --runs 2 --max-evals 20000

    # сравнение с аналогами на лабиринтах авторов (как в статье — 40 запусков)
    python run_experiment.py --maze medium hard --method novelty fitness random mapelites --runs 40

    # лабиринты разной сложности (сначала python make_mazes.py)
    python run_experiment.py --maze grid3x3 grid4x4 grid5x5 --method novelty fitness --runs 10

Результаты: <out>/<maze>/<method>/seed*.pkl, <out>/summary.csv и рисунки в <out>/figures.
"""
import argparse
from pathlib import Path

from nsmaze.cli_utils import add_params_args, params_from_args, write_csv
from nsmaze.evolution import METHODS, RunResult, run_many
from nsmaze.maze import Maze
from nsmaze import plots


def _safe(fn, *args):
    """Рисунок не должен обрывать серию: при ошибке печатаем предупреждение и идём дальше.

    Результаты (.pkl) к этому моменту уже сохранены; рисунки можно перерисовать: python make_figures.py <папка>.
    """
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001
        print(f"  ! не удалось сохранить рисунок {args[-1]}: {e}\n    (перерисовать: python make_figures.py <папка результатов>)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maze", nargs="+", default=["hard"], help="имена из папки mazes/ или пути к файлам")
    ap.add_argument("--method", nargs="+", default=["novelty"], choices=METHODS)
    ap.add_argument("--runs", type=int, default=5, help="число независимых запусков на метод")
    ap.add_argument("--seed0", type=int, default=0, help="первое зерно генератора")
    ap.add_argument("--workers", type=int, default=None, help="число процессов (по умолчанию — все ядра)")
    ap.add_argument("--out", default="results/experiment")
    ap.add_argument("--verbose", action="store_true")
    add_params_args(ap)
    a = ap.parse_args()

    out = Path(a.out).resolve()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    seeds = list(range(a.seed0, a.seed0 + a.runs))
    rows = []
    for maze_name in a.maze:
        maze = Maze.load(maze_name)
        by_method = {}
        for method in a.method:
            p = params_from_args(a, method)
            p.verbose = a.verbose
            res = run_many(maze_name, p, seeds, workers=a.workers, out_dir=out / maze.name / method)
            by_method[method] = res
            rows += [r.summary() for r in res]
            write_csv(rows, out / "summary.csv")          # таблица сохраняется после каждого метода
            _safe(plots.figure_run, res[0], maze, out / "figures" / f"run_{maze.name}_{method}_seed{res[0].seed}.png")
        _safe(plots.figure_heatmaps_compare, by_method, maze, out / "figures" / f"heatmaps_{maze.name}.png")
    _safe(plots.figure_methods_bars, rows, out / "figures" / "methods.png")

    print("\nИтог (решено / медиана оценок / среднее покрытие):")
    for maze_name in dict.fromkeys(r["maze"] for r in rows):
        for method in a.method:
            rs = [r for r in rows if r["maze"] == maze_name and r["p_method"] == method]
            ev = sorted(r["evals"] for r in rs if r["solved"])
            med = ev[len(ev) // 2] if ev else "—"
            cov = sum(r["coverage"] for r in rs) / len(rs)
            print(f"  {maze_name:10s} {method:10s} {sum(r['solved'] for r in rs):3d}/{len(rs):<3d}  {med!s:>7}  {cov:.2f}")
    print(f"\nТаблица: {out / 'summary.csv'}\nРисунки: {out / 'figures'}")


if __name__ == "__main__":
    main()
