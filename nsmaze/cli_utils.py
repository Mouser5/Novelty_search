"""Общие функции для скриптов командной строки."""
from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

from .evolution import Params


def add_params_args(ap):
    g = ap.add_argument_group("параметры модели")
    for f in dataclasses.fields(Params):
        if f.name in ("method", "verbose", "config_path"):
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            g.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes", "да"), default=None,
                           help=f"(по умолчанию {f.default})")
        else:
            g.add_argument(flag, type=_caster(f), default=None, help=f"(по умолчанию {f.default})")
    g.add_argument("--config", dest="config_path", default=None, help="ini-файл NEAT")


def _caster(f):
    t = str(f.type)
    if "int" in t and "float" not in t:
        return int
    if "float" in t:
        return float
    return str


def params_from_args(args, method: str) -> Params:
    kw = {"method": method}
    for f in dataclasses.fields(Params):
        v = getattr(args, f.name, None)
        if v is not None and f.name != "method":
            kw[f.name] = v
    return Params(**kw)


def cast_value(param: str, s: str):
    f = {f.name: f for f in dataclasses.fields(Params)}[param]
    if str(f.type) in ("bool",):
        return s.lower() in ("1", "true", "yes")
    return _caster(f)(s)


def write_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter=";")
        w.writeheader()
        w.writerows(rows)
