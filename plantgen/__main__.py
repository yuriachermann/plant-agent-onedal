"""Generate a synthetic plant and write it as CSV.

    python -m plantgen --size M --seed 7 --out data/plant-M
    python -m plantgen --machines 60 --days 45 --out data/plant-60
"""
import argparse
import json

from .generator import generate_plant
from .io import save_plant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["S", "M", "L"], default="M")
    ap.add_argument("--machines", type=int, help="overrides --size")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = generate_plant(a.machines or a.size, days=a.days, seed=a.seed)
    save_plant(p, a.out)
    print(json.dumps(p.summary(), indent=2))
    print("planted faults:", json.dumps({k: v for k, v in p.truth.items() if k != "onset_day"}))


if __name__ == "__main__":
    main()
