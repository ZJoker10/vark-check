#!/usr/bin/env python3
"""Seed the synthetic cohort and print the lecturer digest.

    python scripts/seed_demo.py
    python scripts/seed_demo.py --reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from varkcheck import compare, seed  # noqa: E402
from varkcheck.config import Settings  # noqa: E402
from varkcheck.store import Store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed synthetic demo students.")
    parser.add_argument("--reset", action="store_true", help="delete existing records first")
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()

    settings = Settings.load()
    store = Store(settings.db_path)

    if args.reset:
        store.clear_demo_data(keep_cache=True)

    created = seed.seed(store, count=args.count or settings.synthetic_students, seed_value=settings.demo_seed)
    print(f"seeded {created['students']} synthetic students across {created['concepts']} concepts")

    print("\nlecturer digest (top 3):")
    for rank, entry in enumerate(compare.lecturer_digest(store, top_n=3), start=1):
        print(
            f"  {rank}. {entry['concept']}  "
            f"avg={entry.get('avg_pct', 0) * 100:.0f}%  "
            f"flags={entry.get('flags', 0)}  <- {entry.get('source', '')}"
        )

    print("\nrecords:", store.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
