#!/usr/bin/env python3
"""Build (or rebuild) the retrieval index from data/course/.

    python scripts/ingest.py            # rebuild if the material changed
    python scripts/ingest.py --force    # always rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from varkcheck import rag  # noqa: E402
from varkcheck.config import Settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Index the course material.")
    parser.add_argument("--force", action="store_true", help="rebuild even if the hash matches")
    parser.add_argument("--chunk-chars", type=int, default=None)
    parser.add_argument("--space", default="tfidf", choices=["tfidf", "sentence-transformers"])
    args = parser.parse_args()

    settings = Settings.load()
    index, rebuilt = rag.load_or_build(
        settings.course_dir,
        settings.index_dir,
        chunk_chars=args.chunk_chars or settings.chunk_chars,
        force=args.force,
        space_kind=args.space,
    )

    files = index.sources_digest()
    print(f"index: {'rebuilt' if rebuilt else 'up to date'}")
    print(f"space: {index.space.kind}")
    print(f"files: {len(files)}  chunks: {len(index.chunks)}  "
          f"vocab: {len(getattr(index.space, 'idf', {}))}")
    for entry in files:
        print(f"  - {entry['doc']}: {len(entry['sections'])} sections")

    print("\ntopics the student can pick:")
    for topic in index.sections():
        print(f"  - {topic['citation']} ({topic['chars']} chars)")

    if not index.chunks:
        print("\nNo chunks. Is data/course/ empty?", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
