"""Record operator-cancelled candidate proposals without fake outcomes."""

from __future__ import annotations

import argparse
import json

from verl_vla.trainer.grfpo.collection_store import skip_window_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("window_dir")
    parser.add_argument("candidate_ids", nargs="+")
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    skipped = skip_window_candidates(args.window_dir, args.candidate_ids, reason=args.reason)
    print(json.dumps({"reason": args.reason, "skipped": skipped}, sort_keys=True))


if __name__ == "__main__":
    main()
