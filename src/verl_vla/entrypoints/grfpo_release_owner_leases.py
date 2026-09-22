"""Release shared-queue leases after an authoritative collector death."""

from __future__ import annotations

import argparse
import json

from verl_vla.trainer.grfpo.collection_store import abandon_owner_leases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("window_dir")
    parser.add_argument("owner_id")
    args = parser.parse_args()
    released = abandon_owner_leases(args.window_dir, args.owner_id)
    print(json.dumps({"owner_id": args.owner_id, "released": released}, sort_keys=True))


if __name__ == "__main__":
    main()
