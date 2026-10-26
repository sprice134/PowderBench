#!/usr/bin/env python3
"""Inspect externally acquired STL inputs without copying the particle scans."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-root", required=True, type=Path)
    parser.add_argument("--powder", required=True)
    parser.add_argument("--min-meshes", type=int, default=1)
    parser.add_argument("--manifest-out", type=Path,
                        help="Optional JSON inventory of input STL names and SHA-256 hashes")
    args = parser.parse_args()
    directory = args.mesh_root / args.powder
    meshes = sorted(directory.glob("*.stl"))
    summary = {"powder": args.powder, "mesh_count": len(meshes),
               "bytes": sum(p.stat().st_size for p in meshes),
               "input_format": "STL"}
    if args.manifest_out:
        records = []
        for mesh in meshes:
            digest = hashlib.sha256()
            with mesh.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            records.append({"file": mesh.name, "bytes": mesh.stat().st_size,
                            "sha256": digest.hexdigest()})
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps({**summary, "meshes": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if len(meshes) >= args.min_meshes else 1


if __name__ == "__main__":
    raise SystemExit(main())
