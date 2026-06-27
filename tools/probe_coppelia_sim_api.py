#!/usr/bin/env python3
"""List CoppeliaSim sim.* API symbols matching gravity/dynamics keywords."""
from __future__ import annotations

import re
import sys

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 23000
    client = RemoteAPIClient(port=port)
    sim = client.getObject("sim")
    pat = re.compile(r"gravity|inverse|dynamic|generalized|jointforce|massmatrix", re.I)
    names = sorted(n for n in dir(sim) if pat.search(n))
    for name in names:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
