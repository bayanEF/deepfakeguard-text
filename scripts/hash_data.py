#!/usr/bin/env python3
"""Print SHA-256 and row count of the raw CSVs so the exact data versions can be recorded in the README."""
import hashlib, sys, pandas as pd
for p in sys.argv[1:]:
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    print(f"{p}\n  sha256: {h}\n  rows: {len(pd.read_csv(p))}")
