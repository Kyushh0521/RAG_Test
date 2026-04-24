#!/usr/bin/env python3
import json
from pathlib import Path

P = Path('test.json')
if not P.exists():
    raise SystemExit('test.json not found')
data = json.loads(P.read_text(encoding='utf-8'))
if not isinstance(data, list):
    raise SystemExit('expected top-level array')
# backup
P.with_suffix('.bak').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
out = []
for i, item in enumerate(data):
    if isinstance(item, dict):
        # assign id and place it first
        rest = {k: v for k, v in item.items() if k != 'id'}
        out.append({'id': i, **rest})
    else:
        out.append(item)
P.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Wrote {len(out)} records to {P}')
