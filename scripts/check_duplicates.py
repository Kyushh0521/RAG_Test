import json
from collections import Counter
from pathlib import Path

P = Path('test.json')
if not P.exists():
    raise SystemExit('test.json not found')
data = json.loads(P.read_text(encoding='utf-8'))
if isinstance(data, dict):
    for v in data.values():
        if isinstance(v, list):
            data = v
            break
if not isinstance(data, list):
    data = list(data)

cnt = Counter()
occ = {}
for i, item in enumerate(data, 1):
    name = ''
    if isinstance(item, dict):
        ei = item.get('enterprise_info')
        if isinstance(ei, dict):
            name = ei.get('company_name') or ''
        else:
            name = item.get('company_name') or ''
    cnt[name] += 1
    occ.setdefault(name, []).append(i)

dups = {k: {'count': v, 'indices': occ[k]} for k, v in cnt.items() if v > 1 and k}
rep = {'total': len(data), 'duplicates': dups}
Path('duplicates_report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(rep, ensure_ascii=False))
