#!/usr/bin/env python3
import json
from pathlib import Path


def is_correct(record):
    try:
        out = record.get("output", {})
        # 优先使用 metric_score.acc 判断
        acc = out.get("metric_score", {}).get("em")
        if acc is not None:
            return float(acc) == 1.0
        # 否则回退到 pred 是否包含在 golden_answers
        pred = out.get("pred")
        golds = record.get("golden_answers") or []
        if pred is not None and golds:
            return pred in golds
        return False
    except Exception:
        return False


def main():
    no_rag_path = Path(r"G:/Projects/RAG_Test/output/Naive Gen/intermediate_data.json")
    rag_path = Path(r"G:/Projects/RAG_Test/output/Naive RAG/intermediate_data.json")
    out_path = Path(r"G:/Projects/RAG_Test/output/failed_after_rag.json")

    a = json.loads(no_rag_path.read_text(encoding="utf-8"))
    b = json.loads(rag_path.read_text(encoding="utf-8"))

    dict_a = {rec.get("id"): rec for rec in a if rec.get("id")}
    dict_b = {rec.get("id"): rec for rec in b if rec.get("id")}

    results = []
    for id_, rec_a in dict_a.items():
        rec_b = dict_b.get(id_)
        if not rec_b:
            continue
        if is_correct(rec_a) and not is_correct(rec_b):
            results.append({
                "id": id_,
                "no_rag": rec_a,
                "rag": rec_b,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()
