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
    no_rag_path = Path(r"output/qwen2_5_7b_instruct/Naive Gen/2026_05_26_14_32_10/intermediate_data.json")
    rag_path = Path(r"output/qwen2_5_7b_instruct/Naive RAG/2026_05_27_10_20_16/intermediate_data.json")
    out_path = Path(r"output/qwen2_5_7b_instruct/merge_norag_rag.json")

    a = json.loads(no_rag_path.read_text(encoding="utf-8"))
    b = json.loads(rag_path.read_text(encoding="utf-8"))

    dict_a = {rec.get("id"): rec for rec in a if rec.get("id")}
    dict_b = {rec.get("id"): rec for rec in b if rec.get("id")}

    results = []
    all_ids = sorted(set(dict_a) | set(dict_b))
    for id_ in all_ids:
        rec_a = dict_a.get(id_)
        rec_b = dict_b.get(id_)
        results.append({
            "id": id_,
            "no_rag": rec_a,
            "rag": rec_b,
            "no_rag_correct": is_correct(rec_a) if rec_a else False,
            "rag_correct": is_correct(rec_b) if rec_b else False,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()
