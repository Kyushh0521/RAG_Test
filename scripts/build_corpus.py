import json
from pathlib import Path
import pandas as pd
import os

def build_unified_knowledge_base(file_configs, output_jsonl_path):
    """
    根据给定的文件配置列表，读取多个表格，并统一转换为单一的 jsonl 知识库文件。
    """
    total_success_count = 0
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        
        for config in file_configs:
            input_path = config.get("input_path")
            prefix = config.get("prefix", "doc")
            title_col = config.get("title_col")
            content_cols = config.get("content_cols", [])
            metadata_cols = config.get("metadata_cols", [])
            
            print(f"⏳ 正在处理文件: {input_path} ...")
            
            try:
                suffix = Path(input_path).suffix.lower()
                if suffix == ".csv":
                    df = pd.read_csv(input_path)
                else:
                    df = pd.read_excel(input_path)
            except Exception as e:
                print(f"❌ 读取文件失败跳过 {input_path}，错误信息: {e}")
                continue

            df = df.fillna("")

            total_rows = len(df)
            pad_width = max(3, len(str(total_rows)))

            file_success_count = 0
            
            for i, (_, row) in enumerate(df.iterrows(), start=1):
                doc_id = f"{prefix}_{i:0{pad_width}d}"
                
                title = str(row.get(title_col, "")) if title_col else ""
                
                contents_parts = []
                for col in content_cols:
                    if col in row and str(row[col]).strip() != "":
                        contents_parts.append(str(row[col]).strip())
                contents = ", ".join(contents_parts)
                
                metadata = {
                    "source_file": input_path,
                }
                for col in metadata_cols:
                    if col in row:
                        metadata[col] = row[col]
                
                json_record = {
                    "id": doc_id,
                    "title": title.strip(),
                    "contents": contents.strip(),
                }
                
                if metadata:
                    json_record["metadata"] = metadata

                json_line = json.dumps(json_record, ensure_ascii=False)
                f.write(json_line + "\n")
                file_success_count += 1
                total_success_count += 1
                
            print(f"  ✅ 成功转换 {file_success_count} 条。")

    print(f"\n🎉 所有文件处理完成！总共生成 {total_success_count} 条数据。")
    print(f"📄 最终知识库已保存至: {output_jsonl_path}")


if __name__ == "__main__":
    # 在这里为每一个文件单独配置规则
    FILE_CONFIGS = [
        {
            "input_path": "excel/achievement.xlsx",
            "prefix": "achievement",
            "title_col": "title",
            "content_cols": ["analyse_contect"],
        },
        {
            "input_path": "excel/paper.csv",
            "prefix": "paper",
            "title_col": "title",
            "content_cols": ["abstract"],
            "metadata_cols": ["arxiv_id", "authors", "categories", "pub", "citations", "journal_level"]
        }
    ]
    
    OUTPUT_JSONL = "data/corpus.jsonl"
    build_unified_knowledge_base(FILE_CONFIGS, OUTPUT_JSONL)