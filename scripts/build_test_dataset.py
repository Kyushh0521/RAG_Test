import json
import os

def build_flashrag_test_dataset(input_json_path, output_jsonl_path, q_prefix="test"):
    """
    将原始测试集转换为 FlashRAG 格式的 test.jsonl
    """
    print(f"⏳ 正在读取原始测试集: {input_json_path} ...")
    
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    if not isinstance(data, list):
        print("❌ 错误：输入数据格式应为 JSON 数组。")
        return

    total_items = len(data)
    pad_width = max(3, len(str(total_items)))
    
    print(f"📦 成功加载 {total_items} 条题目，开始构建测试数据集...")

    success_count = 0
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(data, start=1):
            query_id = f"{q_prefix}_{i:0{pad_width}d}"
            
            instruction = item.get("instruction", "")
            info = item.get("enterprise_info", {})
            opts = item.get("options", {})
            
            info_text = (
                f"【企业基础信息】\n"
                f"- 公司名称：{info.get('company_name', '未知')}\n"
                f"- 行业领域：{info.get('industry_domain', '未知')}\n"
                f"- 主营业务：{info.get('main_business', '未知')}\n"
            )
            
            options_text = "【候选选项】\n" + "\n".join([f"{k}. {v}" for k, v in opts.items()])
            
            full_question = f"{instruction}\n\n{info_text}\n\n{options_text}"
            
            golden_answers = [item.get("answer", "")]
            
            metadata = {
                "research_outputs": item.get("research_outputs", []),
                "rationale": item.get("rationale", ""),
            }
            
            flashrag_item = {
                "id": query_id,
                "question": full_question,
                "golden_answers": golden_answers,
                "metadata": metadata
            }
            
            f.write(json.dumps(flashrag_item, ensure_ascii=False) + "\n")
            success_count += 1

    print(f"\n🎉 转换完成！总共生成 {success_count} 条测试数据。")
    print(f"📄 最终测试集已保存至: {output_jsonl_path}")

if __name__ == "__main__":
    INPUT_JSON = "testv2.json"
    OUTPUT_JSONL = "dataset/prediction/testv2.jsonl"

    build_flashrag_test_dataset(INPUT_JSON, OUTPUT_JSONL)