import os
import json
import uuid
import shutil
import argparse
import warnings
import torch.distributed as dist
from datetime import datetime
from prompt_toolkit import PromptSession

warnings.filterwarnings("ignore")

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.prompt import PromptTemplate
from flashrag.pipeline import SequentialPipeline

def get_args():
    parser = argparse.ArgumentParser(description="Interactive RAG Console")
    parser.add_argument("--config_path", type=str, default="config/my_config.yaml", help="YAML 配置文件路径")
    return parser.parse_args()

def main():
    args = get_args()
    
    print("=" * 60)
    print("🚀 欢迎使用交互式 RAG 问答系统！")
    print("=" * 60)
    
    # 1. 基础配置
    base_save_dir = "output/interactive"
    config_dict = {
        "dataset_name": "interactive_run", 
        "save_dir": base_save_dir,
        "save_note": "interactive",
        "metrics": [],
        "save_metric_score": False
    }
    
    if not os.path.exists(args.config_path):
        print(f"⚠️ 警告: 未找到配置文件 {args.config_path}，将采用框架默认配置。")
        
    config = Config(args.config_path, config_dict)
    
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    generator_model = config.get("generator_model").split('/')[-1]
    real_save_dir = os.path.join(base_save_dir, generator_model, current_time)
    
    config["save_dir"] = real_save_dir
    os.makedirs(real_save_dir, exist_ok=True)
    
    # 定义交互式对话中间结果的保存文件路径
    history_records_file = os.path.join(real_save_dir, "interactive_history.json")
    print(f"📂 本次对话的中间结果将实时保存至: {history_records_file}")

    # 用于在内存中缓存当前 Session 的所有对话数据
    all_records = []
    
    # 2. 准备 Prompt 模板
    system_prompt = (
        "你是一个专业的问答助手。请你根据所给提示和参考文档回答用户的问题。\n"
        "回答需准确、简洁。如果参考文档中没有相关信息，请尝试基于自身知识回答，或说明文档中未找到相关答案。\n"
        "以下是提供的文档：\n{reference}"
    )
    
    template = PromptTemplate(
        config=config,
        system_prompt=system_prompt,
        user_prompt="问题：{question}"
    )
    
    # 3. 初始化Pipeline加载所有模型与索引
    print("\n⏳ 正在初始化 RAG 核心组件 (检索器、生成器及索引加载)，请耐心等待...")
    try:
        pipeline = SequentialPipeline(config, template)
        print("✅ 模型及组件加载完成！可以开始提问。\n")
    except Exception as e:
        print(f"❌ Pipeline 初始化失败: {e}")
        return
    
    session = PromptSession()
    # 4. 进入交互式对话循环
    while True:
        try:
            print("-" * 60)
            user_input = session.prompt("👤 请输入您的问题 (输入 'quit' 或 'exit' 退出):\n> ", multiline=False).strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
            
            if not user_input:
                continue
                
            print("\n🔎 正在检索文档并生成回答...")
            
            # 使用临时文件创建一个仅包含当前问题的测试数据集
            item_id = f"interactive_{uuid.uuid4().hex[:8]}"
            temp_dir = f"dataset/interactive_temp_{item_id}"
            os.makedirs(temp_dir, exist_ok=True)
            temp_test_file = os.path.join(temp_dir, "test.jsonl")
            
            with open(temp_test_file, 'w', encoding='utf-8') as f:
                # 构造符合 flashrag 标准格式的一条数据
                dummy_item = {"id": item_id, "question": user_input}
                f.write(json.dumps(dummy_item, ensure_ascii=False) + "\n")
            
            try:
                # 动态指向临时数据集目录
                config["dataset_path"] = temp_dir
                
                # 加载该数据集
                dataset = get_dataset(config)
                test_dataset = dataset["test"]
                
                # 运行 RAG Pipeline
                # SequentialPipeline(dataset) 返回一个 Dataset 包含了修改后的 Items
                result_dataset = pipeline.run(test_dataset)
                
                # 解析并输出生成结果
                if result_dataset and len(result_dataset) > 0:
                    # dataset是迭代对象，通常可以使用 data[0] 获取第一个Item
                    first_item = result_dataset[0]
                    
                    # 获取生成的文本
                    answer = getattr(first_item, 'pred', None) 
                    
                    if answer:
                        print("\n🤖 [LLM 回复]:")
                        print(answer.strip())
                        print()
                    else:
                        print("\n🤖 [系统提示]: 生成结果为空。\n")
                    
                    # FlashRAG 会将检索到的文档列表存储在 item.retrieval_result 中
                    retrieved_docs = getattr(first_item, 'retrieval_result', [])
                    
                    # 构造您需要的精简保存结构
                    save_item = {
                        "id": item_id,
                        "question": user_input,
                        "docs": retrieved_docs,
                        "output": answer.strip() if answer else ""
                    }
                    
                    # 追加到内存列表中，并覆盖写入 JSON 数组文件
                    all_records.append(save_item)
                    with open(history_records_file, 'w', encoding='utf-8') as f_out:
                        json.dump(all_records, f_out, ensure_ascii=False)
                    # =======================================================
                else:
                    print("\n🤖 [系统提示]: 未生成有效回复 (数据集返回空)。\n")
                    
            except Exception as e:
                print(f"\n❌ [执行 RAG 时发生错误]: {str(e)}")
            finally:
                # 清理本轮生成的临时目录与数据
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
        except KeyboardInterrupt:
            print("\n\n👋 感应到打断信号，但不会退出。若要退出请输入 'quit' 或 'exit'。\n")
            continue
        except Exception as e:
            print(f"\n❌ [全局严重错误]: {str(e)}")

    print("👋 感谢使用交互式 RAG 系统，再见！")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
