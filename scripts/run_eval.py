import argparse
import os
import json
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.evaluator import Evaluator
from flashrag.prompt import PromptTemplate
from flashrag.pipeline import SequentialPipeline

# ================= 1. 自定义预测解析函数 =================
import re

def enterprise_pred_parse(dataset):
    """
    针对企业战略预测任务的自定义预测解析函数。
    利用贪婪匹配直接定位并提取最后一次出现的“答案为/是：X”。
    """
    # re.DOTALL 让 .* 能够跨越换行符进行全局贪婪匹配
    # .* 会一直匹配到最后一个符合 "答案[是为][：:]\s*([A-D])" 的位置前
    pattern = re.compile(r".*答案[是为][：:]\s*([A-D])", re.DOTALL)
    
    # 兜底正则：直接找全文最后出现的一个 A-D
    fallback_pattern = re.compile(r".*([A-D])", re.DOTALL)
    
    for item in dataset:
        pred = str(item.pred)
        
        # search() 只返回一个 match 对象，由于 .* 的贪婪性，这必然是最后一个
        match = pattern.search(pred)
        
        if match:
            # group(1) 提取括号里捕获的 A/B/C/D
            answer = match.group(1)
        else:
            # 兜底逻辑：如果找不到标准句式，直接抓取全文最后一个选项字母
            fallback_match = fallback_pattern.search(pred)
            if fallback_match:
                answer = fallback_match.group(1)
            else:
                # 极端情况：啥都没匹配到，保留原样
                answer = pred.strip()
                
        item.update_output('raw_pred', pred)
        item.update_output('pred', answer)
        
    return dataset

# ================= 2. 配置初始化逻辑 =================
def get_merged_config():
    """
    整合 argparse 和 YAML 配置的初始化函数。
    """
    parser = argparse.ArgumentParser("Update configuration from command line arguments.")

    # 基础设置
    parser.add_argument("--method_name", type=str, default="Naive Gen",
                        choices=["Naive Gen", "Naive RAG"],
                        help="测试方法名称")
    parser.add_argument("--config_path", type=str, default="my_config.yaml", help="YAML 配置文件路径")
    
    # 动态覆盖参数
    parser.add_argument("--dataset_name", type=str, choices=["enterprise_prediction"])
    parser.add_argument("--test_sample_num", type=int, default=None, help="最大测试样本数量")
    parser.add_argument("--save_dir", type=str, default="output/")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--generator_model", type=str, help="生成器模型名称")

    args = parser.parse_args()

    # 加载 YAML 配置
    config_dict = {}
    if args.config_path and os.path.exists(args.config_path):
        pass
    else:
        print(f"Warning: 配置文件 {args.config_path} 未找到，将使用默认参数。")

    # 提取 argparse 参数到字典中以便覆盖
    for key, value in vars(args).items():
        if key != "config_path" and value is not None:
            config_dict[key] = value

    # 设置保存路径逻辑
    config_dict["save_note"] = args.method_name
    
    # 初始化 FlashRAG Config (默认读取 base_config.yaml 并用 config_dict 覆盖)
    # 如果你没有单独的 base_config.yaml，可以直接指向你的 my_config.yaml
    final_config = Config(args.config_path, config_dict)
    
    # 更新保存目录，使其包含模型名和方法名
    final_config["save_dir"] = os.path.join(final_config["save_dir"], final_config["generator_model"], final_config["save_note"])
    
    return final_config

# ================= 3. 主执行逻辑 =================
def main():
    # 1. 获取整合后的配置
    config = get_merged_config()
    method_name = config['method_name']
    
    # 2. 加载数据集
    dataset = get_dataset(config)
    test_data = dataset.test
    
    print(f"\n🚀 正在初始化 【{method_name}】 流水线...")
    
    # 3. 根据方法定义专属 Prompt 模板并执行
    if method_name == "Naive Gen":
        template = PromptTemplate(
            config=config,
            system_prompt=("你是一位资深的企业创新战略预测专家。请你基于自身掌握的内部知识，从四个候选选项（A/B/C/D）中，选择最佳答案选项。"
                           "解释你的答案，并必须在回复末尾以 “答案为：X。” 的格式给出最终答案选项，例如：“答案为：A。”。末尾严禁附加任何其他字符。"),
            user_prompt="问题：{question}"
        )
        pipeline = SequentialPipeline(config, template)
        pipeline.naive_run(test_data, pred_process_fun=enterprise_pred_parse)
        
    elif method_name == "Naive RAG":
        template = PromptTemplate(
            config=config,
            system_prompt=("你是一位资深的企业创新战略预测专家。请你根据所给文档回答问题，从四个候选选项（A/B/C/D）中，选择最佳答案选项。"
                           "解释你的答案，并必须在回复末尾以 “答案为：X。” 的格式给出最终答案选项，例如：“答案为：A。”。末尾严禁附加任何其他字符。"
                           "\n以下是提供的文档：\n{reference}"),
            user_prompt="问题：{question}"
        )
        pipeline = SequentialPipeline(config, template)
        pipeline.run(test_data, pred_process_fun=enterprise_pred_parse)
        
    else:
        print(f"❌ 错误：当前脚本已整合为仅支持 'Naive Gen' 和 'Naive RAG'。")
        return

    # 4. 评测与结果保存
    evaluator = Evaluator(config)
    eval_result = evaluator.evaluate(test_data)
    print(f"\n🎉 {method_name} 评测完成！结果: {eval_result}")
    
    test_data.save(config.save_dir)

if __name__ == "__main__":
    main()