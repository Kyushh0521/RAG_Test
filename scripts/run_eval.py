import argparse
from datetime import datetime
import os
import json
import shutil
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
    parser.add_argument("--config_path", type=str, default="config/my_config.yaml", help="YAML 配置文件路径")
    
    # 动态覆盖参数
    parser.add_argument("--dataset_name", type=str, choices=["prediction"])
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

    # 设置诱饵 temp_layer
    config_dict["save_note"] = "temp_layer"
    
    # 1. 初始化 Config (此时 FlashRAG 已经偷偷建了临时文件夹并塞了 yaml 进去)
    final_config = Config(args.config_path, config_dict)
    
    # 记录下这个包含 yaml 的临时“诱饵”目录
    dummy_dir = final_config["save_dir"]
    
    # 2. 计算出我们真正想要的、干净的三层路径
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    clean_base_dir = os.path.dirname(dummy_dir)
    real_save_dir = os.path.join(
        clean_base_dir, 
        final_config["generator_model"], 
        args.method_name, 
        current_time
    )

    # 3. 强行接管 config 的后续保存路径
    final_config["save_dir"] = real_save_dir

    # 4. 手动创建我们真正的新目录
    os.makedirs(real_save_dir, exist_ok=True)

    # 5. 【核心修复】把 yaml 文件从临时目录“搬家”到正确目录
    old_yaml_path = os.path.join(dummy_dir, "config.yaml")
    new_yaml_path = os.path.join(real_save_dir, "config.yaml")
    if os.path.exists(old_yaml_path):
        shutil.move(old_yaml_path, new_yaml_path)

    # 6. 打扫战场：删掉那个没用的临时文件夹
    try:
        os.rmdir(dummy_dir)
    except OSError:
        pass # 如果文件夹里还有其他意外生成的文件，就不强制删，以免误删数据
    
    return final_config

# ================= 3. 主执行逻辑 =================
def main():
    # 1. 获取整合后的配置
    config = get_merged_config()
    method_name = config['method_name']

    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 2. 加载数据集
    dataset = get_dataset(config)
    test_data = dataset["test"]
    
    print(f"\n🚀 正在初始化 【{method_name}】 流水线...")

    # 3. 根据方法定义专属 Prompt 模板并执行
    if method_name == "Naive Gen":
        template = PromptTemplate(
            config=config,
            system_prompt=("你是一位资深的企业创新战略预测专家。请你基于自身掌握的内部知识，从四个候选选项（A/B/C/D）中，选择最佳答案选项。解释你的答案，并必须在回复末尾以 “答案为：X。” 的格式给出最终答案选项，例如：“答案为：A。”。末尾严禁附加任何其他字符。"),
            user_prompt="问题：{question}"
        )
        pipeline = SequentialPipeline(config, template)
        pipeline.naive_run(test_data, pred_process_fun=enterprise_pred_parse)
        
    elif method_name == "Naive RAG":
        system_prompt = "你是一位资深的企业创新战略预测专家。请你根据所给文档回答问题，从四个候选选项（A/B/C/D）中，选择最佳答案选项。解释你的答案，并必须在回复末尾以 “答案为：X。” 的格式给出最终答案选项，例如：“答案为：A。”。末尾严禁附加任何其他字符。\n以下是提供的文档：\n{reference}"

        system_prompt_1 = "你是一位资深的企业创新战略预测专家。请你根据所给文档（可能混杂了无关的噪声或浅层干扰信息）回答问题，从四个候选选项（A/B/C/D）中，选择最佳答案选项。请你辩证地参考这些文档，切勿盲从。注意：1. 甄别并剔除文档中与企业研究领域和主营业务不符的无效信息。2. 以题干中企业现有的信息为第一性原理，结合提供的文档，进行前瞻性的技术演进推演。3. 警惕那些仅仅是“字面关键词拼接”或“缺乏技术支撑的宏观概念”的干扰选项。解释你的答案，并必须在回复末尾以 “答案为：X。” 的格式给出最终答案选项，例如：“答案为：A。”。末尾严禁附加任何其他字符。\n以下是提供的文档：\n{reference}"

        system_prompt_2 = "你是一位资深的企业创新战略预测专家。请综合题干信息与辅助参考文档，完成企业创新方向的预测。在选择最终答案前，请严格按照以下步骤进行深度思考：第一步：提取题干中企业最核心的领域和业务，这是预测的唯一根基。第二步：阅读提供的参考文档，仅提取其中能够与企业底层技术产生逻辑共鸣的深层趋势，忽略表面的行业科普词汇。第三步：基于企业信息，推演其向未来发展的合理技术跨度，这个跨度必须是实质性的创新，而非简单的产能延续。第四步：逐一审视 A/B/C/D 四个选项，排除缺乏底层技术支撑的“空中楼阁”选项和生硬拼接的“字面陷阱”。请解释你的推理过程，并从四个候选选项（A/B/C/D）中选择最佳答案。必须在回复末尾严格以“答案为：X。”的格式给出最终答案选项，例如：“答案为：A。”。末尾严禁附加任何其他字符。\n以下是提供的文档：\n{reference}"

        system_prompt_3 = """你是一位企业创新战略预测专家。你的任务是根据企业的真实情况，选择其最佳创新方向。以下提供的参考文档仅供了解行业背景。在评估选项时，请考虑以下因素：该选项是否符合该企业【现有的核心业务】？（如果不符合，必须排除！）该选项是否是【技术维度的升级】？（如果只是单纯买设备、扩充传统产能，必须排除！）请解释你的推理过程，并从四个候选选项（A/B/C/D）中选择最佳答案。必须在回复末尾严格以“答案为：X。”的格式给出最终答案选项，例如：“答案为：A。”。末尾严禁附加任何其他字符。\n以下是提供的文档：\n{reference}"""

        system_prompt_4 = """你是一位负责为企业进行技术演进把关的评审专家。请根据题目信息和下方的参考文档，选出最佳创新方向。在做决定前，请对每个选项进行严格的分析：1. 企业现在的技术能力，能否实现这个方向吗？2. 这是技术体系的进化，还是仅仅在堆砌传统规模？3. 概念排雷：这个方向是否只是在生硬套用当下热门词汇，但与企业的主线业务脱节？注意：参考文档可能存在误导，如果文档内容与企业实际技术基因冲突，请以企业自身信息为准。请解释你的推理过程，并从四个候选选项（A/B/C/D）中选择最佳答案。必须在回复末尾严格以“答案为：X。”的格式给出最终答案选项，例如：“答案为：A。”。末尾严禁附加任何其他字符。\n以下是提供的文档：\n{reference}"""

        template = PromptTemplate(
            config=config,
            system_prompt=(system_prompt_3),
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

if __name__ == "__main__":
    main()