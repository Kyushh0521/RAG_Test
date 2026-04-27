import argparse
import os
from flashrag.config import Config
from flashrag.pipeline import SequentialPipeline, BasicPipeline
from flashrag.utils import get_generator, get_dataset
from flashrag.evaluator import Evaluator

# ================= 1. 自定义 Naive Generation 流水线 =================
class NaiveGenerationPipeline(BasicPipeline):
    """
    原生纯生成 Pipeline：完全绕过 Retriever 模块，只加载 Generator。
    不仅避免了报错，还能节约向量模型的显存开销。
    """
    def __init__(self, config):
        self.config = config
        # 1. 加载测试数据集
        self.dataset = get_dataset(config)
        # 2. 仅初始化大模型生成器 (vLLM)
        self.generator = get_generator(config)
        # 3. 初始化评测器
        self.evaluator = Evaluator(config)
        
    def run(self):
        test_data = self.dataset.test
        
        # 将无参考知识的 prompt 模板应用到每一个问题上
        prompts = [
            self.config.prompt_template.replace("{question}", item['question']) 
            for item in test_data
        ]
        
        # 调用大模型进行批量纯生成
        responses = self.generator.generate(prompts)
        
        # 将预测结果写回数据集并评测
        test_data.update_output("pred", responses)
        result = self.evaluator.evaluate(test_data)
        
        # 保存结果
        test_data.save(self.config.save_dir)
        return result

# ================= 2. 主执行逻辑 =================
def main():
    parser = argparse.ArgumentParser(description="原生 FlashRAG 对比评测")
    parser.add_argument('--method', type=str, choices=['standard_rag', 'naive_generation'], required=True)
    args = parser.parse_args()

    # 读取基础配置
    config = Config(config_file_path="my_config.yaml")

    if args.method == 'standard_rag':
        print("\n🚀 正在初始化 【Standard RAG】 (SequentialPipeline)...")
        # RAG 专属 Prompt
        config.prompt_template = """你是一位资深的企业创新战略预测专家。请你根据所给文档回答问题，从四个候选选项（A/B/C/D）中，选择最佳答案选项。解释你的答案，并必须在回复末尾是以 “答案为：X。” 的格式给出最终答案选项，例如：“答案为：A。”。末尾严禁附加任何其他字符。问题：{question}。以下是提供的文档：{reference}。"""
        config.save_dir = os.path.join(config.save_dir, "standard_rag")
        
        # 直接使用 FlashRAG 官方的 Standard RAG 流程类
        pipeline = SequentialPipeline(config)
        
    elif args.method == 'naive_generation':
        print("\n🚀 正在初始化 【Naive Generation】 (Custom Generator Pipeline)...")
        # 纯大模型专属 Prompt（移除 reference）
        config.prompt_template = """你是一位资深的企业创新战略预测专家。请你基于自身掌握的内部知识，从四个候选选项（A/B/C/D）中，选择最佳答案选项。解释你的答案，并必须在回复末尾是以 “答案为：X。” 的格式给出最终答案选项，例如：“答案为：A。”。问题：{question}。"""
        config.save_dir = os.path.join(config.save_dir, "naive_generation")
        
        # 使用我们上面写的、只调用 Generator 的原生类
        pipeline = NaiveGenerationPipeline(config)

    # 运行并打印结果
    result = pipeline.run()
    print(f"\n🎉 {args.method} 评测完成！结果: {result}")

if __name__ == "__main__":
    main()