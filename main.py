"""
使用Qwen3系列模型在RAG的基础上构建Agent应用
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime

# 让本地请求不走代理，解决 Ollama 502 报错
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# 强制让 Hugging Face 使用国内镜像站下载模型，彻底解决超时问题！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from agent.agent_core import AgentCore
from agent.memory import memory_manager
from evaluation.metrics import Evaluator, print_evaluation_report

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(config.log_file, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def interactive_mode():
    print("\n" + "="*60)
    print("   Qwen3 Agent RAG 系统 - 交互模式")
    print("="*60)
    print("输入 'quit' 退出，输入 'clear' 清空记忆")
    print("输入 'stats' 查看统计，输入 'eval' 运行评估")
    print("-"*60)

    agent = AgentCore()

    while True:
        try:
            query = input("\n🧑 用户: ").strip()
            if query.lower() == 'quit':
                print("👋 再见！")
                break
            elif query.lower() == 'clear':
                memory_manager.clear()
                print("✅ 记忆已清空")
                continue
            elif query.lower() == 'stats':
                stats = memory_manager.get_statistics()
                print(f"📊 记忆统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
                continue

            elif query.lower() == 'eval':
                evaluator = Evaluator()
                result = evaluator.run_evaluation()
                print_evaluation_report(result)

                # ===== 新增：生成可视化图表 =====
                try:
                    from viz_report import generate_charts_from_result
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs("./results", exist_ok=True)
                    generate_charts_from_result(result, prefix=f"./results/eval_interactive_{timestamp}")
                    print(f"📊 可视化图表已生成到 ./results/ 目录")
                except Exception as e:
                    print(f"⚠️ 可视化生成失败: {e}")
                # ================================
                continue

            elif not query:
                continue

            print("🤖 Agent思考中...", end=" ", flush=True)
            result = agent.process(query)
            print("✅")
            print(f"\n🤖 Agent: {result.final_answer}")
            print(f"\n📊 详情:")
            print(f"  置信度: {result.confidence:.2%}")
            print(f"  响应时间: {result.total_duration:.2f}s")
            print(f"  执行步骤: {len(result.steps)}")
            print(f"  工具调用: {[tc['tool'] for tc in result.tool_calls]}")
            print(f"  重试次数: {result.retries}")
            print(f"  使用记忆: {'✅' if result.memory_used else '❌'}")
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            logger.error(f"处理出错: {e}", exc_info=True)
            print(f"❌ 处理出错: {e}")


def evaluate_mode():
    print("\n" + "="*60)
    print("   Qwen3 Agent RAG 系统 - 评估模式")
    print("="*60)
    evaluator = Evaluator()
    result = evaluator.run_evaluation()
    print_evaluation_report(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"./results/evaluation_{timestamp}.json"
    os.makedirs("./results", exist_ok=True)

    serializable_result = {
        "accuracy": result.accuracy,
        "avg_response_time": result.avg_response_time,
        "multi_hop_solving_rate": result.multi_hop_solving_rate,
        "tool_calling_accuracy": result.tool_calling_accuracy,
        "memory_efficiency": result.memory_efficiency,
        "retry_success_rate": result.retry_success_rate,
        "avg_confidence": result.avg_confidence,
        "avg_tokens_used": result.avg_tokens_used,
        "detail": result.detail,
        "timestamp": timestamp,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 评估结果已保存: {result_path}")

    # 可视化
    from viz_report import generate_charts_from_result
    generate_charts_from_result(result, prefix=f"./results/evaluation_{timestamp}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Qwen3 Agent RAG 系统 - 实验三")
    parser.add_argument("--mode", "-m", choices=["interactive", "evaluate", "batch"], default="interactive",
                        help="运行模式（默认: interactive）")
    parser.add_argument("--queries", "-q", type=str, help="批量处理的问题文件路径")
    parser.add_argument("--model", type=str, default=config.model.model_name, help=f"Qwen3模型名称")
    args = parser.parse_args()

    if args.model:
        config.model.model_name = args.model
    logger.info(f"Qwen3 Agent RAG 系统启动 | 模型: {config.model.model_name} | 模式: {args.mode}")

    if args.mode == "interactive":
        interactive_mode()
    elif args.mode == "evaluate":
        evaluate_mode()
    elif args.mode == "batch":
        if not args.queries:
            logger.error("批量模式需要指定 --queries 参数")
            return
        with open(args.queries, "r", encoding="utf-8") as f:
            queries = json.load(f)
        questions = queries if isinstance(queries, list) else [queries]
        agent = AgentCore()
        results = []
        for i, q in enumerate(questions):
            q_text = q if isinstance(q, str) else q.get("question", "")
            print(f"[{i+1}/{len(questions)}] 处理: {q_text[:50]}...")
            result = agent.process(q_text)
            results.append({"question": q_text, "answer": result.final_answer,
                            "confidence": result.confidence, "response_time": result.total_duration})
        output_path = f"./results/batch_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("./results", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ 批量处理完成，结果已保存: {output_path}")



if __name__ == "__main__":
    main()
