"""PIQA数据集评估脚本 - PPL方法（长度归一化）。"""
import json
import os
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm
import eval_utils
class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


class PIQAEvaluator:
    """使用PPL方法评估PIQA数据集（0-shot，长度归一化）"""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_workers: int = 128, max_workers_per_question: int = 2):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_workers_per_question = min(max_workers_per_question, 2)  # PIQA只有2个选项
        self._correct_count = 0
        self._total_count = 0
        self._failed_extractions = 0
        self._lock = threading.Lock()

    def load_piqa_data(self, dev_file: str, label_file: str = None) -> List[Dict]:
        """从JSONL文件加载PIQA数据（支持单独的labels文件）"""
        data = []
        
        # 如果没有指定labels文件，尝试自动找到它
        if label_file is None:
            # dev.jsonl -> dev-labels.lst
            # train.jsonl -> train-labels.lst
            base_path = dev_file.replace('.jsonl', '-labels.lst')
            if os.path.exists(base_path):
                label_file = base_path
        
        # 读取labels（如果有单独的文件）
        labels = []
        if label_file and os.path.exists(label_file):
            with open(label_file, 'r', encoding='utf-8') as f:
                labels = [int(line.strip()) for line in f if line.strip()]
        
        # 加载数据
        with open(dev_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                item = json.loads(line.strip())
                goal = item.get('goal', '')
                sol1 = item.get('sol1', '')
                sol2 = item.get('sol2', '')
                
                # 优先使用单独的labels文件，其次使用jsonl中的label字段
                if idx < len(labels):
                    label = labels[idx]
                else:
                    label = item.get('label', -1)
                
                if label < 0:
                    answer = 'NULL'
                else:
                    answer = 'AB'[label]
                
                data.append({
                    'goal': goal,
                    'sol1': sol1,
                    'sol2': sol2,
                    'answer': answer,
                })
        
        return data

    def build_prompt(self, goal: str, solution: str) -> str:
        """构建评估prompt（PPL方法）"""
        # 完整的prompt，用于计算solution的logprob
        prompt = f"""{goal}
{solution}"""
        return prompt

    def get_completion_logprob(self, prompt: str, max_retries: int = 3) -> Tuple[float, int]:
        """获取完整prompt的logprob和token长度"""
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=1,
                    logprobs=0,
                    echo=True,
                    temperature=0,
                )

                if not response.choices:
                    raise ModelResponseError('模型返回的choices为空。')

                logprobs_obj = response.choices[0].logprobs
                if logprobs_obj is None:
                    raise ModelResponseError('模型返回的logprobs为空。')

                tokens = logprobs_obj.tokens
                token_logprobs = logprobs_obj.token_logprobs
                if not tokens or not token_logprobs:
                    raise ModelResponseError('模型返回的tokens/logprobs为空。')

                # 计算总logprob（跳过第一个token，因为它的logprob是None）
                total_logprob = sum([lp for lp in token_logprobs[1:] if lp is not None])
                token_length = len(tokens) - 1  # 不计第一个token

                return total_logprob, token_length

            except Exception as e:
                error_msg = str(e)

                if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    wait_time = min(10 * (attempt + 1), 60)
                    time.sleep(wait_time)
                    continue

                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    last_exception = (
                        e if isinstance(e, ModelResponseError) else ModelResponseError(error_msg)
                    )
                    break

        if last_exception is None:
            last_exception = ModelResponseError('logprob计算中出现未知错误。')

        raise last_exception

    def _get_logprob_for_solution(
        self, goal: str, solution: str, label: str
    ) -> Tuple[str, float, int]:
        """获取单个solution的logprob和长度（用于并发调用）"""
        prompt = self.build_prompt(goal, solution)
        logprob, length = self.get_completion_logprob(prompt)
        return label, logprob, length

    def evaluate_single_question(self, question_data: Dict) -> Dict:
        """评估单个问题（并发获取两个solution的logprob）"""
        try:
            goal = question_data['goal']
            sol1 = question_data['sol1']
            sol2 = question_data['sol2']
            reference_answer = question_data['answer'].strip().upper()
            
            option_labels = ['A', 'B']
            solutions = [sol1, sol2]
            option_logprobs = {}
            option_lengths = {}
            option_normalized_logprobs = {}
            option_ppls = {}

            with ThreadPoolExecutor(max_workers=self.max_workers_per_question) as executor:
                futures = {
                    executor.submit(
                        self._get_logprob_for_solution,
                        goal,
                        solution,
                        label,
                    ): (label, solution)
                    for label, solution in zip(option_labels, solutions)
                }

                for future in as_completed(futures):
                    try:
                        label, logprob, length = future.result()
                        option_logprobs[label] = logprob
                        option_lengths[label] = length
                        # 长度归一化
                        normalized_logprob = logprob / length if length > 0 else logprob
                        option_normalized_logprobs[label] = normalized_logprob
                        option_ppls[label] = np.exp(-normalized_logprob)
                    except ModelResponseError:
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        raise
                    except Exception as e:
                        label, _ = futures[future]
                        print(f"\n获取选项 {label} 的logprob时出错: {e}")
                        option_logprobs[label] = -10.0
                        option_lengths[label] = 1
                        option_normalized_logprobs[label] = -10.0
                        option_ppls[label] = np.exp(10.0)
                        with self._lock:
                            self._failed_extractions += 1

            # 确保所有选项都有logprob
            for label in option_labels:
                if label not in option_logprobs:
                    option_logprobs[label] = -10.0
                    option_lengths[label] = 1
                    option_normalized_logprobs[label] = -10.0
                    option_ppls[label] = np.exp(10.0)

            # 选择归一化logprob最大的（perplexity最小的）
            predicted_answer = max(option_normalized_logprobs, key=option_normalized_logprobs.get)
            is_correct = (predicted_answer == reference_answer)

            with self._lock:
                if is_correct:
                    self._correct_count += 1
                self._total_count += 1

            return {
                'goal': goal,
                'predicted_answer': predicted_answer,
                'reference_answer': reference_answer,
                'is_correct': is_correct,
                'logprobs': option_logprobs,
                'lengths': option_lengths,
                'normalized_logprobs': option_normalized_logprobs,
                'ppls': option_ppls,
            }
        except ModelResponseError:
            raise
        except Exception as e:
            print(f"\n评估问题时出错: {e}")
            return {
                'goal': question_data.get('goal', ''),
                'predicted_answer': '',
                'reference_answer': question_data.get('answer', ''),
                'is_correct': False,
                'error': str(e)
            }

    def evaluate_dataset(self, data: List[Dict], max_samples: int = None, seed: int = 42) -> Dict:
        """评估整个数据集"""
        import random

        if max_samples:
            random.seed(seed)
            data = random.sample(data, min(max_samples, len(data)))
            print(f"随机采样: {max_samples} 个样本（seed={seed}）")

        self._correct_count = 0
        self._total_count = 0
        self._failed_extractions = 0

        # 预热检测：先测试3个样本
        print("\n执行预热检测（测试3个样本）...")
        warmup_samples = data[:min(3, len(data))]
        warmup_failed = False
        for i, sample in enumerate(warmup_samples):
            try:
                result = self.evaluate_single_question(sample)
                print(f"✓ 样本{i+1}: 测试通过")
                
                # 检测异常logprobs
                if 'normalized_logprobs' in result:
                    lps = result['normalized_logprobs']
                    if lps and all(v <= -10.0 or v is None for v in lps.values()):
                        print(f"⚠️  警告: 样本{i+1}的logprobs全部异常: {lps}")
                        warmup_failed = True
                        
            except Exception as e:
                print(f"⚠️  样本{i+1}失败: {e}")
                warmup_failed = True
        
        if warmup_failed:
            print(f"\n❌ 预热检测发现异常！")
            print(f"请检查：")
            print(f"  1. API endpoint是否正确")
            print(f"  2. 模型是否正常运行")
            print(f"  3. 网络连接是否正常")
            raise ModelResponseError("预热检测失败，中止评估")
        print("✓ 预热检测通过，开始正式评估...\n")

        # 重置计数器（预热检测已完成，不计入正式统计）
        self._correct_count = 0
        self._total_count = 0
        self._failed_extractions = 0

        # 正式评估：评估所有样本（预热检测不跳过任何数据）
        eval_data = data
        total = len(eval_data)
        
        print(f"开始评估 {len(data)} 个问题...")
        print(f"并发设置: {self.max_workers} 个问题, 每个问题 {self.max_workers_per_question} 个选项")
        print("方法: PPL（长度归一化）")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.evaluate_single_question, item): item for item in data}

            with tqdm(total=len(eval_data), desc="评估进度", unit="问题") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                        
                        with self._lock:
                            current_total = self._total_count
                            current_correct = self._correct_count
                            failed = self._failed_extractions
                        
                        if current_total % 100 == 0 and current_total > 0:
                            current_acc = current_correct / current_total * 100
                            pbar.set_postfix({
                                'accuracy': f'{current_acc:.2f}%',
                                'correct': current_correct,
                                'failed': failed
                            })
                    except ModelResponseError as e:
                        print(f"\n模型响应错误，评估中止: {e}")
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        pbar.close()
                        raise
                    except Exception as e:
                        print(f"\n评估失败: {e}")
                        pbar.update(1)

        return {
            'total': self._total_count,
            'correct': self._correct_count,
            'accuracy': self._correct_count / self._total_count * 100 if self._total_count > 0 else 0.0,
            'failed_extractions': self._failed_extractions,
            'seed': seed if max_samples else None,
            'max_samples': max_samples,
            'results': results
        }


DEFAULT_PIQA_DIR = os.path.join(os.path.dirname(__file__), "datasets", "piqa")


def find_piqa_data_path(dataset_dir: str = None) -> Tuple[Optional[str], Optional[str]]:
    """查找PIQA数据集路径"""
    if dataset_dir is None:
        dataset_dir = DEFAULT_PIQA_DIR
    
    dev_file = os.path.join(dataset_dir, 'dev.jsonl')
    label_file = os.path.join(dataset_dir, 'dev-labels.lst')
    
    if os.path.exists(dev_file):
        return dev_file, label_file if os.path.exists(label_file) else None
    
    return None, None


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="PIQA 评估脚本")

    parser.add_argument(
        "--shot-num",
        type=int,
        default=0,
        help=f"Few-shot 示例数量（默认: 0）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EVAL_MODEL_NAME", "qwen2-5-72b"),
        help="模型名称（默认: eval-qwen2-5-72b）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1"),
        help="API 基础 URL",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=128,
        help="最大并发工作线程数（默认: 128）",
    )
    parser.add_argument(
        "--max-workers-per-question",
        type=int,
        default=2,
        help="每个问题的最大并发工作线程数（默认: 2）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="随机采样大小，None 表示评估所有数据（默认: None）",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help=f"数据集目录（默认: {DEFAULT_PIQA_DIR}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )

    args = parser.parse_args()

    # 查找数据集路径
    dataset_dir = args.dataset_dir or DEFAULT_PIQA_DIR
    dev_file, label_file = find_piqa_data_path(dataset_dir)
    
    if dev_file is None:
        print(f"⚠️  未找到 PIQA 数据集")
        print(f"请先运行以下命令下载数据集:")
        print(f"  python download_datasets.py piqa")
        print(f"或者手动将数据集放置在: {dataset_dir}")
        return

    log_dir = os.path.join(os.path.dirname(__file__), "logs", args.model)
    os.makedirs(log_dir, exist_ok=True)

    output_filename = f"piqa_0shot.json"
    if args.max_samples:
        output_filename = f"piqa_0shot_{args.max_samples}samples.json"
    output_path = os.path.join(log_dir, output_filename)

    print("=" * 70)
    print(f"{args.shot_num}-shot PPL方法（长度归一化）")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"API: completions + echo=True")
    print(f"方法: PPL评估 + 长度归一化")
    print("=" * 70)

    evaluator = PIQAEvaluator(
        base_url=args.base_url,
        model=args.model,
        api_key="EMPTY",
        max_workers=args.max_workers,
        max_workers_per_question=args.max_workers_per_question,
    )

    print(f"\n加载数据: {dev_file}")
    if label_file:
        print(f"加载标签: {label_file}")
    try:
        data = evaluator.load_piqa_data(dev_file, label_file)
        print(f"✓ 成功加载 {len(data)} 条数据")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return

    if args.max_samples:
        print(f"将随机采样 {args.max_samples} 个样本（seed={args.seed}）")

    print("\n开始评估...")

    try:
        results = evaluator.evaluate_dataset(data, max_samples=args.max_samples, seed=args.seed)

        print(f"\n保存结果: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print("评估结果")
        print("=" * 70)
        print(f"总问题数: {results['total']}")
        print(f"正确答案数: {results['correct']}")
        print(f"准确率: {results['accuracy']:.2f}%")
        print(f"{args.shot_num}-shot")
        if args.max_samples:
            print(f"随机采样: {args.max_samples} 个样本（seed={args.seed}）")

        if results.get('failed_extractions', 0) > 0:
            fail_rate = results['failed_extractions'] / (results['total'] * 2) * 100
            print(f"\nLogprob提取失败: {results['failed_extractions']}")
            print(f"失败率: {fail_rate:.2f}%")
            if fail_rate > 5:
                print("⚠️  失败率较高，可能影响准确率")

        print(f"\n详细结果已保存到: {output_path}")
        print(f"日志目录: {log_dir}")
        print("=" * 70)

    except ModelResponseError as e:
        print(f"\n✗ 检测到模型响应错误: {e}")
        print("评估已终止，未生成结果文件。")
    except KeyboardInterrupt:
        print("\n\n⚠️  评估被用户中断")
        if evaluator._total_count > 0:
            partial_results = {
                "accuracy": evaluator._correct_count / evaluator._total_count * 100,
                "correct": evaluator._correct_count,
                "total": evaluator._total_count,
                "note": "部分结果（已中断）",
            }
            partial_path = output_path.replace(".json", "_partial.json")
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(partial_results, f, ensure_ascii=False, indent=2)
            print(f"部分结果已保存到: {partial_path}")
    except Exception as e:
        print(f"\n✗ 评估过程中出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
