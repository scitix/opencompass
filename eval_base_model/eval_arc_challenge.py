"""ARC-Challenge数据集评估脚本 - PPL方法。"""
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


class ARCChallengeEvaluator:
    """使用25-shot PPL方法评估ARC-Challenge数据集"""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_workers: int = 128, max_workers_per_question: int = 4, shot_num: int = 25):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_workers_per_question = min(max_workers_per_question, 5)  # 最多5个选项
        self.shot_num = shot_num
        self._correct_count = 0
        self._total_count = 0
        self._failed_extractions = 0
        self._lock = threading.Lock()
        self._few_shot_examples: List[Dict] = []

    def load_arc_challenge_data(self, test_file: str, dev_file: str = None) -> List[Dict]:
        """从JSONL文件加载ARC-Challenge数据"""
        # 加载few-shot示例（从dev集，只使用4选项的示例以保持一致性）
        if dev_file and os.path.exists(dev_file):
            self._few_shot_examples = []
            with open(dev_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    question = item.get('question', {})
                    if isinstance(question, dict):
                        stem = question.get('stem', '')
                        choices = question.get('choices', [])
                    else:
                        stem = str(question)
                        choices = []
                    
                    # Few-shot示例只使用4选项的，保持格式一致
                    if len(choices) != 4:
                        continue
                    
                    labels = [c.get('label', '') for c in choices]
                    answer_key = item.get('answerKey', '')
                    answer_idx = labels.index(answer_key) if answer_key in labels else -1
                    if answer_idx == -1:
                        continue
                    
                    answer_label = 'ABCD'[answer_idx]
                    
                    self._few_shot_examples.append({
                        'question': stem,
                        'choices': [c.get('text', '') for c in choices],
                        'answerKey': answer_label,
                    })
                    
                    if len(self._few_shot_examples) >= self.shot_num:
                        break
        
        # 加载测试集（支持3、4、5选项）
        data = []
        option_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # 支持最多26个选项
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                item = json.loads(line)
                question = item.get('question', {})
                if isinstance(question, dict):
                    stem = question.get('stem', '')
                    choices = question.get('choices', [])
                else:
                    stem = str(question)
                    choices = []
                
                # 支持3、4、5选项（以及更多选项）
                if len(choices) < 2 or len(choices) > 26:
                    continue
                
                labels = [c.get('label', '') for c in choices]
                answer_key = item.get('answerKey', '')
                answer_idx = labels.index(answer_key) if answer_key in labels else -1
                if answer_idx == -1:
                    continue
                
                answer_label = option_labels[answer_idx]
                
                data.append({
                    'question': stem,
                    'choices': [c.get('text', '') for c in choices],
                    'answerKey': answer_label,
                })
        
        return data

    def build_few_shot_prompt(self) -> str:
        """构建few-shot prompt"""
        if not self._few_shot_examples:
            return ""
        
        option_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        prompt_parts = []
        for example in self._few_shot_examples[:self.shot_num]:
            choices_text = []
            for i, choice_text in enumerate(example['choices']):
                label = option_labels[i]
                choices_text.append(f"{label}. {choice_text}")
            
            prompt_parts.append(
                f"Question: {example['question']}\n"
                + "\n".join(choices_text) + "\n"
                + f"Answer: {example['answerKey']}"
            )
        
        return "\n\n".join(prompt_parts) + "\n\n"

    def build_prompt(self, question: str, choices: List[str], option_label: str) -> str:
        """构建评估prompt（PPL方法）"""
        few_shot = self.build_few_shot_prompt()
        option_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        choices_text = []
        for i, choice_text in enumerate(choices):
            label = option_labels[i]
            choices_text.append(f"{label}. {choice_text}")
        
        choices_str = "\n".join(choices_text)
        prompt = f"""{few_shot}Question: {question}
{choices_str}
Answer: {option_label}"""
        return prompt

    def extract_option_logprob(
        self, tokens: List[str], token_logprobs: List[Optional[float]], option_label: str
    ) -> Optional[float]:
        """从token列表中提取选项标签的logprob（从后向前搜索，因为答案在末尾）"""
        if not tokens or not token_logprobs:
            return None
            
        max_len = min(len(tokens), len(token_logprobs))
        if max_len == 0:
            return None
            
        start_idx = max_len - 1
        end_idx = max(0, max_len - 15)
        
        for i in range(start_idx, end_idx - 1, -1):
            if i < 0 or i >= len(tokens) or i >= len(token_logprobs):
                continue
                
            token = tokens[i]
            token_stripped = token.strip()

            if token_stripped == option_label:
                if token_logprobs[i] is not None:
                    return token_logprobs[i]

            if token == f" {option_label}":
                if token_logprobs[i] is not None:
                    return token_logprobs[i]

            if option_label in token_stripped and len(token_stripped) <= 2:
                if token_logprobs[i] is not None:
                    return token_logprobs[i]

        return None

    def get_option_logprob(self, prompt: str, option_label: str, max_retries: int = 3) -> float:
        """使用completions API获取选项的logprob（echo=True）"""
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=1,
                    logprobs=5,
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

                option_logprob = self.extract_option_logprob(tokens, token_logprobs, option_label)

                if option_logprob is not None:
                    return option_logprob

                raise ModelResponseError(f'无法提取选项 {option_label} 的logprob。')

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
    
    def get_unconditional_logprob(self, question: str, choices: List[str], label: str, max_retries: int = 3) -> float:
        """
        获取无条件logprob（Domain-Conditional方法）
        
        使用通用问题替代具体问题，保持prompt格式一致，只提取label token的logprob
        
        Args:
            question: 原问题（保持格式参考）
            choices: 选项列表
            label: 选项标签 (A/B/C/D等)
        
        Returns:
            float: 无条件logprob（label token的logprob）
        """
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                few_shot = self.build_few_shot_prompt()
                
                # 构建无条件prompt：格式相同，但问题内容为通用问题
                # 关键：保持与conditional prompt相同的格式和结构
                # 动态生成选项，支持任意数量的选项（3、4、5等）
                option_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                choices_text = []
                for i, choice_text in enumerate(choices):
                    if i >= len(option_labels):
                        break
                    label_char = option_labels[i]
                    choices_text.append(f"{label_char}. {choice_text}")
                
                choices_str = "\n".join(choices_text)
                prompt = f"""{few_shot}Question: What is the correct answer?
{choices_str}
Answer: {label}"""
                
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=1,
                    logprobs=5,
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
                    raise ModelResponseError('模型返回的tokens或token_logprobs为空。')

                # 从后往前查找label token的logprob
                # 确保索引范围有效
                max_len = min(len(tokens), len(token_logprobs))
                if max_len == 0:
                    raise ModelResponseError('tokens和token_logprobs长度为0')
                    
                start_idx = max_len - 1
                end_idx = max(0, max_len - 15)
                
                for i in range(start_idx, end_idx - 1, -1):
                    if i >= 0 and i < len(tokens) and i < len(token_logprobs):
                        token = tokens[i].strip()
                        if token == label and token_logprobs[i] is not None:
                            return token_logprobs[i]
                
                # 如果未找到，返回一个中性值而不是抛出异常
                print(f"警告: 未找到label {label} 的token，返回默认值")
                return -1.0

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

        if last_exception:
            print(f"获取无条件logprob失败: {last_exception}")
            return -1.0
        
        return -1.0

    def _get_logprob_for_label(
        self, question: str, choices: List[str], label: str
    ) -> Tuple[str, float]:
        """获取单个选项的logprob（用于并发调用）- 使用无条件归一化（Domain-Conditional方法）"""
        # 获取条件logprob（包含问题上下文）
        prompt = self.build_prompt(question, choices, label)
        conditional_logprob = self.get_option_logprob(prompt, label)
        
        # 获取无条件logprob（使用通用问题，保持格式一致）
        unconditional_logprob = self.get_unconditional_logprob(question, choices, label)
        
        # 无条件归一化: conditional - unconditional
        normalized_logprob = conditional_logprob - unconditional_logprob
        
        return label, normalized_logprob

    def evaluate_single_question(self, question_data: Dict) -> Dict:
        """评估单个问题（并发获取所有选项的logprob）"""
        try:
            question = question_data['question']
            choices = question_data['choices']
            reference_answer = question_data['answerKey'].strip().upper()
            
            option_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(choices)]
            option_logprobs = {}
            option_ppls = {}

            with ThreadPoolExecutor(max_workers=self.max_workers_per_question) as executor:
                futures = {
                    executor.submit(
                        self._get_logprob_for_label,
                        question,
                        choices,
                        label,
                    ): label
                    for label in option_labels
                }

                for future in as_completed(futures):
                    try:
                        label, logprob = future.result()
                        option_logprobs[label] = logprob
                        option_ppls[label] = np.exp(-logprob)
                    except ModelResponseError:
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        raise
                    except Exception as e:
                        label = futures[future]
                        print(f"\n获取选项 {label} 的logprob时出错: {e}")
                        option_logprobs[label] = -10.0
                        option_ppls[label] = np.exp(10.0)
                        with self._lock:
                            self._failed_extractions += 1

            # 确保所有选项都有logprob
            for label in option_labels:
                if label not in option_logprobs:
                    option_logprobs[label] = -10.0
                    option_ppls[label] = np.exp(10.0)

            predicted_answer = max(option_logprobs, key=option_logprobs.get)
            is_correct = (predicted_answer == reference_answer)

            with self._lock:
                if is_correct:
                    self._correct_count += 1
                self._total_count += 1

            return {
                'question': question,
                'predicted_answer': predicted_answer,
                'reference_answer': reference_answer,
                'is_correct': is_correct,
                'logprobs': option_logprobs,
                'ppls': option_ppls,
            }
        except ModelResponseError:
            raise
        except Exception as e:
            print(f"\n评估问题时出错: {e}")
            return {
                'question': question_data.get('question', ''),
                'predicted_answer': '',
                'reference_answer': question_data.get('answerKey', ''),
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
                print(f"✓ 样本 {i+1}: 通过")
                
                # 检测异常logprobs
                if result.get('logprobs') and all(v <= -10.0 or v is None for v in result.get('logprobs', {}).values()):
                    print(f"⚠️  警告: 样本 {i+1} 有异常logprobs: {result.get('logprobs')}")
                    warmup_failed = True
                        
            except Exception as e:
                print(f"⚠️  样本 {i+1} 失败: {e}")
                warmup_failed = True
        
        if warmup_failed:
            print(f"\n❌ 预热检测发现异常！")
            print(f"请检查:")
            print(f"  1. API端点是否正确")
            print(f"  2. 模型是否正常运行")
            print(f"  3. 网络连接是否稳定")
            raise ModelResponseError("预热检测失败，中止评估")
        print("✓ 预热检测通过，开始正式评估...\n")

        # 重置计数器（预热检测已完成，不计入正式统计）
        self._correct_count = 0
        self._total_count = 0
        self._failed_extractions = 0

        print(f"开始评估 {len(data)} 个问题...")
        print(f"并发设置: {self.max_workers} 个问题, 每个问题 {self.max_workers_per_question} 个选项")
        print("方法: PPL（无条件归一化）")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.evaluate_single_question, item): item for item in data}

            with tqdm(total=len(data), desc="评估进度", unit="问题") as pbar:
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
            'shot_num': self.shot_num,
            'seed': seed if max_samples else None,
            'max_samples': max_samples,
            'results': results
        }


DEFAULT_ARC_EASY_DIR = os.path.join(os.path.dirname(__file__), "datasets", "arc_challenge")


def find_arc_challenge_data_path(dataset_dir: str = None) -> Tuple[Optional[str], Optional[str]]:
    """查找ARC-Challenge数据集路径"""
    if dataset_dir is None:
        dataset_dir = DEFAULT_ARC_EASY_DIR
    
    test_file = os.path.join(dataset_dir, 'ARC-Challenge-Test.jsonl')
    dev_file = os.path.join(dataset_dir, 'ARC-Challenge-Dev.jsonl')
    
    if os.path.exists(test_file) and os.path.exists(dev_file):
        return test_file, dev_file
    
    return None, None


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="ARC-Challenge 评估脚本")
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
        default=4,
        help="每个问题的最大并发工作线程数（默认: 4）",
    )
    parser.add_argument(
        "--shot-num",
        type=int,
        default=25,
        help="Few-shot 示例数量（默认: 25）",
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
        help=f"数据集目录（默认: {DEFAULT_ARC_EASY_DIR}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )

    args = parser.parse_args()

    # 查找数据集路径
    dataset_dir = args.dataset_dir or DEFAULT_ARC_EASY_DIR
    test_file, dev_file = find_arc_challenge_data_path(dataset_dir)
    
    if test_file is None or dev_file is None:
        print(f"⚠️  未找到 ARC-Challenge 数据集")
        print(f"请先运行以下命令下载数据集:")
        print(f"  python download_datasets.py arc_challenge")
        print(f"或者手动将数据集放置在: {dataset_dir}")
        return

    log_dir = os.path.join(os.path.dirname(__file__), "logs", args.model)
    os.makedirs(log_dir, exist_ok=True)

    shot_desc = f"{args.shot_num}-shot" if args.shot_num > 0 else "0-shot"
    output_filename = f"arc_challenge_{args.shot_num}shot.json"
    if args.max_samples:
        output_filename = f"arc_challenge_{args.shot_num}shot_{args.max_samples}samples.json"
    output_path = os.path.join(log_dir, output_filename)

    print("=" * 70)
    print(f"ARC-Challenge 评估 - {args.model} - {shot_desc} PPL方法")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"API: completions + echo=True")
    print(f"方法: {shot_desc} prompting + PPL评估")
    print(f"Few-shot来源: {dev_file}")
    print("=" * 70)

    evaluator = ARCChallengeEvaluator(
        base_url=args.base_url,
        model=args.model,
        api_key="EMPTY",
        max_workers=args.max_workers,
        max_workers_per_question=args.max_workers_per_question,
        shot_num=args.shot_num
    )

    print(f"\n加载数据: {test_file}")
    print(f"加载 Few-shot 示例: {dev_file}")
    try:
        data = evaluator.load_arc_challenge_data(test_file, dev_file)
        print(f"✓ 成功加载 {len(data)} 条测试数据")
        print(f"✓ 成功加载 {len(evaluator._few_shot_examples)} 个 few-shot 示例")
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
        print(f"Shot 配置: {shot_desc}")
        if args.max_samples:
            print(f"随机采样: {args.max_samples} 个样本（seed={args.seed}）")

        if results.get('failed_extractions', 0) > 0:
            fail_rate = results['failed_extractions'] / (results['total'] * len('ABCDE')) * 100
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
                "shot_num": args.shot_num,
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

