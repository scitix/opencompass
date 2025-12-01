"""WinoGrande数据集评估脚本 - PPL方法（长度归一化）。"""
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


class WinoGrandeEvaluator:
    """使用PPL方法评估WinoGrande数据集（两个前缀，选择perplexity最低的）"""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_workers: int = 128, max_workers_per_question: int = 2, shot_num: int = 10):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_workers_per_question = min(max_workers_per_question, 2)
        self.shot_num = shot_num
        self._correct_count = 0
        self._total_count = 0
        self._failed_extractions = 0
        self._lock = threading.Lock()
        self._few_shot_examples: List[Dict] = []
        
        # 预加载tokenizer（只加载一次）
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
        except Exception as e:
            print(f"警告: 无法加载tokenizer: {e}")
            print("将使用简化的logprob提取方法")
            self._tokenizer = None

    def load_winogrande_data(self, val_file: str, train_file: str = None) -> List[Dict]:
        """从JSONL文件加载WinoGrande数据"""
        # 加载few-shot示例（从训练集）
        if train_file and os.path.exists(train_file):
            self._few_shot_examples = []
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    sentence = item.get('sentence', '')
                    option1 = item.get('option1', '')
                    option2 = item.get('option2', '')
                    answer = item.get('answer', '')
                    
                    if not sentence or "_" not in sentence:
                        continue
                    if not option1 or not option2 or not answer:
                        continue
                    
                    try:
                        answer_idx = int(answer) - 1
                        if answer_idx not in (0, 1):
                            continue
                    except ValueError:
                        continue
                    
                    self._few_shot_examples.append({
                        'sentence': sentence,
                        'option1': option1,
                        'option2': option2,
                        'answer': 'AB'[answer_idx],
                    })
                    
                    if len(self._few_shot_examples) >= self.shot_num:
                        break
        
        # 加载验证集
        data = []
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                sentence = item.get('sentence', '')
                option1 = item.get('option1', '')
                option2 = item.get('option2', '')
                answer = item.get('answer', '')
                
                if not sentence or "_" not in sentence:
                    continue
                if not option1 or not option2 or not answer:
                    continue
                
                try:
                    answer_idx = int(answer) - 1
                    if answer_idx not in (0, 1):
                        continue
                except ValueError:
                    continue
                
                data.append({
                    'sentence': sentence,
                    'option1': option1,
                    'option2': option2,
                    'answer': 'AB'[answer_idx],
                })
        
        return data

    def build_few_shot_prompt(self) -> str:
        """构建few-shot prompt"""
        if not self._few_shot_examples:
            return ""
        
        prompt_parts = []
        for example in self._few_shot_examples[:self.shot_num]:
            sentence = example['sentence']
            option1 = example['option1']
            option2 = example['option2']
            answer = example['answer']
            
            # 替换下划线为正确答案
            sentence_with_answer = sentence.replace('_', option1 if answer == 'A' else option2)
            prompt_parts.append(sentence_with_answer)
        
        return "\n\n".join(prompt_parts) + "\n\n" if prompt_parts else ""

    def build_prompt(self, sentence_with_option: str) -> str:
        """构建评估prompt（PPL方法）"""
        few_shot = self.build_few_shot_prompt()
        prompt = f"{few_shot}{sentence_with_option}"
        return prompt

    def get_completion_logprob_with_prefix(self, prefix_with_option: str, completion: str, max_retries: int = 3) -> Tuple[float, int]:
        """
        获取completion在给定prefix下的logprob（符合论文要求）
        
        按照论文要求：
        - prefix_with_option: sentence前半部分 + option（"_"之前 + 选项词）
        - completion: sentence后半部分（"_"之后的内容）
        
        只计算completion部分的logprob
        """
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                # 构建完整的prompt = prefix_with_option + completion
                full_prompt = prefix_with_option + completion
                
                response = self.client.completions.create(
                    model=self.model,
                    prompt=full_prompt,
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

                # 使用预加载的tokenizer来确定prefix的token数量
                if self._tokenizer is not None:
                    try:
                        prefix_tokens = self._tokenizer.encode(prefix_with_option, add_special_tokens=False)
                        prefix_len = len(prefix_tokens)
                        
                        # 提取completion部分的logprobs
                        # +1 因为第一个token的logprob是None
                        completion_logprobs = []
                        for i in range(prefix_len + 1, len(token_logprobs)):
                            if token_logprobs[i] is not None:
                                completion_logprobs.append(token_logprobs[i])
                        
                        if not completion_logprobs:
                            # 如果无法提取，使用全部token
                            completion_logprobs = [lp for lp in token_logprobs[1:] if lp is not None]
                        
                        total_completion_logprob = sum(completion_logprobs)
                        completion_length = max(1, len(completion_logprobs))
                        
                        return total_completion_logprob, completion_length
                        
                    except Exception:
                        pass  # 回退到简单方法
                
                # tokenizer不可用或失败，回退到简单方法（使用全句logprob）
                total_logprob = sum([lp for lp in token_logprobs[1:] if lp is not None])
                token_length = max(1, len(tokens) - 1)
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

    def _get_logprob_for_option(
        self, sentence: str, option: str, label: str
    ) -> Tuple[str, float, int]:
        """
        获取单个选项的logprob和长度（用于并发调用）
        
        按照论文要求：
        - prefix = sentence前半部分 + option
        - completion = sentence后半部分
        - 只计算completion在该prefix条件下的logprob
        """
        # 分离prefix和completion
        if '_' not in sentence:
            raise ValueError(f"句子中没有找到'_'占位符: {sentence}")
        
        parts = sentence.split('_', 1)
        sentence_before = parts[0]  # "_"之前
        sentence_after = parts[1] if len(parts) > 1 else ""  # "_"之后（completion）
        
        # 构建带few-shot的prefix
        few_shot_prompt = self.build_few_shot_prompt()
        prefix_with_option = few_shot_prompt + "\n" + sentence_before + option if few_shot_prompt else sentence_before + option
        
        # 计算completion在该prefix下的logprob
        logprob, length = self.get_completion_logprob_with_prefix(prefix_with_option, sentence_after)
        return label, logprob, length

    def evaluate_single_question(self, question_data: Dict) -> Dict:
        """评估单个问题（并发获取两个选项的logprob）"""
        try:
            sentence = question_data['sentence']
            option1 = question_data['option1']
            option2 = question_data['option2']
            reference_answer = question_data['answer'].strip().upper()
            
            option_labels = ['A', 'B']
            options = [option1, option2]
            option_logprobs = {}
            option_lengths = {}
            option_normalized_logprobs = {}
            option_ppls = {}

            with ThreadPoolExecutor(max_workers=self.max_workers_per_question) as executor:
                futures = {
                    executor.submit(
                        self._get_logprob_for_option,
                        sentence,
                        option,
                        label,
                    ): (label, option)
                    for label, option in zip(option_labels, options)
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
                'sentence': sentence,
                'option1': option1,
                'option2': option2,
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
                'sentence': question_data.get('sentence', ''),
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
        for i, sample in enumerate(warmup_samples):
            try:
                result = self.evaluate_single_question(sample)
                sentence_preview = sample.get('sentence', '')[:50]
                pred = result.get('predicted_answer', '')
                print(f"✓ 样本{i+1}: 句子='{sentence_preview}...' 预测={pred}")
                
                # 检测异常输出
                if not pred or pred.strip() == "":
                    raise ModelResponseError(f"预热检测失败：模型返回空响应")
                    
                # 检查logprobs是否正常
                logprobs = result.get('logprobs', {})
                if not logprobs or all(v == -10.0 for v in logprobs.values()):
                    raise ModelResponseError(f"预热检测失败：所有logprobs异常")
                    
            except ModelResponseError as e:
                print(f"\n❌ 预热检测失败！")
                print(f"错误: {e}")
                print(f"请检查：")
                print(f"  1. API endpoint是否正确")
                print(f"  2. 模型是否正常运行")
                print(f"  3. 网络连接是否正常")
                raise
            except Exception as e:
                print(f"\n❌ 预热检测时出现未预期错误！")
                print(f"错误: {e}")
                raise ModelResponseError(f"预热检测失败: {e}")
        print("✓ 预热检测通过，开始正式评估...\n")

        # 重置计数器（预热检测已完成，不计入正式统计）
        self._correct_count = 0
        self._total_count = 0
        self._failed_extractions = 0

        # 正式评估：评估所有样本（预热检测不跳过任何数据）
        eval_examples = data
        print(f"正式评估 {len(eval_examples)} 条样本（包括预热样本）\n")
        total = len(eval_examples)
        
        print(f"开始评估 {total} 个问题...")
        print(f"并发设置: {self.max_workers} 个问题, 每个问题 {self.max_workers_per_question} 个选项")
        print(f"方法: PPL（长度归一化，{self.shot_num}-shot）")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.evaluate_single_question, item): item for item in eval_examples}

            with tqdm(total=total, desc="评估进度", unit="问题") as pbar:
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


DEFAULT_WINOGRANDE_DIR = os.path.join(os.path.dirname(__file__), "datasets", "winogrande")


def find_winogrande_data_path(dataset_dir: str = None) -> Tuple[Optional[str], Optional[str]]:
    """查找WinoGrande数据集路径"""
    if dataset_dir is None:
        dataset_dir = DEFAULT_WINOGRANDE_DIR
    
    val_file = os.path.join(dataset_dir, 'dev.jsonl')
    train_file = os.path.join(dataset_dir, 'train_xs.jsonl')
    
    if os.path.exists(val_file):
        return val_file, train_file if os.path.exists(train_file) else None
    
    return None, None


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="WinoGrande 评估脚本")
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
        "--shot-num",
        type=int,
        default=10,
        help="Few-shot 示例数量（默认: 5）",
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
        help=f"数据集目录（默认: {DEFAULT_WINOGRANDE_DIR}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )

    args = parser.parse_args()

    # 查找数据集路径
    dataset_dir = args.dataset_dir or DEFAULT_WINOGRANDE_DIR
    val_file, train_file = find_winogrande_data_path(dataset_dir)
    
    if val_file is None:
        print(f"⚠️  未找到 WinoGrande 数据集")
        print(f"请先运行以下命令下载数据集:")
        print(f"  python download_datasets.py winogrande")
        print(f"或者手动将数据集放置在: {dataset_dir}")
        return

    log_dir = os.path.join(os.path.dirname(__file__), "logs", args.model)
    os.makedirs(log_dir, exist_ok=True)

    shot_desc = f"{args.shot_num}-shot" if args.shot_num > 0 else "0-shot"
    output_filename = f"winogrande_{args.shot_num}shot.json"
    if args.max_samples:
        output_filename = f"winogrande_{args.shot_num}shot_{args.max_samples}samples.json"
    output_path = os.path.join(log_dir, output_filename)

    print("=" * 70)
    print(f"WinoGrande 评估 - {args.model} - {shot_desc} PPL方法（长度归一化）")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"API: completions + echo=True")
    print(f"方法: {shot_desc} PPL评估 + 长度归一化")
    if train_file:
        print(f"Few-shot来源: {train_file}")
    print("=" * 70)

    evaluator = WinoGrandeEvaluator(
        base_url=args.base_url,
        model=args.model,
        api_key="EMPTY",
        max_workers=args.max_workers,
        max_workers_per_question=args.max_workers_per_question,
        shot_num=args.shot_num
    )

    print(f"\n加载数据: {val_file}")
    if train_file:
        print(f"加载 Few-shot 示例: {train_file}")
    try:
        data = evaluator.load_winogrande_data(val_file, train_file)
        print(f"✓ 成功加载 {len(data)} 条验证数据")
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
