"""MMLU-Redux数据集评估脚本 - Base模型标准PPL方法。"""
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import random
import os
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import eval_utils
class ModelResponseError(RuntimeError):
    """Raised when the model response does not contain usable logprob data."""


class MMLUReduxEvaluator:
    """使用标准PPL方法评估MMLU-Redux数据集"""
    
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", 
                 max_workers: int = 32, max_workers_per_question: int = 10,
                 shot_num: int = 5, mmlu_dataset_dir: Optional[str] = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_workers_per_question = max_workers_per_question
        self.shot_num = shot_num
        self.mmlu_dataset_dir = mmlu_dataset_dir
        self._correct_count = 0
        self._total_count = 0
        self._failed_extractions = 0
        self._lock = threading.Lock()
        self._few_shot_examples: Optional[List[Dict]] = None
        self._few_shot_prompt: Optional[str] = None
        
    def load_mmlu_redux_data(self, parquet_path: str) -> List[Dict]:
        """从本地parquet文件加载MMLU-Redux数据"""
        df = pd.read_parquet(parquet_path)
        
        data = []
        for _, row in df.iterrows():
            # MMLU-Redux的choices是一个numpy array或列表
            choices = row.get('choices', [])
            if isinstance(choices, np.ndarray):
                choices = choices.tolist()
            if not isinstance(choices, list):
                choices = list(choices)
            
            correct_answer = row.get('correct_answer', '')
            
            # 将choices转换为A, B, C, D格式
            options = {}
            answer_label = None
            for i, choice in enumerate(choices):
                label = chr(65 + i)  # A, B, C, D, ...
                options[label] = str(choice)
                if str(choice) == str(correct_answer):
                    answer_label = label
            
            # 优先使用 category 字段，如果没有则使用 config
            # 确保 category 不为空或 None
            category = row.get('category')
            if category is None or str(category).strip() == "":
                category = row.get('config')
            if category is None or str(category).strip() == "":
                category = 'unknown'
            category = str(category).strip()
            
            data.append({
                'question': str(row.get('question', '')),
                'choices': choices,
                'options': options,
                'correct_answer': str(correct_answer),
                'answer': answer_label if answer_label else 'A',
                'num_options': len(choices),
                'config': category,  # 使用 category 作为 config
                'category': category  # 同时保留 category 字段
            })
        
        return data
    
    def _load_mmlu_few_shot_examples(self) -> List[Dict]:
        """从 MMLU 数据集加载前N条作为通用few-shot示例"""
        if self._few_shot_examples is not None:
            return self._few_shot_examples
        
        if not self.mmlu_dataset_dir or not os.path.isdir(self.mmlu_dataset_dir):
            raise FileNotFoundError(
                f"MMLU 数据集目录不存在: {self.mmlu_dataset_dir}，无法加载 few-shot 示例"
            )
        
        # 加载 MMLU dev 集
        dev_dir = os.path.join(self.mmlu_dataset_dir, "dev")
        if not os.path.isdir(dev_dir):
            dev_dir = os.path.join(self.mmlu_dataset_dir, "validation")
        
        if not os.path.isdir(dev_dir):
            raise FileNotFoundError(
                f"MMLU dev/validation 目录不存在: {dev_dir}"
            )
        
        examples = []
        # 尝试查找 arrow 文件（优先）或 jsonl 文件
        arrow_files = sorted([f for f in os.listdir(dev_dir) if f.endswith('.arrow')])
        jsonl_files = sorted([f for f in os.listdir(dev_dir) if f.endswith('.jsonl')])
        
        if not arrow_files and not jsonl_files:
            raise FileNotFoundError(f"MMLU dev 目录中没有找到 arrow 或 JSONL 文件: {dev_dir}")
        
        examples = []
        
        # 简单地加载前N条作为通用few-shot示例
        if arrow_files:
            for arrow_file in arrow_files:
                if len(examples) >= self.shot_num:
                    break
                    
                file_path = os.path.join(dev_dir, arrow_file)
                try:
                    with pa.memory_map(file_path, "r") as source:
                        try:
                            reader = pa.ipc.open_file(source)
                        except pa.ArrowInvalid:
                            reader = pa.ipc.open_stream(source)
                        table = reader.read_all()
                        
                        # 读取前N条
                        for i in range(min(table.num_rows, self.shot_num - len(examples))):
                            row = {col: table[col][i].as_py() for col in table.column_names}
                            
                            question = row.get("question", "")
                            choices = row.get("choices", [])
                            answer = row.get("answer")
                            
                            if not question or not choices:
                                continue
                            
                            # 标准化答案标签
                            if isinstance(answer, int) and 0 <= answer < len(choices):
                                answer_label = chr(65 + answer)  # 0->A, 1->B, etc.
                            elif isinstance(answer, str):
                                answer_label = answer.strip().upper()
                                if answer_label not in ["A", "B", "C", "D"]:
                                    continue
                            else:
                                continue
                            
                            examples.append({
                                "question": question,
                                "choices": choices,
                                "answer": answer_label
                            })
                            
                except Exception as e:
                    print(f"警告: 读取 {arrow_file} 时出错: {e}")
                    continue
        
        # 如果 arrow 文件没有足够数据，尝试从 jsonl 文件加载
        elif jsonl_files:
            for jsonl_file in jsonl_files:
                if len(examples) >= self.shot_num:
                    break
                
                file_path = os.path.join(dev_dir, jsonl_file)
                
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(examples) >= self.shot_num:
                            break
                            
                        line = line.strip()
                        if not line:
                            continue
                        sample = json.loads(line)
                        question = sample.get("question") or sample.get("input", "")
                        choices = sample.get("choices", [])
                        if len(choices) != 4:
                            continue
                        answer = sample.get("answer") or sample.get("target", "")
                        if not question or not answer:
                            continue
                        
                        # 标准化答案标签
                        if isinstance(answer, int) and 0 <= answer < 4:
                            answer_label = ["A", "B", "C", "D"][answer]
                        elif isinstance(answer, str):
                            answer_label = answer.strip().upper()
                            if answer_label not in ["A", "B", "C", "D"]:
                                continue
                        else:
                            continue
                        
                        examples.append({
                            "question": question,
                            "choices": choices,
                            "answer": answer_label
                        })
        
        self._few_shot_examples = examples[:self.shot_num]
        return self._few_shot_examples
    
    def _build_few_shot_prompt(self) -> str:
        """构建通用few-shot prompt（从MMLU dev集前N条）"""
        if self._few_shot_prompt is not None:
            return self._few_shot_prompt
            
        examples = self._load_mmlu_few_shot_examples()
        
        if not examples:
            self._few_shot_prompt = ""
            return ""
        
        lines = ["The following are multiple choice questions (with answers).\n"]
        option_labels = ["A", "B", "C", "D"]
        
        for example in examples:
            lines.append(example["question"])
            for idx, choice_text in enumerate(example["choices"]):
                if idx < len(option_labels):
                    lines.append(f"{option_labels[idx]}. {choice_text}")
            lines.append(f"Answer: {example['answer']}")
            lines.append("")
        
        self._few_shot_prompt = "\n".join(lines).strip() + "\n\n"
        return self._few_shot_prompt
    
    def build_prompt(self, question: str, options: Dict[str, str], 
                     option_label: str, subject: str) -> str:
        """构建评估prompt，包含从 MMLU dev集前N条作为通用few-shot示例"""
        few_shot = self._build_few_shot_prompt()
        
        # 构建选项字符串
        options_str = ""
        for label in sorted(options.keys()):
            options_str += f"{label}. {options[label]}\n"
 
        prompt = f"""{few_shot}{question}
{options_str}Answer: {option_label}"""
        return prompt
    
    def extract_option_logprob(self, tokens: List[str], token_logprobs: List[Optional[float]], 
                               option_label: str) -> Optional[float]:
        """
        从 token 列表中提取选项标签的 logprob
        """
        # 反向查找最后15个token中包含选项标签的token
        for i in range(len(tokens) - 1, max(0, len(tokens) - 15), -1):
            token = tokens[i]
            token_stripped = token.strip()
            
            # 精确匹配
            if token_stripped == option_label:
                if i < len(token_logprobs) and token_logprobs[i] is not None:
                    return token_logprobs[i]
            
            # 带空格匹配
            if token == f' {option_label}':
                if i < len(token_logprobs) and token_logprobs[i] is not None:
                    return token_logprobs[i]
            
            # 包含匹配（且token很短，避免误匹配单词）
            if option_label in token_stripped and len(token_stripped) <= 2:
                if i < len(token_logprobs) and token_logprobs[i] is not None:
                    return token_logprobs[i]
        
        return None
    
    def get_option_logprob(self, prompt: str, option_label: str, 
                          max_retries: int = 3) -> float:
        """
        获取选项的 logprob（使用completions API + echo=True）
        """
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

                option_logprob = self.extract_option_logprob(
                    tokens, token_logprobs, option_label
                )

                if option_logprob is not None:
                    return option_logprob

                raise ModelResponseError(
                    f"无法从模型响应中提取选项 {option_label} 的logprob。"
                )
                    
            except Exception as e:
                error_msg = str(e)
                
                if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    wait_time = min(10 * (attempt + 1), 60)
                    time.sleep(wait_time)
                    continue
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    last_exception = e if isinstance(e, ModelResponseError) else ModelResponseError(error_msg)
                    break

        if last_exception is None:
            last_exception = ModelResponseError('未知原因导致logprob计算失败。')

        raise last_exception
    
    def get_option_logprob_with_length(self, prompt: str, option_label: str, max_retries: int = 3) -> Tuple[float, int]:
        """获取选项的logprob和token长度（用于长度归一化）"""
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
                    token_length = len(tokens) - 1
                    return option_logprob, token_length

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
    
    def _get_logprob_for_label(self, question: str, options: Dict[str, str], 
                               label: str, subject: str) -> Tuple[str, float, int]:
        """为单个选项获取logprob和长度（用于并发调用）"""
        prompt = self.build_prompt(question, options, label, subject)
        logprob, length = self.get_option_logprob_with_length(prompt, label)
        return label, logprob, length
    
    def evaluate_single_question(self, question_data: Dict) -> Tuple[str, Dict, Dict]:
        """
        评估单个问题（并发获取所有选项的logprob）
        """
        question = question_data['question']
        options = question_data['options']
        option_labels = sorted(options.keys())
        # 优先使用 category，如果没有则使用 config
        subject = question_data.get('category') or question_data.get('config', 'this subject')
        
        # 并发获取每个选项的logprob
        option_logprobs = {}
        option_normalized_logprobs = {}
        option_ppls = {}
        
        max_workers = min(self.max_workers_per_question, len(option_labels))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._get_logprob_for_label,
                    question, options, label, subject
                ): label
                for label in option_labels
            }
            
            for future in as_completed(futures):
                try:
                    label, logprob, length = future.result()
                    option_logprobs[label] = logprob
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
                    label = futures[future]
                    print(f"\n获取选项 {label} 的logprob时出错: {e}")
                    option_logprobs[label] = -10.0
                    option_normalized_logprobs[label] = -10.0
                    option_ppls[label] = np.exp(10.0)
                    with self._lock:
                        self._failed_extractions += 1
        
        # 确保所有选项都有结果
        for label in option_labels:
            if label not in option_normalized_logprobs:
                option_logprobs[label] = -10.0
                option_normalized_logprobs[label] = -10.0
                option_ppls[label] = np.exp(10.0)
        
        # 使用归一化后的logprob选择选项
        predicted = max(option_normalized_logprobs, key=option_normalized_logprobs.get)
        
        return predicted, option_logprobs, option_normalized_logprobs, option_ppls
    
    def _evaluate_single_question_with_result(self, question_data: Dict) -> Dict:
        """评估单个问题并返回结果字典"""
        try:
            predicted, option_logprobs, option_normalized_logprobs, option_ppls = self.evaluate_single_question(question_data)
            correct_answer = question_data['answer'].strip().upper()
            is_correct = (predicted == correct_answer)
            
            with self._lock:
                if is_correct:
                    self._correct_count += 1
                self._total_count += 1
            
            return {
                'question': question_data['question'],
                'options': question_data['options'],
                'correct_answer': question_data['correct_answer'],
                'predicted': predicted,
                'correct': correct_answer,
                'is_correct': is_correct,
                'logprobs': option_logprobs,
                'normalized_logprobs': option_normalized_logprobs,
                'ppls': option_ppls,
                'num_options': question_data['num_options'],
                'category': question_data.get('category') or question_data.get('config', 'unknown'),
                'config': question_data.get('config', 'unknown')
            }
        except ModelResponseError:
            raise
        except Exception as e:
            print(f"\n评估问题时出错: {e}")
            return {
                'question': question_data.get('question', ''),
                'predicted': 'ERROR',
                'correct': question_data.get('answer', ''),
                'is_correct': False,
                'error': str(e)
            }
    
    def evaluate_dataset(self, data: List[Dict], max_samples: int = None, seed: int = 42) -> Dict:
        """评估整个数据集"""
        if max_samples:
            # 随机采样，而不是按顺序取前N个
            random.seed(seed)
            data = random.sample(data, min(max_samples, len(data)))
        
        results = []
        
        self._correct_count = 0
        self._total_count = 0
        self._failed_extractions = 0
        
        # 预热检测：先测试3个样本
        print("\n执行预热检测（测试3个样本）...")
        warmup_samples = data[:min(3, len(data))]
        warmup_failed = False
        for i, sample in enumerate(warmup_samples):
            try:
                predicted, option_logprobs, option_normalized_logprobs, option_ppls = self.evaluate_single_question(sample)
                print(f"✓ 样本{i+1}: 测试通过")
                
                # 检测异常logprobs
                if option_normalized_logprobs and all(v <= -10.0 or v is None for v in option_normalized_logprobs.values()):
                    print(f"⚠️  警告: 样本{i+1}的logprobs全部异常: {option_normalized_logprobs}")
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
        
        print(f"开始评估 {total} 个问题...")
        if max_samples:
            print(f"随机采样: {max_samples} 个样本（seed={seed}）")
        print(f"并发设置: {self.max_workers} 个问题并发, 每个问题最多 {self.max_workers_per_question} 个选项并发")
        shot_desc_eval = f"{self.shot_num}-shot" if self.shot_num > 0 else "0-shot"
        print(f"评估方法: {shot_desc_eval} PPL (completions API + echo=True)")
        
        pbar = tqdm(total=total, desc="评估进度")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_data = {
                executor.submit(self._evaluate_single_question_with_result, q): q 
                for q in data
            }
            
            for future in as_completed(future_to_data):
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
                    print(f"\n模型响应异常，评估中止: {e}")
                    for f in future_to_data:
                        if not f.done():
                            f.cancel()
                    pbar.close()
                    raise
                except Exception as e:
                    print(f"\n评估失败: {e}")
                    pbar.update(1)
        
        pbar.close()
        
        correct = self._correct_count
        accuracy = correct / total * 100 if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'failed_extractions': self._failed_extractions,
            'shot_num': self.shot_num,
            'seed': seed if max_samples else None,
            'max_samples': max_samples,
            'results': results
        }

DEFAULT_MMLU_REDUX_DIR = os.path.join(os.path.dirname(__file__), "datasets", "mmlu_redux")


def find_mmlu_redux_data_path(dataset_dir: str = None) -> Optional[str]:
    """查找MMLU-Redux数据集路径"""
    if dataset_dir is None:
        dataset_dir = DEFAULT_MMLU_REDUX_DIR
    
    test_path = os.path.join(dataset_dir, 'test-00000-of-00001.parquet')
    if os.path.exists(test_path):
        return test_path
    
    # 尝试其他可能的文件名
    import glob
    test_files = glob.glob(os.path.join(dataset_dir, 'test*.parquet'))
    if test_files:
        return test_files[0]
    
    return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MMLU-Redux 评估脚本")

    parser.add_argument(
        "--shot-num",
        type=int,
        default=5,
        help=f"Few-shot 示例数量（默认: 5）",
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
        default=10,
        help="每个问题的最大并发工作线程数（默认: 10）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="随机采样大小，None 表示评估所有数据（默认: None）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="生成的最大 token 数（默认: 1）",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help=f"数据集目录（默认: {DEFAULT_MMLU_REDUX_DIR}）",
    )
    parser.add_argument(
        "--mmlu-dataset-dir",
        type=str,
        default=None,
        help="MMLU 数据集目录（用于加载 few-shot 示例，默认: datasets/cais_mmlu）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )

    args = parser.parse_args()

    # 查找数据集路径
    dataset_dir = args.dataset_dir or DEFAULT_MMLU_REDUX_DIR
    parquet_path = find_mmlu_redux_data_path(dataset_dir)
    if parquet_path is None:
        print(f"⚠️  未找到 MMLU-Redux 数据集")
        print(f"请先运行以下命令下载数据集:")
        print(f"  python download_datasets.py mmlu_redux")
        print(f"或者手动将数据集放置在: {dataset_dir}")
        return

    log_dir = os.path.join(os.path.dirname(__file__), "logs", args.model)
    os.makedirs(log_dir, exist_ok=True)

    shot_desc = f"{args.shot_num}-shot" if args.shot_num > 0 else "0-shot"
    output_filename = f"mmlu_redux_{args.shot_num}shot.json"
    if args.max_samples:
        output_filename = f"mmlu_redux_{args.shot_num}shot_{args.max_samples}samples.json"
    output_path = os.path.join(log_dir, output_filename)
    
    # 确定 MMLU 数据集目录（用于加载 few-shot 示例）
    if args.mmlu_dataset_dir:
        mmlu_dataset_dir = args.mmlu_dataset_dir
    else:
        mmlu_dataset_dir = os.path.join(os.path.dirname(__file__), "datasets", "cais_mmlu")

    print("=" * 70)
    print(f"MMLU-Redux 评估 - {args.model} - {shot_desc} PPL 方法")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"API: completions + echo=True")
    print(f"方法: {shot_desc} prompting + PPL")
    print("=" * 70)

    evaluator = MMLUReduxEvaluator(
        base_url=args.base_url,
        model=args.model,
        api_key="EMPTY",
        max_workers=args.max_workers,
        max_workers_per_question=args.max_workers_per_question,
        shot_num=args.shot_num,
        mmlu_dataset_dir=mmlu_dataset_dir,
    )
    
    # 如果使用 few-shot，检查 MMLU 数据集是否存在
    if args.shot_num > 0:
        if not os.path.isdir(mmlu_dataset_dir):
            print(f"⚠️  MMLU 数据集目录不存在: {mmlu_dataset_dir}")
            print(f"{args.shot_num}-shot")
            evaluator.shot_num = 0
        else:
            try:
                # 预加载 few-shot 示例
                evaluator._load_mmlu_few_shot_examples()
                print(f"✓ 已加载 {len(evaluator._few_shot_examples)} 个 few-shot 示例（来自 MMLU）")
            except Exception as e:
                print(f"⚠️  加载 few-shot 示例失败: {e}")
                print(f"{args.shot_num}-shot")
                evaluator.shot_num = 0

    print(f"\n加载数据: {parquet_path}")
    try:
        data = evaluator.load_mmlu_redux_data(parquet_path)
        print(f"✓ 成功加载 {len(data)} 个样本")
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
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print("评估结果")
        print("=" * 70)
        print(f"总问题数: {results['total']}")
        print(f"正确答案数: {results['correct']}")
        print(f"准确率: {results['accuracy']:.2f}%")
        print(f"Shot 配置: {shot_desc}")

        if results.get("failed_extractions", 0) > 0:
            # 计算平均选项数
            avg_options = (
                sum(r.get("num_options", 4) for r in results["results"]) / len(results["results"])
                if results["results"]
                else 4
            )
            fail_rate = results["failed_extractions"] / (results["total"] * avg_options) * 100
            print(f"\nLogprob 提取失败次数: {results['failed_extractions']}")
            print(f"失败率: {fail_rate:.2f}%")
            if fail_rate > 5:
                print("⚠️  失败率较高，可能影响准确率")

        if results["results"]:
            # 统计 category（优先）或 config
            category_stats = {}
            for result in results["results"]:
                category = result.get("category") or result.get("config", "unknown")
                if category not in category_stats:
                    category_stats[category] = {"correct": 0, "total": 0}
                category_stats[category]["total"] += 1
                if result.get("is_correct"):
                    category_stats[category]["correct"] += 1

            print("\nCategory 统计（Top 10）:")
            sorted_categories = sorted(
                category_stats.items(),
                key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0,
                reverse=True,
            )
            for category, stats in sorted_categories[:10]:
                acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
                print(f"  {category}: {stats['correct']}/{stats['total']} = {acc:.2f}%")

            if len(sorted_categories) > 10:
                print(f"  ... 还有 {len(sorted_categories) - 10} 个 category")

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
                "shot_num": shot_num,
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

