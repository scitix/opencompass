"""MMLU数据集评估脚本 - Base模型标准PPL方法（最终版本）。"""
import json
import numpy as np
import os
import random
from typing import List, Dict, Tuple, Optional

import pyarrow as pa
from openai import OpenAI
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
# from transformers import AutoTokenizer

DEFAULT_MMLU_DIR = os.path.join(os.path.dirname(__file__), 'datasets', 'cais_mmlu')


class ModelResponseError(RuntimeError):
    """Raised when the model response does not contain usable logprob data."""


class MMLUEvaluator:
    """使用标准PPL方法评估MMLU数据集"""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_workers: int = 128, max_workers_per_question: int = 4,
                 dataset_dir: Optional[str] = None, shot_num: int = 5):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_workers_per_question = min(max_workers_per_question, 4)
        self.dataset_dir = dataset_dir or DEFAULT_MMLU_DIR
        self.shot_num = shot_num
        # self._tokenizer = AutoTokenizer.from_pretrained(
        #     "Qwen/Qwen2.5-7B", trust_remote_code=True)
        self._correct_count = 0
        self._total_count = 0
        self._failed_extractions = 0
        self._lock = threading.Lock()
        self._dataset_loaded = False
        self._fewshot_by_subject: Dict[str, List[Dict]] = {}
        self._few_shot_cache: Dict[str, str] = {}
        self._test_data: List[Dict] = []

    def _load_dataset(self):
        """加载本地 MMLU 数据集并建立索引"""
        if self._dataset_loaded:
            return

        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(
                f"未在 {self.dataset_dir} 找到本地 MMLU 数据集，请先下载 cais/mmlu。")

        dev_records = self._load_split('dev')
        if not dev_records:
            dev_records = self._load_split('validation')
        if not dev_records:
            raise FileNotFoundError(
                f"在 {self.dataset_dir} 未找到 'dev' 或 'validation' 拆分，无法构建 few-shot 提示。")

        test_records = self._load_split('test')
        if not test_records:
            raise FileNotFoundError(
                f"在 {self.dataset_dir} 未找到 'test' 拆分，无法执行评估。")

        self._fewshot_by_subject = self._group_by_subject(dev_records)
        self._test_data = self._convert_split(test_records)

        for subject in list(self._fewshot_by_subject.keys()):
            self._few_shot_cache[subject] = self._build_few_shot_prompt(subject)

        if 'miscellaneous' not in self._few_shot_cache:
            self._few_shot_cache['miscellaneous'] = self._build_few_shot_prompt('miscellaneous')

        self._dataset_loaded = True

    def _load_split(self, split_name: str) -> List[Dict]:
        split_dir = os.path.join(self.dataset_dir, split_name)
        if not os.path.isdir(split_dir):
            return []

        records: List[Dict] = []
        jsonl_files = [
            fname for fname in os.listdir(split_dir)
            if fname.endswith('.jsonl')
        ]

        if jsonl_files:
            for filename in sorted(jsonl_files):
                subject = os.path.splitext(filename)[0]
                file_path = os.path.join(split_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        sample = json.loads(line)
                        self._append_record(records, sample, subject)
        else:
            arrow_files = [
                fname for fname in os.listdir(split_dir)
                if fname.endswith('.arrow')
            ]
            for filename in sorted(arrow_files):
                file_path = os.path.join(split_dir, filename)
                with pa.memory_map(file_path, 'r') as source:
                    try:
                        reader = pa.ipc.open_file(source)
                    except pa.ArrowInvalid:
                        reader = pa.ipc.open_stream(source)
                    table = reader.read_all()
                for sample in table.to_pylist():
                    self._append_record(records, sample)
        return records

    def _append_record(self, records: List[Dict], sample: Dict, default_subject: Optional[str] = None):
        question = sample.get('question') or sample.get('input')
        choices = sample.get('choices') or [
            sample.get('A'), sample.get('B'), sample.get('C'), sample.get('D')
        ]
        if not question or not choices or len(choices) != 4:
            return
        choices = ["" if c is None else str(c) for c in choices]
        answer = sample.get('answer')
        if answer is None:
            answer = sample.get('target')
        if answer is None:
            return
        subject = sample.get('subject') or default_subject or 'miscellaneous'

        answer_label = self._normalize_answer_label(answer)
        if answer_label is None:
            return

        records.append({
            'question': str(question),
            'choices': choices,
            'answer': answer_label,
            'subject': subject
        })

    @staticmethod
    def _normalize_answer_label(answer) -> Optional[str]:
        if isinstance(answer, int):
            if 0 <= answer < 4:
                return ['A', 'B', 'C', 'D'][answer]
            return None
        if isinstance(answer, str):
            ans = answer.strip().upper()
            if ans in ['A', 'B', 'C', 'D']:
                return ans
            try:
                idx = int(ans)
                if 0 <= idx < 4:
                    return ['A', 'B', 'C', 'D'][idx]
            except Exception:
                return None
        return None

    @staticmethod
    def _group_by_subject(records: List[Dict]) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = {}
        for row in records:
            subject = row.get('subject', 'miscellaneous') or 'miscellaneous'
            grouped.setdefault(subject, []).append(row)
        return grouped

    @staticmethod
    def _convert_split(records: List[Dict]) -> List[Dict]:
        converted: List[Dict] = []
        for row in records:
            choices = row['choices']
            if len(choices) != 4:
                continue
            converted.append({
                'question': row['question'],
                'A': choices[0],
                'B': choices[1],
                'C': choices[2],
                'D': choices[3],
                'answer': row['answer'],
                'subject': row.get('subject', 'miscellaneous')
            })
        return converted

    def _select_examples(self, subject: str) -> List[Dict]:
        subject_examples = list(self._fewshot_by_subject.get(subject, []))
        if len(subject_examples) >= self.shot_num:
            return subject_examples[:self.shot_num]

        supplemental: List[Dict] = []
        if subject != 'miscellaneous':
            supplemental = self._fewshot_by_subject.get('miscellaneous', [])

        combined = subject_examples + [ex for ex in supplemental if ex not in subject_examples]
        return combined[:self.shot_num]

    def _build_few_shot_prompt(self, subject: str) -> str:
        examples = self._select_examples(subject)
        if not examples:
            return ""

        lines = [
            f"The following are multiple choice questions (with answers) about {subject}.\n"
        ]
        option_labels = ['A', 'B', 'C', 'D']
        for example in examples:
            lines.append(example['question'])
            for idx, choice_text in enumerate(example['choices']):
                lines.append(f"{option_labels[idx]}. {choice_text}")
            lines.append(f"Answer: {example['answer']}")
            lines.append("")

        return "\n".join(lines).strip() + "\n\n"

    def load_mmlu_data(self) -> List[Dict]:
        """加载测试集并准备 few-shot 示例"""
        self._load_dataset()
        return list(self._test_data)

    def get_few_shot_examples(self, subject: str = "miscellaneous") -> str:
        """获取 few-shot 示例（来自本地 dev/validation 集，按科目划分）"""
        self._load_dataset()
        subject_key = subject or 'miscellaneous'
        if subject_key not in self._few_shot_cache:
            self._few_shot_cache[subject_key] = self._build_few_shot_prompt(subject_key)
        return self._few_shot_cache.get(subject_key, "")
    
    def build_prompt(self, question: str, option_a: str, option_b: str, 
                     option_c: str, option_d: str, option_label: str, 
                     subject: str = "miscellaneous") -> str:
        """构建评估prompt（带 few-shot 示例）"""
        few_shot = self.get_few_shot_examples(subject)
        
        prompt = f"""{few_shot}{question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
Answer: {option_label}"""
        return prompt
    
    def extract_option_logprob(self, tokens: List[str], token_logprobs: List[Optional[float]], 
                               option_label: str) -> Optional[float]:
        """
        从 token 列表中提取选项标签的 logprob
        
        改进的提取逻辑：从后往前查找，因为答案在最后
        """
        # 反向查找最后15个token中包含选项标签的token
        for i in range(len(tokens) - 1, max(0, len(tokens) - 15), -1):
            token = tokens[i]
            token_stripped = token.strip()
            
            # 精确匹配
            if token_stripped == option_label:
                if i < len(token_logprobs) and token_logprobs[i] is not None:
                    return token_logprobs[i]
            
            # 带空格匹配（常见格式）
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
                # tokenized = self._tokenizer(prompt, add_special_tokens=False)
                # token_length = len(tokenized['input_ids'])
                # print(f"[Token Length] option {option_label}: {token_length} tokens")
                # context_length = len(prompt)
                # print(f"[Context Length] option {option_label}: {context_length} characters")
                # 使用 completions API 的 echo=True 功能
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=1,
                    logprobs=5,  # 获取top 5 logprobs
                    echo=True,   # 关键：返回 prompt 的 logprobs
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
                
                # 速率限制处理
                if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    wait_time = min(10 * (attempt + 1), 60)
                    time.sleep(wait_time)
                    continue
                
                # 其他错误
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    last_exception = e if isinstance(e, ModelResponseError) else ModelResponseError(error_msg)
                    break

        if last_exception is None:
            last_exception = ModelResponseError('未知原因导致logprob计算失败。')

        raise last_exception
    
    def _get_logprob_for_label(self, question: str, option_a: str, option_b: str, 
                               option_c: str, option_d: str, label: str, 
                               subject: str = "miscellaneous") -> Tuple[str, float]:
        """为单个选项获取logprob（用于并发调用）"""
        prompt = self.build_prompt(question, option_a, option_b, option_c, option_d, label, subject)
        logprob = self.get_option_logprob(prompt, label)
        return label, logprob
    
    def evaluate_single_question(self, question_data: Dict) -> Tuple[str, Dict, Dict]:
        """
        评估单个问题（并发获取4个选项的logprob）
        """
        question = question_data['question']
        subject = question_data.get('subject', 'miscellaneous')
        
        # 并发获取每个选项的logprob
        option_logprobs = {}
        option_ppls = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers_per_question) as executor:
            futures = {
                executor.submit(
                    self._get_logprob_for_label,
                    question, question_data['A'], question_data['B'],
                    question_data['C'], question_data['D'], label, subject
                ): label
                for label in ['A', 'B', 'C', 'D']
            }
            
            for future in as_completed(futures):
                try:
                    label, logprob = future.result()
                    option_logprobs[label] = logprob
                    option_ppls[label] = np.exp(-logprob)
                except ModelResponseError:
                    # 直接将错误向上抛出，终止整个评估流程
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
        
        # 确保所有4个选项都有结果
        for label in ['A', 'B', 'C', 'D']:
            if label not in option_logprobs:
                option_logprobs[label] = -10.0
                option_ppls[label] = np.exp(10.0)
        
        # 选择 logprob 最高的选项（PPL 最低）
        predicted = max(option_logprobs, key=option_logprobs.get)
        
        return predicted, option_logprobs, option_ppls
    
    def _evaluate_single_question_with_result(self, question_data: Dict) -> Dict:
        """评估单个问题并返回结果字典"""
        try:
            predicted, option_logprobs, option_ppls = self.evaluate_single_question(question_data)
            correct_answer = question_data['answer'].strip().upper()
            is_correct = (predicted == correct_answer)
            
            with self._lock:
                if is_correct:
                    self._correct_count += 1
                self._total_count += 1
            
            return {
                'question': question_data['question'],
                'options': {
                    'A': question_data['A'],
                    'B': question_data['B'],
                    'C': question_data['C'],
                    'D': question_data['D']
                },
                'predicted': predicted,
                'correct': correct_answer,
                'is_correct': is_correct,
                'logprobs': option_logprobs,
                'ppls': option_ppls,
                'subject': question_data['subject']
            }
        except ModelResponseError:
            # 直接抛出给上层，终止评估流程
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
        
        total = len(data)
        results = []
        
        self._correct_count = 0
        self._total_count = 0
        self._failed_extractions = 0
        
        print(f"开始评估 {total} 个问题...")
        print(f"并发设置: {self.max_workers} 个问题并发, 每个问题 {self.max_workers_per_question} 个选项并发")
        print(f"评估方法: 标准 PPL (completions API + echo=True)")
        
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

def main():
    """主函数"""
    MODEL = "qwen2-5-72b"
    BASE_URL = "http://172.18.178.129:8000/v1"
    SHOT_NUM = 5  # few-shot数量
    DATASET_PATH = DEFAULT_MMLU_DIR
    
    MAX_WORKERS = 128
    MAX_WORKERS_PER_QUESTION = 4
    
    # 创建日志目录结构：logs/{MODEL}/
    log_dir = os.path.join("logs", MODEL)
    os.makedirs(log_dir, exist_ok=True)
    OUTPUT_PATH = os.path.join(log_dir, f"mmlu_{SHOT_NUM}shot.json")
    
    print("="*70)
    print(f"MMLU 评估 - {MODEL} - {SHOT_NUM}-shot PPL 方法")
    print("="*70)
    print(f"模型: {MODEL}")
    print(f"API: completions + echo=True")
    print(f"方法: {SHOT_NUM}-shot few-shot prompting + PPL")
    print(f"原理: 对每个选项计算 log P(选项标签|few-shot示例+问题+选项)")
    print(f"      选择 log-likelihood 最高的选项")
    print("="*70)
    
    evaluator = MMLUEvaluator(
        BASE_URL, 
        MODEL, 
        max_workers=MAX_WORKERS,
        max_workers_per_question=MAX_WORKERS_PER_QUESTION,
        dataset_dir=DATASET_PATH,
        shot_num=SHOT_NUM
    )
    
    print(f"\n加载数据集: {DATASET_PATH}")
    try:
        data = evaluator.load_mmlu_data()
        print(f"✓ 成功加载 {len(data)} 条数据")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return
    
    print("\n开始评估...")
    
    max_samples = 500  # 测试 500 个样本（随机采样）
    seed = 42  # 随机种子，确保可重复
    #max_samples = None  # 评估全部数据
    
    try:
        results = evaluator.evaluate_dataset(data, max_samples=max_samples, seed=seed)
        
        print(f"\n保存结果: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*70)
        print("评估结果")
        print("="*70)
        print(f"总问题数: {results['total']}")
        print(f"正确答案数: {results['correct']}")
        print(f"准确率: {results['accuracy']:.2f}%")
        print(f"Shot配置: {SHOT_NUM}-shot")
        
        if results.get('failed_extractions', 0) > 0:
            fail_rate = results['failed_extractions'] / (results['total'] * 4) * 100
            print(f"\nLogprob提取失败: {results['failed_extractions']} 次")
            print(f"提取失败率: {fail_rate:.2f}%")
            if fail_rate > 5:
                print("⚠️  提取失败率较高，可能影响准确性")
        
        # 按科目统计
        if results['results']:
            subject_stats = {}
            for result in results['results']:
                subject = result.get('subject', 'unknown')
                if subject not in subject_stats:
                    subject_stats[subject] = {'correct': 0, 'total': 0}
                subject_stats[subject]['total'] += 1
                if result.get('is_correct'):
                    subject_stats[subject]['correct'] += 1
            
            print("\n按科目统计（Top 10）:")
            sorted_subjects = sorted(subject_stats.items(), 
                                   key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0, 
                                   reverse=True)
            for subject, stats in sorted_subjects[:10]:
                acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
                print(f"  {subject}: {stats['correct']}/{stats['total']} = {acc:.2f}%")
            
            if len(sorted_subjects) > 10:
                print(f"  ... 还有 {len(sorted_subjects) - 10} 个科目")
        
        print(f"\n详细结果已保存到: {OUTPUT_PATH}")
        print(f"日志目录: {log_dir}")
        print("="*70)
        
    except ModelResponseError as e:
        print(f"\n✗ 评估过程中检测到模型响应异常: {e}")
        print("评估已终止，未生成结果文件。")
    except KeyboardInterrupt:
        print("\n\n⚠️  评估被用户中断")
        # 保存已有结果
        if evaluator._total_count > 0:
            partial_results = {
                'accuracy': evaluator._correct_count / evaluator._total_count * 100,
                'correct': evaluator._correct_count,
                'total': evaluator._total_count,
                'shot_num': SHOT_NUM,
                'note': 'Partial results (interrupted)'
            }
            partial_path = OUTPUT_PATH.replace('.json', '_partial.json')
            with open(partial_path, 'w', encoding='utf-8') as f:
                json.dump(partial_results, f, ensure_ascii=False, indent=2)
            print(f"部分结果已保存到: {partial_path}")
    except Exception as e:
        print(f"\n✗ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
