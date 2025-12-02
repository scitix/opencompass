"""CMMLU数据集评估脚本 - 5-shot生成方法（EM评估）。"""
import csv
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
import eval_utils


class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


class CMMLUEvaluator:
    """使用5-shot生成方法评估CMMLU数据集"""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_workers: int = 32, max_tokens: int = 512, shot_num: int = 5):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.shot_num = shot_num
        self._correct_count = 0
        self._total_count = 0
        self._lock = threading.Lock()
        self._few_shot_cache: Dict[str, List[Dict]] = {}
        
        # CMMLU subject mapping（参考OpenCompass的cmmlu_0shot_cot_gen_305931.py）
        self.subject_mapping = {
            'agronomy': '农学',
            'anatomy': '解剖学',
            'ancient_chinese': '古汉语',
            'arts': '艺术学',
            'astronomy': '天文学',
            'business_ethics': '商业伦理',
            'chinese_civil_service_exam': '中国公务员考试',
            'chinese_driving_rule': '中国驾驶规则',
            'chinese_food_culture': '中国饮食文化',
            'chinese_foreign_policy': '中国外交政策',
            'chinese_history': '中国历史',
            'chinese_literature': '中国文学',
            'chinese_teacher_qualification': '中国教师资格',
            'clinical_knowledge': '临床知识',
            'college_actuarial_science': '大学精算学',
            'college_education': '大学教育学',
            'college_engineering_hydrology': '大学工程水文学',
            'college_law': '大学法律',
            'college_mathematics': '大学数学',
            'college_medical_statistics': '大学医学统计',
            'college_medicine': '大学医学',
            'computer_science': '计算机科学',
            'computer_security': '计算机安全',
            'conceptual_physics': '概念物理学',
            'construction_project_management': '建设工程管理',
            'economics': '经济学',
            'education': '教育学',
            'electrical_engineering': '电气工程',
            'elementary_chinese': '小学语文',
            'elementary_commonsense': '小学常识',
            'elementary_information_and_technology': '小学信息技术',
            'elementary_mathematics': '初等数学',
            'ethnology': '民族学',
            'food_science': '食品科学',
            'genetics': '遗传学',
            'global_facts': '全球事实',
            'high_school_biology': '高中生物',
            'high_school_chemistry': '高中化学',
            'high_school_geography': '高中地理',
            'high_school_mathematics': '高中数学',
            'high_school_physics': '高中物理学',
            'high_school_politics': '高中政治',
            'human_sexuality': '人类性行为',
            'international_law': '国际法学',
            'journalism': '新闻学',
            'jurisprudence': '法理学',
            'legal_and_moral_basis': '法律与道德基础',
            'logical': '逻辑学',
            'machine_learning': '机器学习',
            'management': '管理学',
            'marketing': '市场营销',
            'marxist_theory': '马克思主义理论',
            'modern_chinese': '现代汉语',
            'nutrition': '营养学',
            'philosophy': '哲学',
            'professional_accounting': '专业会计',
            'professional_law': '专业法学',
            'professional_medicine': '专业医学',
            'professional_psychology': '专业心理学',
            'public_relations': '公共关系',
            'security_study': '安全研究',
            'sociology': '社会学',
            'sports_science': '体育学',
            'traditional_chinese_medicine': '中医中药',
            'virology': '病毒学',
            'world_history': '世界历史',
            'world_religions': '世界宗教'
        }


    # ========== PPL评估方法 ==========
    def extract_option_logprob(
        self, tokens: List[str], token_logprobs: List[Optional[float]], option_label: str
    ) -> Optional[float]:
        """从token列表中提取选项标签的logprob"""
        for i in range(len(tokens) - 1, max(0, len(tokens) - 15), -1):
            token = tokens[i]
            token_stripped = token.strip()
            
            if token_stripped == option_label:
                if i < len(token_logprobs) and token_logprobs[i] is not None:
                    return token_logprobs[i]
            
            if token == f" {option_label}":
                if i < len(token_logprobs) and token_logprobs[i] is not None:
                    return token_logprobs[i]
            
            if option_label in token_stripped and len(token_stripped) <= 2:
                if i < len(token_logprobs) and token_logprobs[i] is not None:
                    return token_logprobs[i]
        
        return None

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


    def load_cmmlu_data(self, data_dir: str, subject_name: str = None) -> List[Dict]:
        """从CSV文件加载CMMLU数据"""
        data = []
        
        if subject_name:
            subjects = [subject_name]
        else:
            # 加载所有subjects
            test_dir = os.path.join(data_dir, 'test')
            if os.path.exists(test_dir):
                subjects = [f.replace('.csv', '') for f in os.listdir(test_dir) 
                           if f.endswith('.csv')]
            else:
                subjects = []
        
        for subject in subjects:
            test_file = os.path.join(data_dir, 'test', f'{subject}.csv')
            dev_file = os.path.join(data_dir, 'dev', f'{subject}.csv')
            
            if not os.path.exists(test_file):
                continue
            
            # 加载dev集用于few-shot
            dev_examples = []
            if os.path.exists(dev_file):
                with open(dev_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # skip header
                    for row in reader:
                        if len(row) < 7:
                            continue
                        dev_examples.append({
                            'question': row[1],
                            'A': row[2],
                            'B': row[3],
                            'C': row[4],
                            'D': row[5],
                            'answer': row[6],
                        })
            
            # 缓存few-shot示例
            self._few_shot_cache[subject] = dev_examples[:self.shot_num]
            
            # 加载test集
            with open(test_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if len(row) < 7:
                        continue
                    data.append({
                        'question': row[1],
                        'A': row[2],
                        'B': row[3],
                        'C': row[4],
                        'D': row[5],
                        'answer': row[6],
                        'subject': subject,
                    })
        
        return data

    def build_few_shot_prompt(self, subject: str) -> str:
        """构建few-shot prompt"""
        examples = self._few_shot_cache.get(subject, [])
        if not examples:
            return ""
        
        prompt_parts = []
        for example in examples:
            prompt_parts.append(
                f"{example['question']}\n"
                f"A) {example['A']}\n"
                f"B) {example['B']}\n"
                f"C) {example['C']}\n"
                f"D) {example['D']}\n"
                f"答案: {example['answer']}"
            )
        
        return "\n\n".join(prompt_parts) + "\n\n"

    def build_prompt(self, question: str, A: str, B: str, C: str, D: str, subject: str, option_label: str = None) -> str:
        """构建评估prompt（PPL方法需要option_label）"""
        few_shot = self.build_few_shot_prompt(subject)
        subject_zh = self.subject_mapping.get(subject, subject)
        
        prompt = f"""{few_shot}以下是关于{subject_zh}的单项选择题，请直接给出正确答案的选项。

题目：{question}
A. {A}
B. {B}
C. {C}
D. {D}
答案是：{option_label if option_label else ''}"""
        return prompt

    def postprocess_answer(self, text: str) -> str:
        """后处理答案，提取A/B/C/D（参考OpenCompass的match_answer_pattern）"""
        # 匹配"答案: A"格式
        match = re.search(r'(?i)答案\s*:\s*[\W]*([A-D])[\W]*', text)
        if match:
            return match.group(1).upper()
        
        # 如果没有找到，尝试查找第一个A/B/C/D
        match = re.search(r'[ABCD]', text.upper())
        if match:
            return match.group(0)
        
        return ""

    def generate_answer(self, prompt: str, subject: str, max_retries: int = 3) -> str:
        """生成答案"""
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    stop=None,
                )

                if not response.choices:
                    raise ModelResponseError('模型返回的choices为空。')

                generated_text = response.choices[0].text
                return self.postprocess_answer(generated_text)

            except Exception as e:
                error_msg = str(e)

                if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    wait_time = min(10 * (attempt + 1), 60)
                    time.sleep(wait_time)
                    continue

                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    last_exception = e
                    break

        if last_exception is None:
            last_exception = ModelResponseError('未知原因导致生成失败。')

        raise last_exception

    def _get_logprob_for_label(self, question: str, A: str, B: str, C: str, D: str, 
                               subject: str, label: str):
        """为单个选项获取logprob和长度"""
        from typing import Tuple
        prompt = self.build_prompt(question, A, B, C, D, subject, label)
        logprob, length = self.get_option_logprob_with_length(prompt, label)
        return label, logprob, length

    def evaluate_single_question(self, question_data: Dict) -> Dict:
        """评估单个问题（使用PPL方法+长度归一化）"""
        import numpy as np

        try:
            question = question_data['question']
            A = question_data['A']
            B = question_data['B']
            C = question_data['C']
            D = question_data['D']
            reference_answer = question_data['answer'].strip().upper()
            subject = question_data['subject']
            
            option_logprobs = {}
            option_normalized_logprobs = {}
            option_ppls = {}
            
            # 并发获取所有选项的logprob
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._get_logprob_for_label, question, A, B, C, D, subject, label): label
                    for label in ['A', 'B', 'C', 'D']
                }
                
                for future in as_completed(futures):
                    try:
                        label, logprob, length = future.result()
                        option_logprobs[label] = logprob
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
            
            # 确保所有选项都有结果
            for label in ['A', 'B', 'C', 'D']:
                if label not in option_normalized_logprobs:
                    option_logprobs[label] = -10.0
                    option_normalized_logprobs[label] = -10.0
                    option_ppls[label] = np.exp(10.0)
            
            predicted_answer = max(option_normalized_logprobs, key=option_normalized_logprobs.get)
            is_correct = (predicted_answer == reference_answer)

            with self._lock:
                if is_correct:
                    self._correct_count += 1
                self._total_count += 1

            return {
                'question': question[:200] + '...' if len(question) > 200 else question,
                'predicted_answer': predicted_answer,
                'reference_answer': reference_answer,
                'is_correct': is_correct,
                'subject': subject,
                'logprobs': option_logprobs,
                'normalized_logprobs': option_normalized_logprobs,
                'ppls': option_ppls
            }
        except Exception as e:
            print(f"\n评估问题时出错: {e}")
            return {
                'question': question_data.get('question', ''),
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

        # 正式评估：评估所有样本（预热检测不跳过任何数据）
        eval_data = data
        total = len(eval_data)
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.evaluate_single_question, item): item for item in data}

            with tqdm(total=len(eval_data), desc="评估进度", unit="问题") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    pbar.set_postfix(accuracy=f"{self._correct_count / self._total_count * 100:.2f}%",
                                   correct=self._correct_count)

        # 按subject统计
        subject_stats = {}
        for result in results:
            subject = result.get('subject', 'unknown')
            if subject not in subject_stats:
                subject_stats[subject] = {'total': 0, 'correct': 0}
            subject_stats[subject]['total'] += 1
            if result.get('is_correct', False):
                subject_stats[subject]['correct'] += 1

        return {
            'total': self._total_count,
            'correct': self._correct_count,
            'accuracy': self._correct_count / self._total_count * 100 if self._total_count > 0 else 0.0,
            'shot_num': self.shot_num,
            'seed': seed if max_samples else None,
            'max_samples': max_samples,
            'subject_stats': {k: {
                'total': v['total'],
                'correct': v['correct'],
                'accuracy': v['correct'] / v['total'] * 100 if v['total'] > 0 else 0.0
            } for k, v in subject_stats.items()},
            'results': results
        }


def find_cmmlu_data_path():
    """查找CMMLU数据集路径（只从datasets目录查找）"""
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'cmmlu')
    
    if os.path.exists(datasets_dir) and os.path.exists(os.path.join(datasets_dir, 'test')):
        return datasets_dir
    
    return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CMMLU evaluation script")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EVAL_MODEL_NAME", "qwen2-5-72b"),
        help="Model name (default: eval-qwen2-5-72b)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1"),
        help="API base URL",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Maximum number of concurrent workers (default: 32)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--shot-num",
        type=int,
        default=5,
        help="Number of few-shot examples (default: 5)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Random sample size, None means evaluate all data (default: None)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory (default: datasets/cmmlu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    
    args = parser.parse_args()

    # 自动查找数据路径
    DATA_DIR = args.dataset_dir or find_cmmlu_data_path()
    if DATA_DIR is None:
        print("⚠️  未找到CMMLU数据集")
        print("   请确保数据集在以下位置：")
        print("   datasets/cmmlu/")
        return

    # 创建日志目录结构：logs/{model}/
    log_dir = os.path.join("logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    
    shot_desc = f"{args.shot_num}-shot" if args.shot_num > 0 else "0-shot"
    output_filename = f"cmmlu_{args.shot_num}shot.json"
    if args.max_samples:
        output_filename = f"cmmlu_{args.shot_num}shot_{args.max_samples}samples.json"
    OUTPUT_PATH = os.path.join(log_dir, output_filename)

    print("="*70)
    print(f"CMMLU 评估 - {args.model} - {shot_desc} PPL方法")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"API: completions + echo=True")
    print(f"方法: {shot_desc} prompting + PPL")
    print(f"Few-shot来源: datasets/cmmlu/dev/")
    print("="*70)

    evaluator = CMMLUEvaluator(
        args.base_url,
        args.model,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        shot_num=args.shot_num
    )

    print(f"\n加载数据: {DATA_DIR}")
    try:
        data = evaluator.load_cmmlu_data(DATA_DIR)
        print(f"✓ 成功加载 {len(data)} 条测试数据")
        
        # 统计各subject数量
        subject_counts = {}
        for item in data:
            subject = item.get('subject', 'unknown')
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        print(f"\nSubject分布:")
        for subject, count in sorted(subject_counts.items())[:10]:
            print(f"  - {subject}: {count} 条")
        if len(subject_counts) > 10:
            print(f"  ... 共 {len(subject_counts)} 个subjects")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n开始评估...")

    try:
        results = evaluator.evaluate_dataset(data, max_samples=args.max_samples, seed=args.seed)

        print(f"\n保存结果: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\n" + "="*70)
        print("评估结果")
        print("="*70)
        print(f"总问题数: {results['total']}")
        print(f"正确答案数: {results['correct']}")
        print(f"准确率: {results['accuracy']:.2f}%")
        print(f"Shot配置: {shot_desc}")
        if args.max_samples:
            print(f"随机采样: {args.max_samples} 个样本（seed={args.seed}）")
        
        print(f"\n各Subject准确率（前10个）:")
        sorted_subjects = sorted(results['subject_stats'].items(), 
                               key=lambda x: x[1]['total'], reverse=True)[:10]
        for subject, stats in sorted_subjects:
            print(f"  - {subject}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")

        print(f"\n详细结果已保存到: {OUTPUT_PATH}")
        print(f"日志目录: {log_dir}")
        print("="*70)

    except ModelResponseError as e:
        print(f"\n✗ 评估过程中检测到模型响应异常: {e}")
        print("评估已终止，未生成结果文件。")
    except KeyboardInterrupt:
        print("\n\n⚠️  评估被用户中断")
    except Exception as e:
        print(f"\n✗ 评估过程中出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()


