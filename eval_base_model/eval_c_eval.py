"""C-Eval数据集评估脚本 - 5-shot生成方法。"""
import csv
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
import threading
import eval_utils
# C-Eval subjects mapping
SUBJECT_MAPPING = {
    "computer_network": ("Computer Network", "计算机网络", "STEM"),
    "operating_system": ("Operating System", "操作系统", "STEM"),
    "computer_architecture": ("Computer Architecture", "计算机组成", "STEM"),
    "college_programming": ("College_Programming", "大学编程", "STEM"),
    "college_physics": ("College_Physics", "大学物理", "STEM"),
    "college_chemistry": ("College_Chemistry", "大学化学", "STEM"),
    "advanced_mathematics": ("Advanced_Mathematics", "高等数学", "STEM"),
    "probability_and_statistics": ("Probability_and_Statistics", "概率统计", "STEM"),
    "discrete_mathematics": ("Discrete_Mathematics", "离散数学", "STEM"),
    "electrical_engineer": ("Electrical_Engineer", "注册电气工程师", "STEM"),
    "metrology_engineer": ("Metrology_Engineer", "注册计量师", "STEM"),
    "fire_engineer": ("Fire_Engineer", "注册消防工程师", "STEM"),
    "civil_servant": ("Civil_Servant", "公务员", "Other"),
    "accountant": ("Accountant", "会计师", "Other"),
    "physician": ("Physician", "医师资格", "Other"),
}


class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


class CEvalEvaluator:
    """使用5-shot生成方法评估C-Eval数据集"""
    
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", 
                 max_workers: int = 32, max_tokens: int = 2048, shot_num: int = 5):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.shot_num = shot_num
        self._correct_count = 0
        self._total_count = 0
        self._lock = threading.Lock()
        self._few_shot_cache: Dict[str, List[Dict]] = {}
        

    # ========== PPL评估方法 ==========
    def extract_option_logprob(
        self, tokens: List[str], token_logprobs: List[Optional[float]], option_label: str
    ) -> Optional[float]:
        """从token列表中提取选项标签的logprob"""
        # 添加边界检查
        if not tokens or not token_logprobs:
            return None
        
        max_len = min(len(tokens), len(token_logprobs))
        if max_len == 0:
            return None
        
        # 从后往前搜索最后15个tokens
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


    def load_ceval_data(self, data_dir: str, subset_name: str = None) -> List[Dict]:
        """从CSV文件加载C-Eval数据"""
        data = []
        
        if subset_name:
            # 加载单个subject
            subsets = [subset_name]
        else:
            # 加载所有subjects
            subsets = list(SUBJECT_MAPPING.keys())
        
        for subset in subsets:
            # 支持两种文件名格式：
            # 1. 官方格式: {subset}.csv (from HuggingFace)
            # 2. 旧格式: {subset}_test.csv, {subset}_dev.csv
            test_file = os.path.join(data_dir, 'test', f'{subset}.csv')
            dev_file = os.path.join(data_dir, 'dev', f'{subset}.csv')
            
            # 尝试旧格式
            if not os.path.exists(test_file):
                test_file = os.path.join(data_dir, 'test', f'{subset}_test.csv')
                dev_file = os.path.join(data_dir, 'dev', f'{subset}_dev.csv')
            
            if not os.path.exists(test_file):
                print(f"⚠️ 跳过不存在的文件: {subset}.csv 或 {subset}_test.csv")
                continue
            
            # 加载dev集用于few-shot
            dev_examples = []
            if os.path.exists(dev_file):
                with open(dev_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        dev_examples.append({
                            'question': row.get('question', ''),
                            'A': row.get('A', ''),
                            'B': row.get('B', ''),
                            'C': row.get('C', ''),
                            'D': row.get('D', ''),
                            'answer': row.get('answer', ''),
                        })
            
            # 缓存few-shot示例
            self._few_shot_cache[subset] = dev_examples[:self.shot_num]
            
            # 加载test集
            with open(test_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    subject, subject_zh, category = SUBJECT_MAPPING.get(subset, ("", "", ""))
                    data.append({
                        'question': row.get('question', ''),
                        'A': row.get('A', ''),
                        'B': row.get('B', ''),
                        'C': row.get('C', ''),
                        'D': row.get('D', ''),
                        'answer': row.get('answer', ''),
                        'subject': subject,
                        'subject_zh': subject_zh,
                        'category': category,
                        'subset': subset,
                    })
        
        return data
    
    def build_few_shot_prompt(self, subset: str) -> str:
        """构建few-shot prompt"""
        examples = self._few_shot_cache.get(subset, [])
        if not examples:
            return ""
        
        prompt_parts = []
        for example in examples:
            prompt_parts.append(
                f"问题：{example['question']}\n"
                f"A. {example['A']}\n"
                f"B. {example['B']}\n"
                f"C. {example['C']}\n"
                f"D. {example['D']}\n"
                f"答案：{example['answer']}\n"
            )
        
        return "\n".join(prompt_parts) + "\n"
    
    def build_prompt(self, question: str, A: str, B: str, C: str, D: str, subset: str, option_label: str = None) -> str:
        """构建评估prompt（PPL方法需要option_label）"""
        few_shot = self.build_few_shot_prompt(subset)
        prompt = f"""{few_shot}问题：{question}
A. {A}
B. {B}
C. {C}
D. {D}
答案：{option_label if option_label else ''}"""
        return prompt

    def postprocess_answer(self, text: str) -> str:
        """后处理答案，提取A/B/C/D"""
        text = text.strip().upper()
        # 查找第一个A/B/C/D
        match = re.search(r'[ABCD]', text)
        if match:
            return match.group(0)
        # 如果没找到，尝试查找"答案是A"这样的模式
        match = re.search(r'答案[：:]\s*([ABCD])', text)
        if match:
            return match.group(1)
        # 如果还是没找到，返回空字符串
        return ""
    
    def generate_answer(self, prompt: str, max_retries: int = 3) -> str:
        """生成答案"""
        last_exception: Optional[Exception] = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=50,  # C-Eval只需要一个字母
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
                               subset: str, label: str):
        """为单个选项获取logprob和长度"""
        from typing import Tuple
        prompt = self.build_prompt(question, A, B, C, D, subset, label)
        logprob, length = self.get_option_logprob_with_length(prompt, label)
        return label, logprob, length

    def evaluate_single_question(self, question_data: Dict) -> Dict:
        """评估单个问题（使用PPL方法，直接比较logprob）"""
        import numpy as np

        try:
            question = question_data['question']
            A = question_data['A']
            B = question_data['B']
            C = question_data['C']
            D = question_data['D']
            reference_answer = question_data['answer'].strip().upper()
            subset = question_data.get('subset', '')
            
            option_logprobs = {}
            option_normalized_logprobs = {}
            option_ppls = {}
            
            # 并发获取所有选项的logprob
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._get_logprob_for_label, question, A, B, C, D, subset, label): label
                    for label in ['A', 'B', 'C', 'D']
                }
                
            for future in as_completed(futures):
                try:
                    label, logprob, length = future.result()
                    option_logprobs[label] = logprob
                    # C-Eval: 不使用长度归一化，因为所有选项标签长度相同（都是单个字母）
                    # 直接使用原始logprob比较
                    normalized_logprob = logprob
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
                'question': question,
                'A': A,
                'B': B,
                'C': C,
                'D': D,
                'predicted_answer': predicted_answer,
                'reference_answer': reference_answer,
                'is_correct': is_correct,
                'subject': question_data.get('subject', ''),
                'subject_zh': question_data.get('subject_zh', ''),
                'category': question_data.get('category', ''),
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
        
        results = []
        
        self._correct_count = 0
        self._total_count = 0
        
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
        
        print(f"开始评估 {total} 个问题...")
        if max_samples:
            print(f"随机采样: {max_samples} 个样本（seed={seed}）")
        print(f"并发设置: {self.max_workers} 个问题并发")
        print(f"评估方法: {self.shot_num}-shot PPL（选择logprob最高的选项）")
        
        pbar = tqdm(total=total, desc="评估进度")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_data = {
                executor.submit(self.evaluate_single_question, q): q 
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
                    
                    if current_total % 100 == 0 and current_total > 0:
                        current_acc = current_correct / current_total * 100
                        pbar.set_postfix({
                            'accuracy': f'{current_acc:.2f}%',
                            'correct': current_correct
                        })
                except Exception as e:
                    print(f"\n评估失败: {e}")
                    pbar.update(1)
        
        pbar.close()
        
        correct = self._correct_count
        accuracy = correct / total * 100 if total > 0 else 0.0
        
        # 按category统计
        category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        for result in results:
            category = result.get('category', 'unknown')
            category_metrics[category]['total'] += 1
            if result.get('is_correct'):
                category_metrics[category]['correct'] += 1
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'shot_num': self.shot_num,
            'seed': seed if max_samples else None,
            'max_samples': max_samples,
            'category_metrics': {k: {
                'accuracy': 100 * v['correct'] / v['total'] if v['total'] > 0 else 0.0,
                'correct': v['correct'],
                'total': v['total']
            } for k, v in category_metrics.items()},
            'results': results
        }


def find_ceval_data_path():
    """查找C-Eval数据集路径（只从datasets目录查找）"""
    # 只从datasets目录查找
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'ceval', 'formal_ceval')
    
    if os.path.exists(datasets_dir) and os.path.exists(os.path.join(datasets_dir, "test")):
        return datasets_dir
    
    return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="C-Eval evaluation script")
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
        default=2048,
        help="Maximum tokens to generate (default: 2048)",
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
        help="Dataset directory (default: datasets/ceval/formal_ceval)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    
    args = parser.parse_args()
    
    # 自动查找数据路径，如果找不到则使用默认路径
    DATA_DIR = args.dataset_dir or find_ceval_data_path()
    if DATA_DIR is None:
        DATA_DIR = "/volume/ai-infra/zkjia/projects/opencompass/data/ceval/formal_ceval"  # 默认路径
        print(f"⚠️  未找到C-Eval数据集，将使用默认路径: {DATA_DIR}")
        print("   如果目录不存在，请使用--dataset-dir参数指定路径")
    
    # 创建日志目录结构：logs/{model}/
    log_dir = os.path.join("logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    
    shot_desc = f"{args.shot_num}-shot" if args.shot_num > 0 else "0-shot"
    output_filename = f"c_eval_{args.shot_num}shot.json"
    if args.max_samples:
        output_filename = f"c_eval_{args.shot_num}shot_{args.max_samples}samples.json"
    OUTPUT_PATH = os.path.join(log_dir, output_filename)
    
    print("="*70)
    print(f"C-Eval 评估 - {args.model} - {shot_desc} PPL方法")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"API: completions + echo=True")
    print(f"方法: {shot_desc} few-shot prompting + PPL")
    print("="*70)
    
    evaluator = CEvalEvaluator(
        args.base_url, 
        args.model, 
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        shot_num=args.shot_num
    )
    
    print(f"\n加载数据: {DATA_DIR}")
    try:
        data = evaluator.load_ceval_data(DATA_DIR)
        print(f"✓ 成功加载 {len(data)} 条数据")
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
        
        # 按category统计
        if 'category_metrics' in results:
            print("\n按Category统计:")
            for category, metrics in results['category_metrics'].items():
                print(f"  {category}: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.2f}%")
        
        print(f"\n详细结果已保存到: {OUTPUT_PATH}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  评估被用户中断")
    except Exception as e:
        print(f"\n✗ 评估过程中出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

