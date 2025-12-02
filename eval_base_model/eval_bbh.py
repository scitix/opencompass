"""BBH数据集评估脚本 - 3-shot生成方法（CoT）。"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from openai import OpenAI
from tqdm import tqdm
import eval_utils
class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


class BBHEvaluator:
    """使用3-shot CoT生成方法评估BBH数据集（参考OpenCompass实现）"""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_workers: int = 32, max_tokens: int = 1024, shot_num: int = 3,
                 task_name: str = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.shot_num = shot_num
        self.task_name = task_name
        self._correct_count = 0
        self._total_count = 0
        self._lock = threading.Lock()
        self._failed_extractions = 0

        # 加载任务特定的few-shot examples（参考OpenCompass的lib_prompt）
        self.few_shot_prompt = self._load_few_shot_prompt()

    def _load_few_shot_prompt(self) -> str:
        """加载任务特定的few-shot prompt（参考OpenCompass的lib_prompt）"""
        if not self.task_name or self.shot_num == 0:
            return ""

        # 查找lib_prompt文件
        lib_prompt_path = os.path.join(
            os.path.dirname(__file__),
            'datasets',
            'bbh',
            'lib_prompt',
            f'{self.task_name}.txt'
        )

        if os.path.exists(lib_prompt_path):
            with open(lib_prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            print(f"⚠️  未找到任务 {self.task_name} 的few-shot prompt文件")
            print(f"   路径: {lib_prompt_path}")
            return ""

    def load_bbh_data(self, json_path: str) -> List[Dict]:
        """从JSON文件加载BBH数据"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        result = []
        for item in data.get('examples', []):
            result.append({
                'input': item.get('input', ''),
                'target': item.get('target', ''),
            })
        return result

    def build_prompt(self, input_text: str) -> str:
        """构建prompt（参考OpenCompass的实现）"""
        if self.shot_num == 0 or not self.few_shot_prompt:
            # 0-shot: 直接问答
            return f"Q: {input_text}\nA: Let's think step by step."
        else:
            # Few-shot: 使用任务特定的CoT examples
            # 格式: "Follow the given examples and answer the question.\n{few_shot_examples}\n\nQ: {input}\nA: Let's think step by step."
            return f"Follow the given examples and answer the question.\n{self.few_shot_prompt}\n\nQ: {input_text}\nA: Let's think step by step."

    def extract_answer(self, text: str) -> Optional[str]:
        """从生成的文本中提取答案（参考OpenCompass的bbh后处理）

        BBH任务的答案通常在"So the answer is"或"the answer is"之后
        """
        # 先截断到第一个新问题（如果模型生成了多个问题）
        if '\n\nQ:' in text:
            text = text.split('\n\nQ:')[0]

        # 策略1: 查找"So the answer is"模式（最常见）
        patterns = [
            r'[Ss]o the answer is[:\s]+([^\n\.]+)',
            r'[Tt]he answer is[:\s]+([^\n\.]+)',
            r'[Aa]nswer is[:\s]+([^\n\.]+)',
            r'[Aa]nswer:[:\s]+([^\n\.]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                ans = match.group(1).strip()
                # 清理答案
                ans = ans.rstrip('.')
                ans = ans.strip()
                # 移除可能的引号、括号等
                ans = ans.strip('"\'()[]{}')
                return ans

        # 策略2: 如果没找到标准格式，尝试提取最后一句话
        # 按句子分割（以句号、问号、感叹号分割）
        sentences = re.split(r'[\.!\?]\s+', text)
        if sentences:
            last_sentence = sentences[-1].strip()
            # 如果最后一句话很短（可能是答案）
            if len(last_sentence) < 50:
                return last_sentence.rstrip('.').strip()

        # 策略3: 如果都没找到，返回None（表示提取失败）
        return None

    def is_equal(self, pred: str, refer: str) -> bool:
        """判断两个答案是否相等（参考OpenCompass的evaluator）"""
        if pred is None:
            return False

        # 标准化：转小写、去空格、去标点
        def normalize(s):
            s = s.lower().strip()
            # 移除常见的标点符号
            s = s.strip('.,;:!?\'"()[]{}')
            # 移除多余空格
            s = ' '.join(s.split())
            return s

        pred_norm = normalize(pred)
        refer_norm = normalize(refer)

        # 精确匹配
        if pred_norm == refer_norm:
            return True

        # 对于选择题，提取选项字母
        # 例如："(A)" -> "a", "A" -> "a"
        pred_option = re.search(r'\(?([A-Ea-e])\)?', pred)
        refer_option = re.search(r'\(?([A-Ea-e])\)?', refer)

        if pred_option and refer_option:
            return pred_option.group(1).lower() == refer_option.group(1).lower()

        # 对于是/否问题
        yes_patterns = ['yes', 'true', 'correct', 'valid']
        no_patterns = ['no', 'false', 'incorrect', 'invalid']

        pred_is_yes = any(p in pred_norm for p in yes_patterns)
        pred_is_no = any(p in pred_norm for p in no_patterns)
        refer_is_yes = any(p in refer_norm for p in yes_patterns)
        refer_is_no = any(p in refer_norm for p in no_patterns)

        if (pred_is_yes and refer_is_yes) or (pred_is_no and refer_is_no):
            return True

        # 包含关系（参考答案包含在预测中，或反之）
        if refer_norm in pred_norm or pred_norm in refer_norm:
            return True

        return False

    def generate_solution(self, input_text: str, max_retries: int = 3) -> str:
        """生成解决方案"""
        prompt = self.build_prompt(input_text)
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                # 添加停止条件，避免模型生成多个问题
                stop_sequences = ["\n\nQ:", "\nQ:", "\n\n\n"]

                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    stop=stop_sequences,
                )

                if not response.choices:
                    raise ModelResponseError('模型返回的choices为空。')

                generated_text = response.choices[0].text
                generated_text = generated_text.strip()

                # 如果生成了多个问题，只保留第一个问题的答案
                if '\n\nQ:' in generated_text:
                    generated_text = generated_text.split('\n\nQ:')[0].strip()

                return generated_text

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

    def evaluate_single_problem(self, problem_data: Dict) -> Dict:
        """评估单个问题"""
        try:
            input_text = problem_data['input']
            reference_answer = problem_data.get('target', '')

            generated_solution = self.generate_solution(input_text)

            # 提取答案
            predicted_answer = self.extract_answer(generated_solution)

            # 判断是否相等
            is_correct = False
            if predicted_answer is not None:
                is_correct = self.is_equal(predicted_answer, reference_answer)
            else:
                with self._lock:
                    self._failed_extractions += 1

            with self._lock:
                if is_correct:
                    self._correct_count += 1
                self._total_count += 1

            return {
                'input': input_text,
                'predicted_answer': predicted_answer,
                'reference_answer': reference_answer,
                'generated_solution': generated_solution,
                'is_correct': is_correct,
            }
        except Exception as e:
            print(f"\n评估问题时出错: {e}")
            return {
                'input': problem_data.get('input', ''),
                'predicted_answer': None,
                'reference_answer': problem_data.get('target', ''),
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

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.evaluate_single_problem, item): item for item in data}

            with tqdm(total=len(data), desc="评估进度", unit="问题") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    pbar.set_postfix(accuracy=f"{self._correct_count / self._total_count * 100:.2f}%",
                                   correct=self._correct_count)

        return {
            'total': self._total_count,
            'correct': self._correct_count,
            'accuracy': self._correct_count / self._total_count * 100 if self._total_count > 0 else 0.0,
            'shot_num': self.shot_num,
            'seed': seed if max_samples else None,
            'max_samples': max_samples,
            'failed_extractions': self._failed_extractions,
            'results': results
        }


def find_bbh_tasks():
    """查找所有BBH任务"""
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'bbh')
    if not os.path.exists(datasets_dir):
        return []

    tasks = []
    for filename in os.listdir(datasets_dir):
        if filename.endswith('.json') and filename != 'README.md':
            task_name = filename[:-5]  # 移除.json后缀
            tasks.append(task_name)

    return sorted(tasks)


def find_bbh_data_path(task_name: str):
    """查找BBH数据集路径"""
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'bbh')
    json_path = os.path.join(datasets_dir, f'{task_name}.json')

    if os.path.exists(json_path):
        return json_path

    return None


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="BBH evaluation script")
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
        default=1024,
        help="Maximum tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--shot-num",
        type=int,
        default=3,
        help="Number of few-shot examples (default: 3)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Random sample size, None means evaluate all data (default: None)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific BBH task to evaluate (default: None, evaluate all tasks)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )

    args = parser.parse_args()

    # 查找所有BBH任务
    all_tasks = find_bbh_tasks()
    if not all_tasks:
        print("⚠️  未找到BBH数据集")
        print("   请确保数据集在 datasets/bbh/ 目录下")
        return

    # 确定要评估的任务
    if args.task:
        if args.task in all_tasks:
            tasks_to_eval = [args.task]
        else:
            print(f"⚠️  未找到任务 '{args.task}'")
            print(f"   可用任务: {', '.join(all_tasks[:5])}... (共{len(all_tasks)}个)")
            return
    else:
        tasks_to_eval = all_tasks

    # 创建日志目录结构：logs/{model}/
    log_dir = os.path.join("logs", args.model)
    os.makedirs(log_dir, exist_ok=True)

    shot_desc = f"{args.shot_num}-shot" if args.shot_num > 0 else "0-shot"

    print("="*70)
    print(f"BBH 评估 - {args.model} - {shot_desc} CoT生成方法")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"API: completions")
    print(f"方法: {shot_desc} CoT prompting + 答案提取")
    print(f"任务数: {len(tasks_to_eval)}")
    if args.task:
        print(f"指定任务: {args.task}")
    print("="*70)

    # 评估每个任务
    all_results = {}
    total_correct = 0
    total_count = 0

    for idx, task_name in enumerate(tasks_to_eval, 1):
        print(f"\n[{idx}/{len(tasks_to_eval)}] 评估任务: {task_name}")
        print("-" * 70)

        # 查找数据路径
        json_path = find_bbh_data_path(task_name)
        if json_path is None:
            print(f"⚠️  未找到任务数据: {task_name}.json")
            continue

        # 创建评估器
        evaluator = BBHEvaluator(
            args.base_url,
            args.model,
            max_workers=args.max_workers,
            max_tokens=args.max_tokens,
            shot_num=args.shot_num,
            task_name=task_name
        )

        # 加载数据
        try:
            data = evaluator.load_bbh_data(json_path)
            print(f"✓ 加载 {len(data)} 条数据")
        except Exception as e:
            print(f"✗ 加载数据失败: {e}")
            continue

        # 评估
        try:
            results = evaluator.evaluate_dataset(data, max_samples=args.max_samples, seed=args.seed)

            # 存储完整的任务结果（包含所有详细数据）
            all_results[task_name] = results
            total_correct += results['correct']
            total_count += results['total']

            print(f"✓ 准确率: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
            if results['failed_extractions'] > 0:
                print(f"  提取失败: {results['failed_extractions']} 个")

        except Exception as e:
            print(f"✗ 评估失败: {e}")
            import traceback

            traceback.print_exc()

    # 保存所有结果到一个文件
    if all_results:
        overall_accuracy = total_correct / total_count * 100 if total_count > 0 else 0.0

        output_filename = f"bbh_{args.shot_num}shot.json"
        if args.max_samples:
            output_filename = f"bbh_{args.shot_num}shot_{args.max_samples}samples.json"
        output_path = os.path.join(log_dir, output_filename)

        # 构建完整的结果结构
        full_results = {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_count': total_count,
            'num_tasks': len(all_results),
            'shot_num': args.shot_num,
            'seed': args.seed if args.max_samples else None,
            'max_samples': args.max_samples,
            'tasks': all_results  # 包含所有子任务的完整结果
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)

        print("\n" + "="*70)
        print("总体评估结果")
        print("="*70)
        print(f"任务数: {len(all_results)}")
        print(f"总问题数: {total_count}")
        print(f"总正确数: {total_correct}")
        print(f"总体准确率: {overall_accuracy:.2f}%")
        print(f"Shot配置: {shot_desc}")
        if args.max_samples:
            print(f"随机采样: {args.max_samples} 个样本/任务（seed={args.seed}）")
        print(f"\n结果已保存到: {output_path}")
        print("="*70)

        # 打印每个任务的准确率
        print("\n各任务准确率:")
        for task_name, task_result in sorted(all_results.items()):
            print(f"  {task_name:45s}: {task_result['accuracy']:6.2f}%")


if __name__ == "__main__":
    main()
