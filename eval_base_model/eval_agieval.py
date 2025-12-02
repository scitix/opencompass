"""AGIEval数据集评估脚本 - 0-shot生成方法（EM评估）。"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from openai import OpenAI, APIConnectionError
from tqdm import tqdm
import eval_utils
# ============== Math Equivalence (from hendrycks/math) ==============
# Source: https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
# flake8: noqa

def _fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        a = int(a)
        b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if '\\text{ ' in string:
        splits = string.split('\\text{ ')
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if '\\sqrt' not in string:
        return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != '{':
            a = split[0]
            new_substr = '\\sqrt{' + a + '}' + split[1:]
        else:
            new_substr = '\\sqrt' + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace('\n', '')

    # remove inverse spaces
    string = string.replace('\\!', '')

    # replace \\ with \
    string = string.replace('\\\\', '\\')

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # remove dollar signs
    string = string.replace('\\$', '')

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace('\%', '')

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(' ', '')

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == '0.5':
        string = '\\frac{1}{2}'

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print('WARNING: Both None')
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2

# ====================================================================


# 导入agieval_post_process（如果存在）
try:
    from agieval_post_process import parse_math_answer as oc_parse_math_answer, find_first_capital_letter
except ImportError:
    def oc_parse_math_answer(setting_name, text):
        return text.strip()
    def find_first_capital_letter(text):
        for c in text:
            if c in 'ABCDEFG':
                return c
        return ''


class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


class AGIEvalEvaluator:
    """使用few-shot生成方法评估AGIEval数据集"""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_workers: int = 32, max_tokens: int = 1024, 
                 shot_num: int = 0, data_dir: str = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.shot_num = shot_num
        self.data_dir = data_dir
        self._correct_count = 0
        self._total_count = 0
        self._lock = threading.Lock()
        self._few_shot_cache: Dict[str, List[Dict]] = {}  # 按子集缓存

    def load_agieval_data(self, data_dir: str, subset_name: str = None) -> List[Dict]:
        """从JSONL文件加载AGIEval数据"""
        data = []
        
        if subset_name:
            subsets = [subset_name]
        else:
            # 加载所有子集
            subsets = [f.replace('.jsonl', '') for f in os.listdir(data_dir) 
                       if f.endswith('.jsonl')]
        
        for subset in subsets:
            jsonl_path = os.path.join(data_dir, f'{subset}.jsonl')
            if not os.path.exists(jsonl_path):
                continue
            
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    passage = item.get('passage', '')
                    question = item.get('question', '')
                    options = item.get('options', [])
                    label = item.get('label', '') or item.get('answer', '')
                    
                    # 处理label（可能是列表）
                    if isinstance(label, list):
                        label = ''.join(label)
                    
                    # 构建完整问题
                    full_question = passage + question if passage else question
                    options_text = '\n'.join(options) if options else ''
                    
                    data.append({
                        'question': full_question,
                        'options': options_text,
                        'label': label,
                        'subset': subset,
                    })
        
        return data

    def _prepare_few_shot(self, subset: str) -> List[Dict]:
        """准备few-shot示例"""
        if subset in self._few_shot_cache:
            return self._few_shot_cache[subset]
        
        if self.shot_num == 0:
            self._few_shot_cache[subset] = []
            return []
        
        # 加载该子集的所有数据
        all_data = self.load_agieval_data(self.data_dir, subset_name=subset)
        
        if len(all_data) < self.shot_num:
            raise ValueError(f"{subset} 数据不足 {self.shot_num} 条")
        
        # 选择前 shot_num 个作为示例
        few_shot_examples = all_data[:self.shot_num]
        self._few_shot_cache[subset] = few_shot_examples
        
        return few_shot_examples

    def build_prompt(self, question: str, options: str, subset: str) -> str:
        """构建prompt（混合策略：平衡简洁性和引导性）"""
        # 数据集分类
        english_qa_datasets = [
            'lsat-ar', 'lsat-lr', 'lsat-rc', 'logiqa-en', 'sat-math', 'sat-en',
            'aqua-rat', 'sat-en-without-passage'
        ]
        chinese_qa_datasets = [
            'logiqa-zh', 'jec-qa-kd', 'jec-qa-ca', 'gaokao-chinese',
            'gaokao-geography', 'gaokao-history', 'gaokao-biology', 'gaokao-chemistry',
            'gaokao-physics', 'gaokao-mathqa', 'gaokao-english'
        ]
        english_cloze_datasets = ['math']
        chinese_cloze_datasets = ['gaokao-mathcloze']
        
        # 对于英文QA（需要推理的），使用引导式prompt
        if subset in english_qa_datasets and options:
            option_list = options.split('\n')
            option_string = 'ABCDEFG'
            count = len(option_list) if option_list else 5
            # 引导格式帮助模型更好地理解选择题
            prompt = f"{question}\n{options}\nAmong A through {option_string[count-1]}, the answer is"
        
        # 对于中文QA，使用简单引导
        elif subset in chinese_qa_datasets and options:
            option_list = options.split('\n')
            option_string = 'ABCDEFG'
            count = len(option_list) if option_list else 4
            prompt = f"{question}\n{options}\n从A到{option_string[count-1]}, 我们应选择"
        
        # 对于Cloze（填空题），直接问答
        elif subset in english_cloze_datasets:
            prompt = f"{question}\nThe answer is"
        
        elif subset in chinese_cloze_datasets:
            prompt = f"{question}\n答案："
        
        else:
            # 默认格式
            prompt = f"{question}\n{options}\nThe answer is " if options else f"{question}\nThe answer is"
        
        return prompt

    def postprocess_answer(self, text: str, subset: str) -> str:
        """后处理答案（简单有效的版本）"""
        text = text.strip()
        
        # 数据集分类
        english_cloze_datasets = ['math']
        chinese_cloze_datasets = ['gaokao-mathcloze']
        multi_choice_datasets = ['jec-qa-kd', 'jec-qa-ca', 'gaokao-physics']
        
        # Cloze任务：使用OpenCompass的parse_math_answer
        if subset in english_cloze_datasets or subset in chinese_cloze_datasets:
            result = oc_parse_math_answer('zero-shot', text)
            return result if result else text  # 如果提取失败，返回原文
        
        # 多选题：提取所有选项字母
        if subset in multi_choice_datasets:
            # 策略1: 匹配连续字母 "ABC" or "ABCD"
            match = re.search(r'\b([A-G]{2,})\b', text.upper())
            if match:
                return match.group(1)
            
            # 策略2: 匹配括号格式 (A)(B)(C)
            matches = re.findall(r'\(([A-G])\)', text.upper())
            if matches:
                return ''.join(matches)
            
            # 策略3: 匹配独立字母 A B C
            matches = re.findall(r'\b([A-G])\b', text.upper())
            if matches:
                return ''.join(matches)
            
            return ''
        
        # 单选题：智能提取（避免提取prompt中的选项）
        # 策略1: 优先匹配常见的答案表达模式
        answer_patterns = [
            r'(?:answer|choice|option|选择|答案)(?:\s+is|\s+:)?\s*\(?([A-G])\)?',  # "answer is B" or "答案：B"
            r'选\s*([A-G])',  # "选B"
            r'(?:choose|pick|select)\s+\(?([A-G])\)?',  # "choose B"
            r'^\s*\(?([A-G])\)?[\s\.\,]',  # 开头就是选项 "B. because..."
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # 策略2: 如果上述模式都没匹配，找最后一个独立的字母
        # （假设prompt中的选项在前，答案在后）
        matches = re.findall(r'\b([A-G])\b', text.upper())
        if matches:
            return matches[-1]  # 返回最后一个
        
        # 策略3: 兜底，找任何字母
        match = re.search(r'[A-G]', text.upper())
        if match:
            return match.group(0)
        
        return ''
    
    def generate_answer(self, prompt: str, subset: str, max_retries: int = 3) -> str:
        """生成答案"""
        last_exception: Optional[Exception] = None

        # 使用简单的stop策略（恢复到59.80%的配置）
        stop_tokens = ["\n", "---"]

        for attempt in range(max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    stop=stop_tokens,
                )

                if not response.choices:
                    raise ModelResponseError('模型返回的choices为空。')

                generated_text = response.choices[0].text
                return self.postprocess_answer(generated_text, subset)

            except APIConnectionError as e:
                # 连接错误特殊处理
                error_msg = f"API连接失败: {str(e)}\n"
                error_msg += f"API端点: {self.client.base_url}\n"
                error_msg += "请检查:\n"
                error_msg += "  1. API服务是否正在运行\n"
                error_msg += "  2. --base-url参数是否正确\n"
                error_msg += "  3. 网络连接是否正常"
                raise ModelResponseError(error_msg) from e

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

    def is_answer_correct(self, pred: str, ref: str, subset: str) -> bool:
        """判断答案是否正确（使用OpenCompass标准）"""
        pred = str(pred).strip()
        ref = str(ref).strip()
        
        # 分类数据集
        english_cloze_datasets = ['math']
        chinese_cloze_datasets = ['gaokao-mathcloze']
        multi_choice_datasets = ['jec-qa-kd', 'jec-qa-ca', 'gaokao-physics']
        
        # 对于填空题（数学题），使用OpenCompass的is_equiv函数
        if subset in english_cloze_datasets or subset in chinese_cloze_datasets:
            # 使用OpenCompass的数学等价性检查
            return is_equiv(pred, ref)
        
        # 对于多选题，排序后比较
        if subset in multi_choice_datasets:
            # 提取所有字母并排序
            pred_letters = sorted(set(c for c in pred.upper() if c in 'ABCDEFG'))
            ref_letters = sorted(set(c for c in ref.upper() if c in 'ABCDEFG'))
            return pred_letters == ref_letters
        
        # 对于单选题，直接比较（大小写不敏感）
        return pred.upper() == ref.upper()

    def evaluate_single_question(self, question_data: Dict) -> Dict:
        """评估单个问题"""
        try:
            question = question_data['question']
            options = question_data['options']
            reference_answer = question_data['label']
            subset = question_data['subset']

            prompt = self.build_prompt(question, options, subset)
            predicted_answer = self.generate_answer(prompt, subset)

            is_correct = self.is_answer_correct(predicted_answer, reference_answer, subset)

            with self._lock:
                if is_correct:
                    self._correct_count += 1
                self._total_count += 1

            return {
                'question': question[:200] + '...' if len(question) > 200 else question,
                'options': options[:200] + '...' if len(options) > 200 else options,
                'predicted_answer': predicted_answer,
                'reference_answer': reference_answer,
                'is_correct': is_correct,
                'subset': subset,
            }
        except Exception as e:
            print(f"\n评估问题时出错: {e}")
            return {
                'question': question_data.get('question', ''),
                'predicted_answer': '',
                'reference_answer': question_data.get('label', ''),
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
                question = sample['question']
                options = sample['options']
                subset = sample['subset']
                prompt = self.build_prompt(question, options, subset)
                predicted_answer = self.generate_answer(prompt, subset)
                print(f"✓ 样本 {i+1}: 通过")

                # 检测异常答案（空答案可能表示模型未正常响应）
                if not predicted_answer or predicted_answer.strip() == '':
                    print(f"⚠️  警告: 样本 {i+1} 返回空答案")
                    warmup_failed = True

            except Exception as e:
                print(f"⚠️  样本 {i+1} 失败: {e}")
                warmup_failed = True

        if warmup_failed:
            print(f"\n❌ 预热检测发现异常!")
            print(f"请检查:")
            print(f"  1. API 端点是否正确")
            print(f"  2. 模型是否正常运行")
            print(f"  3. 网络连接是否稳定")
            raise ModelResponseError("预热检测失败，中止评估")
        print("✓ 预热检测通过，开始正式评估...\n")

        # 重置计数器（预热检测已完成，不计入正式统计）
        self._correct_count = 0
        self._total_count = 0

        # 正式评估：评估所有样本（预热检测不跳过任何数据）
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.evaluate_single_question, item): item for item in data}

            with tqdm(total=len(data), desc="评估进度", unit="问题") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    if self._total_count > 0:
                        pbar.set_postfix(accuracy=f"{self._correct_count / self._total_count * 100:.2f}%",
                                       correct=self._correct_count)

        # 按子集统计
        subset_stats = {}
        for result in results:
            subset = result.get('subset', 'unknown')
            if subset not in subset_stats:
                subset_stats[subset] = {'total': 0, 'correct': 0}
            subset_stats[subset]['total'] += 1
            if result.get('is_correct', False):
                subset_stats[subset]['correct'] += 1

        return {
            'total': self._total_count,
            'correct': self._correct_count,
            'accuracy': self._correct_count / self._total_count * 100 if self._total_count > 0 else 0.0,
            'shot_num': self.shot_num,
            'seed': seed if max_samples else None,
            'max_samples': max_samples,
            'subset_stats': {k: {
                'total': v['total'],
                'correct': v['correct'],
                'accuracy': v['correct'] / v['total'] * 100 if v['total'] > 0 else 0.0
            } for k, v in subset_stats.items()},
            'results': results
        }


DEFAULT_AGIEVAL_DIR = os.path.join(os.path.dirname(__file__), "datasets", "agieval")


def find_agieval_data_path(dataset_dir: str = None) -> Optional[str]:
    """查找AGIEval数据集路径"""
    if dataset_dir is None:
        dataset_dir = DEFAULT_AGIEVAL_DIR
    
    if os.path.exists(dataset_dir):
        # 检查是否有 JSONL 文件
        jsonl_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jsonl')]
        if jsonl_files:
            return dataset_dir
    
    return None


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="AGIEval 评估脚本")

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
        default=32,
        help="最大并发工作线程数（默认: 32）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="生成的最大 token 数（默认: 512）",
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
        help=f"数据集目录（默认: {DEFAULT_AGIEVAL_DIR}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )

    args = parser.parse_args()

    # 查找数据集路径
    dataset_dir = args.dataset_dir or DEFAULT_AGIEVAL_DIR
    data_dir = find_agieval_data_path(dataset_dir)
    
    if data_dir is None:
        print(f"⚠️  未找到 AGIEval 数据集")
        print(f"请先运行以下命令下载数据集:")
        print(f"  python download_datasets.py agieval")
        print(f"或者手动将数据集放置在: {dataset_dir}")
        return

    log_dir = os.path.join(os.path.dirname(__file__), "logs", args.model)
    os.makedirs(log_dir, exist_ok=True)

    output_filename = f"agieval_{args.shot_num}shot.json"
    if args.max_samples:
        output_filename = f"agieval_{args.shot_num}shot_{args.max_samples}samples.json"
    output_path = os.path.join(log_dir, output_filename)

    print("=" * 70)
    print(f"{args.shot_num}-shot 生成方法")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"API: completions")
    print(f"{args.shot_num}-shot prompting + EM评估")
    print("=" * 70)

    evaluator = AGIEvalEvaluator(
        base_url=args.base_url,
        model=args.model,
        api_key="EMPTY",
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        shot_num=args.shot_num,
        data_dir=data_dir
    )

    print(f"\n加载数据: {data_dir}")
    try:
        data = evaluator.load_agieval_data(data_dir)
        print(f"✓ 成功加载 {len(data)} 条数据")
        
        # 统计各子集数量
        subset_counts = {}
        for item in data:
            subset = item.get('subset', 'unknown')
            subset_counts[subset] = subset_counts.get(subset, 0) + 1
        print(f"\n子集分布:")
        for subset, count in sorted(subset_counts.items()):
            print(f"  - {subset}: {count} 条")
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
        
        print(f"\n各子集准确率:")
        for subset, stats in sorted(results['subset_stats'].items()):
            print(f"  - {subset}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")

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
