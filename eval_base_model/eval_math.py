"""Math数据集评估脚本 - 4-shot生成方法。"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from openai import OpenAI
from tqdm import tqdm
import threading
import eval_utils
# 4-shot few-shot examples
# 注意：OpenCompass配置中第一个示例的prompt有`.}}`，但实际渲染时应该是`.`
FEW_SHOT_EXAMPLES = """Problem:
Find the domain of the expression $\\frac{{\\sqrt{{x-2}}}}{{\\sqrt{{5-x}}}}$.
Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{{[2,5)}}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\\det \\mathbf{{A}} = 2$ and $\\det \\mathbf{{B}} = 12,$ then find $\\det (\\mathbf{{A}} \\mathbf{{B}}).$
Solution:
We have that $\\det (\\mathbf{{A}} \\mathbf{{B}}) = (\\det \\mathbf{{A}})(\\det \\mathbf{{B}}) = (2)(12) = \\boxed{{24}}$.
Final Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$: \\begin{{align*}} 30n&=480\\\\ \\Rightarrow\\qquad n&=480/30=\\boxed{{16}} \\end{{align*}}
Final Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations: \\begin{{align*}} 6x-4y&=a,\\\\ 6y-9x &=b. \\end{{align*}}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{{a}}{{b}},$ assuming $b$ is nonzero.
Solution:
If we multiply the first equation by $-\\frac{{3}}{{2}}$, we obtain $$6y-9x=-\\frac{{3}}{{2}}a.$$Since we also know that $6y-9x=b$, we have $$-\\frac{{3}}{{2}}a=b\\Rightarrow\\frac{{a}}{{b}}=\\boxed{{-\\frac{{2}}{{3}}}}.$$
Final Answer: The final answer is $-\\frac{{2}}{{3}}$. I hope it is correct.

"""


class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


class MathEvaluator:
    """使用4-shot生成方法评估Math数据集"""
    
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_workers: int = 32, max_tokens: int = 2048, shot_num: int = 4):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.shot_num = shot_num
        self._correct_count = 0
        self._total_count = 0
        self._lock = threading.Lock()
        
    def load_math_data(self, json_path: str) -> List[Dict]:
        """从JSON文件加载Math数据，并提取boxed答案（与OpenCompass一致）"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = []
        for key, item in data.items():
            # 从solution中提取boxed答案（与OpenCompass的MATHDataset.load一致）
            solution_text = item.get('solution', '')
            extracted_answer = self.extract_boxed_answer(solution_text)
            if extracted_answer is None:
                # 如果没有boxed答案，使用整个solution（但这种情况应该很少）
                extracted_answer = solution_text
            
            result.append({
                'problem': item['problem'],
                'solution': extracted_answer,  # 存储提取的答案，而不是完整solution
                'solution_full': solution_text,  # 保留完整solution用于调试
            })
        return result
    
    def build_prompt(self, problem: str) -> str:
        """构建4-shot prompt"""
        return f"""{FEW_SHOT_EXAMPLES}Problem:
{problem}
Solution:
"""
    
    def last_boxed_only_string(self, string: str) -> Optional[str]:
        """查找最后一个\\boxed字符串（与OpenCompass一致）
        
        注意：对于base模型，如果生成了多个问题，应该提取第一个问题的答案。
        但为了与OpenCompass保持一致，这里仍然使用rfind查找最后一个。
        实际使用时，应该通过停止条件避免生成多个问题。
        """
        # 先尝试查找第一个boxed（如果模型生成了多个问题，应该只取第一个）
        # 查找第一个"Problem:"出现的位置，如果存在，只在这之前查找boxed
        first_problem_idx = string.find('\n\nProblem:')
        if first_problem_idx > 0:
            # 如果找到了新的问题，只在前面的部分查找boxed
            search_string = string[:first_problem_idx]
        else:
            search_string = string
        
        idx = search_string.rfind('\\boxed')
        if idx < 0:
            idx = search_string.rfind('\\fbox')
            if idx < 0:
                return None
        
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(search_string):
            if search_string[i] == '{':
                num_left_braces_open += 1
            if search_string[i] == '}':
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        
        if right_brace_idx is None:
            return None
        
        return search_string[idx:right_brace_idx + 1]
    
    def remove_boxed(self, s: str) -> Optional[str]:
        """移除\\boxed标记（与OpenCompass一致）"""
        left = '\\boxed{'
        try:
            if s.startswith(left) and s.endswith('}'):
                return s[len(left):-1]
        except Exception:
            pass
        return None
    
    def extract_boxed_answer(self, text: str, strip_double_curly_brace: bool = True) -> Optional[str]:
        """从文本中提取\\boxed答案（与OpenCompass一致）"""
        boxed_str = self.last_boxed_only_string(text)
        if boxed_str is None:
            return None
        
        answer = self.remove_boxed(boxed_str)
        if answer is None:
            return None
        
        # 默认strip_double_curly_brace=True，因为OpenCompass的math_postprocess_v2使用了这个参数
        if strip_double_curly_brace:
            # 处理双重花括号的情况，如 \boxed{{2}} -> 2
            match = re.match(r'^\{(.*)\}$', answer)
            if match:
                answer = match.group(1)
        
        return answer
    
    def normalize_final_answer(self, final_answer: str) -> str:
        """标准化最终答案"""
        SUBSTITUTIONS = [
            ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''),
            (r'\ ', ''), (' ', ''), ('mbox', 'text'),
            (',\\text{and}', ','), ('\\text{and}', ','),
            ('\\text{m}', '\\text{}'), ('\\le', '<')
        ]
        REMOVED_EXPRESSIONS = [
            'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
            'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes',
            'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals',
            'edges', 'students', 'childrentickets', 'multiples', '\\text{s}',
            '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}',
            '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;', r',\!',
            '{,}', '"', '\\dots', '\n', '\r', '\f'
        ]
        
        for before, after in SUBSTITUTIONS:
            final_answer = final_answer.replace(before, after)
        for expr in REMOVED_EXPRESSIONS:
            final_answer = final_answer.replace(expr, '')
        
        # Extract answer that is in LaTeX math
        final_answer = re.sub(r'(\\text\{)\((.*?)\)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
        
        if len(re.findall(r'finalansweris(.*)', final_answer)) > 0:
            final_answer = re.findall(r'finalansweris(.*)', final_answer)[-1]
        
        if len(re.findall(r'answer?is:?(.*)', final_answer)) > 0:
            final_answer = re.findall(r'answer?is:?(.*)', final_answer)[-1]
        
        if len(re.findall(r'oxed\{(.*?)\}', final_answer)) > 0:
            final_answer = re.findall(r'oxed\{(.*?)\}', final_answer)[-1]
        
        if len(re.findall(r'\$(.*?)\$', final_answer)) > 0:
            final_answer = re.findall(r'\$(.*?)\$', final_answer)[-1]
        
        final_answer = final_answer.strip()
        if 'rac' in final_answer and '\\frac' not in final_answer:
            final_answer = final_answer.replace('rac', '\\frac')
        
        # Normalize shorthand TeX
        final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
        final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
        final_answer = final_answer.replace('$', '')
        
        # Normalize 100,000 -> 100000
        if final_answer.replace(',', '').isdigit():
            final_answer = final_answer.replace(',', '')
        
        return final_answer
    
    def _strip_string(self, string: str) -> str:
        """标准化字符串用于比较"""
        string = string.replace('\n', '')
        string = string.replace('\\!', '')
        string = string.replace('\\\\', '\\')
        string = string.replace('tfrac', 'frac')
        string = string.replace('dfrac', 'frac')
        string = string.replace('\\left', '')
        string = string.replace('\\right', '')
        string = string.replace('^{\\circ}', '')
        string = string.replace('^\\circ', '')
        string = string.replace('\\$', '')
        if '\\text{ ' in string:
            splits = string.split('\\text{ ')
            if len(splits) == 2:
                string = splits[0]
        string = string.replace('\\%', '')
        string = string.replace(r'\%', '')
        string = string.replace(' .', ' 0.')
        string = string.replace('{.', '{0.')
        if len(string) > 0 and string[0] == '.':
            string = '0' + string
        if len(string.split('=')) == 2:
            if len(string.split('=')[0]) <= 2:
                string = string.split('=')[1]
        # Fix sqrt
        if '\\sqrt' in string:
            splits = string.split('\\sqrt')
            new_string = splits[0]
            for split in splits[1:]:
                if split[0] != '{':
                    a = split[0]
                    new_substr = '\\sqrt{' + a + '}' + split[1:]
                else:
                    new_substr = '\\sqrt' + split
                new_string += new_substr
            string = new_string
        string = string.replace(' ', '')
        # Fix fracs
        substrs = string.split('\\frac')
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += '\\frac'
                if len(substr) > 0 and substr[0] == '{':
                    new_str += substr
                else:
                    if len(substr) >= 2:
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
        # Fix a/b
        if len(string.split('/')) == 2:
            a, b = string.split('/')
            try:
                a = int(a)
                b = int(b)
                if string == '{}/{}'.format(a, b):
                    string = '\\frac{' + str(a) + '}{' + str(b) + '}'
            except (ValueError, AssertionError):
                pass
        if string == '0.5':
            string = '\\frac{1}{2}'
        return string
    
    def is_equiv(self, str1: str, str2: str) -> bool:
        """判断两个答案是否等价"""
        if str1 is None and str2 is None:
            return True
        if str1 is None or str2 is None:
            return False
        
        try:
            ss1 = self._strip_string(str1)
            ss2 = self._strip_string(str2)
            if ss1 == ss2:
                return True
            ss1 = self.normalize_final_answer(ss1)
            ss2 = self.normalize_final_answer(ss2)
            if ss1 == ss2:
                return True
        except Exception:
            pass
        
        try:
            ss1 = self.normalize_final_answer(str1)
            ss2 = self.normalize_final_answer(str2)
            if ss1 == ss2:
                return True
        except Exception:
            pass
        
        return str1 == str2
    
    def generate_solution(self, problem: str, max_retries: int = 3) -> str:
        """生成解决方案"""
        prompt = self.build_prompt(problem)
        last_exception: Optional[Exception] = None
        
        for attempt in range(max_retries):
            try:
                # 添加停止条件，避免模型生成多个问题
                # 当模型生成新的"Problem:"时停止
                stop_sequences = ["\n\nProblem:", "Problem:\n"]
                
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
                # 查找第一个"Problem:"出现的位置
                first_problem_idx = generated_text.find('\n\nProblem:')
                if first_problem_idx > 0:
                    # 只保留第一个问题之前的内容
                    generated_text = generated_text[:first_problem_idx].strip()
                
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
            problem = problem_data['problem']
            reference_answer = problem_data.get('solution', '')  # 已经是提取的答案
            
            generated_solution = self.generate_solution(problem)
            
            # 提取答案（优先提取boxed答案，默认strip_double_curly_brace=True）
            predicted_answer = self.extract_boxed_answer(generated_solution, strip_double_curly_brace=True)
            if predicted_answer is None:
                # 如果没有boxed答案，尝试其他提取方法
                # 查找"The final answer is"或"Final Answer"
                final_answer_patterns = [
                    r'The final answer is\s*\$?([^$\.]+)',
                    r'Final Answer[:\s]+([^\.\n]+)',
                    r'final answer[:\s]+([^\.\n]+)',
                ]
                for pattern in final_answer_patterns:
                    match = re.search(pattern, generated_solution, re.IGNORECASE)
                    if match:
                        predicted_answer = match.group(1).strip()
                        # 移除可能的$符号
                        predicted_answer = predicted_answer.strip('$').strip()
                        break
                
                # 如果还是没找到，尝试使用math_postprocess的逻辑
                if predicted_answer is None or not predicted_answer:
                    # 参考OpenCompass的math_postprocess
                    for maybe_ans in generated_solution.split('.'):
                        if 'final answer' in maybe_ans.lower():
                            predicted_answer = self.normalize_final_answer(maybe_ans)
                            break
                    if not predicted_answer:
                        predicted_answer = self.normalize_final_answer(generated_solution.split('.')[0])
            
            # 清理预测答案
            if predicted_answer:
                predicted_answer = predicted_answer.strip()
                # 移除可能的boxed标记（但要小心，不要移除LaTeX中的花括号）
                # 只移除开头的\boxed{和结尾的单独}
                if predicted_answer.startswith('\\boxed{'):
                    predicted_answer = predicted_answer[7:]
                if predicted_answer.endswith('}') and predicted_answer.count('{') == predicted_answer.count('}'):
                    predicted_answer = predicted_answer[:-1]
            
            # 判断是否等价（reference_answer已经是提取的答案）
            is_correct = self.is_equiv(predicted_answer, reference_answer)
            
            with self._lock:
                if is_correct:
                    self._correct_count += 1
                self._total_count += 1
            
            return {
                'problem': problem,
                'predicted_answer': predicted_answer,
                'reference_answer': reference_answer,
                'generated_solution': generated_solution,
                'is_correct': is_correct,
            }
        except Exception as e:
            print(f"\n评估问题时出错: {e}")
            return {
                'problem': problem_data.get('problem', ''),
                'predicted_answer': '',
                'reference_answer': problem_data.get('solution', ''),
                'is_correct': False,
                'error': str(e)
            }
    
    def evaluate_dataset(self, data: List[Dict], max_samples: int = None, seed: int = 42) -> Dict:
        """评估整个数据集"""
        import random
        if max_samples:
            random.seed(seed)
            data = random.sample(data, min(max_samples, len(data)))
        
        total = len(data)
        results = []
        
        self._correct_count = 0
        self._total_count = 0
        
        print(f"开始评估 {total} 个问题...")
        if max_samples:
            print(f"随机采样: {max_samples} 个样本（seed={seed}）")
        print(f"并发设置: {self.max_workers} 个问题并发")
        print(f"评估方法: {self.shot_num}-shot 生成")
        
        pbar = tqdm(total=total, desc="评估进度")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_data = {
                executor.submit(self.evaluate_single_problem, q): q 
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
                    
                    if current_total % 50 == 0 and current_total > 0:
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
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'shot_num': self.shot_num,
            'seed': seed if max_samples else None,
            'max_samples': max_samples,
            'results': results
        }


def find_math_data_path():
    """查找Math数据集路径（只从datasets目录查找）"""
    # 只从datasets目录查找
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'math')
    json_path = os.path.join(datasets_dir, 'math.json')
    
    if os.path.exists(json_path):
        return json_path
    
    return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MATH evaluation script")
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
        default=4,
        help="Number of few-shot examples (default: 4)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Random sample size, None means evaluate all data (default: None)",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="Dataset JSON file path (default: datasets/math/math.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    
    args = parser.parse_args()
    
    # 自动查找数据路径，如果找不到则使用默认路径
    JSON_PATH = args.dataset_file or find_math_data_path()
    if JSON_PATH is None:
        JSON_PATH = "/volume/ai-infra/zkjia/projects/opencompass/data/math/math.json"  # 默认路径
        print(f"⚠️  未找到Math数据集，将使用默认路径: {JSON_PATH}")
        print("   如果文件不存在，请使用--dataset-file参数指定路径")
    
    # 创建日志目录结构：logs/{model}/
    log_dir = os.path.join("logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    
    shot_desc = f"{args.shot_num}-shot" if args.shot_num > 0 else "0-shot"
    output_filename = f"math_{args.shot_num}shot.json"
    if args.max_samples:
        output_filename = f"math_{args.shot_num}shot_{args.max_samples}samples.json"
    OUTPUT_PATH = os.path.join(log_dir, output_filename)
    
    print("="*70)
    print(f"Math 评估 - {args.model} - {shot_desc} 生成方法")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"API: completions")
    print(f"方法: {shot_desc} few-shot prompting + 数学等价性验证")
    print("="*70)
    
    evaluator = MathEvaluator(
        args.base_url,
        args.model,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        shot_num=args.shot_num
    )
    
    print(f"\n加载数据: {JSON_PATH}")
    try:
        data = evaluator.load_math_data(JSON_PATH)
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

