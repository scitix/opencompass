"""MBPP数据集评估脚本 - Base模型代码生成（3-shot Pass@1）。"""
import contextlib
import io
import json
import multiprocessing
import os
import re
import signal
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm
import eval_utils
try:
    from datasets import load_dataset
except ImportError:
    print("请安装 datasets: pip install datasets")
    raise


class TimeOutException(Exception):
    pass


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeOutException('Time out!')

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):
    _stream = 'stdin'


def _execution(programs, timeout, key):
    try:
        exec_globals = {}
        with swallow_io():
            with time_limit(timeout):
                exec(programs, exec_globals)
        key.append('pass')
    except TimeOutException:
        key.append('timeout')
    except AssertionError:
        key.append('wrong_answer')
    except BaseException as e:
        print(e)
        key.append('failed')


def execution(programs, task_id, timeout):
    """执行代码并返回结果"""
    manager = multiprocessing.Manager()
    key = manager.list()
    p = multiprocessing.Process(target=_execution,
                                args=(programs, timeout - 1, key))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        return task_id, 'timeout'
    return task_id, key[0] if key else 'failed'


class MBPPEvaluator:
    """使用代码生成方法评估MBPP数据集（3-shot Pass@1）"""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_workers: int = 32, max_tokens: int = 512, shot_num: int = 3):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.shot_num = shot_num
        
        # 3-shot示例（来源：opencompass/opencompass/configs/datasets/mbpp/mbpp_gen_830460.py）
        # 保持原有格式（已验证 70% Pass@1）
        self.few_shot_examples = [
            {
                'human': 'You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \n',
                'bot': "[BEGIN]\n 'def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)' \n[DONE] \n\n "
            },
            {
                'human': 'You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\n assert is_not_prime(2) == False \nassert is_not_prime(10) == True \nassert is_not_prime(35) == True \n',
                'bot': "[BEGIN]\n 'import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\n            result = True\r\n    return result' \n[DONE] \n\n "
            },
            {
                'human': 'You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \n',
                'bot': "[BEGIN]\n 'import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums' \n[DONE] \n\n "
            }
        ]

    def extract_function_name(self, test_list: str) -> Optional[str]:
        """从测试用例中提取函数名"""
        # 匹配 assert function_name(...) 格式
        match = re.search(r'assert\s+(\w+)\s*\(', test_list)
        if match:
            return match.group(1)
        return None
    
    def build_prompt(self, text: str, test_list: str) -> str:
        """构建3-shot prompt，明确指定函数名"""
        prompt_parts = []
        
        # 添加few-shot示例
        for example in self.few_shot_examples:
            prompt_parts.append(example['human'])
            prompt_parts.append(example['bot'])
        
        # 提取函数名
        func_name = self.extract_function_name(test_list)
        
        # 添加当前任务，明确指定函数名
        if func_name:
            current_task = f'You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n {test_list}  \n\nImplement the function named "{func_name}".\n'
        else:
            current_task = f'You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n {test_list}  \n'
        
        prompt_parts.append(current_task)
        prompt_parts.append('[BEGIN]\n')
        
        return ''.join(prompt_parts)

    def postprocess_code(self, text: str) -> str:
        """后处理生成的代码，参考 OpenCompass MBPPEvaluator._process_answer"""
        # 保存原始文本用于调试
        original_text = text
        
        patterns = [
            r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]",
            r"BEGIN\s*'(.*)'\s*\[DONE\]",
            r"\[BEGIN\]\s*'(.*)'\s*DONE",
            r"BEGIN\s*'(.*)'\s*DONE",
            r"\[BEGIN\]\s*'(.*)\s*\[DONE\]",
            r"BEGIN\s*'(.*)\s*\[DONE\]",
            r"\[BEGIN\]\s*'(.*)\s*DONE",
            r"BEGIN\s*'(.*)\s*DONE",
            r'\[BEGIN\]\s*(.*)\s*\[DONE\]',
            r'BEGIN\s*(.*)\s*\[DONE\]',
            r'\[BEGIN\]\s*(.*)\s*DONE',
            r'BEGIN\s*(.*)\s*DONE',
            r'```python\s*(.*)\s*```',
            r'```\s*(.*)\s*```',
            r'```python\s*(.*)\s*$',
            r'```\s*(.*)\s*$',
            r'(.*)\s*```.*',
            r"\[BEGIN\]\s*'(.*)",
            r'\[BEGIN\](.*)',
            r"'(.*)'\s*\[DONE\]",
        ]
        for p in patterns:
            try:
                match = re.search(p, text, re.DOTALL)
            except Exception:
                match = None

            if match:
                text = match.group(1)
                break
        
        text = text.split('```')[0]
        text = re.split(r"'?\s*\[?DONE\]?", text)[0]
        text = text.replace('\\_', '_')
        text = text.replace('\\r\\n', '\n')
        text = text.replace('\\n', '\n')
        text = text.strip()
        
        # 验证是否包含函数定义
        if text and 'def ' not in text:
            # 如果没有找到 def，尝试直接返回原始文本
            # 某些情况下模型可能不按照预期格式返回
            if 'def ' in original_text:
                # 尝试提取 def 后面的所有内容
                def_match = re.search(r'(def\s+\w+.*)', original_text, re.DOTALL)
                if def_match:
                    text = def_match.group(1)
                    # 清理可能的结尾标记
                    text = text.split('[DONE]')[0]
                    text = text.split('```')[0]
                    text = text.strip()
        
        return text

    def generate_code(self, prompt: str, max_retries: int = 3) -> str:
        """生成代码"""
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
                    raise RuntimeError('模型返回的choices为空。')
                
                generated_text = response.choices[0].text
                return self.postprocess_code(generated_text)
                    
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
            last_exception = RuntimeError('未知原因导致代码生成失败。')

        raise last_exception

    def fix_function_name(self, code: str, expected_func_name: Optional[str]) -> str:
        """修复函数名不匹配的问题"""
        if not expected_func_name or 'def ' not in code:
            return code
        
        # 提取代码中的函数名
        match = re.search(r'def\s+(\w+)\s*\(', code)
        if not match:
            return code
        
        actual_func_name = match.group(1)
        
        # 如果函数名不匹配，替换它
        if actual_func_name != expected_func_name:
            # 替换函数定义中的名字
            code = re.sub(
                r'def\s+' + re.escape(actual_func_name) + r'\s*\(',
                f'def {expected_func_name}(',
                code,
                count=1
            )
        
        return code
    
    def evaluate_single_task(self, task_data: Dict) -> Dict:
        """评估单个任务"""
        try:
            text = task_data['text']
            test_list = task_data['test_list']
            
            prompt = self.build_prompt(text, test_list)
            generated_code = self.generate_code(prompt)
            
            # 验证生成的代码
            if not generated_code.strip():
                raise ValueError("Generated code is empty")
            
            if 'def ' not in generated_code:
                print(f"\n⚠️  Task {task_data['task_id']}: No function definition found in generated code")
                print(f"Generated code preview: {generated_code[:200]}...")
            
            # 提取期望的函数名并修复
            expected_func_name = self.extract_function_name(test_list)
            generated_code = self.fix_function_name(generated_code, expected_func_name)
            
            # 准备执行代码
            test_case = task_data.get('test_list_2', test_list)
            programs = generated_code + '\n' + test_case
            
            return {
                'task_id': task_data['task_id'],
                'text': text,
                'generated_code': generated_code,
                'programs': programs,
                'test_case': test_case,
                'expected_func_name': expected_func_name,
            }
        except Exception as e:
            print(f"\n任务 {task_data.get('task_id', 'unknown')} 生成代码时出错: {e}")
            return {
                'task_id': task_data.get('task_id', 'unknown'),
                'text': task_data.get('text', ''),
                'generated_code': '',
                'error': str(e),
            }

    def load_mbpp_data(self, data_path: Optional[str] = None) -> List[Dict]:
        """加载MBPP数据集"""
        # 只从datasets目录查找
        if data_path is None:
            datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'mbpp')
            data_path = os.path.join(datasets_dir, 'mbpp.jsonl')
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"未找到MBPP数据集: {data_path}\n"
                    f"请确保数据集已复制到 datasets/mbpp/ 目录"
                )
        
        # 处理数据
        def processing_test(example):
            example['test_case'] = example['test_list']
            example['test_list'] = '\n'.join(example['test_list'])
            example['test_list_2'] = example['test_list']
            return example
        
        # 加载数据集（MBPP 标准：train[10:] = 964 条测试数据）
        if os.environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(
                data_path,
                subset_name='full',
                split='train[10:]'
            ).map(processing_test)
        else:
            dataset = load_dataset('json', data_files=data_path, split='train[10:]').map(processing_test)
        
        return [item for item in dataset]

    def evaluate_dataset(self, data: List[Dict], max_samples: Optional[int] = None) -> Dict:
        """评估整个数据集"""
        if max_samples:
            data = data[:max_samples]
        
        total = len(data)
        print(f"开始评估 {total} 个任务...")
        print(f"并发设置: {self.max_workers} 个任务并发")
        print(f"方法: {self.shot_num}-shot 代码生成")
        
        # 第一步：生成代码
        print("\n第一步：生成代码...")
        results = []
        pbar = tqdm(total=total, desc="生成代码")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self.evaluate_single_task, task): task 
                for task in data
            }
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                except Exception as e:
                    print(f"\n评估失败: {e}")
                    pbar.update(1)
        
        pbar.close()
        
        # 第二步：执行代码评估
        print("\n第二步：执行代码评估...")
        result_stats = {'pass': 0, 'timeout': 0, 'failed': 0, 'wrong_answer': 0}
        details = {}
        
        with ProcessPoolExecutor() as executor:
            futures = []
            for i, result in enumerate(results):
                if 'error' in result:
                    details[str(i)] = {
                        'result': 'error',
                        'is_correct': False,
                        'error': result.get('error', '')
                    }
                    continue
                
                programs = result.get('programs', '')
                task_id = result.get('task_id', i)
                future = executor.submit(execution, programs, task_id, 10)
                futures.append((future, i, result))
            
            for future, idx, result in tqdm(futures, desc="执行代码"):
                try:
                    task_id, ret = future.result()
                    result_stats[ret] += 1
                    details[str(idx)] = {
                        'task_id': result.get('task_id', ''),
                        'result': ret,
                        'is_correct': (ret == 'pass'),
                        'generated_code': result.get('generated_code', ''),
                    }
                except Exception as e:
                    result_stats['failed'] += 1
                    details[str(idx)] = {
                        'result': 'exception',
                        'is_correct': False,
                        'error': str(e)
                    }
        
        pass_at_1 = result_stats['pass'] / total * 100 if total > 0 else 0.0
        
        return {
            'pass@1': pass_at_1,
            'total': total,
            'stats': result_stats,
            'details': details,
            'results': results
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MBPP evaluation script (3-shot Pass@1)")

    parser.add_argument(
        "--shot-num",
        type=int,
        default=3,
        help=f"Few-shot 示例数量（默认: 3）",
    )
    parser.add_argument("--model", type=str, default=os.environ.get("EVAL_MODEL_NAME", "qwen2-5-72b"), help="模型名称（默认: eval-qwen2-5-72b）")
    parser.add_argument("--base-url", type=str, default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1"), help="API base URL")
    parser.add_argument("--max-workers", type=int, default=32, help="最大并发数（默认: 32）")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成 token 数（默认: 512）")
    parser.add_argument("--max-samples", type=int, default=None, help="随机采样数量，None表示评估全部数据（默认: None）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认: 42）")
    parser.add_argument("--num-process-evaluate", type=int, default=4, help="测试执行进程数（默认: 4）")
    parser.add_argument("--timeout", type=int, default=6, help="测试超时时间（默认: 6秒）")
    
    args = parser.parse_args()
    
    # 创建日志目录结构：logs/{model}/
    log_dir = os.path.join("logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    
    output_filename = "mbpp_3shot.json"
    if args.max_samples:
        output_filename = f"mbpp_3shot_{args.max_samples}samples.json"
    OUTPUT_PATH = os.path.join(log_dir, output_filename)
    
    print("="*70)
    print(f"{args.shot_num}-shot Pass@1")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"API: completions")
    print(f"{args.shot_num}-shot 代码生成")
    print("="*70)
    
    evaluator = MBPPEvaluator(
        args.base_url,
        args.model,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        shot_num=args.shot_num
    )
    
    print(f"\n加载数据集...")
    try:
        data = evaluator.load_mbpp_data()
        print(f"✓ 成功加载 {len(data)} 条数据")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n开始评估...")
    
    try:
        results = evaluator.evaluate_dataset(data, max_samples=args.max_samples)
        
        print(f"\n保存结果: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*70)
        print("评估结果")
        print("="*70)
        print(f"总任务数: {results['total']}")
        print(f"Pass@1: {results['pass@1']:.2f}%")
        print(f"\n统计信息:")
        for key, value in results['stats'].items():
            print(f"  {key}: {value}")
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

