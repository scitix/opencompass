"""LiveCodeBench-Base数据集评估脚本 - Base模型代码生成（3-shot Pass@1）。"""
import ast
import json
import multiprocessing
import os
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from decimal import Decimal
from io import StringIO
from types import ModuleType
from typing import Dict, List, Optional
from unittest.mock import mock_open, patch
from openai import OpenAI
from tqdm import tqdm
import eval_utils

try:
    from datasets import load_dataset
except ImportError:
    print("请安装 datasets: pip install datasets")
    raise


# LiveCodeBench导入字符串
LIVECODEBENCH_IMPORT_STRING = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"


def _unsafe_execute_fn_call(fn, single_input: str, expect_output: str) -> tuple[bool, str]:
    """执行函数调用并比较输出（模块级函数，用于多进程）"""
    try:
        args = [json.loads(line) for line in single_input.split("\n")]
        exp_outputs = json.loads(expect_output)

        outputs = fn(*args)
        # 不惩罚模型如果它产生元组而不是列表
        if isinstance(outputs, tuple):
            outputs = list(outputs)

        if outputs != exp_outputs:
            return False, f"output {outputs} != expect {exp_outputs}"
        return True, ""
    except Exception as e:
        return False, f"[{type(e).__name__}] {e}"


def _get_function(compiled_sol, fn_name: str):
    """从编译后的模块中获取函数（模块级函数，用于多进程）"""
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return None


def _compile_code(code: str):
    """编译代码为模块（模块级函数，用于多进程）"""
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            # LeetCode风格的解决方案
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol
        assert compiled_sol is not None
    except Exception:
        return None
    return compiled_sol


def _make_function(code: str) -> str:
    """将代码包装成函数（用于stdin/stdout模式）
    
    如果代码是函数定义，提取函数体并转换return为print
    """
    import ast
    import re
    
    # 如果代码已经是一个函数定义，提取函数体
    if code.strip().startswith('def '):
        try:
            # 使用AST解析函数定义
            tree = ast.parse(code)
            if tree.body and isinstance(tree.body[0], ast.FunctionDef):
                func_def = tree.body[0]
                
                # 提取函数体，将return转换为print
                body_lines = []
                for node in func_def.body:
                    if isinstance(node, ast.Return):
                        # 将return转换为print
                        if node.value:
                            # 获取return的值
                            return_value = ast.unparse(node.value)
                            body_lines.append(f"    print({return_value})")
                        else:
                            body_lines.append("    print()")
                    else:
                        # 其他语句保持原样，但需要调整缩进
                        stmt_code = ast.unparse(node)
                        # 去除原有的函数体缩进（通常是4个空格）
                        if stmt_code.startswith('    '):
                            stmt_code = stmt_code[4:]
                        body_lines.append(f"    {stmt_code}")
                
                body_code = '\n'.join(body_lines)
                return f"def wrapped_function():\n{body_code}"
        except:
            # AST解析失败，使用简单的字符串处理
            pass
        
        # 简单的字符串处理：提取函数体，将return转换为print
        lines = code.split('\n')
        func_def_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                func_def_line = i
                break
        
        if func_def_line is not None:
            body_lines = []
            base_indent = None
            for i in range(func_def_line + 1, len(lines)):
                line = lines[i]
                stripped = line.strip()
                if not stripped:
                    continue
                
                if base_indent is None:
                    base_indent = len(line) - len(line.lstrip())
                
                # 去除函数体缩进
                if base_indent > 0 and line.startswith(' ' * base_indent):
                    body_line = line[base_indent:]
                else:
                    body_line = line
                
                # 将return转换为print
                if stripped.startswith('return '):
                    return_value = stripped[7:].rstrip(';')
                    body_lines.append(f"    print({return_value})")
                elif stripped == 'return':
                    body_lines.append("    print()")
                else:
                    body_lines.append(f"    {body_line}")
            
            body_code = '\n'.join(body_lines)
            return f"def wrapped_function():\n{body_code}"
    
    # 否则直接包装成函数
    return f"def wrapped_function():\n    " + code.replace('\n', '\n    ')


def _call_method(method, inputs: str):
    """调用方法并模拟stdin输入（模块级函数，用于多进程）"""
    from io import StringIO
    from unittest.mock import patch, mock_open
    
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    
    inputs_line_iterator = iter(inputs.split("\n"))
    
    # 创建自定义stdin mock
    class MockStdinWithBuffer:
        def __init__(self, inputs: str):
            self.inputs = inputs
            self._stringio = StringIO(inputs)
        
        def read(self, *args):
            return self.inputs
        
        def readline(self, *args):
            try:
                return next(inputs_line_iterator) + "\n"
            except StopIteration:
                return ""
        
        def readlines(self, *args):
            return inputs.split("\n")
    
    mock_stdin = MockStdinWithBuffer(inputs)
    
    with patch("builtins.open", mock_open(read_data=inputs)):
        with patch("sys.stdin", mock_stdin):
            with patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator, "") + "\n"):
                with patch("sys.stdin.readlines", lambda *args: inputs.split("\n")):
                    with patch("sys.stdin.read", lambda *args: inputs):
                        try:
                            return method()
                        except SystemExit:
                            pass
    return None


class _Capturing(list):
    """捕获stdout输出"""
    def __enter__(self):
        import sys
        from io import StringIO
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        import sys
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


def _get_stripped_lines(text: str) -> List[str]:
    """获取去除空白后的行列表"""
    return [line.rstrip() for line in text.rstrip().split('\n')]


def _unsafe_execute_stdio(method, single_input: str, expect_output: str) -> tuple[bool, str]:
    """执行stdin/stdout模式的代码（模块级函数，用于多进程）"""
    with _Capturing() as captured_output:
        try:
            _call_method(method, single_input)
        except Exception as e:
            return False, f"[{type(e).__name__}] {e}"
    
    output = captured_output[0] if captured_output else ""
    stripped_output_lines = _get_stripped_lines(output)
    stripped_expect_outputs_lines = _get_stripped_lines(expect_output)
    
    if len(stripped_output_lines) != len(stripped_expect_outputs_lines):
        return False, f"output line count mismatch: {len(stripped_output_lines)} != {len(stripped_expect_outputs_lines)}"
    
    for out_line, exp_line in zip(stripped_output_lines, stripped_expect_outputs_lines):
        if out_line == exp_line:
            continue
        # 如果输出不匹配，尝试数值比较（处理浮点数精度问题）
        try:
            out_val = float(out_line)
            exp_val = float(exp_line)
            if abs(out_val - exp_val) < 1e-9:
                continue
        except:
            pass
        return False, f"output line '{out_line}' != expect '{exp_line}'"
    
    return True, ""


def _unsafe_execute(code: str, inputs: List[str], expect_outputs: List[str], fn_name: Optional[str]) -> tuple[bool, str]:
    """执行代码并验证所有测试用例（模块级函数，用于多进程）
    
    支持两种模式：
    1. 函数调用模式（fn_name不为None）：调用指定函数并验证返回值
    2. stdin/stdout模式（fn_name为None）：模拟stdin输入，捕获stdout输出并验证
    """
    if len(inputs) != len(expect_outputs):
        return False, "failed: number of inputs and outputs mismatch"

    if fn_name is not None:
        # 函数调用模式
        code_to_compile = LIVECODEBENCH_IMPORT_STRING + "\n\n" + code
        compiled_sol = _compile_code(code_to_compile)
        if compiled_sol is None:
            return False, "failed: compile error"
        fn = _get_function(compiled_sol, fn_name)
        if fn is None:
            return False, "failed: no function defined"

        # 执行所有测试用例
        for single_input, single_output in zip(inputs, expect_outputs):
            ok, msg = _unsafe_execute_fn_call(fn, single_input, single_output)
            if not ok:
                return False, f"failed: {msg}"
        return True, ""
    else:
        # stdin/stdout模式：将代码包装成函数，然后执行
        code_to_compile = LIVECODEBENCH_IMPORT_STRING + "\n\n" + _make_function(code)
        compiled_sol = _compile_code(code_to_compile)
        if compiled_sol is None:
            return False, "failed: compile error"
        
        # 获取包装后的函数
        fn = getattr(compiled_sol, "wrapped_function", None)
        if fn is None:
            return False, "failed: no function defined"

        # 执行所有测试用例
        for single_input, single_output in zip(inputs, expect_outputs):
            ok, msg = _unsafe_execute_stdio(fn, single_input, single_output)
            if not ok:
                return False, f"failed: {msg}"
        return True, ""


def _subprocess_target(q, code: str, inputs: List[str],
                      expect_outputs: List[str], fn_name: Optional[str]):
    """子进程执行目标（模块级函数，用于多进程）"""
    try:
        ok, msg = _unsafe_execute(code, inputs, expect_outputs, fn_name)
        q.put((ok, msg))
    except Exception as e:
        import traceback
        error_msg = f"failed: [{type(e).__name__}] {e}\n{traceback.format_exc()}"
        q.put((False, error_msg))


class LiveCodeBenchEvaluator:
    """使用代码生成方法评估LiveCodeBench数据集（3-shot Pass@1）"""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_workers: int = 32, max_tokens: int = 1024,
                 num_process_evaluate: int = 4, timeout: int = 6,
                 shot_num: int = 3, cot: bool = False):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.num_process_evaluate = num_process_evaluate
        self.timeout = timeout
        self.shot_num = shot_num
        self.cot = cot
        
        # 3-shot示例（注意：LiveCodeBench在opencompass中没有官方的few-shot示例）
        # 这些示例是根据LiveCodeBench的prompt格式自行创建的通用示例
        # 参考格式：opencompass/opencompass/datasets/livecodebench/prompts.py
        # 如果需要使用官方few-shot，需要从LiveCodeBench数据集或论文中获取
        self.few_shot_examples = [
            {
                'question': 'Write a function to find the maximum element in a list.',
                'starter_code': 'def find_max(lst):\n    # YOUR CODE HERE',
                'answer': 'def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)'
            },
            {
                'question': 'Write a function to check if a number is prime.',
                'starter_code': 'def is_prime(n):\n    # YOUR CODE HERE',
                'answer': 'def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True'
            },
            {
                'question': 'Write a function to reverse a string.',
                'starter_code': 'def reverse_string(s):\n    # YOUR CODE HERE',
                'answer': 'def reverse_string(s):\n    return s[::-1]'
            }
        ]

    def build_prompt(self, question_content: str, format_prompt: str) -> str:
        """构建3-shot prompt（支持CoT和non-CoT）"""
        prompt_parts = []
        
        # 添加few-shot示例
        for example in self.few_shot_examples:
            prompt_parts.append(f"### Question:\n{example['question']}\n\n")
            if example.get('starter_code'):
                prompt_parts.append(f"### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n")
                prompt_parts.append(f"```python\n{example['starter_code']}\n```\n\n")
            else:
                prompt_parts.append("### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n")
                prompt_parts.append("```python\n# YOUR CODE HERE\n```\n\n")
            
            # 根据CoT设置添加答案部分
            if self.cot:
                prompt_parts.append("### Answer:\n")
                prompt_parts.append("Let's think step by step first, provide a concise plan:\n")
                prompt_parts.append("```text\n1. Analyze the problem\n2. Design the solution\n3. Implement the code\n```\n")
                prompt_parts.append("Then provide the final solution: (use the provided format with backticks)\n\n")
                prompt_parts.append(f"```python\n{example['answer']}\n```\n\n\n")
            else:
                prompt_parts.append("### Answer: (use the provided format with backticks)\n\n")
                prompt_parts.append(f"```python\n{example['answer']}\n```\n\n\n")
        
        # 添加当前任务
        prompt_parts.append(f"### Question:\n{question_content}\n\n")
        prompt_parts.append(format_prompt)
        
        # 根据CoT设置添加答案部分
        if self.cot:
            prompt_parts.append("### Answer:\n")
            prompt_parts.append("Let's think step by step first, provide a concise plan:\n")
            prompt_parts.append("```text\n1. ...\n2. ...\n3. ...\n```\n")
            prompt_parts.append("Then provide the final solution: (use the provided format with backticks)\n\n")
        else:
            prompt_parts.append("### Answer: (use the provided format with backticks)\n\n")
        
        return ''.join(prompt_parts)

    def postprocess_code(self, text: str) -> str:
        """后处理生成的代码，提取代码块"""
        # 尝试提取代码块
        blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
        if len(blocks) >= 1:
            text = blocks[0]
        else:
            # 如果没有找到代码块，尝试提取第一个代码块
            blocks = re.findall(r'```\n(.*?)```', text, re.DOTALL)
            if len(blocks) >= 1:
                text = blocks[0]
        
        # 清理代码
        text = text.strip()
        # 移除可能的函数签名重复
        lines = text.split('\n')
        if len(lines) > 1 and lines[0].strip().startswith('def'):
            # 检查是否有重复的函数定义
            def_count = sum(1 for line in lines if line.strip().startswith('def'))
            if def_count > 1:
                # 保留第一个函数定义及其内容
                result_lines = []
                in_first_function = True
                for line in lines:
                    if line.strip().startswith('def'):
                        if in_first_function:
                            result_lines.append(line)
                            in_first_function = False
                        else:
                            break
                    elif not in_first_function:
                        break
                    else:
                        result_lines.append(line)
                text = '\n'.join(result_lines)
        
        return text.strip()

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

    def evaluate_single_task(self, task_data: Dict) -> Dict:
        """评估单个任务"""
        try:
            question_content = task_data['question_content']
            format_prompt = task_data.get('format_prompt', '')
            
            prompt = self.build_prompt(question_content, format_prompt)
            generated_code = self.generate_code(prompt)
            
            return {
                'question_id': task_data.get('question_id', ''),
                'question_content': question_content,
                'generated_code': generated_code,
                'evaluation_sample': task_data.get('evaluation_sample', ''),
            }
        except Exception as e:
            print(f"\n任务 {task_data.get('question_id', 'unknown')} 生成代码时出错: {e}")
            return {
                'question_id': task_data.get('question_id', 'unknown'),
                'question_content': task_data.get('question_content', ''),
                'generated_code': '',
                'error': str(e),
            }

    def _kill_proc(self, p: multiprocessing.Process):
        """终止进程"""
        if not p:
            return
        if p.is_alive():
            p.terminate()
            p.join(0.1)
        if p.is_alive():
            try:
                os.kill(p.pid, signal.SIGKILL)
            except Exception:
                pass
            p.join(0.1)
        try:
            p.close()
        except Exception:
            pass


    def codegen_check_correctness(self, sample, generation, timeout, debug=True):
        """检查代码正确性（本地执行，不依赖外部服务）"""
        try:
            # 解析测试用例
            test_data = json.loads(sample['input_output'])
            inputs = test_data.get('inputs', [])
            outputs = test_data.get('outputs', [])
            fn_name = test_data.get('fn_name')
            
            if not generation or not generation.strip():
                return [0], {}  # 空代码视为失败
            
            # 使用多进程执行代码（隔离执行环境）
            # 使用fork模式而不是spawn，spawn在某些环境下可能有问题
            try:
                ctx = multiprocessing.get_context("fork")
            except ValueError:
                # 如果fork不可用（如Windows），回退到spawn
                ctx = multiprocessing.get_context("spawn")
            q = ctx.Queue()  # 使用Queue而不是SimpleQueue，因为Queue.get()支持timeout参数
            p = ctx.Process(
                target=_subprocess_target,
                args=(q, generation, inputs, outputs, fn_name)
            )
            p.start()

            try:
                # 等待结果，带超时
                import threading
                result_container = [None]
                exception_container = [None]

                def get_result():
                    try:
                        result_container[0] = q.get(timeout=timeout)
                    except Exception as e:
                        exception_container[0] = e

                thread = threading.Thread(target=get_result)
                thread.daemon = True
                thread.start()
                thread.join(timeout=timeout + 1)

                # 等待进程结束（不阻塞太久）
                p.join(timeout=0.5)

                if thread.is_alive():
                    # 超时
                    if debug:
                        print(f"线程超时，进程状态: alive={p.is_alive()}, exitcode={p.exitcode}")
                    self._kill_proc(p)
                    return [0], {}  # 超时视为失败

                if exception_container[0]:
                    if debug:
                        print(f"队列获取异常: {exception_container[0]}")
                        print(f"进程状态: alive={p.is_alive()}, exitcode={p.exitcode}")
                    raise exception_container[0]

                if result_container[0] is None:
                    if debug:
                        print(f"结果容器为空（可能超时），进程状态: alive={p.is_alive()}, exitcode={p.exitcode}")
                    self._kill_proc(p)
                    return [0], {}

                ok, msg = result_container[0]

                # 转换结果格式：[1]表示通过，[0]表示失败
                if ok:
                    return [1], {}
                else:
                    if debug:
                        print(f"测试失败: {msg}")
                    return [0], {}

            except Exception as e:
                self._kill_proc(p)
                if debug:
                    import traceback
                    print(f"执行代码时出错: {e}")
                    traceback.print_exc()
                return [0], {}
            finally:
                try:
                    q.close()
                except Exception:
                    pass
            
        except Exception as e:
            if debug:
                print(f"检查代码正确性时出错: {e}")
            return [0], {}

    def evaluate_generations_by_problem(self, args):
        """评估单个问题的所有生成"""
        problem_generations, sample, debug, timeout = args
        res = []
        metadata = []
        for o_idx, o in enumerate(problem_generations):
            curr_res = [-2]
            try:
                curr_res, curr_metadata = self.codegen_check_correctness(
                    sample, o, timeout=timeout, debug=debug)
                if debug:
                    print(f'\nSuccessful compilation of task {o_idx}!')
                fixed = []
                for e in curr_res:
                    if isinstance(e, bool):
                        e = 1 if e else 0
                    fixed.append(e)
                curr_res = fixed
            except Exception as e:
                if debug:
                    print(f'Compilation failed: {repr(e)}\n')
                curr_metadata = {}
            finally:
                res.append(curr_res)
                metadata.append(curr_metadata)
        return res, metadata

    def load_livecodebench_data(self, data_path: Optional[str] = None, 
                                version_tag: str = 'release_v4',
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                dataset_dir: Optional[str] = None) -> List[Dict]:
        """加载LiveCodeBench数据集（支持日期过滤）
        
        使用 livecodebench/code_generation_lite 数据集（LiveCodeBench-Base）
        支持的版本：
        - release_v1: 2023年5月至2024年3月，400个问题
        - release_v2: 2023年5月至2024年5月，511个问题
        - release_v3: 2023年5月至2024年7月，612个问题
        - release_v4: 2023年5月至2024年9月，713个问题（包含2024-08-01数据）
        - release_v5: 2023年5月至2025年1月，880个问题（包含完整目标日期范围）
        
        日期过滤范围：2024-08-01 至 2024-11-01
        默认使用 release_v4（包含8月和9月数据）
        """
        # 优先从本地文件加载
        if dataset_dir is None:
            dataset_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'livecodebench')
        
        local_file = os.path.join(dataset_dir, f'test_{version_tag}.jsonl')
        
        if data_path is None:
            # 尝试从本地文件加载
            if os.path.exists(local_file):
                print(f"从本地文件加载: {local_file}")
                dataset = load_dataset('json', data_files=local_file, split='train')
            else:
                # 尝试从 HuggingFace 加载
                print(f"本地文件不存在，从 HuggingFace 加载 (version_tag={version_tag})...")
                print(f"提示: 可以运行 'python download_datasets.py livecodebench' 下载到本地")
                try:
                    dataset = load_dataset('livecodebench/code_generation_lite', split='test', version_tag=version_tag)
                except Exception as e:
                    raise FileNotFoundError(
                        f"无法加载LiveCodeBench数据集 (version_tag={version_tag}): {e}\n"
                        f"提示: 请运行 'python download_datasets.py livecodebench' 下载数据集到本地"
                    )
        else:
            dataset = load_dataset(data_path, split='test', trust_remote_code=True)
        
        # 日期过滤（LiveCodeBench-Base要求：2024-08-01到2024-11-01）
        # 注意：contest_date字段是datetime对象，可以直接比较
        original_len = len(dataset)
        if start_date is not None or end_date is not None:
            if len(dataset) > 0:
                sample = dataset[0]
                if "contest_date" not in sample:
                    print(f"警告: 数据集中没有 'contest_date' 字段，跳过日期过滤")
                    print(f"可用字段: {list(sample.keys())}")
                else:
                    # 显示过滤前的数据量和日期范围信息
                    all_dates = [e["contest_date"] for e in dataset if "contest_date" in e]
                    if all_dates:
                        # contest_date可能是datetime对象或字符串，需要统一处理
                        def parse_date(d):
                            if isinstance(d, datetime):
                                return d
                            elif isinstance(d, str):
                                try:
                                    return datetime.fromisoformat(d.split('T')[0])
                                except:
                                    return datetime.strptime(d, "%Y-%m-%d")
                            return d
                        
                        parsed_dates = [parse_date(d) for d in all_dates]
                        min_date = min(parsed_dates)
                        max_date = max(parsed_dates)
                        print(f"过滤前数据量: {original_len}")
                        print(f"数据集日期范围: {min_date.date()} 至 {max_date.date()}")
                    
                    # contest_date可能是datetime对象或字符串，需要统一处理
                    def parse_date_for_filter(d):
                        if isinstance(d, datetime):
                            return d
                        elif isinstance(d, str):
                            try:
                                return datetime.fromisoformat(d.split('T')[0])
                            except:
                                return datetime.strptime(d, "%Y-%m-%d")
                        return d
                    
                    if start_date is not None:
                        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
                        dataset = dataset.filter(lambda e: p_start_date <= parse_date_for_filter(e["contest_date"]))
                        print(f"应用开始日期过滤 ({start_date}): {len(dataset)} 条数据")
                    if end_date is not None:
                        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
                        dataset = dataset.filter(lambda e: parse_date_for_filter(e["contest_date"]) <= p_end_date)
                        print(f"应用结束日期过滤 ({end_date}): {len(dataset)} 条数据")
                    
                    if len(dataset) == 0:
                        print(f"\n⚠️  警告: 日期过滤后没有数据！")
                        print(f"   请检查数据集是否包含 {start_date} 至 {end_date} 范围的数据")
                        print(f"   当前数据集日期范围: {min_date.date()} 至 {max_date.date()}")
        
        # 转换数据格式
        def transform(item):
            if item.get('starter_code'):
                format_prompt = f'### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n'
                format_prompt += f"```python\n{item['starter_code']}\n```\n\n"
            else:
                format_prompt = f'### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n'
                format_prompt += '```python\n# YOUR CODE HERE\n```\n\n'
            
            item['format_prompt'] = format_prompt
            
            # 加载测试用例
            public_test_cases = json.loads(item['public_test_cases'])
            try:
                import pickle
                import zlib
                import base64
                private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(item['private_test_cases'].encode('utf-8'))
                        )
                    )
                )
            except:
                private_test_cases = []
            
            metadata = json.loads(item['metadata'])
            evaluation_sample = json.dumps({
                'inputs': [t['input'] for t in public_test_cases + private_test_cases],
                'outputs': [t['output'] for t in public_test_cases + private_test_cases],
                'fn_name': metadata.get('func_name', None),
            })
            item['evaluation_sample'] = evaluation_sample
            
            return item
        
        dataset = dataset.map(transform)
        return [item for item in dataset]

    def evaluate_dataset(self, data: List[Dict], max_samples: Optional[int] = None) -> Dict:
        """评估整个数据集"""
        if max_samples:
            data = data[:max_samples]
        
        total = len(data)
        print(f"开始评估 {total} 个任务...")
        print(f"并发设置: {self.max_workers} 个任务并发")
        print(f"{self.shot_num}-shot 代码生成")
        
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
        
        samples_list = []
        generations_list = []
        result_indices = []  # 记录有效结果在results中的索引
        
        for idx, result in enumerate(results):
            if 'error' not in result:
                try:
                    # 确保evaluation_sample存在
                    if 'evaluation_sample' in result:
                        samples_list.append({'input_output': result.get('evaluation_sample', ''), 
                                           'question_id': result.get('question_id', '')})
                        generations_list.append(result.get('generated_code', ''))
                        result_indices.append(idx)
                except Exception as e:
                    print(f"处理结果{idx}时出错: {e}")
                    pass
        
        if not samples_list:
            return {
                'pass@1': 0.0,
                'total': total,
                'note': 'No valid generations to evaluate',
                'results': results
            }
        
        # 使用线程池执行评估（代码执行在子进程中完成，主线程使用线程池管理）
        eval_results = {}
        metadata_results = {}
        
        def evaluate_single(index):
            """评估单个生成的代码"""
            if index >= len(samples_list) or index >= len(generations_list):
                return [-1], {}
            sample = samples_list[index]
            generation = generations_list[index] if generations_list[index] else ""
            return self.codegen_check_correctness(sample, generation, self.timeout, debug=False)
        
        with tqdm(total=len(samples_list), desc="执行代码") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(evaluate_single, index): index
                    for index in range(len(samples_list))
                }
                
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        eval_results[index], metadata_results[index] = future.result()
                    except Exception as e:
                        print(f"\n评估索引{index}时出错: {e}")
                        eval_results[index] = [-1]
                        metadata_results[index] = {}
                    pbar.update(1)
        
        # 计算Pass@1
        # eval_results的索引对应samples_list的索引，需要映射回results
        pass_at_1s = []
        for idx in sorted(eval_results.keys()):
            execution_result = eval_results[idx]
            if execution_result and len(execution_result) > 0:
                # execution_result是[1]或[0]或[-1]格式
                # [1]表示通过，[0]表示失败，[-1]表示评估错误
                test_result = execution_result[0]
                passed = (test_result == 1)
                pass_at_1s.append(1.0 if passed else 0.0)
            else:
                pass_at_1s.append(0.0)
        
        pass_at_1 = sum(pass_at_1s) / len(pass_at_1s) * 100 if pass_at_1s else 0.0
        
        # 构建详细信息，将评估结果映射回原始results
        details = []
        for idx, result in enumerate(results):
            detail = {
                'question_id': result.get('question_id', ''),
                'generated_code': result.get('generated_code', ''),
                'correct': False,
            }
            # 如果这个result在有效结果列表中
            if idx in result_indices:
                eval_idx = result_indices.index(idx)
                if eval_idx < len(pass_at_1s):
                    detail['correct'] = pass_at_1s[eval_idx] == 1.0
                    if eval_idx in eval_results:
                        detail['eval_result'] = eval_results[eval_idx]
            details.append(detail)
        
        return {
            'pass@1': pass_at_1,
            'total': total,
            'details': details,
            'results': results
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LiveCodeBench-Base evaluation script (3-shot Pass@1)")

    parser.add_argument(
        "--shot-num",
        type=int,
        default=3,
        help=f"Few-shot 示例数量（默认: 3）",
    )
    parser.add_argument("--model", type=str, default=os.environ.get("EVAL_MODEL_NAME", "qwen2-5-72b"), help="模型名称（默认: eval-qwen2-5-72b）")
    parser.add_argument("--base-url", type=str, default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1"), help="API base URL")
    parser.add_argument("--max-workers", type=int, default=32, help="最大并发数（默认: 32）")
    parser.add_argument("--max-tokens", type=int, default=1024, help="最大生成 token 数（默认: 1024）")
    parser.add_argument("--max-samples", type=int, default=None, help="随机采样数量，None表示评估全部数据（默认: None）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认: 42）")
    parser.add_argument("--num-process-evaluate", type=int, default=4, help="测试执行进程数（默认: 4，已废弃，使用max-workers）")
    parser.add_argument("--timeout", type=int, default=6, help="测试超时时间（默认: 6秒）")
    parser.add_argument("--cot", action="store_true", help="使用Chain-of-Thought方法（默认: False，使用non-CoT）")
    parser.add_argument("--start-date", type=str, default="2024-08-01", help="数据集开始日期（格式: YYYY-MM-DD，默认: 2024-08-01）")
    parser.add_argument("--end-date", type=str, default="2024-11-01", help="数据集结束日期（格式: YYYY-MM-DD，默认: 2024-11-01）")
    parser.add_argument("--version-tag", type=str, default="release_v4", 
                       help="数据集版本标签（默认: release_v4）\n"
                            "可选: release_v1 (2023-05至2024-03, 400题), release_v2 (至2024-05, 511题),\n"
                            "      release_v3 (至2024-07, 612题), release_v4 (至2024-09, 713题),\n"
                            "      release_v5 (至2025-01, 880题)")
    
    args = parser.parse_args()
    
    # 创建日志目录结构：logs/{model}/
    log_dir = os.path.join("logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    
    # 根据CoT设置生成输出文件名
    method_suffix = "cot" if args.cot else "noncot"
    output_filename = f"livecodebench_3shot_{method_suffix}.json"
    if args.max_samples:
        output_filename = f"livecodebench_3shot_{method_suffix}_{args.max_samples}samples.json"
    OUTPUT_PATH = os.path.join(log_dir, output_filename)
    
    print("="*70)
    print(f"{args.shot_num}-shot Pass@1 ({'CoT' if args.cot else 'non-CoT'})")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"API: completions")
    print(f"{args.shot_num}-shot 代码生成")
    print(f"方法: {'Chain-of-Thought (CoT)' if args.cot else 'Direct Generation (non-CoT)'}")
    print(f"数据集版本: {args.version_tag}")
    print(f"日期范围: {args.start_date} 至 {args.end_date}")
    print("="*70)
    
    evaluator = LiveCodeBenchEvaluator(
        args.base_url, 
        args.model, 
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
        shot_num=args.shot_num,
        cot=args.cot
    )
    
    print(f"\n加载数据集...")
    print(f"数据集版本: {args.version_tag}")
    try:
        data = evaluator.load_livecodebench_data(
            version_tag=args.version_tag,
            start_date=args.start_date,
            end_date=args.end_date
        )
        print(f"✓ 成功加载 {len(data)} 条数据（日期范围: {args.start_date} 至 {args.end_date}）")
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
        if 'note' in results:
            print(f"\n注意: {results['note']}")
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

